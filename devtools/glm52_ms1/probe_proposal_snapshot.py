#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One HTTP request + offline first Attention comparison. Temporary sync tool.

--replay DIR uses saved evidence only. No model import, tensor replacement,
input logprobs, request retry, or correctness/performance PASS.
"""

# ruff: noqa: E402 -- bound BLAS threads before NumPy or its dependent imports.

import argparse
import json
import os
import urllib.request
from pathlib import Path

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from collect_acceptance_trace import PROMPT, request_json
from dspark_attention_reference import (
    attention,
    fused_prepare,
    input_norm,
    paged_slots,
    slot_contract,
)
from dspark_context_snapshot import file_hash, require, write_json
from dspark_local_reference import (
    bf16_round,
    linear,
    rope_cos_sin,
    tensor_difference,
)
from probe_context_snapshot import Arrays
from probe_quarot_vocab import Checkpoint

DEFAULT_STATE = Path("/home/tyj/glm52-ms1")


def checkpoint_rows(root, snapshot, arrays):
    """Independently resolve exact known checkpoint aliases, never suffix-match."""
    registry = []
    checkpoint = Checkpoint(snapshot["draft_path"], registry)
    for name, identity in snapshot["draft_artifact_files"].items():
        require(Path(name).name == name, "Invalid checkpoint file name")
        stat = (checkpoint.path / name).stat()
        require(
            stat.st_size == identity["bytes"]
            and stat.st_mtime_ns == identity["mtime_ns"],
            "Checkpoint file changed since snapshot",
        )
    require(
        checkpoint.config_evidence["sha256"]
        == snapshot["artifact_configs"]["draft"]["sha256"],
        "Draft config changed since snapshot",
    )
    d = snapshot["geometry"]
    ids = arrays.get("input_ids").astype(np.int64).tolist()
    reader, key = checkpoint.find_tensor(
        ("embed_tokens.weight", "model.embed_tokens.weight")
    )
    require(
        reader.matrix_info(key)["shape"][1] == d["hidden"],
        "Checkpoint embedding width differs",
    )
    embedding = reader.read_tensor(key, ids)
    shards = []
    for proj in ("q", "k", "v"):
        reader, key = checkpoint.find_tensor(
            (
                f"layers.0.self_attn.{proj}_proj.weight",
                f"model.layers.0.self_attn.{proj}_proj.weight",
            )
        )
        info = reader.matrix_info(key)
        require(
            info["shape"] == [16 * d["heads"] * d["head_dim"], d["hidden"]],
            "Checkpoint QKV shape/TP16 differs",
        )
        # Rank0 owns the first 4 full heads in each separate q/k/v matrix.
        shards.append(reader.read_tensor(key, range(d["heads"] * d["head_dim"])))
    raw = {
        "files": registry,
        "indices": checkpoint.indices,
        "scope": "Selected embedding rows and complete rank0 layer0 Q/K/V row shards only",
    }
    write_json(root / "checkpoint-reads.json", raw)
    return (
        embedding,
        np.concatenate(shards, axis=0),
        {
            "file": "checkpoint-reads.json",
            "sha256": file_hash(root / "checkpoint-reads.json"),
            "config_sha256": checkpoint.config_evidence["sha256"],
            "tensors": [
                {"key": r["key"], "shape": r["shape"], "selected_rows": len(r["rows"])}
                for f in registry
                for r in f["reads"]
            ],
        },
    )


def validate(root, config, snapshot, arrays):
    require(
        snapshot["status"] == "PROPOSAL_SNAPSHOT_COLLECTED",
        f"Observer status: {snapshot['status']}; {snapshot.get('errors')}",
    )
    require(
        snapshot["observer_sha256"]
        == config["tool_hashes"]["dspark_proposal_snapshot.py"],
        "Observer fingerprint differs",
    )
    require(arrays.total == snapshot["saved_bytes"], "Saved byte count differs")
    require(
        snapshot["rank"] == snapshot["layer_id"] == 0
        and snapshot["stage"] == "first_proposal_layer0",
        "Scope differs",
    )
    request = json.loads((root / "request.json").read_text())
    response = json.loads((root / "response.json").read_text())
    require(
        request["rid"] == response["id"] == snapshot["rid"] == config["rid"],
        "RID differs",
    )
    require(snapshot["run_id"] == config["run_id"], "Run ID differs")
    require(
        request["cache_salt"] == config["cache_salt"]
        and request["messages"] == [{"role": "user", "content": PROMPT}]
        and request["temperature"] == 0
        and request["max_tokens"] == 64,
        "Fixed request differs",
    )
    choice = response["choices"][0]
    require(
        choice["finish_reason"] in ("length", "stop")
        and choice["meta_info"]["cached_tokens"] == 0,
        "Incomplete or non-cold response",
    )
    require(
        choice["prompt_token_ids"] == snapshot["prefill"]["prompt_ids"],
        "Observed/API prompt differs",
    )
    require(snapshot["prefill"]["prefix_lens"] == [0], "Non-cold observed prefill")
    for name in (
        "input_embedding",
        "norm_input",
        "norm_output",
        "qkv_input",
        "qkv_output",
        "fused_input",
        "prepared_q",
        "prepared_k",
        "prepared_v",
        "fia_query",
        "pool_k",
        "pool_v",
        "attention_output",
    ):
        require(
            arrays.dtype(name) == "torch.bfloat16",
            f"Diagnostic BF16 boundary differs: {name}",
        )
    return response


def compare_snapshot(root):
    config = json.loads((root / "config.json").read_text())
    scripts = Path(__file__).resolve().parent
    for name, digest in config["tool_hashes"].items():
        require(
            Path(name).name == name and file_hash(scripts / name) == digest,
            f"Replay tool differs from launch: {name}",
        )
    snapshot = json.loads((root / "snapshot.json").read_text())
    arrays = Arrays(root, snapshot, config["max_bytes"])
    response = validate(root, config, snapshot, arrays)
    get = arrays.get
    geometry = snapshot["geometry"]
    heads, dim, width = (geometry[k] for k in ("heads", "head_dim", "queries"))
    prefix = len(snapshot["prefill"]["prompt_ids"])
    ids, pos = get("input_ids"), get("positions")
    expected_slots = get("logical_slots")
    actual_slots = paged_slots(
        get("block_table")[0],
        snapshot["page_size"],
        snapshot["actual_seq_lengths_kv"][0],
        snapshot["pool_capacity"],
    )
    contracts = slot_contract(
        expected_slots, actual_slots, get("out_cache_loc"), prefix
    )
    contracts.update(
        {
            "slot_derivation_matches_observer": actual_slots.tolist()
            == snapshot["actual_slots"],
            "query_count": ids.size == width,
            "anchor_matches_bonus": bool(ids[0] == get("bonus_tokens").reshape(-1)[0]),
            "anchor_matches_first_output_token": int(ids[0])
            == response["choices"][0]["response_token_ids"][0],
            "remaining_ids_are_masks": bool(
                np.all(ids[1:] == snapshot["mask_token_id"])
            ),
            "prefix_cpu_gpu_and_forward": all(
                np.array_equal(get(n), [prefix])
                for n in ("prefix_lens", "prefix_lens_cpu", "forward_seq_lens")
            ),
            "forward_cpu_includes_query_once": bool(
                np.array_equal(get("forward_seq_lens_cpu"), [prefix + width])
            ),
            "query_positions": bool(
                np.array_equal(pos, np.arange(prefix, prefix + width))
            ),
            "verify_window_has_bonus_width": get("verify_positions").shape
            == (1, width + 1),
            "query_positions_from_verify_prefix": bool(
                np.array_equal(pos, get("verify_positions")[0, :width])
            ),
            "current_slots_from_verify_prefix": bool(
                np.array_equal(get("out_cache_loc"), get("verify_cache_loc")[0, :width])
            ),
            "actual_query_length": snapshot["actual_seq_lengths_q"] == [width],
            "actual_kv_length": snapshot["actual_seq_lengths_kv"] == [prefix + width],
            "request_map_length": expected_slots.size == prefix + width,
            "draft_spec_width": snapshot["spec_draft_token_num"] == width,
            "full_visibility": snapshot["mask_is_none"]
            and snapshot["sparse_mode"] == 0
            and snapshot["sliding_window_size"] == -1,
            "attention_scale": snapshot["scale"] == dim**-0.5,
            "checkpoint_local_embedding": snapshot["embedding_is_checkpoint_local"],
        }
    )
    comparisons = {}

    def compare(name, reference, actual):
        comparisons[name] = tensor_difference(reference, actual)

    embedding, raw_weight, checkpoint = checkpoint_rows(root, snapshot, arrays)
    compare("checkpoint_embedding_vs_layer_input", embedding, get("input_embedding"))
    compare("checkpoint_qkv_shard_vs_runtime_weight", raw_weight, get("qkv_weight"))
    compare("embedding_to_norm_input", get("input_embedding"), get("norm_input"))

    def bias(name):
        return get(name) if snapshot["bias"][name]["present"] else None

    norm_stored = None
    norm_mode = snapshot["input_norm_reference_mode"]
    if norm_mode != "uncovered":
        norm_math, norm_stored = input_norm(
            get("norm_input"),
            get("input_norm_weight"),
            snapshot["input_norm_eps"],
            bias=bias("input_norm_bias"),
            mode=norm_mode,
        )
        compare("norm_same_input_math", norm_math, get("norm_output"))
        compare("norm_same_input_bf16_storage", norm_stored, get("norm_output"))
    compare("norm_output_to_qkv_input", get("norm_output"), get("qkv_input"))
    projection = linear(get("qkv_input"), raw_weight)
    compare("qkv_same_input_math", projection, get("qkv_output"))
    compare("qkv_same_input_bf16_storage", bf16_round(projection), get("qkv_output"))
    compare("consumed_qkv_to_fused_input", get("qkv_output"), get("fused_input"))
    for store in (False, True):
        prepared = fused_prepare(
            get("fused_input"),
            get("q_norm_weight"),
            get("k_norm_weight"),
            get("fused_cos"),
            get("fused_sin"),
            snapshot["fused_eps"],
            heads,
            dim,
            bf16_storage=store,
            q_bias=bias("q_norm_bias"),
            k_bias=bias("k_norm_bias"),
        )
        for label, value in zip(("q", "k", "v"), prepared):
            compare(
                f"fused_{label}_same_input_{'bf16_storage' if store else 'math'}",
                value,
                get("prepared_" + label).reshape(value.shape),
            )
    c, s = rope_cos_sin(
        pos,
        dim,
        snapshot["rope_base"],
        bf16_cache=arrays.dtype("fused_cos") == "torch.bfloat16",
    )
    compare(
        "rope_cos_from_positions",
        np.concatenate((c, c), -1),
        get("fused_cos").reshape(len(pos), dim),
    )
    compare(
        "rope_sin_from_positions",
        np.concatenate((s, s), -1),
        get("fused_sin").reshape(len(pos), dim),
    )
    for label in ("q", "k", "v"):
        compare(
            "prepared_to_backend_" + label,
            get("prepared_" + label).reshape(-1, heads, dim),
            get("backend_" + label).reshape(-1, heads, dim),
        )
    compare(
        "backend_q_to_fia_query",
        get("backend_q").reshape(-1, heads, dim),
        get("fia_query"),
    )
    for label in ("k", "v"):
        compare(
            "current_" + label + "_write",
            get("backend_" + label).reshape(-1, heads, dim),
            get("current_pool_" + label),
        )
    compare(
        "fia_local_to_returned_output", get("fia_local_output"), get("attention_output")
    )
    attention_status = "NOT_REPLAYED_UNCOVERED_VISIBILITY"
    if contracts["full_visibility"]:
        q, k, v = (get(n) for n in ("fia_query", "pool_k", "pool_v"))
        out = get("attention_output").reshape(-1, heads, dim)
        for dtype in (np.float64, np.float32):
            computed = attention(q, k, v, snapshot["scale"], compute_dtype=dtype)
            compare(
                "attention_same_actual_inputs_" + np.dtype(dtype).name, computed, out
            )
            compare(
                "attention_same_actual_inputs_"
                + np.dtype(dtype).name
                + "_bf16_storage",
                bf16_round(computed),
                out,
            )
        attention_status = "SAME_ACTUAL_INPUTS_COMPARED"
        if norm_stored is not None and all(
            contracts[k]
            for k in (
                "same_ordered_slots",
                "current_matches_request_tail",
                "actual_kv_length",
                "actual_query_length",
                "query_count",
                "query_positions",
            )
        ):
            # Separate propagation experiment: keep ACTUAL context KV and norm
            # input; recompute current Q/K/V through norm and checkpoint QKV.
            composed_qkv = bf16_round(linear(norm_stored, raw_weight))
            cq, ck, cv = fused_prepare(
                composed_qkv,
                get("q_norm_weight"),
                get("k_norm_weight"),
                get("fused_cos"),
                get("fused_sin"),
                snapshot["fused_eps"],
                heads,
                dim,
                bf16_storage=True,
                q_bias=bias("q_norm_bias"),
                k_bias=bias("k_norm_bias"),
            )
            composed = attention(
                cq,
                np.concatenate((k[:prefix], ck)),
                np.concatenate((v[:prefix], cv)),
                snapshot["scale"],
            )
            compare(
                "composed_preparation_actual_context_attention_bf16_storage",
                bf16_round(composed),
                out,
            )
    result = {
        "status": "PROPOSAL_COMPARISON_COLLECTED",
        "run_id": config["run_id"],
        "rid": config["rid"],
        "scope": {
            "rank": 0,
            "layer_id": 0,
            "prefix_tokens": prefix,
            "query_tokens": int(ids.size),
            "head_dim": dim,
            "local_heads": heads,
        },
        "contracts": contracts,
        "structural_issues": [
            k for k, v in contracts.items() if isinstance(v, (bool, np.bool_)) and not v
        ],
        "attention_replay": attention_status,
        "bias": snapshot["bias"],
        "fused_bias_enabled": snapshot["fused_bias_enabled"],
        "input_norm_reference_mode": norm_mode,
        "comparisons": comparisons,
        "checkpoint": checkpoint,
        "observed_source": snapshot["source"],
        "acceptance": response.get("sglext", {}).get("spec_tokens_details", {}),
        "limits": [
            "No invented tolerance or correctness PASS; continuous local differences only.",
            "First proposal/layer0/rank0 only, before output projection and cross-rank reduction; later layers/Markov/verify/commit are not validated.",
            "Attention replay uses actual Q and consumed pool K/V; this does not establish target hidden semantics or context weight correctness.",
            "CPU FP32/FP64 and BF16 storage emulation do not reproduce NPU accumulation order.",
            "Observation adds synchronization and is not a performance or acceptance qualification run.",
        ],
    }
    # Convert NumPy scalar bools from structural comparisons; no ndarray payloads.
    result = json.loads(json.dumps(result, default=lambda x: x.item()))
    write_json(root / "comparison.json", result)
    return result


def collect(root, url, timeout=600, opener=None):
    config = json.loads((root / "config.json").read_text())
    require(
        Path(config["run_dir"]).resolve() == root.resolve(), "Run directory mismatch"
    )
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    info = request_json(opener, url.rstrip("/") + "/server_info", timeout=timeout)
    write_json(root / "server_info.json", info)
    expected = dict(
        device="npu",
        speculative_algorithm="DSPARK",
        tp_size=16,
        dp_size=1,
        nnodes=1,
        disable_cuda_graph=True,
    )
    require(
        all(info.get(k) == v for k, v in expected.items()),
        "Service differs from approved scope",
    )
    with (root / "client.claim").open("x") as f:
        f.write("One request only. Replay saved evidence; do not resubmit.\n")
    payload = dict(
        rid=config["rid"],
        cache_salt=config["cache_salt"],
        model="GLM-5.2-w8a8",
        messages=[dict(role="user", content=PROMPT)],
        temperature=0,
        max_tokens=64,
        stream=False,
        return_spec_tokens_details=True,
        return_meta_info=True,
        return_token_ids=True,
    )
    write_json(root / "request.json", payload)
    response = request_json(
        opener, url.rstrip("/") + "/v1/chat/completions", payload, timeout
    )
    write_json(root / "response.json", response)
    return compare_snapshot(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    p.add_argument("--url", default="http://61.47.19.71:8810")
    p.add_argument("--replay", type=Path)
    args = p.parse_args()
    root = (
        args.replay.expanduser().resolve()
        if args.replay
        else Path(
            json.loads((args.state_dir / "proposal-snapshot-current.json").read_text())[
                "config"
            ]
        ).parent
    )
    print(f"Evidence: {root}", flush=True)
    try:
        result = compare_snapshot(root) if args.replay else collect(root, args.url)
    except Exception as exc:
        write_json(
            root / "comparison-error.json",
            dict(
                status="PROPOSAL_COMPARISON_FAILED",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        print(f"PROPOSAL_COMPARISON_FAILED: {exc}")
        return 1
    print(result["status"])
    print("Structural issues:", result["structural_issues"])
    print("Input norm reference:", result["input_norm_reference_mode"])
    print("Observed bias:", result["bias"])
    for name, value in result["comparisons"].items():
        print(
            f"{name}: relative_l2={value['relative_l2']} max_abs={value['max_abs_error']}"
        )
    print(
        "Send comparison.json and snapshot.json; keep .npy and checkpoint-reads.json in the intranet. No performance/correctness PASS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
