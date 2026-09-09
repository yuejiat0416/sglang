#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline FIA residual distribution from a saved proposal snapshot.

CPU/NumPy only. No HTTP, model/checkpoint imports, NPU operations or recapture.
Source evidence stays unchanged; output goes to a new subdirectory.
Temporary sync-branch tool, permanently excluded from the upstream PR.
"""

# ruff: noqa: E402 -- bound BLAS threads before importing NumPy.

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
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
from dspark_attention_reference import attention, paged_slots, slot_contract
from dspark_local_reference import bf16_round, tensor_difference
from fia_residual_metrics import analyze_residual

HERE = Path(__file__).resolve().parent
MAX_SELECTED_BYTES = 2 * 1024**2
OP_PLUGIN_REFERENCE = (
    "https://github.com/Ascend/op-plugin/blob/"
    "8b9c8534fa41eff367a41c155843daa530ab3a08/"
    "test/test_custom_ops/test_npu_fused_infer_attention_score.py#L89"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    require(path.stat().st_size <= 2 * 1024**2, f"Metadata too large: {path.name}")
    return json.loads(path.read_text())


def read_selected(root, snapshot, name, identity, budget):
    """Validate only the selected small arrays; never read the QKV checkpoint."""
    info = snapshot["arrays"][name]
    require(info["file"] == name + ".npy", f"Invalid array file: {name}")
    path = root / info["file"]
    size = path.stat().st_size
    require(
        size == info["bytes"] and 0 < size <= MAX_SELECTED_BYTES,
        f"Array size changed or too large: {name}",
    )
    budget[0] += size
    require(budget[0] <= MAX_SELECTED_BYTES, "Selected arrays exceed CPU budget")
    require(digest(path) == info["sha256"], f"Array hash changed: {name}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    require(list(arr.shape) == info["shape"], f"Array shape changed: {name}")
    if info["encoding"] == "bf16_uint16":
        require(
            arr.dtype == np.uint16 and info["dtype"] == "torch.bfloat16",
            f"Invalid BF16 array: {name}",
        )
        value = (np.asarray(arr, dtype=np.uint32) << 16).view(np.float32)
    else:
        require(
            info["encoding"] == "numpy"
            and arr.dtype.kind in "iu"
            and info["dtype"] in ("torch.int32", "torch.int64")
            and arr.dtype.itemsize == int(info["dtype"][-2:]) // 8,
            f"Expected integer metadata: {name}",
        )
        value = np.array(arr, copy=True)
    require(np.isfinite(value).all(), f"Non-finite array: {name}")
    identity[name] = {k: info[k] for k in ("file", "bytes", "sha256", "shape", "dtype")}
    return value


def load_snapshot(root):
    root = root.expanduser().resolve(strict=True)
    config, snapshot, previous = (
        read_json(root / name)
        for name in ("config.json", "snapshot.json", "comparison.json")
    )
    require(
        snapshot["status"] == "PROPOSAL_SNAPSHOT_COLLECTED"
        and previous["status"] == "PROPOSAL_COMPARISON_COLLECTED"
        and snapshot["errors"] == [],
        "Snapshot/comparison incomplete",
    )
    for key in ("run_id", "rid"):
        require(config[key] == snapshot[key] == previous[key], f"Mixed {key}")
    require(
        snapshot["observer_sha256"]
        == config["tool_hashes"]["dspark_proposal_snapshot.py"],
        "Recorded observer identity differs",
    )
    # We add a new analyser rather than rewriting the capture/replay tools.
    for name in ("dspark_attention_reference.py", "dspark_local_reference.py"):
        require(
            digest(HERE / name) == config["tool_hashes"][name],
            f"Reference changed: {name}",
        )
    require(
        snapshot["rank"] == snapshot["layer_id"] == 0
        and snapshot["stage"] == "first_proposal_layer0"
        and snapshot["geometry"]
        == {"hidden": 6144, "heads": 4, "head_dim": 192, "queries": 8},
        "Outside approved first-proposal geometry",
    )
    prefix = len(snapshot["prefill"]["prompt_ids"])
    require(
        0 < prefix <= 120
        and snapshot["prefill"]["prefix_lens"] == [0]
        and snapshot["prefill"]["rid"] == snapshot["rid"],
        "Expected short cold prefill",
    )
    require(
        snapshot["backend_branch"] == "ordinary_mha_fia_tnd"
        and snapshot["mask_is_none"]
        and snapshot["sparse_mode"] == 0
        and snapshot["sliding_window_size"] == -1
        and snapshot["attn_type"] == "AttentionType.ENCODER_ONLY"
        and snapshot["scale"] == 192**-0.5
        and snapshot["actual_seq_lengths_q"] == [8]
        and snapshot["actual_seq_lengths_kv"] == [prefix + 8],
        "Uncovered FIA visibility, scale or sequence lengths",
    )
    require(
        previous["structural_issues"] == []
        and previous["attention_replay"] == "SAME_ACTUAL_INPUTS_COMPARED",
        "Earlier comparison did not establish the expected local structure",
    )
    require(
        previous["scope"]
        == {
            "rank": 0,
            "layer_id": 0,
            "prefix_tokens": prefix,
            "query_tokens": 8,
            "head_dim": 192,
            "local_heads": 4,
        },
        "Earlier comparison scope differs",
    )
    names = (
        "fia_query",
        "pool_k",
        "pool_v",
        "attention_output",
        "positions",
        "block_table",
        "logical_slots",
        "out_cache_loc",
    )
    identity, budget = {}, [0]
    arrays = {n: read_selected(root, snapshot, n, identity, budget) for n in names}
    for name in ("fia_query", "pool_k", "pool_v", "attention_output"):
        require(identity[name]["dtype"] == "torch.bfloat16", f"Not BF16: {name}")
    for name, shape in {
        "fia_query": (8, 4, 192),
        "pool_k": (prefix + 8, 4, 192),
        "pool_v": (prefix + 8, 4, 192),
        "attention_output": (8, 768),
        "positions": (8,),
        "logical_slots": (prefix + 8,),
        "out_cache_loc": (8,),
    }.items():
        require(arrays[name].shape == shape, f"Wrong geometry: {name}")
    require(
        np.array_equal(arrays["positions"], np.arange(prefix, prefix + 8)),
        "Positions differ",
    )
    table = arrays["block_table"]
    require(table.ndim == 2 and table.shape[0] == 1, "Invalid block table")
    actual_slots = paged_slots(
        table[0], snapshot["page_size"], prefix + 8, snapshot["pool_capacity"]
    )
    contracts = slot_contract(
        arrays["logical_slots"], actual_slots, arrays["out_cache_loc"], prefix
    )
    require(
        all(v for v in contracts.values() if isinstance(v, (bool, np.bool_)))
        and actual_slots.tolist() == snapshot["actual_slots"],
        "Saved cache mapping differs",
    )
    return root, snapshot, previous, arrays, identity


def reference_profile(q, k, v, scale, prefix, positions):
    """Reference-only diagnostics, NOT captured FIA probabilities/intermediates."""
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    scores = np.einsum("thd,shd->hts", q, k) * scale
    unnorm = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = unnorm / unnorm.sum(axis=-1, keepdims=True)
    logp = np.zeros_like(probs)
    np.log(probs, out=logp, where=probs > 0)
    rows = []
    for t, position in enumerate(positions):
        for h in range(q.shape[1]):
            s, p = scores[h, t], probs[h, t]
            rows.append(
                {
                    "query_index": t,
                    "position": int(position),
                    "head_index": h,
                    "score_min": float(s.min()),
                    "score_max": float(s.max()),
                    "probability_max": float(p.max()),
                    "max_probability_kv_index": int(p.argmax()),
                    "entropy_nats": float(-np.sum(p * logp[h, t])),
                    "context_probability_mass": float(p[:prefix].sum()),
                    "current_block_probability_mass": float(p[prefix:].sum()),
                    "v_max_abs": float(np.abs(v[:, h]).max()),
                }
            )
    return {
        "origin": "CPU FP64 reference only; not an observation of FIA internals",
        "rows": rows,
    }


def op_plugin_style_reference(q, k, v, scale):
    """One source-backed diagnostic candidate, not FIA's internal golden.

    Ascend op-plugin 8b9c8534 supported_op_exec_ntd uses input-dtype QK and
    scale outputs, FP32 softmax cast back to input dtype, then PV. Here we
    adapt those BF16 storage boundaries to the ACTUAL snapshot scale and
    unpaged effective K/V. CPU FP32 matmuls do not simulate NPU accumulation.
    We do not search cast combinations or tune the scale to fit the output.
    """
    q, k, v = (np.asarray(a, dtype=np.float32) for a in (q, k, v))
    scores = bf16_round(np.einsum("thd,shd->hts", q, k))
    scores = bf16_round(scores * np.float32(scale))
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    probs = bf16_round(probs)
    return bf16_round(np.einsum("hts,shd->thd", probs, v))


def analyze(snapshot_dir):
    root, snapshot, previous, arrays, identity = load_snapshot(snapshot_dir)
    q, k, v = (arrays[n] for n in ("fia_query", "pool_k", "pool_v"))
    actual = arrays["attention_output"].reshape(q.shape)
    pos, scale = arrays["positions"], snapshot["scale"]
    ref64 = attention(q, k, v, scale, compute_dtype=np.float64)
    ref32 = attention(q, k, v, scale, compute_dtype=np.float32)
    stored64, stored32 = bf16_round(ref64), bf16_round(ref32)
    residual = analyze_residual(stored64, actual, pos)
    candidate = op_plugin_style_reference(q, k, v, scale)
    old_key = "attention_same_actual_inputs_float64_bf16_storage"
    prior = previous["comparisons"][old_key]
    recomputed = tensor_difference(stored64, actual)
    return {
        "status": "FIA_OFFLINE_ANALYSIS_COLLECTED",
        "run_id": snapshot["run_id"],
        "rid": snapshot["rid"],
        "source_snapshot_dir": str(root),
        "scope": previous["scope"],
        "metadata_sha256": {
            n: digest(root / n)
            for n in ("config.json", "snapshot.json", "comparison.json")
        },
        "selected_arrays": identity,
        "selected_array_bytes": sum(x["bytes"] for x in identity.values()),
        "analyser_sources": {
            n: digest(HERE / n)
            for n in (
                "analyze_fia_snapshot.py",
                "fia_residual_metrics.py",
                "dspark_attention_reference.py",
                "dspark_local_reference.py",
            )
        },
        "versions": {"python": sys.version, "numpy": np.__version__},
        "comparison_parties": {
            "actual": "Captured original NPU FIA output, before output projection",
            "reference": "CPU stable softmax(Q K.T * scale) V using SAME actual Q and read-back pool K/V",
            "storage": "Reference rounded to BF16 at output only; no invented intermediate casts",
        },
        "prior_comparison": {
            "field": old_key,
            "reported": prior,
            "recomputed": recomputed,
            "relative_l2_change": recomputed["relative_l2"] - prior["relative_l2"]
            if recomputed["relative_l2"] is not None
            and prior["relative_l2"] is not None
            else None,
        },
        "reference_comparisons": {
            "float64_vs_actual": tensor_difference(ref64, actual),
            "float32_vs_actual": tensor_difference(ref32, actual),
            "float64_vs_float32": tensor_difference(ref64, ref32),
            "stored_float64_vs_stored_float32": tensor_difference(stored64, stored32),
            "float64_vs_its_output_storage": tensor_difference(ref64, stored64),
        },
        "residual": residual,
        "op_plugin_style_candidate": {
            "source": OP_PLUGIN_REFERENCE,
            "basis": "op-plugin gitlink of Ascend/pytorch 5dd8ef3f9b375b5ae4a83538d5785754148c3302 (version.txt=2.10.0.post4)",
            "calculation": "BF16(QK_FP32) -> BF16(* actual_scale) -> softmax_FP32 -> BF16 -> PV_FP32 -> BF16",
            "adaptations": "Actual scale replaces source test's 1/0.0078125; effective unpaged K/V; CPU arithmetic; BF16 input-dtype interpretation",
            "source_limits": "The corresponding NTD_TND source test is skipped and uses all-ones inputs and V dimension 128; it does not validate this A3 paged TND/192 case",
            "claim": "One official-test-style numerical candidate; not current kernel internals, a precision tolerance or a correctness PASS",
            "residual": analyze_residual(candidate, actual, pos),
            "math_reference_vs_candidate": tensor_difference(stored64, candidate),
        },
        "reference_attention_profile": reference_profile(
            q, k, v, scale, k.shape[0] - q.shape[0], pos
        ),
        "limits": [
            "No tolerance, precision PASS, model correctness, acceptance or performance qualification.",
            "BF16 representable-step distances describe output values, not a hardware precision guarantee.",
            "FIA internal QK/softmax/PV dtype and accumulation order are not observed or emulated.",
            "Reference probabilities/scores are calculated data, never FIA internal snapshots.",
            "Only selected small arrays were rehashed; no full checkpoint or full snapshot array audit.",
            "First proposal/layer0/rank0 only; actual context KV does not establish target hidden semantics.",
            "No inference request, original-file overwrite, NPU operation or production code change.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.snapshot_dir)
        root = Path(result["source_snapshot_dir"])
        suffix = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        output = root / ("fia-analysis-" + suffix)
        payload = (
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        output.mkdir()
        report = output / "report.json"
        report.write_text(payload)
    except Exception as exc:
        print(
            f"FIA_OFFLINE_ANALYSIS_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    print(result["status"])
    print(f"Report: {report}")
    print(
        "Stored reference vs actual:",
        json.dumps(result["prior_comparison"]["recomputed"]),
    )
    print(
        "Official-test-style candidate vs actual:",
        json.dumps(result["op_plugin_style_candidate"]["residual"]["overall"]),
    )
    print("Send this report.json; no restart or additional inference request was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
