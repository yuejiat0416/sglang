#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One fixed HTTP request, then offline context comparison. No model imports.

Default: use the observer launch's current run. --replay DIR only recomputes an
existing snapshot and saved response; never resubmit a claimed request.
Temporary sync-branch diagnostics, excluded from the upstream feature branch.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

# Set before NumPy/BLAS imports. CPU replay is deliberately bounded to one thread.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

from collect_acceptance_trace import PROMPT, request_json
from dspark_context_snapshot import file_hash, require, selected_rows, write_json

DEFAULT_STATE = Path("/home/tyj/glm52-ms1")


class Arrays:
    def __init__(self, root, snapshot, max_bytes):
        import numpy as np

        self.np = np
        self.root, self.info = root, snapshot["arrays"]
        total = 0
        for name, entry in self.info.items():
            require(
                entry["file"] == name + ".npy"
                and Path(entry["file"]).name == entry["file"],
                "Invalid array filename",
            )
            path = root / entry["file"]
            total += path.stat().st_size
            require(total <= max_bytes, "Saved snapshot exceeds budget")
            require(
                path.stat().st_size == entry["bytes"]
                and file_hash(path) == entry["sha256"],
                f"Array file changed: {name}",
            )
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            require(list(a.shape) == entry["shape"], f"Array shape differs: {name}")
            require(entry["encoding"] in ("numpy", "bf16_uint16"), "Unknown encoding")
            if entry["encoding"] == "bf16_uint16":
                require(
                    a.dtype == np.uint16 and entry["dtype"] == "torch.bfloat16",
                    "Bad BF16 encoding",
                )
            else:
                require(a.dtype.kind in "fiu", "Unsupported saved dtype")
        self.total = total

    def get(self, name, rows=None):
        np = self.np
        e = self.info[name]
        a = np.load(self.root / e["file"], mmap_mode="r", allow_pickle=False)
        if rows is not None:
            a = a[rows]
        if e["encoding"] == "bf16_uint16":
            return (np.asarray(a, dtype=np.uint32) << 16).view(np.float32)
        return np.array(a, copy=True)

    def dtype(self, name):
        return self.info[name]["dtype"]


def validate_response(root, config, snapshot, arrays):
    request = json.loads((root / "request.json").read_text())
    response = json.loads((root / "response.json").read_text())
    require(
        request["rid"] == config["rid"] == response["id"] == snapshot["rid"],
        "Request/response/snapshot RID mismatch",
    )
    require(snapshot["run_id"] == config["run_id"], "Run identity mismatch")
    require(request["cache_salt"] == config["cache_salt"], "Cache salt mismatch")
    require(
        request["messages"] == [{"role": "user", "content": PROMPT}]
        and request["temperature"] == 0
        and request["max_tokens"] == 64,
        "Fixed request differs",
    )
    choice = response["choices"][0]
    require(
        choice["finish_reason"] in ("stop", "length"), "Request did not finish normally"
    )
    require(
        choice["meta_info"]["cached_tokens"] == 0, "Response was not a cold prefill"
    )
    ids = arrays.get("prompt_ids").tolist()
    require(
        choice["prompt_token_ids"] == ids, "Observed prompt token IDs differ from API"
    )
    require(
        len(ids) == snapshot["token_count"]
        and snapshot["rows"] == selected_rows(len(ids)),
        "Invalid selected rows",
    )
    require(
        snapshot["rank"] == 0
        and snapshot["stage"] == "first_prefill"
        and snapshot["prefix_lens"] == [0],
        "Unexpected capture scope",
    )
    return response


def compare_snapshot(root):
    import dspark_local_reference as ref
    import numpy as np

    config = json.loads((root / "config.json").read_text())
    snap = json.loads((root / "snapshot.json").read_text())
    require(
        file_hash(__file__) == config["client_sha256"],
        "Client differs from diagnostic launch; preserve matching tools",
    )
    require(
        file_hash(ref.__file__) == config["reference_sha256"],
        "Reference differs from diagnostic launch",
    )
    require(
        snap["status"] == "CONTEXT_SNAPSHOT_COLLECTED",
        f"Snapshot incomplete: {snap.get('errors')}",
    )
    require(
        snap.get("branch") in {"fused_kv_norm_rope_write", "stacked_ctx_kv"},
        "Uncovered runtime branch",
    )
    require(
        snap["observer_sha256"] == config["observer_sha256"],
        "Observer source differs from launch",
    )
    arrays = Arrays(root, snap, config["max_bytes"])
    require(arrays.total == snap["saved_bytes"], "Saved byte count differs")
    response = validate_response(root, config, snap, arrays)
    result = {
        "status": "CONTEXT_COMPARISON_COLLECTED",
        "run_id": config["run_id"],
        "rid": config["rid"],
        "scope": {
            k: snap[k] for k in ("rank", "stage", "rows", "token_count", "branch")
        },
        "observed_source": snap["source"],
        "collector_sha256": file_hash(__file__),
        "reference_sha256": file_hash(ref.__file__),
        "numpy_version": np.__version__,
        "comparisons": {},
        "layout_checks": {},
        "layers": [],
        "acceptance": response.get("sglext", {}).get("spec_tokens_details"),
        "limits": [
            "Continuous differences only; no invented model/kernel tolerance or automatic correctness PASS.",
            "Only first prefill, rank0 and three rows; no full draft, decode/commit, graph or performance validation.",
            "Fused K norm/RoPE/write has no separate real intermediate output; compare the group, not an invented internal snapshot.",
            "Same loaded weights do not establish checkpoint mapping, QuaRot equivalence or target capture semantics.",
            "BF16 boundary emulation does not reproduce NPU accumulation order; observation changes timing.",
        ],
    }
    diff = ref.tensor_difference
    h, z, norm_in, ctx = (
        arrays.get(k) for k in ("hidden", "fc_output", "norm_input", "context")
    )
    require(
        h.shape[0] == z.shape[0] == ctx.shape[0] == len(snap["rows"]),
        "Activation rows differ",
    )
    # Stream FC output rows: do not expand the entire 360 MiB BF16 matrix to FP32.
    blocks = []
    out_size = arrays.info["fc_weight"]["shape"][0]
    for lo in range(0, out_size, 256):
        w = arrays.get("fc_weight", slice(lo, min(lo + 256, out_size)))
        blocks.append(ref.linear(h, w))
    z_ref = np.concatenate(blocks, axis=1)
    comp = result["comparisons"]
    comp["fc_math_vs_actual"] = diff(z_ref, z)
    comp["fc_bf16_storage_vs_actual"] = diff(ref.bf16_round(z_ref), z)
    comp["actual_fc_output_vs_actual_norm_input"] = diff(z, norm_in)
    nw = arrays.get("hidden_norm_weight")
    c_ref = ref.rms_norm(norm_in, nw, snap["hidden_norm_eps"])
    comp["norm_same_actual_input_math"] = diff(c_ref, ctx)
    comp["norm_same_actual_input_bf16_storage"] = diff(ref.bf16_round(c_ref), ctx)
    c_chain = ref.rms_norm(
        ref.bf16_round(z_ref), nw, snap["hidden_norm_eps"], bf16_storage=True
    )
    comp["fc_norm_composed_bf16_storage"] = diff(c_chain, ctx)
    positions = arrays.get("positions")
    projected = arrays.get("kv_projected")
    layers = snap["layers"]
    total_kv = sum(2 * layer["kv_size"] for layer in layers)
    require(
        projected.shape == (len(snap["rows"]), total_kv), "Bad stacked KV dimensions"
    )
    require(
        arrays.dtype("kv_projected") == "torch.bfloat16",
        "Reference storage mode requires BF16 KV",
    )
    stacked_path = snap["branch"] == "stacked_ctx_kv"
    packed_prefix = "stacked" if stacked_path else "fused"
    actual_eps = snap[packed_prefix + "_eps"]
    actual_neox = (
        layers[0]["is_neox_style"] if stacked_path else snap["fused_is_neox_style"]
    )
    if stacked_path:
        result["layout_checks"].update(
            positions_match_rope=np.array_equal(
                positions, arrays.get("stacked_rope_positions")
            ),
        )
    else:
        result["layout_checks"].update(
            positions_match_fused=np.array_equal(
                positions, arrays.get("fused_positions")
            ),
            addresses_match_fused=np.array_equal(
                arrays.get("cache_locs"), arrays.get("fused_locs")
            ),
            fused_meta_matches_pool=snap["fused_meta_matches_pool"],
        )
    offset = 0
    for i, layer in enumerate(layers):
        q, kv, d = layer["q_size"], layer["kv_size"], layer["head_dim"]
        raw = arrays.get(f"qkv_weight_{i}")
        require(raw.shape[0] == q + 2 * kv, "QKV layout mismatch")
        k_w, v_w = raw[q : q + kv], raw[q + kv : q + 2 * kv]
        packed_w = arrays.get(packed_prefix + "_weight", slice(offset, offset + 2 * kv))
        actual = projected[:, offset : offset + 2 * kv]
        k_actual, v_actual = (
            a.reshape(-1, layer["local_heads"], d) for a in np.split(actual, 2, axis=1)
        )
        k_ref = ref.linear(ctx, k_w).reshape(k_actual.shape)
        v_ref = ref.linear(ctx, v_w).reshape(v_actual.shape)
        pool_k, pool_v = arrays.get(f"pool_k_{i}"), arrays.get(f"pool_v_{i}")
        cache = arrays.get(f"cos_sin_{i}")
        fused_cache = arrays.get(packed_prefix + "_cos_sin")
        knw = arrays.get(f"k_norm_weight_{i}")
        fused_knw = arrays.get(packed_prefix + "_knw")[i]
        # Group isolation uses the exact *actual fused host inputs*.
        kn = ref.rms_norm(k_actual, fused_knw, actual_eps, bf16_storage=True)
        kr = ref.apply_rope(
            kn,
            fused_cache[:, : d // 2],
            fused_cache[:, d // 2 :],
            is_neox_style=actual_neox,
            bf16_storage=True,
        )
        math_kn = ref.rms_norm(k_actual, fused_knw, actual_eps)
        math_kr = ref.apply_rope(
            math_kn,
            fused_cache[:, : d // 2],
            fused_cache[:, d // 2 :],
            is_neox_style=actual_neox,
        )
        gen_c, gen_s = ref.rope_cos_sin(
            positions,
            d,
            layer["base"],
            bf16_cache=arrays.dtype(f"cos_sin_{i}") == "torch.bfloat16",
        )
        chain_k = ref.bf16_round(ref.linear(c_chain, k_w)).reshape(k_actual.shape)
        chain_v = ref.bf16_round(ref.linear(c_chain, v_w)).reshape(v_actual.shape)
        chain_norm = ref.rms_norm(chain_k, knw, layer["k_norm_eps"], bf16_storage=True)
        chain_rope = ref.apply_rope(
            chain_norm,
            cache[:, : d // 2],
            cache[:, d // 2 :],
            is_neox_style=layer["is_neox_style"],
            bf16_storage=True,
        )
        extra = {}
        if stacked_path:
            norm_actual = arrays.get("stacked_rope_input")[
                :, i * kv : (i + 1) * kv
            ].reshape(k_actual.shape)
            write_k, write_v = arrays.get(f"write_k_{i}"), arrays.get(f"write_v_{i}")
            rotated_actual_input = ref.apply_rope(
                norm_actual,
                fused_cache[:, : d // 2],
                fused_cache[:, d // 2 :],
                is_neox_style=actual_neox,
                bf16_storage=True,
            )
            extra = {
                "k_norm_same_actual_input": diff(kn, norm_actual),
                "rope_same_actual_input": diff(rotated_actual_input, write_k),
                "k_copy_to_pool": diff(write_k, pool_k),
                "v_output_to_pool": diff(write_v, pool_v),
            }
        result["layers"].append(
            {
                "layer_id": layer["layer_id"],
                "checks": {
                    "raw_kv_slice_matches_runtime_weight": bool(
                        np.array_equal(np.concatenate((k_w, v_w)), packed_w)
                    ),
                    "norm_weight_matches_runtime": bool(np.array_equal(knw, fused_knw)),
                    "rope_cache_matches_runtime": bool(
                        np.array_equal(cache, fused_cache)
                    ),
                    "eps_matches_runtime": layer["k_norm_eps"] == actual_eps,
                    "rope_pairing_matches_runtime": layer["is_neox_style"]
                    == actual_neox,
                },
                "k_projection_math": diff(k_ref, k_actual),
                "v_projection_math": diff(v_ref, v_actual),
                "k_projection_bf16_storage": diff(ref.bf16_round(k_ref), k_actual),
                "v_projection_bf16_storage": diff(ref.bf16_round(v_ref), v_actual),
                "k_norm_rope_write_group_math": diff(math_kr, pool_k),
                "k_norm_rope_write_group_bf16_storage": diff(kr, pool_k),
                "v_copy_to_pool": diff(v_actual, pool_v),
                "composed_hidden_to_pool_k": diff(chain_rope, pool_k),
                "composed_hidden_to_pool_v": diff(chain_v, pool_v),
                "rope_generated_rows": diff(
                    np.concatenate((gen_c, gen_s), axis=1), cache
                ),
                "stacked_actual_boundaries": extra,
            }
        )
        offset += 2 * kv
    # NumPy bool scalars are not JSON serializable.
    result["layout_checks"] = {k: bool(v) for k, v in result["layout_checks"].items()}
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
    expected = {
        "device": "npu",
        "speculative_algorithm": "DSPARK",
        "tp_size": 16,
        "dp_size": 1,
        "nnodes": 1,
        "disable_cuda_graph": True,
    }
    require(
        all(info.get(k) == v for k, v in expected.items()),
        "Service configuration does not match approved scope",
    )
    with (root / "client.claim").open("x") as f:
        f.write("One request only; use --replay for existing evidence.\n")
    payload = {
        "rid": config["rid"],
        "cache_salt": config["cache_salt"],
        "model": "GLM-5.2-w8a8",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        "return_spec_tokens_details": True,
        "return_meta_info": True,
        "return_token_ids": True,
    }
    write_json(root / "request.json", payload)
    response = request_json(
        opener, url.rstrip("/") + "/v1/chat/completions", payload, timeout
    )
    write_json(root / "response.json", response)
    return compare_snapshot(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--url", default="http://61.47.19.71:8810")
    parser.add_argument(
        "--replay", type=Path, help="Existing evidence directory; no HTTP request"
    )
    args = parser.parse_args()
    if args.replay:
        root = args.replay.expanduser().resolve()
    else:
        current = json.loads(
            (args.state_dir / "context-snapshot-current.json").read_text()
        )
        root = Path(current["config"]).parent
    print(f"Evidence: {root}", flush=True)
    try:
        result = compare_snapshot(root) if args.replay else collect(root, args.url)
    except Exception as exc:
        result = {
            "status": "CONTEXT_COMPARISON_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(root / "comparison-error.json", result)
    print(result["status"])
    if result["status"] != "CONTEXT_COMPARISON_COLLECTED":
        print(result["error"])
        return 1
    for key, value in result["comparisons"].items():
        print(
            f"{key}: relative_l2={value['relative_l2']} max_abs={value['max_abs_error']}"
        )
    for layer in result["layers"]:
        print(
            f"Layer {layer['layer_id']}: K norm/RoPE/write relative_l2={layer['k_norm_rope_write_group_bf16_storage']['relative_l2']}; V exact={layer['v_copy_to_pool']['exact_equal']}; layout={layer['checks']}"
        )
    print(
        "Diagnostic only; send comparison.json and snapshot.json. Keep .npy weight/data files in the intranet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
