# SPDX-License-Identifier: Apache-2.0
"""CPU fixed-input diagnosis for the GLM DSpark target-hidden/FC boundary.

No model/runtime imports, target activation capture, checkpoint writes or
automatic correction. Full FC outputs use an input-side algebraic equivalent;
only selected output rows undergo an actual weight-side Q transformation.
"""

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import probe_quarot_vocab as vocab


def bf16_round(values):
    """Emulate finite FP32-to-BF16 round-to-nearest-even; return FP32 values.

    This only models parameter storage, not NPU matmul accumulation or casts.
    NaNs remain NaNs, including when their payload occupies only the low bits.
    """
    np = vocab.load_numpy()
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    rounded = np.where(np.isnan(values), np.uint32(0x7FC00000), rounded)
    return rounded.astype(np.uint32, copy=False).view(np.float32)


def rms_norm(values, weight, eps):
    """FP32 mathematical reference, including epsilon and learned norm weight."""
    np = vocab.load_numpy()
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected finite positive RMSNorm epsilon")
    values = np.asarray(values, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2 or weight.shape != (values.shape[-1],):
        raise ValueError("Expected [rows, width] values and [width] norm weight")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = np.mean(values * values, axis=-1, keepdims=True)
        return values / np.sqrt(variance + np.float32(eps)) * weight


def evaluate_fc(fc, q, norm_weight, eps, inputs, selected_rows):
    """Compare predefined Q-folding hypotheses; never fit a correction factor."""
    np = vocab.load_numpy()
    if inputs.ndim != 3 or fc.ndim != 2:
        raise ValueError("Expected inputs[T, features, width] and FC[outputs, inputs]")
    tokens, features, width = inputs.shape
    if (
        min(inputs.shape) <= 0
        or q.shape != (width, width)
        or fc.shape[1] != features * width
        or fc.shape[0] <= 0
        or norm_weight.shape != (fc.shape[0],)
    ):
        raise ValueError("FC, Q, hidden_norm and input shapes are incompatible")
    if (
        not selected_rows
        or len(set(selected_rows)) != len(selected_rows)
        or any(type(i) is not int or not 0 <= i < fc.shape[0] for i in selected_rows)
    ):
        raise ValueError("Expected distinct valid FC output row IDs")
    ids = list(range(tokens))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = inputs.reshape(tokens * features, width)
        hq = h @ q
        h_back = hq @ q.T
        reference = inputs.reshape(tokens, -1) @ fc.T
        unchanged = hq.reshape(tokens, -1) @ fc.T
        # Associativity avoids converting all output rows (five 6144^3 GEMMs
        # for this checkpoint). This does NOT model full converted-weight
        # rounding, and does not prove that real target hidden equals H @ Q.
        equivalent = h_back.reshape(tokens, -1) @ fc.T
        selected_fc = fc[selected_rows]
        folded = np.empty_like(selected_fc)
        for feature in range(features):
            block = slice(feature * width, (feature + 1) * width)
            folded[:, block] = selected_fc[:, block] @ q
        actual_fp32 = hq.reshape(tokens, -1) @ folded.T
        actual_bf16 = hq.reshape(tokens, -1) @ bf16_round(folded).T

    norm_reference = rms_norm(reference, norm_weight, eps)
    return {
        "comparisons": {
            "unchanged_fc_pre_norm": vocab.compare_rows(reference, unchanged, ids),
            "q_fold_equivalent_pre_norm": vocab.compare_rows(
                reference, equivalent, ids
            ),
            "unchanged_fc_post_norm": vocab.compare_rows(
                norm_reference, rms_norm(unchanged, norm_weight, eps), ids
            ),
            "q_fold_equivalent_post_norm": vocab.compare_rows(
                norm_reference, rms_norm(equivalent, norm_weight, eps), ids
            ),
        },
        "sampled_weights": {
            "folded_fp32_vs_equivalent": vocab.compare_rows(
                equivalent[:, selected_rows], actual_fp32, ids
            ),
            "folded_bf16_vs_fp32": vocab.compare_rows(actual_fp32, actual_bf16, ids),
            "folded_bf16_vs_reference": vocab.compare_rows(
                reference[:, selected_rows], actual_bf16, ids
            ),
        },
        "q_roundtrip": vocab.compare_rows(h, h_back, list(range(tokens * features))),
        "selected_fc_output_rows": selected_rows,
        "finite": {
            "fc": bool(np.isfinite(fc).all()),
            "q": bool(np.isfinite(q).all()),
            "norm_weight": bool(np.isfinite(norm_weight).all()),
            "inputs": bool(np.isfinite(inputs).all()),
        },
    }


def read_hidden_norm(checkpoint, width):
    """Read the exact 1D norm tensor without changing the old matrix reader."""
    aliases = (
        "hidden_norm.weight",
        "model.hidden_norm.weight",
        "encoder.output_norm_enc.weight",
        "model.encoder.output_norm_enc.weight",
    )
    matches = set()
    if checkpoint.indices:
        for index in checkpoint.indices:
            for key in aliases:
                filename = index["content"]["weight_map"].get(key)
                if filename is not None:
                    if not isinstance(filename, str):
                        raise ValueError(f"Invalid norm shard name: {key}")
                    matches.add(((checkpoint.path / filename).resolve(), key))
    else:
        for path in sorted(checkpoint.path.glob("*.safetensors")):
            reader = checkpoint.get_file(path)
            matches.update(
                (reader.path, key) for key in aliases if key in reader.header
            )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one unambiguous hidden_norm tensor, found {matches}"
        )
    path, key = matches.pop()
    reader = checkpoint.get_file(path)
    info = reader.header.get(key)
    if not isinstance(info, dict):
        raise ValueError(f"Missing indexed norm tensor: {key}")
    dtype, shape, offsets = (
        info.get("dtype"),
        info.get("shape"),
        info.get("data_offsets"),
    )
    if (
        shape != [width]
        or dtype not in vocab.DTYPE_BYTES
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(i) is not int for i in offsets)
        or not 0 <= offsets[0] <= offsets[1]
        or offsets[1] - offsets[0] != width * vocab.DTYPE_BYTES[dtype]
        or reader.data_start + offsets[1] > reader.evidence()["file_size"]
    ):
        raise ValueError(f"Invalid hidden_norm shape, dtype or payload: {key}")
    offset, length = reader.data_start + offsets[0], offsets[1] - offsets[0]
    record = {"key": key, **info, "rows": None, "ranges": []}
    reader.reads.append(record)
    with reader.path.open("rb") as stream:
        reader._check_identity(stream)
        raw = vocab.read_bytes(stream, offset, length)
        record["ranges"].append(
            {"offset": offset, "length": length, "sha256": vocab.sha256(raw)}
        )
        reader._check_identity(stream)
    np = vocab.load_numpy()
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype("<u4")
        np.left_shift(bits, 16, out=bits)
        return bits.view("<f4")
    return np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4"}[dtype]).astype(
        np.float32, copy=False
    )


def q_structure(q):
    """All row norms and absolute entry range; no full Q @ Q.T or fitting."""
    np = vocab.load_numpy()
    if not np.isfinite(q).all():
        return {"finite": False}
    squared_norms = []
    abs_min, abs_max = math.inf, 0.0
    # Bound FP64 temporary memory instead of promoting the entire matrix.
    for start in range(0, q.shape[0], 64):
        block = q[start : start + 64].astype(np.float64)
        squared_norms.extend(np.einsum("ij,ij->i", block, block).tolist())
        magnitudes = np.abs(block)
        abs_min = min(abs_min, float(magnitudes.min()))
        abs_max = max(abs_max, float(magnitudes.max()))
    return {
        "finite": True,
        "abs_entry_min": abs_min,
        "abs_entry_max": abs_max,
        "row_squared_norm": {
            "min": min(squared_norms),
            "median": float(np.median(squared_norms)),
            "max": max(squared_norms),
            "count": len(squared_norms),
        },
        "scope": "All row norms only; off-diagonal orthogonality is not proved",
    }


def run_probe(args, report):
    if args.threads < 1 or not 1 <= args.weight_rows <= 64:
        raise ValueError("threads must be positive; weight-rows must be in [1, 64]")
    np = vocab.load_numpy(args.threads)
    report["versions"]["numpy"] = np.__version__
    report["thread_environment"] = {
        key: os.environ.get(key) for key in vocab.THREAD_ENV
    }
    target = vocab.Checkpoint(args.target, report["files"])
    draft = vocab.Checkpoint(args.draft, report["files"])
    # Reject nonstandard NaN/Inf config values before attaching them to the
    # report; otherwise even the FAILED report would not be valid JSON.
    json.dumps(target.config_evidence, allow_nan=False)
    json.dumps(draft.config_evidence, allow_nan=False)
    report["configs"] = {
        "target": target.config_evidence,
        "draft": draft.config_evidence,
    }
    report["indices"] = {
        label: [
            {
                "path": x["path"],
                "sha256": x["sha256"],
                "weight_map_entries": len(x["content"]["weight_map"]),
            }
            for x in checkpoint.indices
        ]
        for label, checkpoint in (("target", target), ("draft", draft))
    }
    fc_reader, fc_key = draft.find_tensor(
        ("fc.weight", "model.fc.weight", "encoder.fc.weight", "model.encoder.fc.weight")
    )
    config = draft.config.get("transformer_layer_config", draft.config)
    width = int(config["hidden_size"])
    layers = draft.config["aux_hidden_state_layer_ids"]
    eps = float(config.get("rms_norm_eps", 1e-6))
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected finite positive RMSNorm epsilon")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Expected nonempty aux_hidden_state_layer_ids")
    if fc_reader.matrix_info(fc_key)["shape"] != [width, len(layers) * width]:
        raise ValueError(
            "FC shape does not match draft hidden size / captured features"
        )
    description, description_evidence = vocab.read_json_file(
        target.path / "quant_model_description.json"
    )
    json.dumps(description, allow_nan=False)
    q_path = description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    if not isinstance(q_path, str):
        raise ValueError("Invalid Q path")
    report["quant_description"] = {
        "path": description_evidence["path"],
        "sha256": description_evidence["sha256"],
        "selected": {
            k: description[k] for k in ("optional", "is_rot_used") if k in description
        },
    }
    q_reader = target.get_file(target.path / q_path)
    if q_reader.matrix_info("global_rotation")["shape"] != [width, width]:
        raise ValueError("Q shape does not match the FC input block width")
    print("READ Q, full draft FC and hidden_norm (CPU only)", flush=True)
    q = q_reader.read_tensor("global_rotation")
    fc = fc_reader.read_tensor(fc_key)
    norm_weight = read_hidden_norm(draft, width)
    report["q_structure"] = q_structure(q)
    inputs = (
        np.random.default_rng(0)
        .standard_normal((3, len(layers), width))
        .astype(np.float32)
    )
    scales = np.array([1.0, 0.01, 0.0001], dtype=np.float32)
    inputs *= scales[:, None, None]
    selected = np.linspace(
        0, width - 1, min(args.weight_rows, width), dtype=np.int64
    ).tolist()
    report["fixed_inputs"] = {
        "seed": 0,
        "shape": list(inputs.shape),
        "scales": scales.tolist(),
        "sha256": vocab.sha256(inputs.tobytes()),
        "meaning": "Synthetic original-coordinate inputs; not captured target hidden or token IDs",
    }
    report["rms_norm_eps"] = eps
    print(
        "COMPARE full FC / hidden_norm and selected converted weight rows", flush=True
    )
    report["results"] = evaluate_fc(fc, q, norm_weight, eps, inputs, selected)
    results = report["results"]
    comparisons = [results["q_roundtrip"]] + [
        item
        for name in ("comparisons", "sampled_weights")
        for item in results[name].values()
    ]
    review = not all(results["finite"].values()) or any(
        x["needs_review"] for x in comparisons
    )
    report["status"] = (
        "NUMERICAL_REVIEW_REQUIRED" if review else "FIXED_INPUT_DIAGNOSTIC_COLLECTED"
    )
    return 2 if review else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("/workspace/weight/GLM-5.2-w8a8")
    )
    parser.add_argument(
        "--draft", type=Path, default=Path("/workspace/weight/GLM-5.2-DSpark-NPU-0805")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/home/tyj/glm52-ms1/evidence")
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--weight-rows", type=int, default=16)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"quarot-fc-{stamp}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {out}", flush=True)
    started = time.monotonic()
    report = {
        "status": "STARTED",
        "files": [],
        "inputs": {"target": str(args.target), "draft": str(args.draft)},
        "requested_threads": args.threads,
        "requested_weight_rows": args.weight_rows,
        "versions": {"python": platform.python_version()},
        "notes": [
            "No self-defined precision threshold; collection is not a correctness PASS.",
            "Full outputs compare H F.T, Hq F.T and (Hq Q.T) F.T with actual hidden_norm.",
            "Input-side equivalence is not a complete converted FC or its BF16 rounding.",
            "Actual converted weight rows are sampled; BF16 storage is simulated with FP32 GEMM.",
            "No real target hidden, NPU arithmetic, full draft, acceptance or performance validation.",
            "No Q inverse, fitted correction, R application, checkpoint write or model import.",
        ],
    }
    code = 1
    try:
        report["git_head"] = vocab.git_head()
        report["runner_sha256"] = vocab.sha256(Path(__file__).read_bytes())
        report["reader_sha256"] = vocab.sha256(Path(vocab.__file__).read_bytes())
        code = run_probe(args, report)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
        print(f"FAILED: {exc}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(report["status"], flush=True)
    results = report.get("results", {})
    for name in ("comparisons", "sampled_weights"):
        for label, item in results.get(name, {}).items():
            summary = item["summary"]
            print(
                f"{label}: relative_l2 per input = {[r['relative_l2'] for r in item['rows']]}; max = {summary['relative_l2']['max']}"
            )
    print("Diagnostic only: no runtime adaptation or acceptance-rate result.")
    return code


if __name__ == "__main__":
    sys.exit(main())
