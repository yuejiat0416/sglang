#!/usr/bin/env python3
"""Offline paired accuracy comparison and coverage inventory; no NPU access."""

import argparse
import json
from pathlib import Path

from client_common import write_json
from config import MODES


def compare(baseline, candidate):
    issues = []
    if not baseline.get("mode", "").startswith("target-"):
        issues.append("Baseline must be target-only")
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if not value.get("complete"):
            issues.append(f"{name} collection incomplete")
    for key in (
        "dataset",
        "fixture_sha256",
        "case_ids",
        "prompt_protocol",
        "max_tokens",
        "repeats",
        "concurrency",
        "temperature",
        "top_p",
        "eos",
        "cache_policy",
    ):
        if baseline.get("protocol", {}).get(key) != candidate.get("protocol", {}).get(
            key
        ):
            issues.append(f"Different protocol: {key}")
    for key in (
        "device",
        "model_path",
        "tp_size",
        "dp_size",
        "nnodes",
        "quantization",
        "served_model_name",
    ):
        if baseline.get("server_configuration", {}).get(key) != candidate.get(
            "server_configuration", {}
        ).get(key):
            issues.append(f"Different target/topology: {key}")

    def index(value):
        rows = value.get("accuracy", {}).get("per_question", [])
        keys = [(r["case_id"], r["repeat"]) for r in rows]
        if not rows or len(keys) != len(set(keys)):
            issues.append("Empty or duplicate case/repeat rows")
        return dict(zip(keys, rows))

    left, right = index(baseline), index(candidate)
    if set(left) != set(right):
        issues.append("Case/repeat sets differ")
    rows = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        prompts = a.get("prompt_token_ids"), b.get("prompt_token_ids")
        if any(p is None for p in prompts):
            issues.append(f"{key}: prompt token IDs unavailable")
        elif prompts[0] != prompts[1]:
            issues.append(f"{key}: actual prompt token IDs differ")
        tokens = a.get("response_token_ids"), b.get("response_token_ids")
        equal = None if any(t is None for t in tokens) else tokens[0] == tokens[1]
        divergence = None
        if equal is False:
            divergence = next(
                (i for i, (x, y) in enumerate(zip(*tokens)) if x != y),
                min(map(len, tokens)),
            )
        rows.append(
            {
                "case_id": key[0],
                "repeat": key[1],
                "baseline_correct": a["score"].get("correct"),
                "candidate_correct": b["score"].get("correct"),
                "lost_correct": a["score"].get("correct") is True
                and b["score"].get("correct") is not True,
                "recovered_correct": a["score"].get("correct") is not True
                and b["score"].get("correct") is True,
                "response_tokens_equal": equal,
                "first_different_token_index": divergence,
                "baseline_finish": a.get("finish_reason"),
                "candidate_finish": b.get("finish_reason"),
            }
        )
    score_a = baseline.get("accuracy", {}).get("accuracy")
    score_b = candidate.get("accuracy", {}).get("accuracy")
    return {
        "status": "PAIRED_COMPARISON_COLLECTED"
        if not issues
        else "COMPARISON_NOT_COMPARABLE",
        "comparable": not issues,
        "baseline_mode": baseline.get("mode"),
        "candidate_mode": candidate.get("mode"),
        "dataset": baseline.get("dataset"),
        "sample_count": len(rows),
        "accuracy_delta_candidate_minus_baseline": score_b - score_a
        if not issues and score_a is not None and score_b is not None
        else None,
        "lost_correct": sum(r["lost_correct"] for r in rows),
        "recovered_correct": sum(r["recovered_correct"] for r in rows),
        "per_question": rows,
        "issues": issues,
        "qualification": "NOT_ASSIGNED",
        "limits": [
            "Identical small-sample accuracy does not prove no model-wide regression or a one-percent bound.",
            "Token differences identify an investigation point, not by themselves an inference bug.",
            "Target/draft weight identity and runtime source still require both node manifests and launch logs.",
        ],
    }


def inventory(root):
    records = []
    for path in sorted(Path(root).rglob("summary.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            records.append(
                {"path": str(path), "status": "UNREADABLE", "error": str(exc)}
            )
            continue
        records.append(
            {
                "path": str(path),
                "mode": data.get("mode"),
                "dataset": data.get("dataset"),
                "status": data.get("status"),
                "complete": data.get("complete"),
                "stage": data.get("stage"),
                "cache_percent": data.get(
                    "cache_percent", data.get("cache_percent_label")
                ),
            }
        )
    missing = []
    for mode in MODES:
        for dataset in ("gsm8k", "gpqa"):
            if not any(
                r.get("mode") == mode
                and r.get("dataset") == dataset
                and r.get("complete")
                for r in records
            ):
                missing.append(f"{mode}/{dataset}")
    missing_prefix = []
    for mode in MODES:
        for stage in ("quick", "load"):
            for percent in (0, 50, 90):
                if not any(
                    r.get("mode") == mode
                    and r.get("stage") == stage
                    and r.get("cache_percent") == percent
                    and r.get("complete")
                    for r in records
                ):
                    missing_prefix.append(f"{mode}/{stage}/cache{percent}")
    return {
        "status": "EVIDENCE_INVENTORY",
        "root": str(Path(root).resolve()),
        "records": records,
        "missing_complete_dataset_runs": missing,
        "missing_complete_prefix_runs": missing_prefix,
        "qualification": "NOT_ASSIGNED",
        "note": "Records are never selected by best score. Review every run, prefix/load/graph evidence and both node manifests separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pair = sub.add_parser("compare")
    pair.add_argument("--baseline", type=Path, required=True)
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    listing = sub.add_parser("inventory")
    listing.add_argument("--evidence", type=Path, required=True)
    listing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
        if args.action == "compare"
        else inventory(args.evidence)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result["status"], args.output)
    return int(result.get("comparable") is False)


if __name__ == "__main__":
    raise SystemExit(main())
