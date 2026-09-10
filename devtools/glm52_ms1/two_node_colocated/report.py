#!/usr/bin/env python3
"""Offline paired accuracy comparison and coverage inventory; no NPU access."""

import argparse
import json
import re
from pathlib import Path

from client_common import write_json
from config import MODES


def latest_attempts(root):
    """Use latest timestamped attempt, including failures; never select best score."""
    groups, history = {}, []
    for path in sorted(
        (Path(root) / "evidence/two-node-colocated").glob("*/summary.json")
    ):
        value = json.loads(path.read_text())
        mode = value.get("mode")
        kind = value.get("dataset") or value.get("stage")
        if mode not in MODES or kind not in {"gsm8k", "gpqa", "check", "quick", "load"}:
            continue
        stamp = re.search(r"(\d{8}T\d{6}Z)-[a-f0-9]+$", path.parent.name)
        if stamp is None:
            raise ValueError(f"Missing run timestamp: {path}")
        record = {"path": str(path), "timestamp": stamp.group(1), "value": value}
        history.append({k: v for k, v in record.items() if k != "value"})
        groups.setdefault((mode, kind), []).append(record)
    selected = {}
    for key, records in groups.items():
        records.sort(key=lambda row: row["timestamp"])
        if len(records) > 1 and records[-1]["timestamp"] == records[-2]["timestamp"]:
            raise ValueError(
                f"Ambiguous simultaneous runs for {key}; do not choose by score"
            )
        selected[key] = records[-1]
    return selected, history


def compare_performance(baseline, candidate):
    """Compare one mode/cache cell using recorded native timings and actual IDs."""
    issues = []
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        if not record.get("complete") or record.get("invalid_requests"):
            issues.append(f"{label} measurement incomplete or invalid")
    for key in (
        "expected_cached_tokens",
        "measured_requests",
        "concurrency",
        "used_dp_lanes",
        "requested_minimum_prompts",
        "requested_minimum_duration_seconds",
    ):
        if baseline.get(key) != candidate.get(key):
            issues.append(f"Different performance protocol: {key}")

    def inputs(record):
        return [
            (r.get("index"), r.get("dp_rank"), r.get("input_sha256"))
            for r in record.get("results", [])
        ]

    if not inputs(baseline) or any(
        row[2] is None for row in inputs(baseline) + inputs(candidate)
    ):
        issues.append("Actual input fingerprints missing")
    elif inputs(baseline) != inputs(candidate):
        issues.append("Actual input tokens / DP assignments differ")
    bm, cm = (
        baseline.get("native_bench_metrics", {}),
        candidate.get("native_bench_metrics", {}),
    )

    def ratio(numerator, denominator):
        return (
            numerator / denominator
            if not issues
            and isinstance(numerator, (int, float))
            and isinstance(denominator, (int, float))
            and denominator > 0
            else None
        )

    return {
        "comparable": not issues,
        "issues": issues,
        "output_throughput_ratio": ratio(
            cm.get("output_throughput"), bm.get("output_throughput")
        ),
        "ttft_speedup": ratio(bm.get("mean_ttft_ms"), cm.get("mean_ttft_ms")),
        "tpot_speedup": ratio(bm.get("mean_tpot_ms"), cm.get("mean_tpot_ms")),
    }


def campaign_report(root):
    root.mkdir(parents=True, exist_ok=True)
    selected, history = latest_attempts(root)
    missing, accuracy, performance = [], [], []
    for dataset in ("gsm8k", "gpqa"):
        for graph in ("eager", "graph"):
            a = selected.get((f"target-{graph}", dataset))
            b = selected.get((f"dspark-{graph}", dataset))
            if not a or not b:
                missing.append(f"{dataset}: target-{graph} / dspark-{graph}")
                continue
            pair = compare(a["value"], b["value"])
            pair.update(baseline_path=a["path"], candidate_path=b["path"])
            for label, record in (("baseline", a), ("candidate", b)):
                pair[label + "_accuracy"] = record["value"].get("accuracy", {})
                pair[label + "_graph"] = record["value"].get("graph", {})
            accuracy.append(pair)
    for mode in MODES:
        attempt = selected.get((mode, "load"))
        for percent in (0, 50, 90):
            if not attempt:
                missing.append(f"performance: {mode}/cache{percent}")
                continue
            path = Path(attempt["path"]).parent / f"cache{percent}/summary.json"
            if not path.exists():
                missing.append(f"performance: {mode}/cache{percent}")
                continue
            cell = json.loads(path.read_text())
            cell["complete"] = cell.get("complete") and attempt["value"].get("complete")
            baseline_mode = "target-graph" if mode.endswith("graph") else "target-eager"
            baseline = selected.get((baseline_mode, "load"))
            comparison = {
                "comparable": False,
                "issues": ["Matching target-only run missing"],
            }
            baseline_path = None
            if baseline:
                bp = Path(baseline["path"]).parent / f"cache{percent}/summary.json"
                if bp.exists():
                    baseline_path = str(bp)
                    basecell = json.loads(bp.read_text())
                    basecell["complete"] = basecell.get("complete") and baseline[
                        "value"
                    ].get("complete")
                    comparison = compare_performance(basecell, cell)
                    for key in (
                        "device",
                        "model_path",
                        "tp_size",
                        "dp_size",
                        "nnodes",
                        "quantization",
                        "page_size",
                        "context_length",
                        "max_running_requests",
                    ):
                        left = (
                            baseline["value"].get("server_configuration", {}).get(key)
                        )
                        right = (
                            attempt["value"].get("server_configuration", {}).get(key)
                        )
                        if left is None or right is None:
                            comparison["issues"].append(
                                f"Missing target/topology: {key}"
                            )
                        elif left != right:
                            comparison["issues"].append(
                                f"Different target/topology: {key}"
                            )
                    if comparison["issues"]:
                        comparison.update(
                            comparable=False,
                            output_throughput_ratio=None,
                            ttft_speedup=None,
                            tpot_speedup=None,
                        )
            performance.append(
                {
                    "mode": mode,
                    "cache": percent,
                    "path": str(path),
                    "complete": cell.get("complete"),
                    "accept_rate": cell.get("accept_rate"),
                    "above_50_percent": cell.get("strictly_above_0_5"),
                    "counts": {
                        k: cell.get(k)
                        for k in ("accepted_drafts", "proposed_drafts", "verify_rounds")
                    },
                    "metrics": cell.get("native_bench_metrics", {}),
                    "graph": cell.get("graph_evidence", {}),
                    "baseline_mode": baseline_mode,
                    "baseline_path": baseline_path,
                    "comparison": comparison,
                    "expected_cached_tokens": cell.get("expected_cached_tokens"),
                    "measurement_seconds": cell.get("measurement_seconds"),
                }
            )
    result = {
        "status": "TEST_MATRIX_RECORDED",
        "missing": missing,
        "accuracy": accuracy,
        "performance": performance,
        "history": history,
        "qualification": "NOT_ASSIGNED",
        "selection": "Latest attempt per mode/stage, including failures; all paths retained",
    }
    write_json(root / "comparison.json", result)

    def number(value, factor=1):
        return "—" if value is None else f"{value * factor:.3f}"

    lines = [
        "# 双机测试对比",
        "",
        "采用每个模式/阶段最新一次尝试，包括失败；所有历史路径保存在 comparison.json。",
        "10题抽样不证明整体精度不下降；本页不自动授予准出。",
        "",
        "## 精度：每数据集10题，eager/graph分别配对",
        "",
        "| 数据集 | 模式 | Target正确 | DSpark正确 | 丢失正确题 | 未解析/截断(target, draft) | 协议可比 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in accuracy:
        a, b = row["baseline_accuracy"], row["candidate_accuracy"]
        lines.append(
            f"| {row['dataset']} | {row['candidate_mode']} | {a.get('correct')}/{a.get('samples')} | {b.get('correct')}/{b.get('samples')} | {row['lost_correct']} | {a.get('unresolved')}/{a.get('truncated')}, {b.get('unresolved')}/{b.get('truncated')} | {row['comparable']} |"
        )
    lines += [
        "",
        "## 性能：131072/1024，三档分别统计",
        "",
        "倍率以相同eager/graph的target-only为基线；大于1表示更快。",
        "",
        "| 模式 | Cache标签 | 命中tokens | TTFT均值ms | TPOT均值ms | 输出tok/s | 吞吐倍率 | 接受率% | >50% | 完整/可比 | Target回放观测 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in performance:
        m, c = row["metrics"], row["comparison"]
        lines.append(
            f"| {row['mode']} | {row['cache']} | {row['expected_cached_tokens']} | {number(m.get('mean_ttft_ms'))} | {number(m.get('mean_tpot_ms'))} | {number(m.get('output_throughput'))} | {number(c.get('output_throughput_ratio'))} | {number(row['accept_rate'], 100)} | {row['above_50_percent']} | {row['complete']}/{c['comparable']} | {row['graph'].get('target_replay_observed')} |"
        )
    lines += [
        "",
        "Draft回放尚需两端日志/trace证明；HTTP指标不能单独确认DSpark所有阶段都在回放。",
        "压测口径为固定请求数与并发；正式持续时间/SLO仍需与测试确认，0.5不按warmup或bonus计算。",
        "",
        "## 缺失记录",
        "",
        *(f"- {item}" for item in missing),
        "",
        "详细逐题变化、A/P/N、延迟倍率、协议差异及结果路径见同目录 comparison.json。",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"对比表：{root / 'comparison.md'}")
    print(f"缺失记录：{len(missing)}；未授予精度/性能准出")
    return 0


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
