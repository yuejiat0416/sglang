#!/usr/bin/env python3
"""Run one offline dataset on one existing two-node mode; never restart servers."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from client_common import (
    PARENT,
    benchmark_context,
    fetch_text,
    graph_evidence,
    import_benchmark,
    prepare_run,
    selected_server,
    sha,
    validate_server,
    write_json,
)
from config import MODES, load_config
from offline_dataset import score_response

sys.path.insert(0, str(PARENT))
from bench_gsm8k_modes import capture_responses  # noqa: E402
from gsm8k_mode_stats import summarize_responses  # noqa: E402


def read_fixture(path):
    fixture = json.loads(Path(path).read_text())
    if fixture.get("status") != "DATASET_PREPARED" or fixture.get("dataset") not in {
        "gsm8k",
        "gpqa",
    }:
        raise ValueError("Prepare a GSM8K or GPQA offline fixture first")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset is empty")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if (
            not isinstance(case.get("messages"), list)
            or not case["messages"]
            or "answer" not in case
        ):
            raise ValueError("Fixture must have messages and a locally retained answer")
    return fixture


def make_requests(cases, run_id, max_tokens, repeats=1):
    rows, mapping = [], {}
    for repeat in range(repeats):
        for case in cases:
            rid = f"{run_id}-r{repeat + 1}-{case['id']}"
            # Send only the prompt. The source case and gold answer remain local.
            rows.append(
                {
                    "messages": case["messages"],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "rid": rid,
                    "cache_salt": rid,
                    "return_meta_info": True,
                    "return_spec_tokens_details": True,
                    "return_token_ids": True,
                }
            )
            mapping[rid] = {"case": case, "repeat": repeat + 1}
    return rows, mapping


def bench_arguments(cfg, run, count, max_tokens, concurrency):
    return [
        "--backend",
        "sglang-oai-chat",
        "--host",
        cfg["nodes"][0]["host"],
        "--port",
        str(cfg["port"]),
        "--model",
        cfg["target_model"],
        "--served-model-name",
        cfg["served_model_name"],
        "--tokenizer",
        cfg["tokenizer"],
        "--dataset-name",
        "openai",
        "--dataset-path",
        str(run / "requests.jsonl"),
        "--num-prompts",
        str(count),
        "--sharegpt-output-len",
        str(max_tokens),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "42",
        "--warmup-requests",
        "0",
        "--disable-ignore-eos",
        "--disable-stream",
        "--output-details",
        "--output-file",
        str(run / "benchmark.jsonl"),
    ]


def score_collected(summary, mapping, dataset):
    scored = []
    for row in summary["per_question"]:
        entry = mapping[row["id"]]
        response = row.get("response")
        response = response if isinstance(response, dict) else {}
        choices = response.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            else {}
        )
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        # Do not score hidden reasoning as the final answer. A response that
        # never reaches a final answer remains unscored/incorrect for this sample.
        result = score_response(
            dataset, entry["case"]["answer"], content, row.get("finish_reason")
        )
        if not row.get("valid"):
            result.update(correct=None, status="INVALID_RESPONSE")
        scored.append(
            {
                "id": row["id"],
                "case_id": entry["case"]["id"],
                "repeat": entry["repeat"],
                "score": result,
                "response_token_ids": choice.get("response_token_ids"),
                "prompt_token_ids": choice.get("prompt_token_ids"),
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "completion_tokens": row.get("completion_tokens"),
                "counters": row.get("counters"),
                "valid": row.get("valid"),
                "finish_reason": row.get("finish_reason"),
            }
        )
    return {
        "samples": len(scored),
        "correct": sum(r["score"]["correct"] is True for r in scored),
        "accuracy": sum(r["score"]["correct"] is True for r in scored) / len(scored),
        "unresolved": sum(r["score"]["correct"] is None for r in scored),
        "truncated": sum(r["finish_reason"] == "length" for r in scored),
        "per_question": scored,
        "scope": "Local explicit final-answer scoring; not EvalScope or formal GPQA qualification. All requested samples stay in the denominator.",
    }


def run(args, cfg=None):
    cfg = load_config(args.config) if cfg is None else cfg
    fixture = read_fixture(args.fixture)
    cases = fixture["cases"][: args.limit] if args.limit else fixture["cases"]
    dataset = fixture["dataset"]
    output = prepare_run(cfg, dataset, args.mode)
    rows, mapping = make_requests(cases, output.name, args.max_tokens, args.repeats)
    (output / "requests.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    argv = bench_arguments(cfg, output, len(rows), args.max_tokens, args.concurrency)
    protocol = {
        "dataset": dataset,
        "fixture_sha256": sha(args.fixture),
        "case_ids": [case["id"] for case in cases],
        "prompt_protocol": fixture.get("prompt_protocol"),
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "temperature": 0,
        "top_p": 1,
        "eos": "honored",
        "stream": False,
        "cache_policy": "unique salt for every request",
        "benchmark_argv": argv,
        "source": fixture.get("source"),
    }
    write_json(output / "protocol.json", protocol)
    responses, errors, before = [], [], {}
    metrics_before = metrics_after = None
    try:
        before = json.loads(fetch_text(cfg["base_url"] + "/server_info"))
        write_json(output / "server_info.before.json", before)
        validate_server(before, cfg, args.mode)
        metrics_before = fetch_text(cfg["base_url"] + "/metrics")
        (output / "metrics.before.txt").write_text(metrics_before)
        with benchmark_context():
            bench = import_benchmark()
            write_json(
                output / "benchmark-source.json",
                {"file": bench.__file__, "sha256": sha(bench.__file__)},
            )
            sys.argv = ["sglang.benchmark.serving", *argv]
            with capture_responses(bench, output / "responses.jsonl", responses):
                bench.cli_main()
        result = json.loads((output / "benchmark.jsonl").read_text().splitlines()[-1])
        if result.get("completed") != len(rows):
            raise ValueError(
                f"Only {result.get('completed')}/{len(rows)} requests completed"
            )
        after = json.loads(fetch_text(cfg["base_url"] + "/server_info"))
        write_json(output / "server_info.after.json", after)
        if selected_server(before) != selected_server(after):
            raise ValueError("Server configuration changed while running")
        metrics_after = fetch_text(cfg["base_url"] + "/metrics")
        (output / "metrics.after.txt").write_text(metrics_after)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    collected = summarize_responses(list(mapping), responses, args.mode)
    errors.extend(collected["issues"])
    accuracy = score_collected(collected, mapping, dataset)
    summary = {
        "status": "DATASET_COLLECTED" if not errors else "DATASET_INCOMPLETE",
        "complete": not errors,
        "dataset": dataset,
        "mode": args.mode,
        "protocol": protocol,
        "server_configuration": selected_server(before),
        "acceptance": collected["aggregate"] if not errors else None,
        "partial_valid_counts": collected["partial_valid_counts"],
        "accuracy": accuracy,
        "graph": graph_evidence(metrics_before, metrics_after, args.mode),
        "issues": errors,
        "qualification": "NOT_ASSIGNED",
        "limits": [
            "Default ten questions per dataset; GSM8K and GPQA are never combined.",
            "A ten-question score cannot prove a one-percent regression bound.",
            "Nonstream timings are not TTFT/TPOT performance evidence.",
            "No automatic rerun, server restart, cache flush, or framework changes.",
        ],
    }
    write_json(output / "summary.json", summary)
    print(summary["status"], "correct=", accuracy["correct"], "/", accuracy["samples"])
    print("Acceptance:", json.dumps(summary["acceptance"]))
    for error in errors:
        print("ISSUE:", error)
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="0 explicitly selects the full prepared fixture",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Keep identical across compared modes; truncations remain in report",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 0 or min(args.max_tokens, args.concurrency, args.repeats) < 1:
        parser.error("limit must be >=0; other numeric options must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
