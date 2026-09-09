# SPDX-License-Identifier: Apache-2.0
"""Sequential fixed-input diagnostics against the existing DSpark service.

No model imports, cache resets, retries, server mutations or generated-code
execution. Collection success is not a quality or performance PASS.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import collect_acceptance_trace as collector

PLAN_PATH = Path(__file__).with_name("acceptance-cases.json")


def read_json(path):
    return json.loads(path.read_text())


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_plan(path):
    plan = read_json(path)
    for name in ("max_tokens", "repeats"):
        if type(plan.get(name)) is not int or plan[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not plan.get("suite_id") or not plan.get("expected_server_configuration"):
        raise ValueError("Plan must identify its suite and expected service")
    if not isinstance(plan.get("cases"), list) or not plan["cases"]:
        raise ValueError("Plan must contain cases")
    ids = set()
    for case in plan["cases"]:
        name = case.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(
                "Case IDs must use lowercase letters, digits or underscore"
            )
        if name in ids:
            raise ValueError(f"Duplicate case ID: {name}")
        ids.add(name)
        for key in ("prompt", "category", "manual_review"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Missing {key} for {name}")
    return plan


def validate_server(info, expected):
    mismatches = {
        key: {"expected": value, "observed": info.get(key)}
        for key, value in expected.items()
        if key not in info or info[key] != value
    }
    if mismatches:
        raise ValueError(f"Service configuration differs from fixed plan: {mismatches}")
    for _, payload in collector.trace_payloads(info):
        if payload.get("gamma") != 8 or payload.get("verify_num_draft_tokens") != 9:
            raise ValueError("This fixed diagnostic expects gamma=8 and verify width=9")


def summarize(plan, results):
    total = len(plan["cases"]) * plan["repeats"]
    trusted = [row for row in results if row["status"] == "TRACE_COLLECTED"]
    counts = {
        key: sum(row["summary"]["api"][key] for row in trusted)
        for key in ("accepted_drafts", "proposed_drafts", "verify_rounds")
    }
    complete = len(results) == total and len(trusted) == total
    aggregate = None
    if complete:
        rate = counts["accepted_drafts"] / counts["proposed_drafts"]
        aggregate = {
            **counts,
            "accept_rate": rate,
            "first_draft_accepted_rounds": sum(
                row["summary"]["first_draft_accepted_rounds"] for row in trusted
            ),
            "strictly_above_user_reference_0_5": rate > 0.5,
        }
    comparisons = []
    for case in plan["cases"]:
        rows = [row for row in trusted if row["case_id"] == case["id"]]
        for row in rows[1:]:
            before = rows[0].get("response_tokens_sha256")
            after = row.get("response_tokens_sha256")
            comparisons.append(
                {
                    "case_id": case["id"],
                    "reference_repeat": rows[0]["repeat"],
                    "repeat": row["repeat"],
                    "same_output_token_ids": before == after
                    if before is not None and after is not None
                    else None,
                }
            )
    return {
        "status": "SUITE_COLLECTED" if complete else "SUITE_INCOMPLETE",
        "suite_id": plan["suite_id"],
        "plan_sha256": json_sha256(plan),
        "total_requests": total,
        "attempted_requests": len(results),
        "not_attempted_requests": total - len(results),
        "trusted_requests": len(trusted),
        "aggregate": aggregate,
        "trusted_prefix_totals": counts,
        "length_finished_requests": sum(
            row["summary"].get("finish_reason") == "length" for row in trusted
        ),
        "quality_review": "NOT_REVIEWED",
        "repeat_comparisons": comparisons,
        "cases": results,
        "limits": [
            "Fixed development examples, not a formal business workload or acceptance PASS.",
            "Only complete trusted collection gets an aggregate rate; no cherry-picking after failures.",
            "Natural EOS is allowed; length finish means the output budget was reached, not automatic quality failure or success.",
            "Output/trace agreement is within each request, not target-only equivalence.",
            "No cache flush: later repeats can reuse cache. Inspect saved cached_tokens.",
            "Full-history server_info export and debug recording affect timing; not a performance benchmark.",
            "A timeout may leave a server request active. No retry or next request is sent automatically.",
        ],
    }


def write_answers(output, plan, results):
    lines = [
        "# 固定用例响应：待人工审视",
        "",
        "采集完成不代表回答正确；生成代码仅展示，不执行。",
        "",
    ]
    cases = {case["id"]: case for case in plan["cases"]}
    for row in results:
        case = cases[row["case_id"]]
        lines.extend(
            [
                f"## {row['case_id']} / repeat {row['repeat']}",
                "",
                f"状态：{row['status']}；人工审视：NOT_REVIEWED",
                "",
                f"输入：{case['prompt']}",
                "",
                f"审视要点：{case['manual_review']}",
                "",
            ]
        )
        path = output / row["directory"] / "response.json"
        if path.exists():
            response = read_json(path)
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            lines.extend([f"结束原因：{choice.get('finish_reason')}", ""])
            for key in ("reasoning_content", "content"):
                value = message.get(key)
                if isinstance(value, str):
                    longest = max((len(s) for s in re.findall(r"`+", value)), default=0)
                    fence = "`" * max(3, longest + 1)
                    lines.extend([f"{key}：", "", fence + "text", value, fence, ""])
    (output / "responses.md").write_text("\n".join(lines) + "\n")


def run_suite(url, output, plan, timeout=600, opener=None):
    output.mkdir(parents=True, exist_ok=True)
    collector.write_json(output / "suite-plan.json", plan)
    results = []
    for repeat in range(1, plan["repeats"] + 1):
        for case in plan["cases"]:
            directory = f"{len(results) + 1:02d}-{case['id']}-r{repeat}"
            case_output = output / directory
            case_output.mkdir()
            row = {"case_id": case["id"], "repeat": repeat, "directory": directory}
            print(f"RUN {directory}", flush=True)
            try:
                summary = collector.collect(
                    url,
                    case_output,
                    plan["max_tokens"],
                    timeout=timeout,
                    opener=opener,
                    prompt=case["prompt"],
                    validate_server=lambda info: validate_server(
                        info, plan["expected_server_configuration"]
                    ),
                )
                row.update(status=summary["status"], summary=summary)
                response = read_json(case_output / "response.json")
                choice = response["choices"][0]
                meta = choice.get("meta_info") or {}
                row.update(
                    response_tokens_sha256=json_sha256(choice["response_token_ids"]),
                    cached_tokens=meta.get("cached_tokens"),
                    diagnostic_e2e_latency=meta.get("e2e_latency"),
                    output_budget_reached=choice.get("finish_reason") == "length",
                )
                api = summary["api"]
                print(
                    f"{row['status']} A/P/N={api['accepted_drafts']}/"
                    f"{api['proposed_drafts']}/{api['verify_rounds']} "
                    f"rate={api['accept_rate']:.6%} finish={summary['finish_reason']}",
                    flush=True,
                )
            except (Exception, KeyboardInterrupt) as exc:
                row.update(
                    status="INTERRUPTED"
                    if isinstance(exc, KeyboardInterrupt)
                    else "COLLECTION_FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
                response_path = case_output / "response.json"
                if response_path.exists():
                    response = read_json(response_path)
                    details = response.get("sglext", {}).get("spec_tokens_details", {})
                    if (
                        row["status"] != "INTERRUPTED"
                        and details.get("spec_verify_ct") == 0
                    ):
                        row["status"] = "NO_SPECULATIVE_ROUNDS"
                print(f"{row['status']} {row['error']}", flush=True)
            results.append(row)
            report = summarize(plan, results)
            collector.write_json(case_output / "suite-case.json", row)
            collector.write_json(output / "suite-summary.json", report)
            write_answers(output, plan, results)
            if row["status"] != "TRACE_COLLECTED":
                return report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://61.47.19.71:8810")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("/home/tyj/glm52-ms1/evidence")
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    plan = load_plan(PLAN_PATH)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.evidence_root / f"acceptance-suite-{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    script = Path(__file__).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
    )
    collector.write_json(
        output / "collector.json",
        {
            "collector_git_head": revision.stdout.strip(),
            "url": args.url,
            "files_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (script, Path(collector.__file__), PLAN_PATH)
            },
            "note": "Client files only; the existing server keeps its prior loaded code. No restart is needed for this tool update.",
        },
    )
    print(f"Evidence: {output}", flush=True)
    report = run_suite(args.url, output, plan, timeout=args.timeout)
    print(report["status"])
    if report["aggregate"] is not None:
        print("Aggregate:", json.dumps(report["aggregate"]))
    print(f"Length finishes: {report['length_finished_requests']}")
    print(f"Answers for manual review: {output / 'responses.md'}")
    print(
        "Diagnostic collection only; no quality, performance or formal acceptance PASS."
    )
    return 0 if report["status"] == "SUITE_COLLECTED" else 1


if __name__ == "__main__":
    sys.exit(main())
