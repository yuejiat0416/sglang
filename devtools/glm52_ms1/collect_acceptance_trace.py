# SPDX-License-Identifier: Apache-2.0
"""Collect one chat request and existing DSpark records; no model/runtime imports.

Run against the existing static/eager diagnostic service started with
SGLANG_DSPARK_DEBUG_DUMP=core,reqs. This is evidence collection, not an acceptance
quality test or a performance benchmark. Raw evidence stays outside the repo.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROMPT = "请用三句话解释为什么天空看起来是蓝色的。"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def request_json(opener, url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def trace_payloads(server_info):
    return [
        (index, state["dspark_info_record"])
        for index, state in enumerate(server_info.get("internal_states", []))
        if state.get("dspark_info_record") is not None
    ]


def check_recording(server_info):
    payloads = trace_payloads(server_info)
    if not payloads or not any(
        {"core", "reqs"}.issubset(payload.get("components", []))
        for _, payload in payloads
    ):
        raise ValueError(
            "No core,reqs recording. Restart the existing service with "
            "SGLANG_DSPARK_DEBUG_DUMP=core,reqs. No chat request was sent."
        )
    for _, payload in payloads:
        if payload.get("mode") != "static" or payload.get("simulate_acc_len"):
            raise ValueError(
                "Expected static recording without simulated acceptance. No chat request was sent."
            )


def analyze_trace(server_info, response, rid):
    """Keep every matching row; compare an explicitly labelled API-counted prefix.

    An overlap worker can record finished rows that the scheduler never settles.
    Matching API A/N/histogram is evidence for a prefix, not proof of actual KV
    contents or a licence to silently drop the remaining worker rows.
    """
    if response.get("id") != rid:
        raise ValueError(
            "Response id differs from the requested rid; no fallback matching."
        )
    matches = []
    for state_index, payload in trace_payloads(server_info):
        rows = [
            {"forward_ct": record["forward_ct"], **req}
            for record in payload.get("records", [])
            for req in (record.get("reqs") or [])
            if req.get("rid") == rid
        ]
        if rows:
            matches.append((state_index, payload, rows))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one recording owner for rid, found {len(matches)}. "
            "Raw server_info is retained; do not combine ranks or guess by time."
        )
    state_index, payload, rows = matches[0]
    rows.sort(key=lambda row: row["forward_ct"])
    if len({row["forward_ct"] for row in rows}) != len(rows):
        raise ValueError("Duplicate forward_ct for this rid; records are ambiguous.")
    if payload.get("mode") != "static":
        raise ValueError("This first diagnostic expects static records.")
    if payload.get("simulate_acc_len"):
        raise ValueError("Simulated acceptance is not a model-quality observation.")
    gamma = payload["gamma"]
    if type(gamma) is not int or gamma <= 0:
        raise ValueError("Invalid recorded gamma.")
    width = gamma + 1
    if payload.get("verify_num_draft_tokens") != width:
        raise ValueError("Unexpected static verify width.")

    details = response["sglext"]["spec_tokens_details"]
    n = details["spec_verify_ct"]
    a = details["spec_num_correct_drafts"]
    p = details["spec_num_proposed_drafts"]
    if any(type(x) is not int for x in (n, a, p)) or n <= 0 or p <= 0:
        raise ValueError("Missing or invalid positive API speculative counters.")
    issues = []
    choice = response["choices"][0]
    num_retractions = (choice.get("meta_info") or {}).get("num_retractions")
    if type(num_retractions) is not int or num_retractions != 0:
        issues.append(
            "Retraction count is missing or nonzero; the first N rows may not be the settled rounds"
        )
    for row in rows:
        for key in (
            "correct_drafts",
            "acc_len",
            "prefix_len",
            "verify_len",
            "cap_trim",
        ):
            if type(row.get(key)) is not int:
                raise ValueError(f"Non-integer or missing {key} in a recorded row.")
        if not 0 <= row["correct_drafts"] <= gamma:
            issues.append(
                f"forward_ct {row['forward_ct']}: correct_drafts outside gamma"
            )
        if row["verify_len"] != width or row["cap_trim"] != 0:
            issues.append(
                f"forward_ct {row['forward_ct']}: unexpected static window/cap"
            )
        if row["acc_len"] != row["correct_drafts"] + 1:
            issues.append(
                f"forward_ct {row['forward_ct']}: commit != accepted drafts + 1"
            )

    # These are only candidates for scheduler-accounted rounds until counters
    # agree. Retractions/aborts or incomplete evidence need manual investigation.
    prefix = rows[:n]
    accepted = [row["correct_drafts"] for row in prefix]
    histogram = Counter(accepted)
    api_histogram = details.get("spec_correct_drafts_histogram")
    histogram_matches = isinstance(api_histogram, list) and histogram == Counter(
        {i: count for i, count in enumerate(api_histogram) if count}
    )
    counter_match = (
        len(prefix) == n and sum(accepted) == a and p == n * gamma and histogram_matches
    )
    if not counter_match:
        issues.append(
            "API A/P/N/histogram do not match the candidate prefix; no root-cause verdict"
        )
    discontinuities = [
        {
            "previous_forward_ct": prev["forward_ct"],
            "forward_ct": cur["forward_ct"],
            "expected_prefix_len": prev["prefix_len"] + prev["acc_len"],
            "recorded_prefix_len": cur["prefix_len"],
        }
        for prev, cur in zip(prefix, prefix[1:])
        if cur["prefix_len"] != prev["prefix_len"] + prev["acc_len"]
    ]
    if discontinuities:
        issues.append(
            "Recorded prefix discontinuity: inspect lifecycle and dump before attributing a KV bug"
        )
    generated = [
        token
        for row in prefix
        for token in row["draft_tokens"][: row["correct_drafts"]] + [row["bonus_token"]]
    ]
    output_ids = choice.get("response_token_ids")
    completion_tokens = response.get("usage", {}).get("completion_tokens")
    token_match = (
        isinstance(output_ids, list)
        and len(output_ids) > 0
        and type(completion_tokens) is int
        and len(output_ids) == completion_tokens
        and len(output_ids) - 1 <= len(generated)
        and output_ids[1:] == generated[: len(output_ids) - 1]
    )
    if not token_match:
        issues.append(
            "Response token IDs do not match the recorded output prefix (after the prefill token)"
        )
    if choice.get("finish_reason") not in ("stop", "length"):
        issues.append(
            "Request did not finish normally; do not infer a settled-round prefix"
        )
    if (
        choice.get("finish_reason") == "length"
        and isinstance(output_ids, list)
        and len(output_ids) <= 1 + sum(row["acc_len"] for row in prefix[:-1])
    ):
        issues.append(
            "Length truncation precedes the final counted round; inspect record attribution"
        )
    trailing_zeros = 0
    for count in reversed(accepted):
        if count:
            break
        trailing_zeros += 1
    summary = {
        "status": "TRACE_REVIEW_REQUIRED" if issues else "TRACE_COLLECTED",
        "rid": rid,
        "recording_owner": state_index,
        "mode": payload["mode"],
        "gamma": gamma,
        "server_configuration": {
            key: server_info.get(key)
            for key in (
                "device",
                "speculative_algorithm",
                "model_path",
                "speculative_draft_model_path",
                "tp_size",
                "dp_size",
                "nnodes",
                "disable_cuda_graph",
                "disable_decode_cuda_graph",
                "disable_prefill_cuda_graph",
                "cuda_graph_backend_decode",
                "cuda_graph_backend_prefill",
            )
        },
        "api": {
            "accepted_drafts": a,
            "proposed_drafts": p,
            "verify_rounds": n,
            "accept_rate": a / p,
        },
        "recorded_worker_rounds": len(rows),
        "candidate_prefix_matches_api_counters": counter_match,
        "candidate_prefix_matches_output_token_ids": token_match,
        "num_retractions": num_retractions,
        "candidate_prefix_accepted_drafts_by_round": accepted,
        "first_draft_accepted_rounds": sum(count > 0 for count in accepted),
        "candidate_prefix_trailing_zero_rounds": trailing_zeros,
        "additional_worker_rows": len(rows[n:]),
        "prefix_discontinuities": discontinuities,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "completion_tokens": completion_tokens,
        "candidate_output_tokens_before_finish_trimming": 1 + len(generated),
        "issues": issues,
        "limits": [
            "Counts describe worker acceptance before final EOS/length truncation; do not equate sum(commit) with visible output length.",
            "Additional worker rows can be unsettled overlap work; retained separately, not automatically an inference error.",
            "Matching counters do not verify target logits, actual KV, or correctness; no quality/performance PASS is assigned.",
        ],
    }
    return summary, {"candidate_api_prefix": prefix, "additional_worker_rows": rows[n:]}


def collect(url, output, max_tokens, timeout=600, opener=None):
    # Match the existing curl --noproxy '*' request, independently of user proxy env.
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = url.rstrip("/")
    before = request_json(opener, url + "/server_info", timeout=timeout)
    write_json(output / "server_info.before.json", before)
    check_recording(before)
    rid = "ms1-trace-" + uuid.uuid4().hex
    payload = {
        "rid": rid,
        "model": "GLM-5.2-w8a8",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "return_spec_tokens_details": True,
        "return_meta_info": True,
        "return_token_ids": True,
    }
    write_json(output / "request.json", payload)
    response = request_json(opener, url + "/v1/chat/completions", payload, timeout)
    write_json(output / "response.json", response)
    after = request_json(opener, url + "/server_info", timeout=timeout)
    write_json(output / "server_info.after.json", after)
    summary, trace = analyze_trace(after, response, rid)
    write_json(output / "trace.json", trace)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://61.47.19.71:8810")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("/home/tyj/glm52-ms1/evidence")
    )
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.evidence_root / f"acceptance-trace-{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    print(f"Evidence: {output}", flush=True)
    script = Path(__file__).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    write_json(
        output / "collector.json",
        {
            "collector_git_head": revision.stdout.strip(),
            "collector_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "url": args.url,
            "max_tokens": args.max_tokens,
            "note": "Collector identity is not proof of the running server source; keep its launch/overlay evidence.",
        },
    )
    try:
        summary = collect(args.url, output, args.max_tokens, args.timeout)
    except Exception as exc:
        summary = {
            "status": "COLLECTION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(output / "summary.json", summary)
    print(summary["status"])
    if "api" in summary:
        print(
            "Server configuration:",
            json.dumps(summary["server_configuration"], ensure_ascii=False),
        )
        api = summary["api"]
        print(
            f"API A/P/N: {api['accepted_drafts']}/{api['proposed_drafts']}/{api['verify_rounds']}; acceptance={api['accept_rate']:.6%}"
        )
        print(
            "Accepted drafts by round:",
            summary["candidate_prefix_accepted_drafts_by_round"],
        )
        print("First draft accepted rounds:", summary["first_draft_accepted_rounds"])
        print("API counter match:", summary["candidate_prefix_matches_api_counters"])
        print(
            "Output token match:", summary["candidate_prefix_matches_output_token_ids"]
        )
        print("Additional worker rows:", summary["additional_worker_rows"])
        print("Retractions:", summary["num_retractions"])
        print("Prefix discontinuities:", summary["prefix_discontinuities"])
        for issue in summary["issues"]:
            print("REVIEW:", issue)
    else:
        print(summary["error"])
    return 0 if summary["status"] == "TRACE_COLLECTED" else 1


if __name__ == "__main__":
    sys.exit(main())
