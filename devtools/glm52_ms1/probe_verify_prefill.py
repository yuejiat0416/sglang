"""Compare recorded static verify decisions with fresh target prefill predictions.

Diagnostic only: no model imports, patching, cache flush or automatic retry.
By default this prepares a plan without contacting the server. --run sends at
most six requests, each asking for one output token, to the existing service.
"""

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from collect_acceptance_trace import analyze_trace, request_json, write_json

CONFIG_KEYS = (
    "model_path",
    "tokenizer_path",
    "tokenizer_mode",
    "tokenizer_backend",
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_num_draft_tokens",
    "speculative_draft_model_quantization",
    "device",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "tp_size",
    "dp_size",
    "pp_size",
    "nnodes",
    "dcp_size",
    "enable_dp_attention",
    "page_size",
    "disable_cuda_graph",
    "chunked_prefill_size",
    "chat_template",
    "sampling_defaults",
    "preferred_sampling_params",
    "json_model_override_args",
    "override_config_file",
)
LIMITS = [
    "Same DSPARK service: target prefill versus historical verify; not an independent target-only deployment.",
    "History contains decisions, not logits: only accepted-prefix predictions and the bonus are known.",
    "Repeat agreement is diagnostic, not a numerical tolerance or correctness/acceptance/performance PASS.",
    "Prefill still injects draft hidden; overlap can execute an unsettled extra worker round.",
    "Server configuration and collector hashes do not prove running source or weight bytes.",
    "Sampling-default files describe the current container; their historical loaded bytes are not proved.",
    "Top-k tie ordering can differ from argmax; a tied alternative is not evidence of a verify error.",
]


def token_list(value, name):
    if (
        not isinstance(value, list)
        or not value
        or any(type(v) is not int or v < 0 for v in value)
    ):
        raise ValueError(f"{name} must be a nonempty list of nonnegative token IDs")
    return value


def reconstruct_round(prompt_ids, output_ids, row, gamma):
    token_list(prompt_ids, "prompt_token_ids")
    token_list(output_ids, "response_token_ids")
    drafts = token_list(row["draft_tokens"], "draft_tokens")
    prefix_len, accepted, bonus = (
        row["prefix_len"],
        row["correct_drafts"],
        row["bonus_token"],
    )
    if any(type(v) is not int for v in (gamma, prefix_len, accepted, bonus)):
        raise ValueError("Non-integer round metadata")
    offset = prefix_len - len(prompt_ids)
    if gamma <= 0 or len(drafts) != gamma or not 0 <= accepted <= gamma:
        raise ValueError("Invalid draft width or accepted length")
    if not 0 <= offset < len(output_ids) or bonus < 0:
        raise ValueError("Recorded prefix cannot be reconstructed from response")
    if accepted < gamma and drafts[accepted] == bonus:
        raise ValueError(
            "Rejected draft equals bonus; not the expected greedy contract"
        )
    return {
        "forward_ct": row["forward_ct"],
        "anchor_index": prefix_len,
        "anchor_token": output_ids[offset],
        "gamma": gamma,
        "accepted_drafts": accepted,
        "input_ids": prompt_ids + output_ids[: offset + 1] + drafts,
        "known_predictions": drafts[:accepted] + [bonus],
        "unknown_verify_positions": list(range(accepted + 1, gamma + 1)),
    }


def select_rounds(rows, gamma):
    if not rows:
        raise ValueError("No API-accounted rounds")
    selected = {}
    for label, predicate in (
        ("first", lambda row: True),
        ("first_zero", lambda row: row["correct_drafts"] == 0),
        ("first_full", lambda row: row["correct_drafts"] == gamma),
    ):
        for index, row in enumerate(rows):
            if predicate(row):
                selected.setdefault(index, {"labels": [], "row": row})["labels"].append(
                    label
                )
                break
    return list(selected.values())


def sampling_evidence(request, info):
    evidence = {"sampling_defaults": info.get("sampling_defaults")}
    if info.get("preferred_sampling_params") not in (None, {}):
        raise ValueError(
            "Preferred sampling overrides require separate review before this diagnostic"
        )
    if "repetition_penalty" in request:
        repetition = request["repetition_penalty"]
        evidence["repetition_source"] = "explicit source request"
    elif info.get("sampling_defaults") == "openai":
        repetition = 1.0
        evidence["repetition_source"] = "OpenAI default"
    elif info.get("sampling_defaults") == "model":
        if info.get("override_config_file"):
            raise ValueError(
                "Alternate model configuration requires separate sampling-default review"
            )
        root = Path(info["model_path"])
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(
                "Read model sampling defaults in the running container with the recorded model directory mounted"
            )
        path = root / "generation_config.json"
        evidence["generation_config_path"] = str(path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raw = None
        if raw is not None:
            config = json.loads(raw)
            if not isinstance(config, dict):
                raise ValueError("Invalid generation_config.json")
            evidence["generation_config_sha256"] = hashlib.sha256(raw).hexdigest()
            evidence["generation_config"] = config
            repetition = config.get("repetition_penalty")
            if repetition is None:
                repetition = 1.0
        else:
            evidence["generation_config_sha256"] = None
            repetition = 1.0
        evidence["repetition_source"] = (
            "local model generation config, or OpenAI fallback when absent"
        )
    else:
        raise ValueError("Unknown historical sampling defaults")
    evidence["effective_repetition_penalty"] = repetition
    if repetition != 1:
        raise ValueError(
            f"Historical repetition_penalty={repetition}; raw prefill predictions are not a matched comparison"
        )
    return evidence


def read_source(source):
    data, fingerprints = {}, {}
    for name in ("request.json", "response.json", "server_info.after.json"):
        raw = (source / name).read_bytes()
        data[name] = json.loads(raw)
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
    request, response, info = (data[k] for k in data)
    if request.get("temperature") != 0:
        raise ValueError("Source request must explicitly use temperature=0")
    for key, neutral in (
        ("frequency_penalty", 0),
        ("presence_penalty", 0),
        ("repetition_penalty", 1),
        ("logit_bias", {}),
    ):
        if request.get(key, neutral) != neutral:
            raise ValueError(f"Source {key} is not supported by this diagnostic")
    supported = {
        "rid",
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "return_spec_tokens_details",
        "return_meta_info",
        "return_token_ids",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "logit_bias",
    }
    if set(request) - supported:
        raise ValueError(
            f"Unsupported source request options: {sorted(set(request) - supported)}"
        )
    if not (
        info.get("device") == "npu"
        and info.get("speculative_algorithm") == "DSPARK"
        and info.get("disable_cuda_graph") is True
        and info.get("dp_size") == 1
        and info.get("nnodes") == 1
        and info.get("pp_size", 1) == 1
    ):
        raise ValueError(
            "This diagnostic expects the recorded single-node DP1/PP1 NPU DSPARK eager service"
        )
    summary, trace = analyze_trace(info, response, request["rid"])
    if summary["status"] != "TRACE_COLLECTED":
        raise ValueError(f"Source attribution requires review: {summary['issues']}")
    choice = response["choices"][0]
    prompt = token_list(choice.get("prompt_token_ids"), "prompt_token_ids")
    output = token_list(choice.get("response_token_ids"), "response_token_ids")
    if response["usage"].get("prompt_tokens") != len(prompt):
        raise ValueError("Prompt token count differs from saved token IDs")
    rows = trace["candidate_api_prefix"]
    expected_prefix = len(prompt)
    for row in rows:
        if row["prefix_len"] != expected_prefix:
            raise ValueError("Round prefix is not continuous from the original prompt")
        expected_prefix += row["acc_len"]
    cases = []
    for selection in select_rounds(rows, summary["gamma"]):
        case = reconstruct_round(prompt, output, selection["row"], summary["gamma"])
        case["labels"] = selection["labels"]
        cases.append(case)
    return {
        "source": str(source.resolve()),
        "source_sha256": fingerprints,
        "source_rid": request["rid"],
        "sampling_evidence": sampling_evidence(request, info),
        "server_configuration": {k: info.get(k) for k in CONFIG_KEYS},
        "cases": cases,
        "repeats_per_case": 2,
        "planned_requests": 2 * len(cases),
        "excluded_worker_rows": len(trace["additional_worker_rows"]),
        "limits": LIMITS,
    }


def score_request(case):
    return {
        "rid": "ms1-prefill-" + uuid.uuid4().hex,
        "cache_salt": "ms1-prefill-" + uuid.uuid4().hex,
        "input_ids": case["input_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "return_logprob": True,
        "top_logprobs_num": 5,
        "logprob_start_len": case["anchor_index"],
        "return_text_in_logprobs": False,
        "stream": False,
    }


def validate_top(row):
    if not isinstance(row, list) or len(row) < 2:
        raise ValueError("Missing top predictions; no fallback to output text")
    parsed = []
    for entry in row:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError("Malformed top-logprob entry")
        value, token = entry[:2]
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or type(token) is not int
            or token < 0
        ):
            raise ValueError("Invalid logprob or token ID")
        parsed.append((float(value), token))
    if len({token for _, token in parsed}) != len(parsed):
        raise ValueError("Duplicate token in top predictions")
    if any(a[0] < b[0] for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Top predictions are not ordered")
    return parsed


def compare_response(case, request, response):
    meta = response["meta_info"]
    if meta.get("id") != request["rid"]:
        raise ValueError("Scoring response ID differs from request")
    if (
        meta.get("prompt_tokens") != len(case["input_ids"])
        or meta.get("completion_tokens") != 1
    ):
        raise ValueError("Unexpected scoring input/output length")
    if meta.get("cached_tokens") != 0:
        raise ValueError(
            "Scoring request was not a fresh prefill (cached_tokens must be 0)"
        )
    if meta.get("num_retractions") != 0:
        raise ValueError("Scoring request retracted or lacks retraction evidence")
    if (meta.get("finish_reason") or {}).get("type") not in ("length", "stop"):
        raise ValueError("Scoring request did not finish normally")
    inputs, outputs = meta["input_top_logprobs"], meta["output_top_logprobs"]
    if len(inputs) != case["gamma"] + 1 or inputs[0] is not None or len(outputs) != 1:
        raise ValueError("Unexpected input/output top-logprob alignment")
    input_tokens = meta["input_token_logprobs"]
    suffix = case["input_ids"][case["anchor_index"] :]
    if (
        len(input_tokens) != len(suffix)
        or [entry[1] for entry in input_tokens] != suffix
    ):
        raise ValueError("Scored token positions differ from anchor+draft sequence")
    tops = [validate_top(row) for row in inputs[1:] + outputs]
    sampled = meta["output_token_logprobs"]
    if len(sampled) != 1 or len(sampled[0]) < 2:
        raise ValueError("Expected one sampled output token")
    sampled_token = sampled[0][1]
    if type(sampled_token) is not int or sampled_token not in [
        token for value, token in tops[-1] if value == tops[-1][0][0]
    ]:
        raise ValueError("Sampled output is not a maximal prefill prediction")
    comparisons = []
    for position, expected in enumerate(case["known_predictions"]):
        top = tops[position]
        tied = [token for value, token in top if value == top[0][0]]
        predicted = sampled_token if position == case["gamma"] else top[0][1]
        comparisons.append(
            {
                "verify_position": position,
                "expected_token": expected,
                "prefill_top1": predicted,
                "top1_matches": predicted == expected,
                "expected_tied_at_max": expected in tied,
                "top1_top2_logprob_gap": top[0][0] - top[1][0],
                "expected_rank_in_returned_top5": next(
                    (i + 1 for i, (_, token) in enumerate(top) if token == expected),
                    None,
                ),
            }
        )
    return {
        "rid": request["rid"],
        "comparisons": comparisons,
        "known_top1_mismatches": sum(not row["top1_matches"] for row in comparisons),
        "known_expected_outside_returned_max": sum(
            not row["expected_tied_at_max"] for row in comparisons
        ),
        "prefill_top1_all_positions": [top[0][1] for top in tops[:-1]]
        + [sampled_token],
        "prefill_output_token": sampled_token,
        "unknown_historical_positions": case["unknown_verify_positions"],
    }


def collect(plan, output, url, timeout, opener):
    url = url.rstrip("/")
    results = []
    for moment in ("before", "after"):
        info = request_json(opener, url + "/server_info", timeout=timeout)
        config = {k: info.get(k) for k in CONFIG_KEYS}
        write_json(output / f"server_configuration.{moment}.json", config)
        changed = [
            k for k in CONFIG_KEYS if config[k] != plan["server_configuration"][k]
        ]
        if changed:
            raise ValueError(
                f"Server configuration differs from source evidence: {changed}"
            )
        sampling = plan["sampling_evidence"]
        if "generation_config_path" in sampling:
            path = Path(sampling["generation_config_path"])
            try:
                current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                current_hash = None
            if current_hash != sampling["generation_config_sha256"]:
                raise ValueError(
                    "Model sampling-default file changed during collection"
                )
        if moment == "after":
            break
        for index, case in enumerate(plan["cases"], 1):
            repeats = []
            for repeat in (1, 2):
                stem = f"case{index}-r{repeat}"
                request = score_request(case)
                write_json(output / f"{stem}.request.json", request)
                print(f"RUN {stem} labels={','.join(case['labels'])}", flush=True)
                try:
                    response = request_json(opener, url + "/generate", request, timeout)
                except urllib.error.HTTPError as exc:
                    body = exc.read(65537)
                    write_json(
                        output / f"{stem}.http-error.json",
                        {
                            "status_code": exc.code,
                            "reason": str(exc.reason),
                            "body": body[:65536].decode("utf-8", errors="replace"),
                            "body_truncated": len(body) > 65536,
                        },
                    )
                    raise
                write_json(output / f"{stem}.response.json", response)
                comparison = compare_response(case, request, response)
                write_json(output / f"{stem}.comparison.json", comparison)
                repeats.append(comparison)
            results.append(
                {
                    "forward_ct": case["forward_ct"],
                    "labels": case["labels"],
                    "repeats": repeats,
                    "prefill_repeats_top1_equal": repeats[0][
                        "prefill_top1_all_positions"
                    ]
                    == repeats[1]["prefill_top1_all_positions"],
                }
            )
    return {"status": "COMPARISON_COLLECTED", "results": results, "limits": LIMITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing single-request evidence directory",
    )
    parser.add_argument("--url", default="http://61.47.19.71:8810")
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("/home/tyj/glm52-ms1/evidence")
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Send the prepared scoring requests (at most six)",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("timeout must be positive and finite")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.evidence_root / f"verify-prefill-{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    print(f"Evidence: {output}", flush=True)
    try:
        plan = read_source(args.source)
        plan["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        plan["trace_analyzer_sha256"] = hashlib.sha256(
            Path(__file__).with_name("collect_acceptance_trace.py").read_bytes()
        ).hexdigest()
        write_json(output / "plan.json", plan)
        print(
            f"Prepared {len(plan['cases'])} rounds / {plan['planned_requests']} scoring requests"
        )
        if not args.run:
            report = {"status": "PLAN_PREPARED_NO_REQUESTS", "limits": LIMITS}
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            report = collect(plan, output, args.url, args.timeout, opener)
    except KeyboardInterrupt:
        report = {"status": "INTERRUPTED", "limits": LIMITS}
    except Exception as exc:
        report = {
            "status": "COLLECTION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "limits": LIMITS,
        }
    write_json(output / "report.json", report)
    print(report["status"])
    for result in report.get("results", []):
        print(
            f"{result['labels']}: known-position top1 mismatches={[r['known_top1_mismatches'] for r in result['repeats']]}; outside returned maximum set={[r['known_expected_outside_returned_max'] for r in result['repeats']]}; prefill repeat equality={result['prefill_repeats_top1_equal']}"
        )
    print(
        "For differing IDs, inspect maximum-score ties before interpreting a verify/prefill divergence."
    )
    print("Diagnostic only; no correctness, acceptance-rate or performance PASS.")
    return 1 if report["status"] in ("COLLECTION_FAILED", "INTERRUPTED") else 0


if __name__ == "__main__":
    sys.exit(main())
