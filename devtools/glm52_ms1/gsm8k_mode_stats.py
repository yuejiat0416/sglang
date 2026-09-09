"""Pure CPU summaries for temporary fixed-question serving diagnostics.

This module does not send requests, import inference code, or assign a quality
or performance PASS. Raw responses remain available for independent review.
"""

import json
import math
import re

MODES = {"dspark-eager", "dspark-graph", "target-eager", "target-graph", "nextn-graph"}
COUNTER_ALIASES = {
    "accepted_drafts": ("spec_num_correct_drafts", "spec_accepted_drafts"),
    "proposed_drafts": ("spec_num_proposed_drafts", "spec_proposed_drafts"),
    "verify_rounds": ("spec_verify_ct",),
}
GRAPH_MODES = ("decode_cuda_graph", "decode_none")
METRIC = "sglang:cuda_graph_passes_total"
_SAMPLE = re.compile(
    r"^sglang:cuda_graph_passes_total(?:\{(.*)\})?\s+"
    r"([^\s]+)(?:\s+[^\s]+)?\s*$"
)
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*")\s*')


def _integer(value):
    return type(value) is int and value >= 0


def _question(response, expected_id, speculative):
    issues = []
    result = {
        "id": expected_id,
        "valid": False,
        "issues": issues,
        "finish_reason": None,
        "completion_tokens": None,
        "content": None,
        "reasoning_content": None,
        "counters": {},
        "counter_sources": {},
        "response": response,
    }
    if not isinstance(response, dict):
        issues.append("Response is not an object")
        return result
    if response.get("error"):
        issues.append("Response contains an error")
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        issues.append("Expected one completed chat choice")
        return result
    choice = choices[0]
    result["finish_reason"] = choice.get("finish_reason")
    if not isinstance(result["finish_reason"], str) or not result["finish_reason"]:
        issues.append("Missing final finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        issues.append("Missing chat message")
    else:
        result["content"] = message.get("content")
        result["reasoning_content"] = message.get("reasoning_content")
    meta = choice.get("meta_info")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        issues.append("Invalid choices[0].meta_info")
        meta = {}
    if "id" in meta and meta["id"] != expected_id:
        issues.append("meta_info.id differs from response id")
    usage = response.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        issues.append("Invalid usage")
        usage = {}
    token_values = [
        value["completion_tokens"]
        for value in (usage, meta)
        if "completion_tokens" in value
    ]
    if not token_values or any(not _integer(value) for value in token_values):
        issues.append("Missing or invalid completion_tokens")
    elif len(set(token_values)) != 1:
        issues.append("Conflicting completion_tokens")
    else:
        result["completion_tokens"] = token_values[0]
    sglext = response.get("sglext")
    if sglext is None:
        sglext = {}
    if not isinstance(sglext, dict):
        issues.append("Invalid sglext")
        sglext = {}
    details = sglext.get("spec_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        issues.append("Invalid sglext.spec_tokens_details")
        details = {}
    sources = (("sglext.spec_tokens_details", details), ("choices[0].meta_info", meta))
    for canonical, aliases in COUNTER_ALIASES.items():
        observed = {
            f"{source}.{alias}": data[alias]
            for source, data in sources
            for alias in aliases
            if alias in data
        }
        result["counter_sources"][canonical] = observed
        values = list(observed.values())
        if any(not _integer(value) for value in values):
            issues.append(f"Invalid {canonical}; require nonnegative integer, not bool")
            result["counters"][canonical] = None
        elif len(set(values)) > 1:
            issues.append(f"Conflicting {canonical} aliases/sources")
            result["counters"][canonical] = None
        elif not values:
            result["counters"][canonical] = None if speculative else 0
            if speculative:
                issues.append(
                    f"Missing {canonical}; no reconstruction from rates or lengths"
                )
        else:
            result["counters"][canonical] = values[0]
    counts = result["counters"]
    a, p, n = (counts[key] for key in COUNTER_ALIASES)
    if a is not None and p is not None and a > p:
        issues.append("accepted_drafts exceeds proposed_drafts")
    if p is not None and n is not None and ((p > 0) != (n > 0)):
        issues.append(
            "proposed_drafts and verify_rounds disagree on whether verification occurred"
        )
    if not speculative and any(
        value is not None and value != 0 for value in counts.values()
    ):
        issues.append("Target-only mode returned nonzero speculative counters")
    result["valid"] = not issues
    result["accept_rate"] = a / p if not issues and speculative and p else None
    result["mean_accepted_drafts_per_round"] = (
        a / n if not issues and speculative and n else None
    )
    return result


def summarize_responses(
    expected_ids: list[str], responses: list[dict], mode: str
) -> dict:
    """Summarize exactly the supplied expected requests, weighted by raw counts.

    The caller selects ten dataset questions. This function accepts any nonempty
    unique ID list so small independent unit tests can verify its arithmetic.
    Partial valid counts are retained separately and never presented as the
    complete run's acceptance rate.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if (
        not isinstance(expected_ids, list)
        or not expected_ids
        or any(not isinstance(item, str) or not item for item in expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError(
            "expected_ids must be a nonempty list of unique nonempty strings"
        )
    if not isinstance(responses, list):
        raise ValueError("responses must be a list")
    speculative = not mode.startswith("target-")
    issues, unexpected = [], []
    grouped = {rid: [] for rid in expected_ids}
    for index, response in enumerate(responses):
        rid = response.get("id") if isinstance(response, dict) else None
        if not isinstance(rid, str) or rid not in grouped:
            issues.append(f"Response {index} has a missing or foreign id: {rid!r}")
            unexpected.append(response)
        else:
            grouped[rid].append(response)
    questions = []
    for rid in expected_ids:
        found = grouped[rid]
        if len(found) != 1:
            problem = "Missing response" if not found else "Duplicate responses"
            issues.append(f"{rid}: {problem}")
            questions.append(
                {"id": rid, "valid": False, "issues": [problem], "responses": found}
            )
            continue
        item = _question(found[0], rid, speculative)
        questions.append(item)
        issues.extend(f"{rid}: {issue}" for issue in item["issues"])
    valid = [item for item in questions if item["valid"]]
    complete = not issues
    partial = {
        key: sum(item["counters"][key] for item in valid) for key in COUNTER_ALIASES
    }
    aggregate = dict(partial) if complete else {key: None for key in COUNTER_ALIASES}
    a, p, n = (partial[key] for key in COUNTER_ALIASES)
    aggregate.update(
        {
            "accept_rate": a / p if complete and speculative and p else None,
            "mean_accepted_drafts_per_round": a / n
            if complete and speculative and n
            else None,
            "one_plus_mean_accepted_drafts_per_round": 1 + a / n
            if complete and speculative and n
            else None,
            "applicability": "speculative"
            if speculative
            else "not_applicable_target_only",
        }
    )
    return {
        "status": "GSM8K_RESPONSES_COLLECTED"
        if complete
        else "GSM8K_RESPONSES_INCOMPLETE",
        "complete": complete,
        "mode": mode,
        "expected_count": len(expected_ids),
        "response_count": len(responses),
        "valid_count": len(valid),
        "issues": issues,
        "aggregate": aggregate,
        "partial_valid_counts": partial,
        "per_question": questions,
        "unexpected_responses": unexpected,
        "limits": [
            "Acceptance is sum(accepted draft tokens) / sum(proposed draft tokens), not a mean of question rates.",
            "One plus A/N is theoretical per-round progress before final trimming; it is not API completion_tokens/N.",
            "Request counters do not prove model accuracy, full graph coverage, benchmark equivalence or performance PASS.",
        ],
    }


def _parse_graph(text):
    if text is None:
        return {}, []
    if not isinstance(text, str):
        return {}, ["Metrics input must be text or None"]
    series, issues = {}, []
    for line in text.splitlines():
        if not line.startswith(METRIC) or line.startswith(METRIC + "_"):
            continue
        match = _SAMPLE.fullmatch(line)
        if not match:
            issues.append("Malformed graph counter sample")
            continue
        labels, offset = {}, 0
        raw = match.group(1) or ""
        try:
            while offset < len(raw):
                token = _LABEL.match(raw, offset)
                if token is None:
                    raise ValueError("Malformed graph counter labels")
                name, value = token.groups()
                if name in labels:
                    raise ValueError("Duplicate graph counter label")
                labels[name] = json.loads(value)
                offset = token.end()
                if offset < len(raw):
                    if raw[offset] != ",":
                        raise ValueError("Malformed graph counter label separator")
                    offset += 1
            if labels.get("mode") not in GRAPH_MODES:
                continue
            value = float(match.group(2))
            if not math.isfinite(value) or value < 0:
                raise ValueError("Graph counter must be finite and nonnegative")
            key = tuple(sorted(labels.items()))
            if key in series:
                raise ValueError("Duplicate graph counter series")
            series[key] = value
        except (ValueError, json.JSONDecodeError) as exc:
            issues.append(str(exc))
    return series, issues


def graph_count_delta(before_text, after_text) -> dict:
    """Subtract counters by complete label set, preserving missing/reset evidence.

    Newly materialized labeled counters have baseline zero. Missing counters in
    both snapshots are unavailable; disappearing or decreasing series are errors.
    Counts are scheduler graph passes, not requests or proof of every draft stage.
    """
    if before_text is None or after_text is None:
        return {
            "status": "GRAPH_COUNTERS_UNAVAILABLE",
            "available": False,
            "issues": ["A metrics scrape is unavailable; no zero baseline is inferred"],
            "before_present": None,
            "after_present": None,
            "deltas": {mode: None for mode in GRAPH_MODES},
            "series": [],
            "limits": ["Missing metrics are not measured zero."],
        }
    before, before_issues = _parse_graph(before_text)
    after, after_issues = _parse_graph(after_text)
    issues = [f"before: {item}" for item in before_issues] + [
        f"after: {item}" for item in after_issues
    ]
    rows, deltas = [], {mode: None for mode in GRAPH_MODES}
    for key in sorted(set(before) | set(after)):
        labels = dict(key)
        initial, final = before.get(key, 0), after.get(key)
        delta = None
        if final is None:
            issues.append(f"Graph counter series disappeared: {labels}")
        elif final < initial:
            issues.append(f"Graph counter reset/decreased: {labels}")
        else:
            delta = final - initial
            if delta.is_integer():
                delta = int(delta)
            mode = labels["mode"]
            deltas[mode] = (deltas[mode] or 0) + delta
        rows.append(
            {
                "labels": labels,
                "before": initial,
                "after": final,
                "delta": delta,
                "new_series": key not in before,
            }
        )
    available = bool(after) and not issues
    if issues:
        deltas = {mode: None for mode in GRAPH_MODES}
    return {
        "status": (
            "GRAPH_COUNTERS_INVALID"
            if issues
            else "GRAPH_COUNTERS_COLLECTED"
            if available
            else "GRAPH_COUNTERS_UNAVAILABLE"
        ),
        "available": available,
        "issues": issues,
        "before_present": bool(before),
        "after_present": bool(after),
        "deltas": deltas,
        "series": rows,
        "limits": [
            "Deltas describe all traffic between scrapes for the observed full label sets.",
            "A graph counter increase does not prove every DSpark stage executed inside a graph.",
            "A mode absent from both scrapes is unavailable, not measured zero.",
        ],
    }
