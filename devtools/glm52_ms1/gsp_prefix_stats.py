"""Validate one measured native /generate response for a temporary GSP run.

The caller sends cache warm-up requests separately. A measured response only
establishes the requested scenario when its ID, lengths, cache count and absence
of retractions are observed. This module sets no acceptance or precision gate.
"""

from copy import deepcopy


COUNTER_ALIASES = {
    "accepted_drafts": ("spec_num_correct_drafts", "spec_accepted_drafts"),
    "proposed_drafts": ("spec_num_proposed_drafts", "spec_proposed_drafts"),
    "verify_rounds": ("spec_verify_ct",),
}


def _nonnegative_integer(value):
    return type(value) is int and value >= 0


def summarize_case(
    response, *, rid, input_tokens, output_tokens, expected_cached_tokens
):
    """Return counts and distinguish an invalid response from a wrong scenario.

    A valid response with a short output, unexpected cache count or a retraction
    retains its acceptance counts/rate, but is never labelled as meeting the
    requested case conditions. Missing evidence is invalid rather than zero.
    Alias conflicts and foreign IDs prevent assigning an acceptance rate.
    """
    if not isinstance(rid, str) or not rid:
        raise ValueError("rid must be a nonempty string")
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        if not _nonnegative_integer(value) or value == 0:
            raise ValueError(f"{name} must be a positive integer, not bool")
    if (
        not _nonnegative_integer(expected_cached_tokens)
        or expected_cached_tokens > input_tokens
    ):
        raise ValueError(
            "expected_cached_tokens must be an integer in [0, input_tokens]"
        )

    invalid, mismatches = [], []
    actual = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "cache_hit_ratio": None,
        "finish_reason": None,
        "num_retractions": None,
        "cached_tokens_details": None,
    }
    acceptance = {name: None for name in COUNTER_ALIASES}
    acceptance["accept_rate"] = None
    counter_sources = {}
    meta = {}
    if not isinstance(response, dict):
        invalid.append("Response must be one native /generate JSON object")
    else:
        if response.get("error"):
            invalid.append("Response contains an error")
        candidate_meta = response.get("meta_info")
        if not isinstance(candidate_meta, dict):
            invalid.append("Missing or invalid meta_info")
        else:
            meta = candidate_meta
        if "id" in response and response["id"] != rid:
            invalid.append("Response id does not match measured request rid")

    if meta.get("id") != rid:
        invalid.append("Missing or foreign meta_info.id; warm-up counts cannot be used")
    if meta.get("error"):
        invalid.append("meta_info contains an error")
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "num_retractions",
    ):
        value = meta.get(name)
        if not _nonnegative_integer(value):
            invalid.append(
                f"Missing or invalid {name}; require nonnegative integer, not bool"
            )
        else:
            actual[name] = value
    actual["finish_reason"] = deepcopy(meta.get("finish_reason"))
    actual["cached_tokens_details"] = deepcopy(meta.get("cached_tokens_details"))
    finish = actual["finish_reason"]
    if (
        not isinstance(finish, dict)
        or not isinstance(finish.get("type"), str)
        or finish["type"] not in {"stop", "length"}
    ):
        invalid.append(
            "Missing or unsuccessful native finish_reason (require stop or length)"
        )

    for canonical, aliases in COUNTER_ALIASES.items():
        observed = {alias: meta[alias] for alias in aliases if alias in meta}
        counter_sources[canonical] = deepcopy(observed)
        values = list(observed.values())
        if not values:
            invalid.append(
                f"Missing {canonical}; rates and accept_length are not counters"
            )
        elif any(not _nonnegative_integer(value) for value in values):
            invalid.append(
                f"Invalid {canonical}; require nonnegative integer, not bool"
            )
        elif len(set(values)) != 1:
            invalid.append(f"Conflicting {canonical} aliases")
        else:
            acceptance[canonical] = values[0]

    a, p, n = (acceptance[name] for name in COUNTER_ALIASES)
    if a is not None and p is not None and a > p:
        invalid.append("accepted_drafts exceeds proposed_drafts")
    if p == 0:
        invalid.append("proposed_drafts must be positive for a speculative measurement")
    if n == 0:
        invalid.append("verify_rounds must be positive for a speculative measurement")

    prompt, cached = actual["prompt_tokens"], actual["cached_tokens"]
    if prompt is not None and cached is not None:
        if cached > prompt:
            invalid.append("cached_tokens exceeds prompt_tokens")
        if prompt > 0:
            actual["cache_hit_ratio"] = cached / prompt
    for name, expected in (
        ("prompt_tokens", input_tokens),
        ("completion_tokens", output_tokens),
        ("cached_tokens", expected_cached_tokens),
        ("num_retractions", 0),
    ):
        if actual[name] is not None and actual[name] != expected:
            mismatches.append(f"{name}: expected {expected}, observed {actual[name]}")

    valid = not invalid
    if valid:
        acceptance["accept_rate"] = a / p
    matches = valid and not mismatches
    status = "GSP_RESPONSE_INVALID"
    if valid:
        status = "GSP_CASE_CONDITIONS_MET" if matches else "GSP_CASE_CONDITIONS_NOT_MET"
    return {
        "status": status,
        "rid": rid,
        "valid_response": valid,
        "conditions_match": matches,
        "issues": invalid + mismatches,
        "invalid_response_issues": invalid,
        "condition_mismatches": mismatches,
        "expected": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cached_tokens": expected_cached_tokens,
            "cache_hit_ratio": expected_cached_tokens / input_tokens,
            "num_retractions": 0,
        },
        "actual": actual,
        "acceptance": acceptance,
        "counter_sources": counter_sources,
        "limits": [
            "Acceptance is accepted draft tokens / proposed draft tokens; accept_length is not this ratio.",
            "A recorded cache count does not establish DDR/SSD cache hits; retain reported tier details separately.",
            "Meeting case conditions is not model accuracy, performance or formal acceptance PASS.",
        ],
    }
