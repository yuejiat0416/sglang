"""Small independent contracts for the temporary GSM8K counter summaries."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_mode_stats import graph_count_delta, summarize_responses  # noqa: E402


def response(rid="one", a=1, p=8, n=1):
    return {
        "id": rid,
        "choices": [
            {
                "message": {"content": "Answer", "reasoning_content": "Reason"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 4},
        "sglext": {
            "spec_tokens_details": {
                "spec_num_correct_drafts": a,
                "spec_num_proposed_drafts": p,
                "spec_verify_ct": n,
            }
        },
    }


def sample(value, mode="decode_cuda_graph", rank="0", reverse=False):
    labels = f'mode="{mode}",tp_rank="{rank}"'
    if reverse:
        labels = f'tp_rank="{rank}",mode="{mode}"'
    return f"sglang:cuda_graph_passes_total{{{labels}}} {value}\n"


def test_weighted_aggregate_not_mean_request_rates():
    original = [response("one", 1, 8, 1), response("two", 72, 80, 10)]
    saved = copy.deepcopy(original)
    report = summarize_responses(["one", "two"], original, "dspark-eager")
    assert report["complete"]
    assert report["aggregate"]["accept_rate"] == 73 / 88
    assert report["aggregate"]["mean_accepted_drafts_per_round"] == 73 / 11
    assert report["aggregate"]["one_plus_mean_accepted_drafts_per_round"] == 1 + 73 / 11
    assert report["aggregate"]["accept_rate"] != (1 / 8 + 72 / 80) / 2
    assert report["per_question"][0]["content"] == "Answer"
    assert report["per_question"][0]["reasoning_content"] == "Reason"
    assert report["per_question"][0]["response"] == saved[0]
    assert original == saved
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mode", ["dspark-eager", "dspark-graph", "nextn-graph"])
def test_identical_aliases_across_sources(mode):
    row = response()
    row["choices"][0]["meta_info"] = {
        "spec_accepted_drafts": 1,
        "spec_proposed_drafts": 8,
        "spec_verify_ct": 1,
        "id": "one",
        "completion_tokens": 4,
    }
    report = summarize_responses(["one"], [row], mode)
    assert report["complete"]
    assert len(report["per_question"][0]["counter_sources"]["accepted_drafts"]) == 2


def test_meta_aliases_alone_suffice():
    row = response()
    del row["sglext"]
    row["choices"][0]["meta_info"] = {
        "spec_accepted_drafts": 1,
        "spec_proposed_drafts": 8,
        "spec_verify_ct": 1,
    }
    assert summarize_responses(["one"], [row], "dspark-eager")["complete"]


@pytest.mark.parametrize(
    "missing",
    ["sglext", "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct"],
)
def test_missing_counts_never_reconstructed(missing):
    row = response()
    if missing == "sglext":
        del row["sglext"]
    else:
        del row["sglext"]["spec_tokens_details"][missing]
    row["choices"][0]["meta_info"] = {
        "spec_accept_rate": 0.125,
        "spec_accept_length": 4.0,
    }
    report = summarize_responses(["one"], [row], "dspark-eager")
    assert not report["complete"]
    assert report["aggregate"]["accept_rate"] is None
    assert any("Missing" in issue for issue in report["issues"])


@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "1", None])
def test_invalid_counter_values(bad):
    report = summarize_responses(["one"], [response(a=bad)], "dspark-graph")
    assert not report["complete"]
    assert report["aggregate"]["accepted_drafts"] is None


@pytest.mark.parametrize("a,p,n", [(9, 8, 1), (1, 0, 0), (0, 8, 0), (0, 0, 1)])
def test_inconsistent_counts(a, p, n):
    assert not summarize_responses(["one"], [response(a=a, p=p, n=n)], "dspark-eager")[
        "complete"
    ]


def test_zero_verification_is_collected_with_unavailable_rate():
    report = summarize_responses(["one"], [response(a=0, p=0, n=0)], "dspark-eager")
    assert report["complete"]
    assert report["aggregate"]["accept_rate"] is None
    assert report["aggregate"]["mean_accepted_drafts_per_round"] is None


@pytest.mark.parametrize("same_source", [True, False])
def test_conflicting_count_aliases(same_source):
    row = response()
    if same_source:
        row["sglext"]["spec_tokens_details"]["spec_accepted_drafts"] = 2
    else:
        row["choices"][0]["meta_info"] = {"spec_num_correct_drafts": 2}
    report = summarize_responses(["one"], [row], "dspark-eager")
    assert not report["complete"]
    assert any("Conflicting" in issue for issue in report["issues"])


@pytest.mark.parametrize("mode", ["target-eager", "target-graph"])
@pytest.mark.parametrize("counter_form", ["absent", "zero", "partial_zero"])
def test_target_acceptance_is_not_applicable(mode, counter_form):
    row = response(a=0, p=0, n=0)
    if counter_form == "absent":
        del row["sglext"]
    elif counter_form == "partial_zero":
        del row["sglext"]["spec_tokens_details"]["spec_verify_ct"]
    report = summarize_responses(["one"], [row], mode)
    assert report["complete"]
    assert report["aggregate"]["accept_rate"] is None
    assert report["aggregate"]["applicability"] == "not_applicable_target_only"
    assert report["aggregate"]["mean_accepted_drafts_per_round"] is None


def test_target_nonzero_speculation_is_error():
    report = summarize_responses(["one"], [response()], "target-eager")
    assert not report["complete"]
    assert any("Target-only" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "foreign",
        "missing_id",
        "not_object",
        "error",
        "no_choice",
        "two_choices",
        "unfinished",
        "no_tokens",
        "bad_tokens",
        "conflicting_tokens",
        "meta_id",
    ],
)
def test_response_integrity(kind):
    row = response()
    rows = [row]
    if kind == "missing":
        rows = []
    elif kind == "duplicate":
        rows.append(copy.deepcopy(row))
    elif kind == "foreign":
        rows.append(response("other"))
    elif kind == "missing_id":
        del row["id"]
    elif kind == "not_object":
        rows.append("unexpected")
    elif kind == "error":
        row["error"] = {"message": "failed"}
    elif kind == "no_choice":
        row["choices"] = []
    elif kind == "two_choices":
        row["choices"] *= 2
    elif kind == "unfinished":
        row["choices"][0]["finish_reason"] = None
    elif kind == "no_tokens":
        del row["usage"]
    elif kind == "bad_tokens":
        row["usage"]["completion_tokens"] = True
    elif kind == "conflicting_tokens":
        row["choices"][0]["meta_info"] = {"completion_tokens": 5}
    elif kind == "meta_id":
        row["choices"][0]["meta_info"] = {"id": "other"}
    result = summarize_responses(["one"], rows, "dspark-eager")
    assert not result["complete"]
    assert result["aggregate"]["accept_rate"] is None
    assert result["issues"]


def test_partial_valid_counts_retained_but_not_full_rate():
    report = summarize_responses(["one", "two"], [response()], "dspark-eager")
    assert report["partial_valid_counts"]["accepted_drafts"] == 1
    assert report["aggregate"]["accepted_drafts"] is None
    assert report["aggregate"]["accept_rate"] is None


@pytest.mark.parametrize("ids", [[], ["one", "one"], [None], "one", [""]])
def test_bad_expected_ids_raise(ids):
    with pytest.raises(ValueError):
        summarize_responses(ids, [], "dspark-eager")


def test_bad_mode_raise():
    with pytest.raises(ValueError):
        summarize_responses(["one"], [], "unknown")


def test_graph_full_labels_distinguished_order_irrelevant():
    before = sample(5, rank="0") + sample(20, rank="1") + sample(7, "decode_none")
    after = (
        sample(10, rank="0", reverse=True)
        + sample(22, rank="1")
        + sample(8, "decode_none")
    )
    result = graph_count_delta(before, after)
    assert result["available"]
    assert result["deltas"] == {"decode_cuda_graph": 7, "decode_none": 1}
    assert len(result["series"]) == 3
    json.dumps(result, allow_nan=False)


def test_graph_new_series_zero_baseline_and_absent_mode_unavailable():
    result = graph_count_delta("# HELP other Something\n", sample(9))
    assert result["available"]
    assert result["deltas"] == {"decode_cuda_graph": 9, "decode_none": None}
    assert result["series"][0]["new_series"]


def test_graph_zero_delta_is_measured():
    result = graph_count_delta(sample(9), sample(9))
    assert result["available"]
    assert result["deltas"]["decode_cuda_graph"] == 0


@pytest.mark.parametrize(
    "before,after", [("", ""), (None, sample(9)), (sample(9), None)]
)
def test_graph_missing_metrics_unavailable(before, after):
    result = graph_count_delta(before, after)
    assert not result["available"]
    assert result["status"] == "GRAPH_COUNTERS_UNAVAILABLE"
    assert all(value is None for value in result["deltas"].values())


@pytest.mark.parametrize("after", [sample(2), ""])
def test_graph_reset_or_disappearance_invalid(after):
    result = graph_count_delta(sample(5), after)
    assert not result["available"]
    assert result["status"] == "GRAPH_COUNTERS_INVALID"
    assert all(value is None for value in result["deltas"].values())


def test_graph_no_cross_label_cancellation_of_reset():
    result = graph_count_delta(
        sample(10, rank="0") + sample(10, rank="1"),
        sample(9, rank="0") + sample(100, rank="1"),
    )
    assert not result["available"]
    assert any("reset" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    "text",
    [
        sample("NaN"),
        sample("Inf"),
        sample(-1),
        sample(1) * 2,
        'sglang:cuda_graph_passes_total{mode="decode_none",mode="decode_none"} 1',
        "sglang:cuda_graph_passes_total{bad} 1",
    ],
)
def test_graph_malformed_metrics_invalid(text):
    assert graph_count_delta("", text)["status"] == "GRAPH_COUNTERS_INVALID"


def test_graph_escaped_labels_timestamp_and_other_modes():
    line = 'sglang:cuda_graph_passes_total{mode="decode_cuda_graph",model_name="a\\"b\\\\c"} 1e1 1000\n'
    ignored = 'sglang:cuda_graph_passes_total{mode="prefill_cuda_graph"} 900\nsglang:cuda_graph_passes_total_created{mode="decode_cuda_graph"} 12\n'
    result = graph_count_delta(ignored, line + ignored)
    assert result["available"]
    assert result["deltas"]["decode_cuda_graph"] == 10
    assert result["series"][0]["labels"]["model_name"] == 'a"b\\c'
