"""CPU checks of full-set completeness and exact acceptance/accuracy gates."""

import csv

import pytest

import run_gpqa_diamond as runner


def response(index=0, histogram=None):
    histogram = [0, 0, 0, 1] if histogram is None else histogram
    n = sum(histogram)
    return {
        "id": f"request-{index}",
        "choices": [{"finish_reason": "stop"}],
        "sglext": {
            "spec_tokens_details": {
                "spec_num_correct_drafts": sum(i * x for i, x in enumerate(histogram)),
                "spec_num_proposed_drafts": 5 * n,
                "spec_verify_ct": n,
                "spec_correct_drafts_histogram": histogram,
            }
        },
    }


def report(correct=180):
    identity = {"name": "accuracy", "aggregation": "mean"}
    return {
        "num": 198,
        "primary_metric_identity": identity,
        "metrics": [
            {"identity": identity, "num": 198, "score": round(correct / 198, 4)}
        ],
        "execution_summary": {
            "requested": 198,
            "succeeded": 198,
            "errored": 0,
            "incomplete": False,
        },
    }


def records(histogram=None):
    return [runner.response_counts(response(i, histogram)) for i in range(198)]


@pytest.mark.parametrize(
    "correct,passed",
    [(178, False), (179, True), (180, True), (181, True), (182, True), (183, False)],
)
def test_accuracy_boundaries_on_single_full_set(correct, passed):
    result = runner.summarize(report(correct), records())
    assert result["correct"] == correct
    assert result["accuracy_pass"] is passed
    assert (result["status"] == "PASS") is passed


@pytest.mark.parametrize(
    "histogram,passed", [([0, 0, 1, 1], False), ([0, 0, 1, 2], True), ([1], False)]
)
def test_strict_acceptance_uses_integer_counts_and_short_histograms(histogram, passed):
    result = runner.summarize(report(), records(histogram))
    assert result["acceptance_pass"] is passed


def test_sum_counts_not_mean_of_request_rates():
    rows = records([1])  # 197 short requests with no accepted drafts
    rows[0] = runner.response_counts(response(0, [0, 0, 0, 0, 0, 1000]))
    result = runner.summarize(report(), rows)
    assert result["A"] == 5000 and result["P"] == 5985
    assert result["acceptance_pass"] is True


@pytest.mark.parametrize(
    "key,value",
    [("requested", 197), ("succeeded", 197), ("errored", 1), ("incomplete", True)],
)
def test_partial_evalscope_execution_cannot_pass(key, value):
    data = report()
    data["execution_summary"][key] = value
    with pytest.raises(ValueError, match="未完成"):
        runner.summarize(data, records())


def test_missing_duplicate_and_length_finished_responses():
    rows = records()
    with pytest.raises(ValueError, match="唯一响应"):
        runner.summarize(report(), rows[:-1])
    with pytest.raises(ValueError, match="唯一响应"):
        runner.summarize(report(), rows[:-1] + [rows[0]])
    rows[0]["finish_reason"] = "length"
    result = runner.summarize(report(), rows)
    assert result["questions"] == 198 and result["length_finished"] == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("spec_num_proposed_drafts", 8),
        ("spec_verify_ct", -1),
        ("spec_num_correct_drafts", True),
        ("spec_correct_drafts_histogram", [1]),
        ("spec_correct_drafts_histogram", "bad"),
    ],
)
def test_bad_server_counters_cannot_pass(key, value):
    data = response()
    data["sglext"]["spec_tokens_details"][key] = value
    with pytest.raises(ValueError):
        runner.response_counts(data)


def test_missing_counts_are_not_synthesized_from_output_tokens():
    data = response()
    data.pop("sglext")
    data["usage"] = {"completion_tokens": 1000}
    with pytest.raises(ValueError, match="缺少"):
        runner.response_counts(data)


@pytest.mark.parametrize("count,duplicate", [(198, False), (197, False), (198, True)])
def test_full_csv_only(tmp_path, count, duplicate):
    path = tmp_path / "gpqa.csv"
    fields = [
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(count):
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [f"Question {0 if duplicate else i}", "a", "b", "c", "d"],
                    )
                )
            )
    if count == 198 and not duplicate:
        assert len(runner.check_dataset(path)) == 64
    else:
        with pytest.raises(ValueError):
            runner.check_dataset(path)


def test_wrong_window_fails_before_evaluation():
    info = dict(
        device="npu",
        nnodes=2,
        tp_size=32,
        dp_size=4,
        served_model_name=runner.MODEL,
        speculative_algorithm="DSPARK",
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=6,
        context_length=133120,
        cuda_graph_config={"decode": {"backend": "full"}},
        enable_metrics=True,
    )
    runner.check_server(info)
    info["speculative_num_draft_tokens"] = 9
    with pytest.raises(ValueError, match="speculative_num_draft_tokens"):
        runner.check_server(info)


@pytest.mark.parametrize("backend", ["disabled", None, "unknown"])
def test_graph_run_rejects_disabled_or_unknown_backend(backend):
    info = dict(
        device="npu",
        nnodes=2,
        tp_size=32,
        dp_size=4,
        served_model_name=runner.MODEL,
        speculative_algorithm="DSPARK",
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=6,
        context_length=133120,
        cuda_graph_config={"decode": {"backend": backend}},
    )
    with pytest.raises(ValueError, match="graph"):
        runner.check_server(info)


def metric(value):
    return f'sglang:cuda_graph_passes_total{{mode="decode_cuda_graph",dp_rank="0"}} {value}\n'


@pytest.mark.parametrize("after", [metric(2), metric(1), "", None])
def test_unobserved_or_reset_graph_preserves_scores_but_cannot_pass(after):
    result = runner.add_graph_result(
        {"status": "PASS", "correct": 180}, metric(2), after
    )
    assert result["status"] == "INCOMPLETE"
    assert result["evaluation_status"] == "PASS"
    assert result["correct"] == 180
    assert result["graph"]["target_replay_observed"] is not True


def test_graph_delta_is_recorded_without_claiming_draft_replay():
    result = runner.add_graph_result({"status": "PASS"}, metric(2), metric(12))
    assert result["status"] == "PASS"
    assert result["graph"]["counters"]["deltas"]["decode_cuda_graph"] == 10
    assert result["graph"]["target_replay_observed"] is True
    assert result["graph"]["draft_replay"] == "NOT_EXPOSED_BY_HTTP_API"


def test_eager_comparison_does_not_require_graph_counters(monkeypatch):
    monkeypatch.setattr(runner, "GRAPH", False)
    result = runner.add_graph_result({"status": "PASS"}, None, None)
    assert result["status"] == "PASS"
    assert result["mode"] == "dspark-eager"


def test_missing_metrics_is_rejected_before_full_evaluation():
    info = dict(
        device="npu",
        nnodes=2,
        tp_size=32,
        dp_size=4,
        served_model_name=runner.MODEL,
        speculative_algorithm="DSPARK",
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=6,
        context_length=133120,
        cuda_graph_config={"decode": {"backend": "full"}},
    )
    with pytest.raises(ValueError, match="enable-metrics"):
        runner.check_server(info)


def test_metrics_scrape_error_is_saved_as_unavailable(tmp_path):
    class BrokenOpener:
        def open(self, *args, **kwargs):
            raise OSError("endpoint unavailable")

    assert runner.collect_graph_metrics(BrokenOpener(), tmp_path, "after") is None
    assert "endpoint unavailable" in (tmp_path / "metrics.after.error.json").read_text()
