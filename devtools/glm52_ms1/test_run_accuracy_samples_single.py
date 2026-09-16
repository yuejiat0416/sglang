import csv
import json

import pytest

import run_accuracy_samples_single as runner


def server(speculative=True):
    return {
        "device": "npu",
        "nnodes": 1,
        "tp_size": 16,
        "served_model_name": runner.MODEL,
        "context_length": 69632,
        "speculative_algorithm": "DSPARK" if speculative else None,
        "speculative_num_draft_tokens": 9 if speculative else None,
        "disable_cuda_graph": False,
        "cuda_graph_config": {"decode": {"backend": "full"}},
    }


def response(index=0, speculative=True, a=4, n=1):
    value = {
        "id": f"request-{index}",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"completion_tokens": 10},
    }
    if speculative:
        value["sglext"] = {
            "spec_tokens_details": {
                "spec_num_correct_drafts": a,
                "spec_num_proposed_drafts": 8 * n,
                "spec_verify_ct": n,
            }
        }
    return value


def report(limit, score=0.8):
    identity = {"name": "accuracy", "aggregation": "mean"}
    return {
        "num": limit,
        "primary_metric_identity": identity,
        "metrics": [{"identity": identity, "num": limit, "score": score}],
        "execution_summary": {
            "requested": limit,
            "succeeded": limit,
            "errored": 0,
            "incomplete": False,
        },
    }


def test_server_mode_is_read_from_runtime():
    mode = runner.check_server(server())
    assert mode == {
        "algorithm": "DSPARK",
        "speculative": True,
        "gamma": 8,
        "graph": True,
        "mode": "dspark-graph",
    }
    target = runner.check_server(server(False))
    assert target["mode"] == "target-graph" and target["gamma"] is None


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("nnodes", 2, "单机"),
        ("tp_size", 32, "单机"),
        ("context_length", 16384, "65536"),
        ("served_model_name", "other", "服务名"),
    ],
)
def test_wrong_server_fails_before_requests(field, value, match):
    info = server()
    info[field] = value
    with pytest.raises(ValueError, match=match):
        runner.check_server(info)


def test_response_counts_and_target_only():
    mode = runner.check_server(server())
    row = runner.response_record(response(), mode)
    assert (row["A"], row["P"], row["N"], row["accept_rate"]) == (4, 8, 1, 0.5)
    target = runner.response_record(response(speculative=False), runner.check_server(server(False)))
    assert target["A"] is None and target["accept_rate"] is None


@pytest.mark.parametrize("a,p,n", [(9, 8, 1), (1, 7, 1), (1, 8, -1)])
def test_invalid_acceptance_counts_fail(a, p, n):
    data = response(a=a, n=max(n, 0))
    details = data["sglext"]["spec_tokens_details"]
    details.update(
        spec_num_correct_drafts=a,
        spec_num_proposed_drafts=p,
        spec_verify_ct=n,
    )
    with pytest.raises(ValueError, match="A/P/N|非负"):
        runner.response_record(data, runner.check_server(server()))


def test_summary_uses_weighted_counts_and_small_sample_is_not_a_quality_pass():
    mode = runner.check_server(server())
    rows = [runner.response_record(response(i, a=0), mode) for i in range(9)]
    rows.append(runner.response_record(response(9, a=8), mode))
    result = runner.summarize_report("gpqa_diamond", report(10, 0.7), rows, 10, mode)
    assert result["approximate_correct"] == 7
    assert result["quality_gate"] == "NOT_ASSIGNED_SMALL_SAMPLE"
    assert result["acceptance"]["accepted_drafts"] == 8
    assert result["acceptance"]["proposed_drafts"] == 80
    assert result["acceptance"]["accept_rate"] == 0.1


def test_incomplete_or_duplicate_evalscope_run_fails():
    mode = runner.check_server(server())
    rows = [runner.response_record(response(i), mode) for i in range(10)]
    bad = report(10)
    bad["execution_summary"]["succeeded"] = 9
    with pytest.raises(ValueError, match="没有完成"):
        runner.summarize_report("gsm8k", bad, rows, 10, mode)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="唯一"):
        runner.summarize_report("gsm8k", report(10), rows, 10, mode)


def test_dataset_validators(tmp_path, monkeypatch):
    gsm = tmp_path / "gsm.jsonl"
    gsm.write_text(
        "\n".join(
            json.dumps({"question": f"q{i}", "answer": f"a{i}"}) for i in range(3)
        )
        + "\n"
    )
    monkeypatch.setattr(runner, "GSM8K_ROWS", 3)
    assert runner.validate_gsm8k(gsm)["rows"] == 3

    gpqa = tmp_path / "gpqa.csv"
    with gpqa.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.GPQA_FIELDS)
        writer.writeheader()
        for i in range(2):
            writer.writerow(dict(zip(runner.GPQA_FIELDS, [f"q{i}", "a", "b", "c", "d"])))
    monkeypatch.setattr(runner, "GPQA_ROWS", 2)
    assert runner.validate_gpqa(gpqa)["rows"] == 2
