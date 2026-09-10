"""The simple entry reuses real clients; reports must not hide failed attempts."""

import copy
import json

import pytest

import bench_prefix
import bench_accuracy
import run_tests
from report import campaign_report, compare_performance, latest_attempts
from config import MODES


@pytest.mark.parametrize(
    "action,count,concurrency,stage",
    [
        ("check", 64, 8, "check"),
        ("quick", 1, 1, "quick"),
        ("performance", 64, 8, "load"),
    ],
)
def test_entry_uses_existing_prefix_client_without_json(
    monkeypatch, action, count, concurrency, stage
):
    seen = []
    monkeypatch.setattr(
        bench_prefix, "run", lambda args, config: seen.append((args, config)) or 0
    )
    assert run_tests.main([action]) == 0
    args, cfg = seen[0]
    assert (args.num_prompts, args.concurrency, args.action) == (
        count,
        concurrency,
        stage,
    )
    assert args.cache_hit == "all" and args.duration_seconds is None
    assert not hasattr(args, "config")
    assert cfg["base_url"] == "http://61.47.19.71:8810"


def test_prepare_never_replaces_a_different_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    (tmp_path / "gsm8k-10.json").write_text('{"old": true}')
    assert run_tests.main(["prepare"]) == 2
    assert json.loads((tmp_path / "gsm8k-10.json").read_text()) == {"old": True}


@pytest.mark.parametrize("result", (0, 1))
def test_gsm8k_only_prepares_and_runs_without_gpqa(tmp_path, monkeypatch, result):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    assert run_tests.main(["prepare", "--dataset", "gsm8k"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gsm8k-10.json"]
    fixture = bench_accuracy.read_fixture(tmp_path / "gsm8k-10.json")
    assert fixture["dataset"] == "gsm8k" and len(fixture["cases"]) == 10
    seen = []

    def run(args, cfg):
        seen.append(args)
        return result

    monkeypatch.setattr(bench_accuracy, "run", run)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == result
    assert len(seen) == 1 and seen[0].fixture.name == "gsm8k-10.json"
    assert seen[0].limit == 10 and seen[0].concurrency == 1
    assert seen[0].max_tokens == run_tests.ACCURACY_MAX_TOKENS


def test_missing_selected_fixture_does_not_call_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Missing data must not issue model requests")

    monkeypatch.setattr(bench_accuracy, "run", unexpected_run)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == 2


@pytest.mark.parametrize("action", ("check", "quick", "performance", "report"))
def test_dataset_selection_cannot_be_ignored_by_other_actions(action):
    with pytest.raises(SystemExit) as exc:
        run_tests.main([action, "--dataset", "gsm8k"])
    assert exc.value.code == 2


@pytest.mark.parametrize("first_result,expected_count", [(0, 2), (1, 1)])
def test_accuracy_keeps_datasets_separate_and_stops_after_failure(
    monkeypatch, first_result, expected_count
):
    seen = []
    monkeypatch.setattr(
        bench_accuracy, "read_fixture", lambda path: {"cases": [{}] * 10}
    )

    def run(args, cfg):
        seen.append(args)
        assert cfg["tp_size"] == 32 and cfg["dp_size"] == 8
        return first_result if len(seen) == 1 else 0

    monkeypatch.setattr(bench_accuracy, "run", run)
    assert run_tests.main(["accuracy"]) == first_result
    assert len(seen) == expected_count
    assert [a.fixture.name for a in seen] == ["gsm8k-10.json", "gpqa-10.json"][
        :expected_count
    ]
    assert all(
        a.limit == 10 and a.concurrency == 1 and a.max_tokens == 4096 for a in seen
    )


def test_latest_failure_is_not_replaced_by_previous_success(tmp_path):
    for stamp, complete in (("20260910T010000Z", True), ("20260910T020000Z", False)):
        path = (
            tmp_path
            / "evidence/two-node-colocated"
            / f"gsm8k-dspark-eager-{stamp}-deadbeef"
        )
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps(
                {"dataset": "gsm8k", "mode": "dspark-eager", "complete": complete}
            )
        )
    selected, history = latest_attempts(tmp_path)
    assert len(history) == 2
    assert selected[("dspark-eager", "gsm8k")]["value"]["complete"] is False
    assert campaign_report(tmp_path) == 0
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["missing"] and result["qualification"] == "NOT_ASSIGNED"


def test_performance_comparison_requires_identical_inputs_and_complete_runs():
    baseline = {
        "complete": True,
        "invalid_requests": 0,
        "concurrency": 8,
        "measured_requests": 64,
        "expected_cached_tokens": 65536,
        "used_dp_lanes": list(range(8)),
        "results": [{"index": 0, "dp_rank": 0, "input_sha256": "abc"}],
        "native_bench_metrics": {
            "output_throughput": 100,
            "mean_ttft_ms": 1000,
            "mean_tpot_ms": 50,
        },
    }
    candidate = copy.deepcopy(baseline)
    candidate["native_bench_metrics"] = {
        "output_throughput": 200,
        "mean_ttft_ms": 2000,
        "mean_tpot_ms": 25,
    }
    result = compare_performance(baseline, candidate)
    assert result["comparable"]
    assert (
        result["output_throughput_ratio"],
        result["ttft_speedup"],
        result["tpot_speedup"],
    ) == (2, 0.5, 2)
    candidate["results"][0]["input_sha256"] = "other"
    assert compare_performance(baseline, candidate)["output_throughput_ratio"] is None
    candidate = copy.deepcopy(baseline)
    candidate["complete"] = False
    assert not compare_performance(baseline, candidate)["comparable"]


def test_campaign_performance_matrix_uses_matching_baselines(tmp_path):
    server = dict(
        device="npu",
        model_path="/target",
        tp_size=32,
        dp_size=8,
        nnodes=2,
        quantization="modelslim",
        page_size=128,
        context_length=133120,
        max_running_requests=8,
    )
    for mode in MODES:
        path = (
            tmp_path
            / "evidence/two-node-colocated"
            / f"gsp-prefix-{mode}-20260910T010000Z-deadbeef"
        )
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps(
                dict(
                    mode=mode, stage="load", complete=True, server_configuration=server
                )
            )
        )
        for percent in (0, 50, 90):
            cell = path / f"cache{percent}"
            cell.mkdir()
            speculative = not mode.startswith("target")
            (cell / "summary.json").write_text(
                json.dumps(
                    dict(
                        mode=mode,
                        complete=True,
                        invalid_requests=0,
                        concurrency=8,
                        measured_requests=64,
                        expected_cached_tokens=percent,
                        used_dp_lanes=list(range(8)),
                        requested_minimum_prompts=64,
                        requested_minimum_duration_seconds=None,
                        results=[
                            dict(index=i, dp_rank=i % 8, input_sha256=str(i))
                            for i in range(64)
                        ],
                        native_bench_metrics=dict(
                            output_throughput=200 if speculative else 100,
                            mean_ttft_ms=1000,
                            mean_tpot_ms=25,
                        ),
                        accept_rate=0.6 if speculative else None,
                        strictly_above_0_5=True if speculative else None,
                    )
                )
            )
    campaign_report(tmp_path)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert len(report["performance"]) == 15
    assert all(row["comparison"]["comparable"] for row in report["performance"])
    nextn = [r for r in report["performance"] if r["mode"] == "nextn-graph"]
    assert all(
        r["baseline_mode"] == "target-graph"
        and r["comparison"]["output_throughput_ratio"] == 2
        for r in nextn
    )
    assert all(
        r["above_50_percent"] is None
        for r in report["performance"]
        if r["mode"].startswith("target")
    )
