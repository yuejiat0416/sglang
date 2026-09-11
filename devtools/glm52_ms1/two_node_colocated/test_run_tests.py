"""The simple entry reuses real clients; reports must not hide failed attempts."""

import copy
import json
import shlex

import bench_accuracy
import bench_prefix
import client_common
import pytest
import run_tests
from config import MODES
from report import campaign_report, compare_performance, latest_attempts


def test_check_still_only_uses_read_only_capacity_client(monkeypatch):
    seen = []
    monkeypatch.setattr(
        bench_prefix, "run", lambda args, config: seen.append((args, config)) or 0
    )
    assert run_tests.main(["check"]) == 0
    args, cfg = seen[0]
    assert (args.num_prompts, args.concurrency, args.action) == (
        64,
        4,
        "check",
    )
    assert args.cache_hit == "all" and args.duration_seconds is None
    assert not hasattr(args, "config")
    assert cfg["base_url"] == "http://61.47.19.68:8810"


@pytest.mark.parametrize(
    "action,count,concurrency", [("quick", 8, 1), ("performance", 64, 4)]
)
@pytest.mark.parametrize("failure", ["none", "exit", "request", "short"])
def test_native_cli_subprocess_outputs_and_failure_stop(
    tmp_path, monkeypatch, action, count, concurrency, failure
):
    # Exercise the real subprocess boundary with a tiny CLI stand-in. This
    # deliberately cannot import a tokenizer, old client or NPU package.
    repo = tmp_path / "checkout"
    package = repo / "python/sglang/benchmark"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "serving.py").write_text("""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def arg(name): return args[args.index(name) + 1]
count = int(arg("--num-prompts"))
failure = os.environ["TEST_NATIVE_FAILURE"]
print("native child executed", flush=True)
if failure == "exit": sys.exit(7)
record = {"completed": count, "output_lens": [1024] * count, "errors": [""] * count}
if failure == "request":
    record["completed"] -= 1
    record["errors"][0] = "request failed"
if failure == "short": record["output_lens"][0] = 512
Path(arg("--output-file")).write_text(json.dumps(record) + "\\n")
""")
    monkeypatch.setattr(client_common, "REPO", repo)
    monkeypatch.setattr(run_tests, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("TEST_NATIVE_FAILURE", failure)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Native benchmark must not enter custom GSP client")

    monkeypatch.setattr(bench_prefix, "run", unexpected_run)
    expected = {"none": 0, "exit": 7, "request": 1, "short": 1}[failure]
    assert run_tests.main([action]) == expected
    runs = list((tmp_path / "results/evidence/two-node-colocated").iterdir())
    assert len(runs) == 1
    commands = sorted(runs[0].glob("*.command.txt"))
    assert len(commands) == (3 if failure == "none" else 1)
    for path in commands:
        argv = shlex.split(path.read_text())
        assert argv[1:3] == ["-m", "sglang.benchmark.serving"]

        def value(flag):
            return argv[argv.index(flag) + 1]

        assert value("--dataset-name") == "generated-shared-prefix"
        assert value("--num-prompts") == value("--gsp-prompts-per-group") == str(count)
        assert value("--max-concurrency") == str(concurrency)
        assert (
            int(value("--gsp-system-prompt-len")) + int(value("--gsp-question-len"))
            == 131072
        )
        assert value("--warmup-requests") == "0" and "--flush-cache" in argv
        assert not any(
            x in argv
            for x in ("--extra-request-body", "--disable-stream", "--tokenize-prompt")
        )
        assert (
            "native child executed"
            in path.with_name(path.name.replace(".command.txt", ".log")).read_text()
        )


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


def test_missing_selected_source_does_not_call_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Missing data must not issue model requests")

    monkeypatch.setattr(bench_accuracy, "run", unexpected_run)
    assert run_tests.main(["accuracy", "--dataset", "gpqa"]) == 2


def test_accuracy_prepares_bundled_ten_without_separate_command(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    seen = []
    monkeypatch.setattr(bench_accuracy, "run", lambda args, cfg: seen.append(args) or 0)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == 0
    assert len(bench_accuracy.read_fixture(seen[0].fixture)["cases"]) == 10


def write_gsm_source(path, count):
    path.write_text(
        "".join(
            json.dumps(
                {"question": f"What is {i} plus 1?", "answer": f"Steps\n#### {i + 1}"}
            )
            + "\n"
            for i in range(count)
        )
    )


@pytest.mark.parametrize("count", (100, 1319))
def test_larger_gsm_run_uses_raw_source_once_and_preserves_ten(
    tmp_path, monkeypatch, count
):
    source = tmp_path / "gsm8k-test.jsonl"
    write_gsm_source(source, 1319)
    old = tmp_path / "gsm8k-10.json"
    old.write_text('{"previous_ten_question_fixture": true}')
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    monkeypatch.setattr(run_tests, "GSM8K_LIMIT", count)
    monkeypatch.setattr(run_tests, "GSM8K_SOURCE", str(source))
    seen = []

    def collect(args, cfg):
        fixture = bench_accuracy.read_fixture(args.fixture)
        requests, mapping = bench_accuracy.make_requests(
            fixture["cases"][: args.limit], "test-full", args.max_tokens
        )
        assert len(requests) == len(mapping) == count
        assert fixture["source"]["row_count"] == 1319
        assert fixture["cases"][-1]["answer"] == str(count)
        assert all("Steps" not in r["messages"][0]["content"] for r in requests)
        assert all("answer" not in r for r in requests)
        assert args.concurrency == 1 and args.max_tokens == 4096
        seen.append(args)
        return 0

    monkeypatch.setattr(bench_accuracy, "run", collect)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == 0
    assert len(seen) == 1 and seen[0].fixture.name == f"gsm8k-{count}.json"
    assert old.read_text() == '{"previous_ten_question_fixture": true}'
    assert not (tmp_path / "gpqa-10.json").exists()


@pytest.mark.parametrize("source_kind", ("unset", "missing", "too_short"))
def test_full_gsm_never_falls_back_to_ten_or_sends_partial_dataset(
    tmp_path, monkeypatch, source_kind
):
    source = tmp_path / "gsm8k-test.jsonl"
    if source_kind == "too_short":
        write_gsm_source(source, 10)
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    monkeypatch.setattr(run_tests, "GSM8K_LIMIT", 1319)
    monkeypatch.setattr(
        run_tests, "GSM8K_SOURCE", "" if source_kind == "unset" else str(source)
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Incomplete full dataset must not issue requests")

    monkeypatch.setattr(bench_accuracy, "run", unexpected_run)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == 2
    assert not (tmp_path / "gsm8k-1319.json").exists()


def test_larger_gsm_rechecks_source_before_reusing_fixture(tmp_path, monkeypatch):
    source = tmp_path / "gsm8k-test.jsonl"
    write_gsm_source(source, 100)
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    monkeypatch.setattr(run_tests, "GSM8K_LIMIT", 100)
    monkeypatch.setattr(run_tests, "GSM8K_SOURCE", str(source))
    assert run_tests.main(["prepare", "--dataset", "gsm8k"]) == 0
    old = (tmp_path / "gsm8k-100.json").read_bytes()
    source.write_text(source.read_text().replace("What is 0", "Calculate 0", 1))

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Changed source must not silently reuse saved questions")

    monkeypatch.setattr(bench_accuracy, "run", unexpected_run)
    assert run_tests.main(["accuracy", "--dataset", "gsm8k"]) == 2
    assert (tmp_path / "gsm8k-100.json").read_bytes() == old


@pytest.mark.parametrize("count", (0, -1, True, "1319"))
def test_gsm_count_must_be_explicit_positive_integer(tmp_path, monkeypatch, count):
    monkeypatch.setattr(run_tests, "DATASETS", str(tmp_path))
    monkeypatch.setattr(run_tests, "GSM8K_LIMIT", count)
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
    monkeypatch.setattr(run_tests, "prepare", lambda datasets: 0)
    monkeypatch.setattr(
        bench_accuracy, "read_fixture", lambda path: {"cases": [{}] * 10}
    )

    def run(args, cfg):
        seen.append(args)
        assert cfg["tp_size"] == 32 and cfg["dp_size"] == 4
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
