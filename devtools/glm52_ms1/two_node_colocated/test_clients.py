"""Two-node client identities, gold isolation, partial results and paired scoring."""

import copy
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench_accuracy as runner
import client_common as common
from config import MODES, validate_config
from offline_dataset import prepare_dataset
from report import compare, inventory


@pytest.fixture
def cfg():
    value = json.loads(Path(__file__).with_name("config.example.json").read_text())
    value.update(
        repo="/srv/test/sglang",
        kernel_repo="/srv/test/kernel",
        state="/srv/test/state",
        target_model="/srv/test/target",
        draft_model="/srv/test/draft",
        tokenizer="/srv/test/target",
        served_model_name="test-model",
    )
    for rank, node in enumerate(value["nodes"]):
        node.update(
            host=f"192.0.2.{rank + 1}",
            hccl_socket_ifname="eth0",
            gloo_socket_ifname="eth0",
        )
    return validate_config(value)


def server(cfg, mode):
    ds, nxt = mode.startswith("dspark"), mode.startswith("nextn")
    return {
        "device": "npu",
        "nnodes": 2,
        "tp_size": 32,
        "dp_size": cfg["dp_size"],
        "enable_dp_attention": True,
        "model_path": cfg["target_model"],
        "served_model_name": cfg["served_model_name"],
        "quantization": "modelslim",
        "enable_metrics": True,
        "speculative_algorithm": "DSPARK" if ds else "EAGLE" if nxt else None,
        "speculative_draft_model_path": cfg["draft_model"] if ds else None,
        "speculative_dspark_block_size": 8 if ds else None,
        "speculative_num_draft_tokens": 9 if ds else 5 if nxt else None,
        "speculative_num_steps": 4 if nxt else None,
        "speculative_eagle_topk": 1 if nxt else None,
        "disable_cuda_graph": mode.endswith("eager"),
        "disable_decode_cuda_graph": False,
        "cuda_graph_config": {
            "decode": {"backend": "disabled" if mode.endswith("eager") else "full"}
        },
    }


@pytest.mark.parametrize("mode", MODES)
def test_two_node_modes_and_non_applicable_algorithms(cfg, mode):
    data = server(cfg, mode)
    common.validate_server(data, cfg, mode)
    data["nnodes"] = 1
    with pytest.raises(ValueError, match="nnodes"):
        common.validate_server(data, cfg, mode)


@pytest.mark.parametrize("mode", ("dspark-eager", "dspark-graph"))
@pytest.mark.parametrize("block,draft_tokens", ((8, 9), (5, 6)))
def test_dspark_supported_windows_preserve_reported_values(
    cfg, mode, block, draft_tokens
):
    data = server(cfg, mode)
    data["speculative_dspark_block_size"] = block
    data["speculative_num_draft_tokens"] = draft_tokens
    selected = common.validate_server(data, cfg, mode)
    assert selected["speculative_dspark_block_size"] == block
    assert selected["speculative_num_draft_tokens"] == draft_tokens


@pytest.mark.parametrize(
    "block,draft_tokens", ((5, 9), (8, 6), (5, 5), (6, 7), (None, 6), (5, None))
)
def test_dspark_inconsistent_or_unsupported_windows_block_traffic(
    cfg, block, draft_tokens
):
    data = server(cfg, "dspark-eager")
    data["speculative_dspark_block_size"] = block
    data["speculative_num_draft_tokens"] = draft_tokens
    with pytest.raises(ValueError, match="DSpark block"):
        common.validate_server(data, cfg, "dspark-eager")


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_dp_attention", False),
        ("model_path", "/wrong"),
        ("speculative_draft_model_path", "/wrong-draft"),
        ("quantization", None),
        ("disaggregation_mode", "prefill"),
        ("speculative_num_draft_tokens", 8),
        ("cuda_graph_config", {}),
    ],
)
def test_mislabeled_server_blocks_before_traffic(cfg, key, value):
    data = server(cfg, "dspark-eager")
    data[key] = value
    with pytest.raises(ValueError):
        common.validate_server(data, cfg, "dspark-eager")


def test_gold_never_enters_requests_and_repeats_are_distinct():
    cases = [
        {
            "id": "q1",
            "answer": "SECRET_GOLD",
            "messages": [{"role": "user", "content": "Question"}],
        }
    ]
    rows, mapping = runner.make_requests(cases, "run", 4096, 2)
    assert len(rows) == len(mapping) == 2
    assert "SECRET_GOLD" not in json.dumps(rows)
    assert (
        len({row["rid"] for row in rows})
        == len({row["cache_salt"] for row in rows})
        == 2
    )
    assert all(row["max_tokens"] == 4096 and row["temperature"] == 0 for row in rows)


def test_offline_bundle_and_benchmark_no_remote_alias_or_warmup(cfg, tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(prepare_dataset("gsm8k")))
    bundle = runner.read_fixture(path)
    assert len(bundle["cases"]) == 10
    argv = runner.bench_arguments(cfg, tmp_path, 10, 4096, 1)
    for flag, expected in (
        ("--model", cfg["target_model"]),
        ("--tokenizer", cfg["tokenizer"]),
        ("--warmup-requests", "0"),
        ("--num-prompts", "10"),
    ):
        assert argv[argv.index(flag) + 1] == expected
    assert "--flush-cache" not in argv and "--disable-ignore-eos" in argv


def test_failed_or_truncated_requests_never_disappear_from_score():
    rows, mapping = runner.make_requests(
        [
            {
                "id": "q",
                "answer": "42",
                "messages": [{"role": "user", "content": "question"}],
            }
        ],
        "x",
        8,
        2,
    )
    per = [
        {
            "id": rows[0]["rid"],
            "valid": True,
            "finish_reason": "length",
            "response": {"choices": [{"message": {"content": "#### 42"}}]},
        },
        {"id": rows[1]["rid"], "valid": False, "response": {"choices": ["malformed"]}},
    ]
    scored = runner.score_collected({"per_question": per}, mapping, "gsm8k")
    assert scored["samples"] == 2 and scored["correct"] == 0
    assert scored["truncated"] == 1 and scored["unresolved"] == 2


def paired():
    return {
        "mode": "target-eager",
        "complete": True,
        "dataset": "gsm8k",
        "protocol": {"max_tokens": 4096},
        "server_configuration": {"tp_size": 32},
        "accuracy": {
            "accuracy": 1,
            "per_question": [
                {
                    "case_id": "q",
                    "repeat": 1,
                    "score": {"correct": True},
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": [3, 4],
                }
            ],
        },
    }


def test_comparison_reports_lost_answers_and_first_token_difference():
    left = paired()
    right = copy.deepcopy(left)
    right["mode"] = "dspark-eager"
    right["accuracy"]["accuracy"] = 0
    row = right["accuracy"]["per_question"][0]
    row["score"]["correct"] = False
    row["response_token_ids"] = [3, 5]
    result = compare(left, right)
    assert (
        result["comparable"] and result["accuracy_delta_candidate_minus_baseline"] == -1
    )
    assert (
        result["lost_correct"] == 1
        and result["per_question"][0]["first_different_token_index"] == 1
    )
    assert result["qualification"] == "NOT_ASSIGNED"


@pytest.mark.parametrize(
    "change", ["protocol", "missing_prompt", "prompt_difference", "partial"]
)
def test_incompatible_comparisons_have_no_regression_conclusion(change):
    left, right = paired(), paired()
    if change == "protocol":
        right["protocol"]["max_tokens"] = 1024
    elif change == "missing_prompt":
        right["accuracy"]["per_question"][0]["prompt_token_ids"] = None
    elif change == "prompt_difference":
        right["accuracy"]["per_question"][0]["prompt_token_ids"] = [2, 3]
    else:
        right["complete"] = False
    result = compare(left, right)
    assert (
        not result["comparable"]
        and result["accuracy_delta_candidate_minus_baseline"] is None
    )


def test_client_environment_restored_even_on_failure(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "old")
    monkeypatch.setenv("SGLANG_IS_IN_CI", "true")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(RuntimeError), common.benchmark_context():
        assert common.os.environ["NO_PROXY"] == "*"
        assert common.os.environ["SGLANG_IS_IN_CI"] == "false"
        raise RuntimeError("client stopped")
    assert common.os.environ["NO_PROXY"] == "old"
    assert common.os.environ["SGLANG_IS_IN_CI"] == "true"
    assert "HF_HUB_OFFLINE" not in common.os.environ


def test_inventory_keeps_repeats_and_separates_quick_load_and_cache(tmp_path):
    for name, data in {
        "a": {
            "mode": "dspark-eager",
            "stage": "quick",
            "cache_percent_label": 50,
            "complete": True,
        },
        "b": {
            "mode": "dspark-eager",
            "stage": "load",
            "cache_percent_label": 50,
            "complete": False,
        },
        "c": {"mode": "dspark-eager", "dataset": "gsm8k", "complete": True},
    }.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(data))
    result = inventory(tmp_path)
    assert len(result["records"]) == 3
    assert "dspark-eager/quick/cache50" not in result["missing_complete_prefix_runs"]
    assert "dspark-eager/load/cache50" in result["missing_complete_prefix_runs"]
    assert "dspark-eager/gpqa" in result["missing_complete_dataset_runs"]


def test_full_client_collection_mock_transport_preserves_benchmark_contract(
    cfg, tmp_path, monkeypatch
):
    cfg["state"] = str(tmp_path / "state")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    fixture = tmp_path / "gsm.json"
    fixture.write_text(json.dumps(prepare_dataset("gsm8k", limit=1)))
    info = server(cfg, "dspark-eager")
    monkeypatch.setattr(
        runner,
        "fetch_text",
        lambda url: json.dumps(info) if url.endswith("server_info") else "",
    )
    active = []

    @contextlib.contextmanager
    def capture(bench, path, rows):
        active[:] = [path, rows]
        yield

    def cli():
        args = runner.sys.argv
        requests = Path(args[args.index("--dataset-path") + 1])
        row = json.loads(requests.read_text().strip())
        answer = json.loads(fixture.read_text())["cases"][0]["answer"]
        value = {
            "id": row["rid"],
            "usage": {"completion_tokens": 10},
            "choices": [
                {
                    "message": {"content": "#### " + answer},
                    "finish_reason": "stop",
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": [3, 4],
                }
            ],
            "sglext": {
                "spec_tokens_details": {
                    "spec_num_correct_drafts": 5,
                    "spec_num_proposed_drafts": 8,
                    "spec_verify_ct": 1,
                }
            },
        }
        active[1].append(value)
        active[0].write_text(json.dumps(value) + "\n")
        Path(args[args.index("--output-file") + 1]).write_text('{"completed":1}\n')

    monkeypatch.setattr(runner, "capture_responses", capture)
    monkeypatch.setattr(
        runner,
        "import_benchmark",
        lambda: SimpleNamespace(cli_main=cli, __file__=__file__),
    )
    args = SimpleNamespace(
        config=config,
        fixture=fixture,
        limit=1,
        mode="dspark-eager",
        max_tokens=4096,
        repeats=1,
        concurrency=1,
    )
    assert runner.run(args) == 0
    summary = json.loads(next((tmp_path / "state").rglob("summary.json")).read_text())
    assert summary["complete"] and summary["accuracy"]["correct"] == 1
    assert summary["acceptance"]["accept_rate"] == 5 / 8


def test_wrong_topology_sends_no_traffic_and_retains_incomplete_evidence(
    cfg, tmp_path, monkeypatch
):
    cfg["state"] = str(tmp_path / "state")
    config, fixture = tmp_path / "config.json", tmp_path / "gsm.json"
    config.write_text(json.dumps(cfg))
    fixture.write_text(json.dumps(prepare_dataset("gsm8k", limit=1)))
    data = server(cfg, "dspark-eager")
    data["nnodes"] = 1
    monkeypatch.setattr(runner, "fetch_text", lambda _: json.dumps(data))
    monkeypatch.setattr(
        runner, "import_benchmark", lambda: pytest.fail("must not send traffic")
    )
    args = SimpleNamespace(
        config=config,
        fixture=fixture,
        limit=1,
        mode="dspark-eager",
        max_tokens=4096,
        repeats=1,
        concurrency=1,
    )
    assert runner.run(args) == 1
    summary = json.loads(next((tmp_path / "state").rglob("summary.json")).read_text())
    assert not summary["complete"] and summary["acceptance"] is None
    assert (
        summary["accuracy"]["samples"] == 1 and summary["accuracy"]["unresolved"] == 1
    )
