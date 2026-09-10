"""CPU checks for preview isolation, mode contracts and node identity pairing."""

import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _module("config")
with mock.patch.dict(sys.modules, {"config": config}):
    launch = _module("launch")
    check_node = _module("check_node")


def configuration():
    raw = json.loads((HERE / "config.example.json").read_text())
    raw.update(
        repo="/srv/test/sglang",
        kernel_repo="/srv/test/kernel",
        state="/srv/test/state",
        target_model="/srv/test/target",
        draft_model="/srv/test/draft",
        tokenizer="/srv/test/target",
        served_model_name="test-model",
    )
    raw["nodes"] = [
        {
            "rank": 0,
            "host": "192.0.2.10",
            "hccl_socket_ifname": "eth1",
            "gloo_socket_ifname": "eth0",
        },
        {
            "rank": 1,
            "host": "192.0.2.11",
            "hccl_socket_ifname": "eth2",
            "gloo_socket_ifname": "eth0",
        },
    ]
    return raw


def test_example_requires_actual_hosts_and_interfaces():
    with pytest.raises(ValueError, match="Fill"):
        config.load_config(HERE / "config.example.json")


@pytest.mark.parametrize(
    "key,value",
    [
        ("context_length", 132096),
        ("tp_size", 16),
        ("dp_size", 3),
        ("nnodes", 1),
        ("max_total_tokens", -1),
        ("max_running_requests", True),
        ("mem_fraction_static", 1),
        ("graph_batch_sizes", [16, 8]),
        ("port", 70000),
        ("repo", "relative/repo"),
        ("max_running_requests", 7),
    ],
)
def test_invalid_configuration_fails_before_any_action(key, value):
    raw = configuration()
    raw[key] = value
    with pytest.raises(ValueError):
        config.validate_config(raw)


def test_prefill_budget_is_not_mistaken_for_request_limit():
    cfg = config.validate_config(configuration())
    assert cfg["max_prefill_tokens"] == 69632
    assert cfg["context_length"] > 131072 + 1024
    assert cfg["base_url"] == "http://192.0.2.10:8810"
    assert cfg["dist_init_addr"] == "192.0.2.10:50000"
    assert cfg["profile_status"] == "CANDIDATE_NOT_NPU_VALIDATED"
    assert cfg["requested_max_running_requests_per_dp"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "127.0.0.1"),
        ("host", "0.0.0.0"),
        ("hccl_socket_ifname", "lo"),
        ("gloo_socket_ifname", ""),
    ],
)
def test_remote_network_parameters_are_not_guessed(field, value):
    raw = configuration()
    raw["nodes"][1][field] = value
    with pytest.raises(ValueError):
        config.validate_config(raw)


@pytest.mark.parametrize("mode", config.MODES)
@pytest.mark.parametrize("rank", (0, 1))
def test_five_modes_rank_parameters_and_isolation(mode, rank):
    cfg = config.validate_config(configuration())
    args = launch.server_arguments(cfg, rank, mode)
    env = launch.runtime_environment(cfg, rank, mode)
    assert args[args.index("--node-rank") + 1] == str(rank)
    assert args[args.index("--host") + 1] == cfg["nodes"][rank]["host"]
    assert args[args.index("--dist-init-addr") + 1] == "192.0.2.10:50000"
    assert "--enable-dp-lm-head" in args  # DSpark DP>1 rejects a missing flag.
    assert ("--disable-cuda-graph" in args) == mode.endswith("eager")
    assert env["HCCL_SOCKET_IFNAME"] == cfg["nodes"][rank]["hccl_socket_ifname"]
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert (env.get("SGLANG_NPU_GLM_DSPARK_QUAROT") == "original") == mode.startswith(
        "dspark-"
    )
    if mode.startswith("dspark-"):
        assert args[args.index("--speculative-algorithm") + 1] == "DSPARK"
        assert args[args.index("--speculative-num-draft-tokens") + 1] == "9"
    elif mode.startswith("target-"):
        assert not any(arg.startswith("--speculative") for arg in args)
    else:
        assert args[args.index("--speculative-algorithm") + 1] == "NEXTN"
        assert args[args.index("--speculative-num-steps") + 1] == "4"
        assert args[args.index("--speculative-num-draft-tokens") + 1] == "5"
        assert "--speculative-draft-model-path" not in args


def test_configuration_is_passed_as_argv_not_embedded_in_shell():
    raw = configuration()
    raw["target_model"] = "/srv/test/models/name;$(touch /tmp/should-not-exist)"
    cfg = config.validate_config(raw)
    argv = launch.build_command(cfg, 0, "dspark-eager")
    assert cfg["target_model"] in argv
    assert cfg["target_model"] not in argv[2]
    assert "SGLANG_SIMULATE_ACC_LEN" in argv
    assert "GLM52_CONTEXT_SNAPSHOT_CONFIG" in argv
    assert "ssh" not in argv


def test_preview_does_not_check_hardware_source_vendor_or_start_process(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration()))
    with (
        mock.patch.object(
            launch.subprocess, "Popen", side_effect=AssertionError("started")
        ),
        mock.patch.object(
            launch, "run_foreground", side_effect=AssertionError("started")
        ),
        mock.patch.dict(
            sys.modules,
            {
                "check_node": mock.Mock(
                    collect_local=mock.Mock(side_effect=AssertionError("hardware"))
                )
            },
        ),
    ):
        assert launch.main(["dspark-eager", "--rank", "0", "--config", str(path)]) == 0


def test_foreground_captures_output_and_exact_exit_status(tmp_path):
    process = mock.Mock(stdout=io.StringIO("first\nsecond\n"))
    process.wait.return_value = 7
    with mock.patch.object(launch.subprocess, "Popen", return_value=process) as popen:
        assert (
            launch.run_foreground(
                ["fake-server"], str(tmp_path), tmp_path / "server.log"
            )
            == 7
        )
    assert (tmp_path / "server.log").read_text() == "first\nsecond\n"
    assert popen.call_args.kwargs["start_new_session"] is True


def test_ctrl_c_signals_only_local_group_and_preserves_shutdown_log(tmp_path):
    class InterruptOnce:
        first = True

        def __iter__(self):
            if self.first:
                self.first = False
                raise KeyboardInterrupt
            return iter(["shutdown detail\n"])

        def close(self):
            pass

    process = mock.Mock(stdout=InterruptOnce(), pid=12345)
    process.wait.return_value = 130
    with (
        mock.patch.object(launch.subprocess, "Popen", return_value=process),
        mock.patch.object(launch.os, "killpg") as signal_group,
    ):
        assert (
            launch.run_foreground(["fake"], str(tmp_path), tmp_path / "stop.log") == 130
        )
    signal_group.assert_called_once_with(12345, launch.signal.SIGINT)
    assert (tmp_path / "stop.log").read_text() == "shutdown detail\n"


def report(rank):
    return {
        "rank": rank,
        "ready": True,
        "config": config.validate_config(configuration()),
        "versions": {"torch-npu": "2.10.0.post4"},
        "python": {"version": "same"},
        "repositories": {
            name: {"head": "same", "tracked_status": ""}
            for name in ("sglang", "kernel")
        },
        "sources": {"modelslim": {"sha256": "same"}},
        "tokenizer_files": {"tokenizer.json": {"sha256": "same"}},
        "artifacts": {
            name: {
                "files": {"config.json": {"sha256": "same"}},
                "shards": {"model.safetensors": {"bytes": 9}},
            }
            for name in ("target", "draft")
        },
    }


def test_offline_pair_comparison_detects_source_version_and_weight_metadata_drift():
    a, b = report(0), report(1)
    assert check_node.compare_reports(a, b)["ready"]
    for mutate in (
        lambda r: r["versions"].update({"torch-npu": "other"}),
        lambda r: r["repositories"]["sglang"].update(
            {"tracked_status": " M python/file.py"}
        ),
        lambda r: r["sources"]["modelslim"].update({"sha256": "other"}),
        lambda r: r["artifacts"]["draft"]["shards"]["model.safetensors"].update(
            {"bytes": 10}
        ),
        lambda r: r.update({"rank": 0}),
        lambda r: r.update({"ready": False}),
    ):
        changed = copy.deepcopy(b)
        mutate(changed)
        assert check_node.compare_reports(a, changed)["ready"] is False


def test_artifact_metadata_does_not_hash_weight_payload(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"test-weight")
    read_bytes = Path.read_bytes

    def guarded(path):
        if path == weight:
            raise AssertionError("Full checkpoint bytes must not be read")
        return read_bytes(path)

    with mock.patch.object(Path, "read_bytes", guarded):
        metadata = check_node.artifact_record(tmp_path)
    assert metadata["shards"]["model.safetensors"]["bytes"] == 11
    assert metadata["full_checkpoint_hash"] == "NOT_COMPUTED"


def test_index_cannot_read_outside_model_directory(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside"}})
    )
    with pytest.raises(ValueError, match="outside"):
        check_node.artifact_record(tmp_path)
