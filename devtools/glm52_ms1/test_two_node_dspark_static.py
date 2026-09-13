"""CPU tests of the direct two-node launcher; all route/vendor/device work is fake."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("two_node_dspark_static.sh")


def run_script(*args, overrides=None, script=SCRIPT, legacy=True):
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME")}
    env.update(overrides or {})
    if legacy:
        env["GLM52_LEGACY_LAUNCH"] = "1"
    return subprocess.run(
        ["bash", str(script), *args], env=env, capture_output=True, text=True
    )


def preview(**overrides):
    env = dict(HCCL_SOCKET_IFNAME="eth-test0", GLOO_SOCKET_IFNAME="eth-test1")
    env.update(overrides)
    result = run_script("--print-command", overrides=env)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    return argv, {x.split("=", 1)[0]: x.split("=", 1)[1] for x in argv if "=" in x}


def option(argv, key):
    assert argv.count(key) == 1
    return argv[argv.index(key) + 1]


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("mode", ("dspark", "target-only", "nextn"))
@pytest.mark.parametrize("graph", (0, 1))
def test_modes_ranks_graph_and_original_recipe(rank, mode, graph):
    argv, env = preview(NODE_RANK=str(rank), MODE=mode, GRAPH=str(graph))
    assert option(argv, "--host") == ("61.47.19.71" if rank == 0 else "61.47.19.70")
    assert option(argv, "--dist-init-addr") == "61.47.19.71:50000"
    assert option(argv, "--node-rank") == str(rank)
    assert option(argv, "--nnodes") == "2"
    assert option(argv, "--tp-size") == "32"
    assert option(argv, "--dp-size") == "8"
    assert option(argv, "--context-length") == "133120"
    assert option(argv, "--max-running-requests") == "8"
    assert "--enable-dp-attention" in argv and "--enable-dp-lm-head" in argv
    assert "--enable-metrics" in argv and "--enable-cache-report" in argv
    assert env["HCCL_SOCKET_IFNAME"] == "eth-test0"
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test1"
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert ("--disable-cuda-graph" in argv) == (graph == 0)
    if graph:
        assert option(argv, "--cuda-graph-bs") == "16"
    if mode == "dspark":
        assert option(argv, "--speculative-algorithm") == "DSPARK"
        assert option(argv, "--speculative-dspark-block-size") == "5"
        assert option(argv, "--speculative-num-draft-tokens") == "6"
        assert env["SGLANG_NPU_GLM_DSPARK_QUAROT"] == "original"
    else:
        assert "SGLANG_NPU_GLM_DSPARK_QUAROT" not in env
        assert "--speculative-draft-model-path" not in argv
        if mode == "nextn":
            assert option(argv, "--speculative-algorithm") == "NEXTN"
            assert option(argv, "--speculative-num-steps") == "4"
            assert option(argv, "--speculative-eagle-topk") == "1"
            assert option(argv, "--speculative-num-draft-tokens") == "5"
        else:
            assert not any(x.startswith("--speculative-") for x in argv)


def fake_ip(tmp_path, route="peer via gateway dev eth-route src local", status=0):
    executable = tmp_path / "ip"
    executable.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(route)}\nexit {status}\n"
    )
    executable.chmod(0o755)
    return f"{tmp_path}:{os.environ['PATH']}"


def test_empty_nics_use_route_and_preserve_explicit_other_nic(tmp_path):
    result = run_script(
        "--print-command",
        overrides={"PATH": fake_ip(tmp_path), "GLOO_SOCKET_IFNAME": "eth-explicit"},
    )
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert "HCCL_SOCKET_IFNAME=eth-route" in argv
    assert "GLOO_SOCKET_IFNAME=eth-explicit" in argv
    assert "Kernel" not in result.stderr


@pytest.mark.parametrize(
    "route,status", (("", 1), ("peer dev lo", 0), ("unreachable peer", 0))
)
def test_missing_or_loopback_route_stops_before_launch(tmp_path, route, status):
    result = run_script(
        "--print-command", overrides={"PATH": fake_ip(tmp_path, route, status)}
    )
    assert result.returncode == 2
    assert not result.stdout
    assert "NIC" in result.stderr or "route" in result.stderr


@pytest.mark.parametrize(
    "overrides",
    (
        {"NODE_RANK": "2"},
        {"GRAPH": "2"},
        {"MODE": "wrong"},
        {"HCCL_SOCKET_IFNAME": "lo"},
        {"GLOO_SOCKET_IFNAME": "eth name"},
        {"NODE0_HOST": "same", "NODE1_HOST": "same"},
    ),
)
def test_invalid_settings_fail_before_vendor_or_server(overrides):
    env = dict(HCCL_SOCKET_IFNAME="eth-test0", GLOO_SOCKET_IFNAME="eth-test1")
    env.update(overrides)
    result = run_script("--print-command", overrides=env)
    assert result.returncode == 2
    assert not result.stdout


def test_literal_switches_can_be_edited_without_mode_environment(tmp_path):
    source = SCRIPT.read_text().replace("MODE='dspark'", "MODE='nextn'")
    source = source.replace("GRAPH=0 ", "GRAPH=1 ").replace(
        "NODE_RANK=0 ", "NODE_RANK=1 "
    )
    script = tmp_path / "launch.sh"
    script.write_text(source)
    result = run_script(
        "--print-command",
        script=script,
        overrides={"PATH": fake_ip(tmp_path)},
        legacy=False,
    )
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert option(argv, "--node-rank") == "1"
    assert option(argv, "--speculative-algorithm") == "NEXTN"
    assert option(argv, "--cuda-graph-bs") == "16"


@pytest.mark.parametrize("ip_status,nic_present", ((0, True), (127, True), (127, False)))
def test_explicit_nic_checks_sysfs_before_existing_overlay(
    tmp_path, ip_status, nic_present
):
    # Substitute only the host filesystem root; never access real NICs/CANN/NPUs.
    net = tmp_path / "sys/class/net"
    if nic_present:
        (net / "eth-explicit").mkdir(parents=True)
    script = tmp_path / "launch.sh"
    script.write_text(SCRIPT.read_text().replace("/sys/class/net", str(net)))
    hooks = tmp_path / "hooks.sh"
    hooks.write_text('source() { false; export PYTHONPATH="vendor"; }\n')
    python = tmp_path / "python3"
    log = tmp_path / "args.json"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['LAUNCH_TEST_LOG'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    result = run_script(
        script=script,
        overrides={
            "PATH": fake_ip(tmp_path, status=ip_status),
            "HCCL_SOCKET_IFNAME": "eth-explicit",
            "GLOO_SOCKET_IFNAME": "eth-explicit",
            "BASH_ENV": str(hooks),
            "LAUNCH_TEST_LOG": str(log),
            "SGLANG_REPO": "/test/sglang",
            "KERNEL_REPO": "/test/kernel",
            "MS1_STATE": str(tmp_path / "state"),
        }
    )
    if not nic_present:
        assert result.returncode == 2
        assert "Configured NIC eth-explicit does not exist" in result.stderr
        assert not log.exists()
        return
    assert result.returncode == 0, result.stderr
    argv = json.loads(log.read_text())
    assert argv[0] == "/test/sglang/devtools/glm52_ms1/with_kernel_checkout.py"
    assert option(argv, "--kernel-repo") == "/test/kernel"
    assert not (tmp_path / "state").exists()  # no launcher-generated metadata/config
    assert "check_node.py" not in result.stdout


def test_syntax_and_help():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    result = run_script("--help")
    assert result.returncode == 0
    assert "NODE_RANK=1" in result.stdout


def test_direct_launch_ignores_stale_switches_and_clears_debug(tmp_path):
    result = run_script(
        "--print-command",
        overrides={
            "PATH": fake_ip(tmp_path),
            "NODE_RANK": "1",
            "MODE": "nextn",
            "GRAPH": "1",
            "SGLANG_SIMULATE_ACC_LEN": "8",
            "SGLANG_DSPARK_DEBUG_DUMP": "core",
        },
        legacy=False,
    )
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert option(argv, "--node-rank") == "0"
    assert option(argv, "--speculative-algorithm") == "DSPARK"
    assert "--disable-cuda-graph" in argv
    assert "SGLANG_SIMULATE_ACC_LEN" in argv and "SGLANG_DSPARK_DEBUG_DUMP" in argv
    assert "SGLANG_SIMULATE_ACC_LEN=8" not in argv


def test_active_snapshot_is_rejected_before_network_lookup():
    result = run_script(
        "--print-command",
        overrides={"GLM52_CONTEXT_SNAPSHOT_CONFIG": "/test/config.json"},
        legacy=False,
    )
    assert result.returncode == 2
    assert "snapshot" in result.stderr
    assert not result.stdout
