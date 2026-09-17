# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the formal two-node deployment script."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("two_node_dspark_official.sh")


def preview(
    tmp_path: Path,
    *,
    rank: int = 0,
    mode: str = "dspark",
    graph: int = 0,
) -> list[str]:
    text = SCRIPT.read_text()
    text = text.replace("MODE='dspark'", f"MODE='{mode}'", 1)
    text = text.replace("\nGRAPH=0 ", f"\nGRAPH={graph} ", 1)
    text = text.replace("\nNODE_RANK=0 ", f"\nNODE_RANK={rank} ", 1)
    script = tmp_path / f"dual-{rank}-{mode}-{graph}.sh"
    script.write_text(text)
    result = subprocess.run(
        ["bash", str(script), "--print-command"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)


def option(argv: list[str], key: str) -> str:
    assert argv.count(key) == 1
    return argv[argv.index(key) + 1]


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("mode", ("dspark", "target-only", "nextn"))
@pytest.mark.parametrize("graph", (0, 1))
def test_two_node_modes_graph_and_topology(
    tmp_path: Path, rank: int, mode: str, graph: int
):
    argv = preview(tmp_path, rank=rank, mode=mode, graph=graph)
    assert argv[:2] == [
        "HCCL_SOCKET_IFNAME=enp196s0f0",
        "GLOO_SOCKET_IFNAME=enp196s0f0",
    ]
    argv = argv[2:]
    assert argv[:3] == ["python3", "-m", "sglang.launch_server"]
    assert option(argv, "--host") == (
        "61.47.19.68" if rank == 0 else "61.47.19.70"
    )
    assert option(argv, "--dist-init-addr") == "61.47.19.68:50000"
    assert option(argv, "--node-rank") == str(rank)
    assert option(argv, "--tp-size") == "32"
    assert option(argv, "--dp-size") == "4"
    assert option(argv, "--context-length") == "133120"
    assert option(argv, "--max-total-tokens") == "133120"
    assert option(argv, "--chunked-prefill-size") == "16384"
    assert option(argv, "--max-prefill-tokens") == "131072"
    assert option(argv, "--mem-fraction-static") == "0.73"
    assert option(argv, "--max-running-requests") == "4"
    assert "--enable-dp-attention" in argv
    assert "--enable-dp-lm-head" in argv
    assert "--disable-radix-cache" not in argv
    if graph:
        assert option(argv, "--cuda-graph-bs-decode") == "8"
        assert "--disable-cuda-graph" not in argv
    else:
        assert "--disable-cuda-graph" in argv

    if mode == "dspark":
        assert option(argv, "--speculative-algorithm") == "DSPARK"
        assert option(argv, "--speculative-dspark-block-size") == "8"
        assert option(argv, "--speculative-num-draft-tokens") == "9"
    elif mode == "nextn":
        assert option(argv, "--speculative-algorithm") == "NEXTN"
        assert option(argv, "--speculative-num-steps") == "3"
        assert option(argv, "--speculative-num-draft-tokens") == "4"
    else:
        assert not any(arg.startswith("--speculative-") for arg in argv)


def test_two_node_script_is_direct_and_syntax_valid():
    text = SCRIPT.read_text()
    assert "register_glm52_dspark_kernel.py" in text
    assert "with_kernel_checkout.py" not in text
    assert "ip -o route get" not in text
    assert "export SGLANG_ENABLE_SPEC_V2" not in text
    assert "unset SGLANG_ENABLE_SPEC_V2" in text
    assert "GLM52_LEGACY_LAUNCH" not in text
    assert 'export PYTHONPATH="$SGLANG_REPO/python"' in text
    assert "unset SGLANG_SIMULATE_UNIFORM_EXPERTS" in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True
    )
    assert help_result.returncode == 0
    assert "Set NODE_RANK=0" in help_result.stdout
