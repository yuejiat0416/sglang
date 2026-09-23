# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the formal single-node deployment script."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("single_dspark_official.sh")


def preview(tmp_path: Path, *, mode: str = "dspark", graph: int = 0) -> list[str]:
    text = SCRIPT.read_text()
    text = text.replace("MODE='dspark'", f"MODE='{mode}'", 1)
    text = text.replace("\nGRAPH=0 ", f"\nGRAPH={graph} ", 1)
    script = tmp_path / f"single-{mode}-{graph}.sh"
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


@pytest.mark.parametrize("mode", ("dspark", "target-only", "nextn"))
@pytest.mark.parametrize("graph", (0, 1))
def test_single_modes_and_graph(tmp_path: Path, mode: str, graph: int):
    argv = preview(tmp_path, mode=mode, graph=graph)
    assert argv[:3] == ["python3", "-m", "sglang.launch_server"]
    assert option(argv, "--host") == "61.47.19.68"
    assert option(argv, "--tp-size") == "16"
    assert option(argv, "--node-rank") == "0"
    assert option(argv, "--context-length") == "69632"
    assert option(argv, "--max-total-tokens") == "69632"
    assert option(argv, "--chunked-prefill-size") == "16384"
    assert option(argv, "--max-prefill-tokens") == "280000"
    assert option(argv, "--mem-fraction-static") == "0.825"
    assert option(argv, "--max-running-requests") == "4"
    assert "--enable-metrics" in argv
    assert "--enable-cache-report" in argv
    assert "--cuda-graph-bs" not in argv
    if graph:
        assert option(argv, "--cuda-graph-bs-decode") == "16"
        assert "--disable-cuda-graph" not in argv
    else:
        assert "--disable-cuda-graph" in argv
        assert "--cuda-graph-bs-decode" not in argv

    if mode == "dspark":
        assert option(argv, "--speculative-algorithm") == "DSPARK"
        assert option(argv, "--speculative-dspark-block-size") == "8"
        assert option(argv, "--speculative-num-draft-tokens") == "9"
        assert option(argv, "--speculative-draft-attention-backend") == "ascend"
    elif mode == "nextn":
        assert option(argv, "--speculative-algorithm") == "NEXTN"
        assert option(argv, "--speculative-num-steps") == "3"
        assert option(argv, "--speculative-eagle-topk") == "1"
        assert option(argv, "--speculative-num-draft-tokens") == "4"
        assert "--speculative-draft-model-path" not in argv
    else:
        assert not any(arg.startswith("--speculative-") for arg in argv)


def test_single_script_is_direct_and_syntax_valid():
    text = SCRIPT.read_text()
    assert "register_glm52_dspark_kernel.py" in text
    assert "with_kernel_checkout.py" not in text
    assert "export SGLANG_ENABLE_SPEC_V2" not in text
    assert "unset SGLANG_ENABLE_SPEC_V2" in text
    assert "GLM52_LEGACY_LAUNCH" not in text
    assert "ASCEND_LAUNCH_BLOCKING PYTHONPATH" in text
    assert 'export PYTHONPATH="$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert text.index("ASCEND_LAUNCH_BLOCKING PYTHONPATH") < text.index(
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    ) < text.index('export PYTHONPATH="$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"')
    assert "unset SGLANG_SIMULATE_UNIFORM_EXPERTS" in text
    assert "export SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT=true" in text
    assert (
        "unset SGLANG_RAGGED_VERIFY_MODE "
        "SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT"
    ) in text
    assert "SGLANG_NPU_GLM_DSPARK_QUAROT" not in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True
    )
    assert help_result.returncode == 0
    assert "Edit MODE and GRAPH" in help_result.stdout
