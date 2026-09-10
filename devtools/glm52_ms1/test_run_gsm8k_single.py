# SPDX-License-Identifier: Apache-2.0
"""Check the one-command GSM8K wrapper without contacting a server."""

from pathlib import Path
import shlex
import subprocess


SCRIPT = Path(__file__).with_name("run_gsm8k_single.sh")


def test_wrapper_contains_the_current_single_node_request():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--print-command"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:4] == [
        "python3",
        "devtools/glm52_ms1/bench_gsm8k_modes.py",
        "run",
        "dspark-eager",
    ]
    expected = {
        "--host": "61.47.19.69",
        "--port": "8810",
        "--target": "/home/weights/GLM-5.2-w8a8",
        "--draft": "/home/weights/GLM-5.2-DSpark-NPU-0805",
        "--served-model-name": "GLM-5.2-w8a8",
        "--state": "/home/tyj/glm52-ms1",
        "--max-tokens": "1024",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value


def test_wrapper_rejects_unknown_arguments():
    result = subprocess.run(
        ["bash", str(SCRIPT), "unexpected"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
