"""Inspect Docker arguments with a fake executable; no container is created."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("start_container.sh")


def invoke(tmp_path, missing=None):
    output = tmp_path / "docker-args.json"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['TEST_DOCKER_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        TEST_DOCKER_ARGS=str(output),
        IMAGE="registry.example/test/image:tag",
        CONTAINER_NAME="test-container",
        MS1_STATE="/home/test/run state",
    )
    if missing:
        env.pop(missing, None)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )
    return result, json.loads(output.read_text()) if output.exists() else None


@pytest.mark.parametrize("missing", ["IMAGE", "CONTAINER_NAME", "MS1_STATE"])
def test_required_values_fail_before_docker(tmp_path, missing):
    result, argv = invoke(tmp_path, missing)
    assert result.returncode != 0 and missing in result.stderr
    assert argv is None


def test_required_values_preserve_core_mounts_and_individual_devices(tmp_path):
    result, argv = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert argv[-1] == "registry.example/test/image:tag"
    for i in range(16):
        assert f"--device=/dev/davinci{i}" in argv
    for mount in (
        "/usr/local/sbin:/usr/local/sbin:ro",
        "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
        "/etc/ascend_install.info:/etc/ascend_install.info:ro",
        "/var/queue_schedule:/var/queue_schedule",
        "/home:/home",
    ):
        assert mount in argv
    assert "HF_HOME=/home/test/run state/cache/huggingface" in argv
    assert "PIP_CACHE_DIR=/home/test/run state/cache/pip" in argv
