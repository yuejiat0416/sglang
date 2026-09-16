# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the explicit GLM-5.2 kernel registration step."""

from pathlib import Path

import pytest

from register_glm52_dspark_kernel import KERNEL_FILE, register, sha256


VALID_SOURCE = """\
def split_qkv_rmsnorm_rope(head_dim):
    if head_dim == 192:
        KV_BLOCK_SIZE = head_dim
    return KV_BLOCK_SIZE
"""


def layout(tmp_path: Path, source: str = VALID_SOURCE):
    repo = tmp_path / "kernel"
    source_file = repo / KERNEL_FILE
    source_file.parent.mkdir(parents=True)
    source_file.write_text(source)
    package = tmp_path / "site-packages" / "sgl_kernel_npu"
    target = package / "norm" / "split_qkv_rmsnorm_rope.py"
    target.parent.mkdir(parents=True)
    target.write_text("old installed module\n")
    return repo, package, source_file, target


def test_registration_copies_once_and_then_is_idempotent(tmp_path: Path):
    repo, package, source, target = layout(tmp_path)
    actual, changed = register(repo, package)
    assert actual == target
    assert changed
    assert target.read_bytes() == source.read_bytes()
    assert sha256(target) == sha256(source)

    _, changed = register(repo, package)
    assert not changed


def test_registration_rejects_checkout_without_head192(tmp_path: Path):
    repo, package, _, target = layout(tmp_path, "KV_BLOCK_SIZE = 256\n")
    with pytest.raises(RuntimeError, match="head_dim=192"):
        register(repo, package)
    assert target.read_text() == "old installed module\n"


def test_registration_requires_existing_installed_module(tmp_path: Path):
    repo, package, _, _ = layout(tmp_path)
    (package / "norm" / "split_qkv_rmsnorm_rope.py").unlink()
    with pytest.raises(RuntimeError, match="Installed kernel module"):
        register(repo, package)
