#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the reviewed GLM-5.2 head-dim-192 Python kernel into this container."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
from pathlib import Path


KERNEL_FILE = Path(
    "python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kernel_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def installed_package() -> Path:
    spec = importlib.util.find_spec("sgl_kernel_npu")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if len(locations) != 1:
        raise RuntimeError(
            "Cannot locate one installed sgl_kernel_npu package; check the container image"
        )
    return Path(locations[0]).resolve()


def register(kernel_repo: Path, package: Path | None = None) -> tuple[Path, bool]:
    source = (kernel_repo / KERNEL_FILE).resolve()
    if not source.is_file():
        raise RuntimeError(f"Kernel source does not exist: {source}")

    content = source.read_text(encoding="utf-8")
    required = ("if head_dim == 192:", "KV_BLOCK_SIZE = head_dim")
    if not all(marker in content for marker in required):
        raise RuntimeError(
            "Kernel checkout does not contain the reviewed native head_dim=192 path"
        )

    package = package.resolve() if package else installed_package()
    target = package / "norm" / "split_qkv_rmsnorm_rope.py"
    if not target.is_file():
        raise RuntimeError(f"Installed kernel module does not exist: {target}")

    source_hash = sha256(source)
    changed = sha256(target) != source_hash
    if changed:
        shutil.copyfile(source, target)
        pycache = target.parent / "__pycache__"
        if pycache.is_dir():
            for stale in pycache.glob("split_qkv_rmsnorm_rope.*.pyc"):
                stale.unlink()
    if sha256(target) != source_hash:
        raise RuntimeError(f"Kernel registration verification failed: {target}")
    return target, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register the reviewed GLM-5.2 head_dim=192 Python kernel"
    )
    parser.add_argument("--kernel-repo", type=Path, required=True)
    args = parser.parse_args()

    source = (args.kernel_repo / KERNEL_FILE).resolve()
    target, changed = register(args.kernel_repo)
    print(f"GLM-5.2 kernel source: {source}")
    print(f"GLM-5.2 kernel commit: {kernel_head(args.kernel_repo)}")
    print(f"GLM-5.2 kernel sha256: {sha256(source)}")
    print(f"GLM-5.2 kernel target: {target}")
    print("GLM-5.2 kernel registration: " + ("UPDATED" if changed else "ALREADY_CURRENT"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
