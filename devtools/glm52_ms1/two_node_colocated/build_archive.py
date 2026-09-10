#!/usr/bin/env python3
"""Bundle this temporary suite and its explicit helper dependencies, not evidence."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SUITE_FILES = (
    "README.md",
    "config.example.json",
    "container.sh",
    "config.py",
    "launch.py",
    "check_node.py",
    "client_common.py",
    "offline_dataset.py",
    "bench_accuracy.py",
    "bench_prefix.py",
    "run_suite.py",
    "report.py",
    "build_archive.py",
    "test_clients.py",
    "test_deployment.py",
    "test_offline_dataset.py",
    "test_prefix.py",
    "test_archive.py",
)
HELPERS = (
    "README.md",
    "GSM8K_MODES.md",
    "GSP_PREFIX.md",
    "single_dspark_static.sh",
    "start_container.sh",
    "ms1_target_only.py",
    "target-only.example.json",
    "bench_gsm8k_modes.py",
    "gsm8k_mode_stats.py",
    "gsm8k10.json",
    "gsm8k-LICENSE.txt",
    "bench_gsp_prefix.py",
    "gsp_prefix_stats.py",
    "with_kernel_checkout.py",
    "test_bench_gsm8k_modes.py",
    "test_bench_gsp_prefix.py",
    "test_gsm8k_mode_stats.py",
    "test_gsp_prefix_stats.py",
    "test_single_dspark_static.py",
    "test_start_container.py",
    "test_ms1_target_only.py",
    "test_with_kernel_checkout.py",
)


def selected_files():
    # No globs: local Python/config copies must not silently enter a release.
    files = [HERE / name for name in SUITE_FILES]
    files += [HERE.parent / name for name in HELPERS]
    if any(path.is_symlink() for path in files):
        raise ValueError("Archive sources must be regular files, not symbolic links")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Missing archive dependencies: {missing}")
    return sorted(set(files))


def build(output):
    files = selected_files()
    manifest = {
        "purpose": "Temporary single/two-node colocated self-test tools; extract at a matching SGLang checkout root",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "files": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "excludes": [
            "local config",
            "results",
            "logs",
            "model weights",
            "GPQA data",
            "framework source",
            "kernel source",
        ],
        "requires": "Matching SGLang checkout and kernel checkout; no package install or framework replacement is performed",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    # Exclusive creation protects any previously delivered archive.
    with (
        output.open("xb") as stream,
        gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive,
    ):
        for path in files:
            data = path.read_bytes()
            # Build headers explicitly: do not copy local uid/gid/user/group,
            # timestamp, PAX metadata or absolute paths from the filesystem.
            header = tarfile.TarInfo(str(path.relative_to(REPO)))
            header.size = len(data)
            header.mode = 0o644
            archive.addfile(header, io.BytesIO(data))
        header = tarfile.TarInfo("colocated-tools-manifest.json")
        header.size = len(raw)
        archive.addfile(header, io.BytesIO(raw))
    return {
        "archive": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
