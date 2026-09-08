#!/usr/bin/env python3
"""Run a command with one frozen Python kernel and the installed binary package.

This is a temporary integration tool, not an installer or a service launcher.
The command inherits a private package overlay through PYTHONPATH, including
normal multiprocessing spawn children. The installed package is not modified.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

PACKAGE = "sgl_kernel_npu"
MODULE = f"{PACKAGE}.norm.split_qkv_rmsnorm_rope"
SOURCE = Path("python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py")
LIBRARY = Path("lib/libsgl_kernel_npu.so")
HELPER = Path("utils/triton_utils.py")

# Importing the package loads its existing library. Do not select a device,
# allocate a tensor, call get_device_properties, or execute a kernel here.
IMPORT_PROBE = r"""
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

overlay, installed, expected_hash, output = map(str, sys.argv[1:])
overlay = Path(overlay)
installed = Path(installed)
package = importlib.import_module('sgl_kernel_npu')
module = importlib.import_module('sgl_kernel_npu.norm.split_qkv_rmsnorm_rope')
spec = importlib.util.find_spec(module.__name__)
candidate = overlay / 'sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py'
library = Path(package.__file__).parent / 'lib/libsgl_kernel_npu.so'
helper = Path(inspect.getsourcefile(inspect.unwrap(module.get_device_properties)))
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(Path(package.__file__).parent == overlay / 'sgl_kernel_npu', package.__file__)
require(Path(module.__file__) == candidate, 'Candidate import selected: ' + module.__file__)
require(Path(spec.origin) == candidate, 'Candidate spec selected: ' + spec.origin)
require(hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_hash, 'Candidate hash changed')
require(library.resolve() == (installed / 'lib/libsgl_kernel_npu.so').resolve(), str(library))
require(helper.resolve() == (installed / 'utils/triton_utils.py').resolve(), str(helper))
Path(output).write_text(json.dumps({
    'python': sys.executable,
    'package_file': package.__file__,
    'candidate_module': module.__file__,
    'candidate_spec_origin': spec.origin,
    'library': str(library),
    'library_resolved': str(library.resolve()),
    'helper': str(helper),
    'helper_resolved': str(helper.resolve()),
}, indent=2) + '\n')
"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def installed_package_path():
    # Finding the top-level package does not execute __init__ or import torch.
    spec = importlib.util.find_spec(PACKAGE)
    if spec is None or not spec.origin or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed sgl_kernel_npu package")
    root = Path(spec.origin).resolve().parent
    if not (root / LIBRARY).is_file():
        raise RuntimeError(f"Installed binary library is missing: {root / LIBRARY}")
    if not (root / HELPER).is_file():
        raise RuntimeError(f"Installed kernel helper is missing: {root / HELPER}")
    norm_init = root / "norm/__init__.py"
    if not norm_init.is_file():
        raise RuntimeError(f"Installed norm package is missing: {norm_init}")
    return root


def git_identity(repo):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.rstrip("\n")

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--kernel-repo must identify the kernel repository root")
    return {
        "head": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain=v1", "--untracked-files=normal"),
    }


def build_overlay(installed, overlay, candidate_bytes):
    package = overlay / PACKAGE
    package.mkdir(parents=True)
    for entry in installed.iterdir():
        if entry.name not in {"norm", "__pycache__"}:
            (package / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    norm = package / "norm"
    norm.mkdir()
    for entry in (installed / "norm").iterdir():
        if entry.name not in {SOURCE.name, "__pycache__"}:
            (norm / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    frozen = norm / SOURCE.name
    frozen.write_bytes(candidate_bytes)
    return frozen


def child_environment(overlay):
    env = os.environ.copy()
    original = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(overlay) + (os.pathsep + original if original else "")
    # Symlinked helper directories belong to the installed package. Do not
    # write __pycache__ through those links, including during the import probe.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def verify_import(overlay, installed, source_hash, run_dir, env):
    output = run_dir / "import.json"
    with (
        (run_dir / "import.stdout.txt").open("w") as stdout,
        (run_dir / "import.stderr.txt").open("w") as stderr,
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                IMPORT_PROBE,
                str(overlay),
                str(installed),
                source_hash,
                str(output),
            ],
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=120,
        )
    if result.returncode:
        raise RuntimeError(
            f"Fresh Python import failed ({result.returncode}); see {run_dir / 'import.stderr.txt'}"
        )
    return json.loads(output.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-repo", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Pass a command after --")
    state = args.state_dir.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="kernel-overlay-", dir=state))
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "PREPARING",
        "run_dir": str(run_dir),
        "command": command,
        "command_cwd": os.getcwd(),
        "python": sys.executable,
        "command_result": "NOT_OBSERVED_AFTER_EXEC",
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    print(f"Kernel overlay evidence: {run_dir}", flush=True)
    try:
        repo = args.kernel_repo.expanduser().resolve()
        identity = git_identity(repo)
        source = repo / SOURCE
        candidate_bytes = source.read_bytes()
        installed = installed_package_path()
        installed_source = installed / "norm" / SOURCE.name
        overlay = run_dir / "python"
        frozen = build_overlay(installed, overlay, candidate_bytes)
        try:
            version = importlib.metadata.version("sgl-kernel-npu")
        except importlib.metadata.PackageNotFoundError:
            version = None
        env = child_environment(overlay)
        manifest.update(
            {
                "kernel_repo": str(repo),
                "kernel_git": identity,
                "candidate_source": str(source),
                "candidate_frozen": str(frozen),
                "candidate_sha256": sha256(candidate_bytes),
                "installed_package": str(installed),
                "installed_distribution_version": version,
                "installed_source": str(installed_source),
                "installed_source_sha256": sha256(installed_source.read_bytes()),
                "installed_library": str((installed / LIBRARY).resolve()),
                "overlay": str(overlay),
                "child_pythonpath": env["PYTHONPATH"],
                "child_pythondontwritebytecode": env["PYTHONDONTWRITEBYTECODE"],
                "binary_and_helpers": "SYMLINKED_INSTALLED_FILES_NOT_FROZEN",
            }
        )
        save()
        print(
            f"Installed binary package: {installed} (version {version or 'unavailable'})",
            flush=True,
        )
        print(f"Frozen candidate: {frozen} (git {identity['head']})", flush=True)
        manifest["import_check"] = verify_import(
            overlay, installed, sha256(candidate_bytes), run_dir, env
        )
        manifest["status"] = "READY_TO_EXEC_COMMAND"
        save()
        print(
            "Kernel import verified; starting the command. Service success is not evaluated by this helper.",
            flush=True,
        )
        os.execvpe(command[0], command, env)
    except Exception as exc:
        manifest["status"] = "SETUP_OR_EXEC_FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "traceback.txt").write_text(traceback.format_exc())
        save()
        print(f"Kernel overlay failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
