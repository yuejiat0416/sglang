#!/usr/bin/env python3
"""Read local container metadata, or compare two saved node reports offline.

Reports preserve real local paths, addresses and package/model metadata for
debugging. Keep those reports with the private test evidence, outside archives.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import load_config, node_config

CORE_PATHS = (
    "/usr/local/sbin",
    "/usr/local/Ascend/driver",
    "/usr/local/Ascend/firmware",
    "/etc/ascend_install.info",
    "/var/queue_schedule",
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",
)
VERSIONS = ("torch", "torch-npu", "triton-ascend", "sgl-kernel-npu")
SOURCE_PATHS = (
    "python/sglang/srt/models/dspark.py",
    "python/sglang/srt/models/dflash.py",
    "python/sglang/srt/layers/quantization/modelslim/modelslim.py",
    "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
    "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py",
)
KERNEL_SOURCE = "python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py"


def run_readonly(argv, timeout=30):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "argv": list(argv),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "argv": list(argv),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def file_record(path, hash_contents=True):
    path = Path(path)
    result = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size)
        if hash_contents:
            result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def git_record(repo):
    head = run_readonly(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = run_readonly(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"]
    )
    return {
        "head": head["stdout"].strip() if head["returncode"] == 0 else None,
        "tracked_status": status["stdout"].strip(),
        "errors": [x["stderr"] for x in (head, status) if x["returncode"] != 0],
    }


def artifact_record(root):
    root = Path(root)
    files = {}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "quant_model_description.json",
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    ):
        if (root / name).is_file():
            files[name] = file_record(root / name)
    missing_shards = []
    shard_metadata = {}
    for name in (
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    ):
        if name in files:
            index = json.loads((root / name).read_text())
            shards = set(index.get("weight_map", {}).values())
            for shard in sorted(shards):
                path = (root / shard).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError(
                        "Checkpoint index contains a path outside the model directory"
                    )
                rec = file_record(path, hash_contents=False)
                shard_metadata[shard] = {k: v for k, v in rec.items() if k != "path"}
                if not rec["exists"]:
                    missing_shards.append(shard)
    # One-file draft checkpoints have no index; size is metadata, not proof of equal contents.
    if (root / "model.safetensors").is_file():
        shard_metadata["model.safetensors"] = {
            "exists": True,
            "bytes": (root / "model.safetensors").stat().st_size,
        }
    rotations = {
        name: file_record(root / name, hash_contents=False)
        for name in ("optional/quarot.safetensors", "rot.safetensors")
        if (root / name).is_file()
    }
    return {
        "root": str(root),
        "files": files,
        "shards": shard_metadata,
        "missing_shards": missing_shards,
        "rotation_files": rotations,
        "full_checkpoint_hash": "NOT_COMPUTED",
    }


def collect_local(cfg, rank):
    """Runs only when the owner invokes this tool inside that node's container."""
    node = node_config(cfg, rank)
    issues = []
    required = list(CORE_PATHS) + [
        f"/dev/davinci{i}" for i in range(cfg["npus_per_node"])
    ]
    required += ["/dev/davinci_manager", "/dev/hisi_hdc"]
    paths = {path: Path(path).exists() for path in required}
    for path, exists in paths.items():
        if not exists:
            issues.append(f"Required container mount/device is missing: {path}")
    sources = {name: file_record(Path(cfg["repo"]) / name) for name in SOURCE_PATHS}
    sources["kernel_candidate"] = file_record(Path(cfg["kernel_repo"]) / KERNEL_SOURCE)
    sources["kernel_overlay_helper"] = file_record(
        Path(cfg["repo"]) / "devtools/glm52_ms1/with_kernel_checkout.py"
    )
    for name, record in sources.items():
        if not record["exists"]:
            issues.append(f"Required checkout source is missing: {name}")
    versions = {}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            issues.append(f"Package metadata missing in this Python: {name}")
    configured_python = shutil.which(cfg["python"])
    if (
        configured_python is None
        or Path(configured_python).resolve() != Path(sys.executable).resolve()
    ):
        issues.append(
            "Run check_node with the same Python executable selected by config.python"
        )
    interfaces = {}
    for name in sorted({node["hccl_socket_ifname"], node["gloo_socket_ifname"]}):
        interfaces[name] = run_readonly(["ip", "-j", "address", "show", "dev", name])
        if interfaces[name]["returncode"] != 0:
            issues.append(
                f"Cannot read configured NIC {name}: {interfaces[name]['stderr'].strip()}"
            )
    # --host must exist in this container's host network; the HCCL NIC may be a separate link.
    addresses = run_readonly(["ip", "-j", "address", "show"])
    try:
        local_ips = {
            entry["local"]
            for nic in json.loads(addresses["stdout"])
            for entry in nic.get("addr_info", [])
            if entry.get("family") == "inet"
        }
    except (ValueError, TypeError, KeyError):
        local_ips = set()
    if node["host"] not in local_ips:
        issues.append(
            "Configured host IP is not present in this container's network namespace"
        )
    npu_smi = run_readonly(["npu-smi", "info"])
    if npu_smi["returncode"] != 0:
        issues.append(
            "npu-smi info failed; verify the container's NPU mounts/driver before startup"
        )
    repos = {
        name: git_record(cfg[key])
        for name, key in (("sglang", "repo"), ("kernel", "kernel_repo"))
    }
    for name, record in repos.items():
        if record["head"] is None or record["errors"]:
            issues.append(f"Cannot read {name} Git identity")
    artifacts = {
        name: artifact_record(cfg[key])
        for name, key in (("target", "target_model"), ("draft", "draft_model"))
    }
    for name, record in artifacts.items():
        if (
            "config.json" not in record["files"]
            or not record["shards"]
            or record["missing_shards"]
        ):
            issues.append(f"{name} checkpoint config/shard files are incomplete")
    if "optional/quarot.safetensors" not in artifacts["target"]["rotation_files"]:
        issues.append(
            "Original-coordinate DSpark profile needs the target optional/quarot.safetensors"
        )
    tokenizer = Path(cfg["tokenizer"])
    tokenizer_files = {
        name: file_record(tokenizer / name)
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        if (tokenizer / name).is_file()
    }
    if not ({"tokenizer.json", "tokenizer.model"} & tokenizer_files.keys()):
        issues.append("No local tokenizer.json/tokenizer.model was found")
    spec = importlib.util.find_spec("sgl_kernel_npu")
    installed = Path(spec.origin).parent if spec is not None and spec.origin else None
    installed_files = (
        {
            name: file_record(installed / name, hash_contents=False)
            for name in ("lib/libsgl_kernel_npu.so", "utils/triton_utils.py")
        }
        if installed
        else {}
    )
    if not installed_files or not all(x["exists"] for x in installed_files.values()):
        issues.append(
            "Installed kernel binary/helper required by the checkout overlay are missing"
        )
    return {
        "status": "NODE_METADATA_COLLECTED" if not issues else "NODE_CHECK_BLOCKED",
        "ready": not issues,
        "issues": issues,
        "rank": rank,
        "node": node,
        "config": cfg,
        "python": {"executable": sys.executable, "version": sys.version},
        "versions": versions,
        "required_paths": paths,
        "sources": sources,
        "repositories": repos,
        "artifacts": artifacts,
        "tokenizer_files": tokenizer_files,
        "installed_kernel_files": installed_files,
        "interfaces": interfaces,
        "configured_host_present": node["host"] in local_ips,
        "npu_smi": npu_smi,
        "visibility_environment": {
            k: os.environ[k]
            for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")
            if k in os.environ
        },
        "limits": [
            "Read-only local metadata; no tensor allocation, compilation, collective, load, or generation test.",
            "Matching checkpoint sizes and metadata do not prove matching complete weight bytes.",
            "Current/free HBM and processes are recorded by npu-smi, not inferred from /dev node count.",
            "No remote SSH or remote port probe; inter-node HCCL reachability is not established.",
            "Service capacity must be checked after startup before long requests.",
        ],
    }


def compare_reports(left, right):
    issues = []
    if sorted([left.get("rank", -1), right.get("rank", -1)]) != [0, 1]:
        issues.append("Need one report from rank0 and one from rank1")
    for report in (left, right):
        if report.get("ready") is not True:
            issues.append(f"Rank {report.get('rank')} local checks were blocked")
    for key in ("config", "versions"):
        if left.get(key) != right.get(key):
            issues.append(f"Node {key} differ")
    if left.get("python", {}).get("version") != right.get("python", {}).get("version"):
        issues.append("Python versions differ")
    for name in ("sglang", "kernel"):
        a, b = (r.get("repositories", {}).get(name, {}) for r in (left, right))
        if not a.get("head") or a.get("head") != b.get("head"):
            issues.append(f"{name} Git commits differ or are missing")
        if a.get("tracked_status") or b.get("tracked_status"):
            issues.append(
                f"{name} has tracked edits; compare/commit the intended files before pairing nodes"
            )

    def hashes(records):
        return {name: rec.get("sha256") for name, rec in records.items()}

    for key in ("sources", "tokenizer_files"):
        if hashes(left.get(key, {})) != hashes(right.get(key, {})):
            issues.append(f"{key} fingerprints differ")
    for name in ("target", "draft"):
        a, b = (r.get("artifacts", {}).get(name, {}) for r in (left, right))
        if hashes(a.get("files", {})) != hashes(b.get("files", {})) or a.get(
            "shards"
        ) != b.get("shards"):
            issues.append(f"{name} checkpoint metadata differ")
    return {
        "status": "NODE_PAIR_METADATA_MATCH" if not issues else "NODE_PAIR_BLOCKED",
        "ready": not issues,
        "issues": issues,
        "limits": [
            "Metadata consistency only; not a distributed execution or 128k capacity PASS.",
            "Complete checkpoint bytes and installed binary bytes were not hashed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    local = sub.add_parser(
        "local", help="Run in the node's container using its service Python"
    )
    local.add_argument("--config", required=True, type=Path)
    local.add_argument("--rank", required=True, choices=(0, 1), type=int)
    local.add_argument("--output", type=Path)
    pair = sub.add_parser(
        "compare", help="Offline comparison; no access to either node"
    )
    pair.add_argument("rank0_report", type=Path)
    pair.add_argument("rank1_report", type=Path)
    pair.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "local":
        cfg = load_config(args.config)
        result = collect_local(cfg, args.rank)
        output = args.output or Path(cfg["state"]) / "evidence" / (
            "node-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-rank{args.rank}.json"
        )
    else:
        result = compare_reports(
            json.loads(args.rank0_report.read_text()),
            json.loads(args.rank1_report.read_text()),
        )
        output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Evidence: {output}")
    print(result["status"])
    for issue in result["issues"]:
        print("ISSUE:", issue)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
