#!/usr/bin/env python3
"""Offline A3 target-only preflight, launcher and local smoke client (stdlib only)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request


SOURCE_COMMIT = "dfd9b5c2a4079b8ebf9cf34e92379cdf0871c030"
ENV_KEYS = {"HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"}
FIELDS = {
    "source_dir",
    "source_commit",
    "model_path",
    "image_reference",
    "cann_version",
    "nnodes",
    "tp_size",
    "dp_size",
    "dist_init_addr",
    "host",
    "port",
    "quantization",
    "dtype",
    "kv_cache_dtype",
    "context_length",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "max_running_requests",
    "mem_fraction_static",
    "env",
}
# Paths used by the Python server or its compiled dependencies. Root Python and
# extension modules can also shadow imports because the server runs from cwd.
PRODUCTION_PATHS = (
    "python",
    "3rdparty",
    "proto",
    "rust",
    "sgl-model-gateway",
    "pyproject.toml",
    "setup.cfg",
    *(
        f":(top,glob)*{suffix}"
        for suffix in (".py", ".pyi", ".pyc", ".so", ".pyd", ".dll", ".dylib")
    ),
)


def expected_source_commit(value=None, *, required=False):
    if value is None:
        if required:
            raise ValueError(
                "Git source requires an explicit source_commit (40 hex digits)."
            )
        return SOURCE_COMMIT
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(
            "source_commit must be a full 40-digit hexadecimal commit SHA."
        )
    return value.lower()


def version_identity(commit, expected):
    return {
        "commit": commit,
        "expected_commit": expected,
        "community_baseline_commit": SOURCE_COMMIT,
        "baseline_relation": (
            "exact_community_commit" if commit == SOURCE_COMMIT else "different_commit"
        ),
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def config_from(path):
    cfg = json.loads(path.read_text())
    unknown = set(cfg) - FIELDS - {"_notes"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if set(cfg.get("env", {})) - ENV_KEYS:
        raise ValueError("Only HCCL/GLOO network-interface overrides are supported.")
    source = Path(cfg["source_dir"])
    cfg["source_dir"] = str(
        (path.parent / source).resolve()
        if not source.is_absolute()
        else source.resolve()
    )
    if "source_commit" in cfg:
        cfg["source_commit"] = expected_source_commit(
            cfg["source_commit"], required=True
        )
    return cfg


def git_output(source, *args):
    # Keep the container's scoped GIT_CONFIG_* safe.directory declaration, but
    # do not let repository/index overrides redirect the audit elsewhere.
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Git source audit timed out.") from exc
    if result.returncode:
        raise ValueError(f"Git source audit failed: {result.stderr.strip()}")
    return result.stdout


def git_source_identity(source, expected_commit):
    expected = expected_source_commit(expected_commit, required=True)
    root = Path(git_output(source, "rev-parse", "--show-toplevel").strip()).resolve()
    if root != source:
        raise ValueError(f"source_dir must be the Git repository root: {root}")
    commit = git_output(source, "rev-parse", "--verify", "HEAD^{commit}").strip()
    dirty = git_output(
        source, "diff", "--name-only", "-z", "HEAD", "--", *PRODUCTION_PATHS
    ).split("\0")
    extras = []
    for flags in (
        ("--others", "--exclude-standard"),
        ("--others", "--ignored", "--exclude-standard"),
    ):
        extras.extend(
            git_output(source, "ls-files", "-z", *flags, "--", *PRODUCTION_PATHS).split(
                "\0"
            )
        )
    # Regular bytecode caches validate against their source. Standalone .pyc
    # files outside __pycache__ remain subject to the source audit.
    extras = sorted(
        {
            name
            for name in extras
            if name
            and not ("__pycache__" in Path(name).parts and Path(name).suffix == ".pyc")
        }
    )
    entries = git_output(source, "ls-files", "-v", "-z", "--", *PRODUCTION_PATHS).split(
        "\0"
    )
    hidden_changes = sorted(
        entry[2:]
        for entry in entries
        if entry and (entry[0].islower() or entry[0] == "S")
    )
    # A committed link to an external file still permits unversioned executable
    # contents. Follow internal links by including their target in the audit.
    unsafe_links = []
    for entry in entries:
        if not entry:
            continue
        name = entry[2:]
        file = source / name
        if not file.is_symlink():
            continue
        target = file.resolve()
        if not target.is_relative_to(source) or not target.exists():
            unsafe_links.append(name)
            continue
        target_name = str(target.relative_to(source))
        tracked = git_output(source, "ls-files", "-z", "--", target_name).split("\0")
        changed = git_output(
            source, "diff", "--name-only", "-z", "HEAD", "--", target_name
        )
        if target_name not in tracked or changed:
            unsafe_links.append(name)
    dirty = sorted({name for name in dirty if name})
    return {
        **version_identity(commit, expected),
        "mode": "git",
        "repository_root": str(root),
        "production_scope": list(PRODUCTION_PATHS),
        "modified_files": dirty,
        "extra_files": extras,
        "hidden_index_entries": hidden_changes,
        "unsafe_symlinks": unsafe_links,
        "ok": commit == expected
        and not (dirty or extras or hidden_changes or unsafe_links),
    }


def source_identity(source, expected_commit=None):
    source = source.resolve()
    # .git is a directory in a checkout and a file in a linked worktree. An
    # archive may live inside another repository; its own manifest wins there.
    if (source / ".git").exists() or not (
        source.parent / "source-manifest.json"
    ).is_file():
        return git_source_identity(source, expected_commit)
    expected = expected_source_commit(expected_commit)
    manifest = json.loads((source.parent / "source-manifest.json").read_text())
    if manifest["commit"] != expected:
        raise ValueError("Source manifest does not match the expected source_commit.")
    failures = []
    for name, item in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid source manifest path.")
        file = source / relative
        if item["kind"] == "symlink":
            matches = file.is_symlink() and os.readlink(file) == item["target"]
        else:
            matches = (
                file.is_file()
                and not file.is_symlink()
                and digest(file) == item["sha256"]
            )
        if not matches:
            failures.append(name)
    extras = [
        str(p.relative_to(source))
        for p in source.rglob("*")
        if (p.is_file() or p.is_symlink())
        and "__pycache__" not in p.parts
        and str(p.relative_to(source)) not in manifest["files"]
    ]
    return {
        **version_identity(manifest["commit"], expected),
        "mode": "archive",
        "mismatches": failures,
        "extra_files": extras,
        "ok": not failures and not extras,
    }


def runtime_env(cfg):
    for key in ("SGLANG_EXTERNAL_MODEL_PACKAGE", "SGLANG_DISABLED_MODEL_ARCHS"):
        if os.environ.get(key):
            raise ValueError(
                f"Remove {key} so this run uses the selected source model registry."
            )
    if os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"):
        raise ValueError(
            "Remove SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE so the recorded rendezvous is effective."
        )
    env = dict(os.environ)
    # Avoid importing a second mounted checkout or accessing remote model hubs.
    env["PYTHONPATH"] = str(Path(cfg["source_dir"]) / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("SGLANG_ENABLE_SPEC_V2", None)
    for key, value in cfg.get("env", {}).items():
        if isinstance(value, str) and value and not value.startswith("SET_"):
            env[key] = value
    return env


def run_probe(code, cfg):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=runtime_env(cfg),
            cwd=cfg["source_dir"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stderr": "Probe timed out after 45 seconds."}
    except OSError as exc:
        return {"returncode": -1, "stderr": str(exc)}


def metadata_check(model):
    raw = json.loads((model / "config.json").read_text())
    architecture = raw.get("architectures", [])
    if architecture != ["GlmMoeDsaForCausalLM"]:
        raise ValueError(
            f"Expected GLM DSA target architecture, found {architecture!r}."
        )
    # Inspect the index only; do not load or hash the large weight shards.
    index = model / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if any(Path(s).is_absolute() or ".." in Path(s).parts for s in shards):
            raise ValueError("Invalid weight shard path in index.")
    else:
        shards = [p.name for p in model.glob("*.safetensors")]
    missing = [name for name in shards if not (model / name).is_file()]
    quant_file = model / "quant_model_description.json"
    quant_ok = quant_file.is_file() and isinstance(
        json.loads(quant_file.read_text()), dict
    )
    tokenizer_ok = any(
        (model / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    )
    return {
        "architecture": architecture,
        "model_type": raw.get("model_type"),
        "hidden_size": raw.get("hidden_size"),
        "num_hidden_layers": raw.get("num_hidden_layers"),
        "vocab_size": raw.get("vocab_size"),
        "shard_count": len(shards),
        "missing_shards": missing,
        "modelslim_description_exists": quant_ok,
        "tokenizer_exists": tokenizer_ok,
        "ok": bool(shards) and not missing and quant_ok and tokenizer_ok,
    }


def preflight(cfg, out, node_rank=0):
    build_command(cfg, node_rank)
    runtime_env(cfg)
    report = {
        "declared_image_reference": cfg.get("image_reference"),
        "declared_cann_version": cfg.get("cann_version"),
        "note": "Local report. Image/CANN labels are declarations, not verified ABI compatibility.",
        "python": sys.version,
        "checks": {},
    }
    checks = report["checks"]
    for name, action in (
        (
            "source",
            lambda: source_identity(Path(cfg["source_dir"]), cfg.get("source_commit")),
        ),
        ("model_metadata", lambda: metadata_check(Path(cfg["model_path"]))),
    ):
        try:
            checks[name] = action()
        except (OSError, ValueError, KeyError) as exc:
            checks[name] = {"ok": False, "error": str(exc)}
    if not checks["source"]["ok"]:
        # Do not import code from a checkout whose production identity failed.
        return finish_preflight(report, out)
    report["packages"] = {}
    for name in (
        "torch",
        "torch_npu",
        "transformers",
        "sgl-kernel-npu",
        "triton-ascend",
        "sglang",
    ):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    probe = run_probe(
        "import json, torch, torch_npu; "
        "print(json.dumps({'available':torch.npu.is_available(),'device_count':torch.npu.device_count()}))",
        cfg,
    )
    checks["npu"] = probe
    try:
        device = json.loads(probe.get("stdout", "").strip().splitlines()[-1])
        probe["ok"] = (
            probe["returncode"] == 0
            and device["available"]
            and device["device_count"] >= cfg["tp_size"] // cfg["nnodes"]
        )
        probe["device"] = device
    except (ValueError, IndexError, KeyError):
        probe["ok"] = False
    imports = run_probe(
        "import json, sglang; from sglang.srt.models import glm4_moe; "
        "print(json.dumps({'sglang':sglang.__file__,'glm':glm4_moe.__file__}))",
        cfg,
    )
    checks["imports"] = imports
    try:
        paths = json.loads(imports.get("stdout", "").strip().splitlines()[-1])
        root = Path(cfg["source_dir"]).resolve() / "python"
        imports["ok"] = imports["returncode"] == 0 and all(
            Path(p).resolve().is_relative_to(root) for p in paths.values()
        )
    except (ValueError, IndexError):
        imports["ok"] = False
    return finish_preflight(report, out)


def finish_preflight(report, out):
    checks = report["checks"]
    report["preflight_ok"] = all(item["ok"] for item in checks.values())
    write_json(out / "preflight.json", report)
    print(
        json.dumps(
            {
                "preflight_ok": report["preflight_ok"],
                "checks": {k: v["ok"] for k, v in checks.items()},
            }
        )
    )
    return 0 if report["preflight_ok"] else 1


def build_command(cfg, node_rank):
    for key in (
        "nnodes",
        "tp_size",
        "dp_size",
        "port",
        "context_length",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_running_requests",
    ):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["nnodes"] != 2 or cfg["tp_size"] % 2 or cfg["tp_size"] % cfg["dp_size"]:
        raise ValueError(
            "This recipe requires two nodes and TP divisible by both node count and DP."
        )
    if cfg["max_running_requests"] < cfg["dp_size"]:
        raise ValueError(
            "Global max_running_requests must be >= DP so every worker has a request slot."
        )
    if (cfg["chunked_prefill_size"] // cfg["dp_size"]) % 128 or cfg[
        "chunked_prefill_size"
    ] < cfg["dp_size"] * 128:
        raise ValueError(
            "Per-DP prefill chunk must be positive and aligned to the explicit 128-token page size."
        )
    if node_rank not in (0, 1):
        raise ValueError("node_rank must be 0 or 1.")
    for key in ("model_path", "dist_init_addr", "image_reference"):
        if not isinstance(cfg.get(key), str) or not cfg[key] or "SET_" in cfg[key]:
            raise ValueError(
                f"Fill {key} inside the intranet before generating a launch command."
            )
    if not Path(cfg["model_path"]).is_absolute():
        raise ValueError("model_path must be an existing absolute intranet path.")
    for key in ENV_KEYS:
        value = cfg.get("env", {}).get(key, "")
        if not value or "SET_" in value:
            raise ValueError(
                f"Fill env.{key} with the actual inter-node network interface."
            )
    if not 0 < cfg["mem_fraction_static"] < 1:
        raise ValueError("mem_fraction_static must be between 0 and 1.")
    if cfg["quantization"] != "modelslim":
        raise ValueError(
            "This initial recipe is for the existing ModelSlim target; another format needs a separate recipe."
        )
    argv = [sys.executable, "-m", "sglang.launch_server"]
    for key in (
        "model_path",
        "tp_size",
        "nnodes",
        "dp_size",
        "dist_init_addr",
        "host",
        "port",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "context_length",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "max_running_requests",
        "mem_fraction_static",
    ):
        if cfg[key] is not None:
            argv += ["--" + key.replace("_", "-"), str(cfg[key])]
    argv += [
        "--node-rank",
        str(node_rank),
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-shared-experts-fusion",
        "--page-size",
        "128",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--served-model-name",
        "glm52-target-only",
        "--load-balance-method",
        "round_robin",
    ]
    if cfg["dp_size"] > 1:
        argv += ["--enable-dp-attention"]
    return argv


def request_json(base, path, payload=None):
    url = urllib.parse.urlsplit(base)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Use an explicit local/intranet HTTP server URL without credentials/query/fragment."
        )
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Avoid sending intranet requests through an inherited outbound proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=180
    ) as response:
        body = response.read()
    return json.loads(body) if body else None


def verify_server(info, cfg):
    if not isinstance(info, dict):
        raise ValueError("Server info must be an object.")
    expected = {
        key: cfg[key]
        for key in (
            "model_path",
            "quantization",
            "tp_size",
            "dp_size",
            "nnodes",
            "context_length",
        )
    }
    expected.update(
        device="npu",
        served_model_name="glm52-target-only",
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        enable_dp_attention=cfg["dp_size"] > 1,
    )
    mismatch = [
        key for key, value in expected.items() if key not in info or info[key] != value
    ]
    graphs = info.get("cuda_graph_config", {})
    if any(
        graphs.get(phase, {}).get("backend") != "disabled"
        for phase in ("decode", "prefill")
    ):
        mismatch.append("cuda_graph_config")
    if mismatch:
        raise ValueError(
            f"Queried service does not match the target-only plan: {mismatch}"
        )


def smoke(base, out, cfg, launch_evidence):
    plan = json.loads((launch_evidence / "launch-plan.json").read_text())
    checked = json.loads((launch_evidence / "preflight.json").read_text())
    expected = expected_source_commit(cfg.get("source_commit"))
    source = checked.get("checks", {}).get("source", {})
    if (
        plan.get("config") != cfg
        or plan.get("source_commit") != expected
        or source.get("commit") != expected
        or not source.get("ok")
        or plan.get("source", source) != source
        or not checked.get("preflight_ok")
    ):
        raise ValueError(
            "Use node0 launch evidence matching this configuration and successful preflight."
        )
    request_json(base, "/health")
    info = request_json(base, "/server_info")
    write_json(out / "server-info.json", info)
    verify_server(info, cfg)
    prompts = ["The capital of France is", "1 + 1 =", "请用一句话介绍太阳。"]
    results = []
    for prompt in prompts:
        response = request_json(
            base,
            "/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                "stream": False,
            },
        )
        results.append({"prompt": prompt, "response": response})
        write_json(out / "smoke-responses.json", results)
        if (
            not isinstance(response.get("text"), str)
            or not response["text"].strip()
            or response.get("meta_info", {}).get("completion_tokens", 0) < 1
        ):
            raise ValueError("A smoke request returned no generated text/tokens.")
        if response.get("meta_info", {}).get("finish_reason", {}).get("type") not in (
            "stop",
            "length",
        ):
            raise ValueError("A smoke request did not finish normally (stop/length).")
    write_json(
        out / "smoke-summary.json",
        {
            "status": "SMOKE_ONLY_PASS",
            "requests": len(results),
            "speculative_algorithm": None,
            "launch_evidence": str(launch_evidence.resolve()),
            "source_commit": expected,
            "source": source,
            "identity_note": "HTTP config matching is not proof of process/source identity; correlate this launch log and process locally.",
            "note": "Manual output review required; this is not accuracy/performance or MS1 acceptance.",
        },
    )
    print(
        "SMOKE_ONLY_PASS: 3 sequential requests returned text; inspect outputs locally."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "command", "launch", "smoke"))
    parser.add_argument("--config", type=Path, default=Path("target-only.local.json"))
    parser.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New local evidence directory; must not already exist.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--launch-evidence",
        type=Path,
        help="Same-run node0 launch evidence directory, required by smoke.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        cfg = config_from(args.config.resolve())
        if args.mode == "smoke":
            if args.launch_evidence is None:
                raise ValueError(
                    "smoke requires --launch-evidence from the node0 launch."
                )
            smoke(args.base_url, args.out, cfg, args.launch_evidence)
            return 0
        if args.mode == "preflight":
            return preflight(cfg, args.out, args.node_rank)
        argv = build_command(cfg, args.node_rank)
        source = source_identity(Path(cfg["source_dir"]), cfg.get("source_commit"))
        if not source["ok"]:
            write_json(args.out / "source-check.json", source)
            raise ValueError(
                "Source checkout does not match the expected clean production source."
            )
        write_json(
            args.out / "launch-plan.json",
            {
                "argv": argv,
                "config": cfg,
                "source_commit": source["commit"],
                "source": source,
                "mode": "target-only eager",
                "note": "Candidate topology and capacity, not a performance recipe.",
            },
        )
        print(shlex.join(argv))
        if args.mode == "command":
            return 0
        # Never launch with a mismatched source tree or unresolved preflight.
        if preflight(cfg, args.out, args.node_rank):
            return 1
        print(
            "Starting this foreground server only; use Ctrl-C to stop. Logs stay in server.log.",
            flush=True,
        )
        with (args.out / "server.log").open("w") as log:
            result = subprocess.run(
                argv,
                cwd=cfg["source_dir"],
                env=runtime_env(cfg),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        write_json(args.out / "exit.json", {"returncode": result.returncode})
        return result.returncode
    except KeyboardInterrupt:
        write_json(
            args.out / "exit.json",
            {
                "returncode": 130,
                "note": "Interrupted; inspect this container for remaining server processes.",
            },
        )
        return 130
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        write_json(args.out / "error.json", {"error": str(exc), "mode": args.mode})
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
