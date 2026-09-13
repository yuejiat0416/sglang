"""Shared client-only helpers for the temporary two-node test archive."""

import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = HERE.parents[2]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from gsm8k_mode_stats import graph_count_delta  # noqa: E402

SERVER_KEYS = (
    "device",
    "model_path",
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dspark_block_size",
    "tp_size",
    "dp_size",
    "ep_size",
    "pp_size",
    "nnodes",
    "enable_dp_attention",
    "disaggregation_mode",
    "disable_cuda_graph",
    "disable_decode_cuda_graph",
    "cuda_graph_config",
    "enable_metrics",
    "quantization",
    "served_model_name",
    "page_size",
    "context_length",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "disable_radix_cache",
    "enable_hierarchical_cache",
    "allow_auto_truncate",
    "dcp_size",
    "attn_cp_size",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_text(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response:
        return response.read().decode("utf-8")


def selected_server(info):
    return {key: info.get(key) for key in SERVER_KEYS}


def validate_server(info, cfg, mode):
    """Reject a wrong mode/topology before sending benchmark traffic."""
    from config import MODES

    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    for key, value in (
        ("device", "npu"),
        ("nnodes", 2),
        ("tp_size", cfg["tp_size"]),
        ("dp_size", cfg["dp_size"]),
        ("model_path", cfg["target_model"]),
        ("served_model_name", cfg["served_model_name"]),
        ("quantization", "modelslim"),
        ("enable_metrics", True),
    ):
        if info.get(key) != value:
            raise ValueError(f"Server {key}={info.get(key)!r}; expected {value!r}")
    if info.get("disaggregation_mode") not in (None, "null"):
        raise ValueError("This archive describes colocated serving, not P/D separation")
    if info.get("pp_size", 1) not in (None, 1):
        raise ValueError(
            "PP>1 is outside this archive profile; no framework restriction is added"
        )
    if cfg["dp_size"] > 1 and info.get("enable_dp_attention") is not True:
        raise ValueError(
            "Expected resolved DP Attention for the selected multi-DP profile"
        )
    dspark, nextn = mode.startswith("dspark"), mode.startswith("nextn")
    expected = {"DSPARK"} if dspark else {"NEXTN", "EAGLE"} if nextn else {"", "NONE"}
    if str(info.get("speculative_algorithm") or "").upper() not in expected:
        raise ValueError("Server algorithm does not match the requested mode")
    if dspark:
        if info.get("speculative_draft_model_path") != cfg["draft_model"]:
            raise ValueError(
                f"Expected DSpark speculative_draft_model_path={cfg['draft_model']!r}"
            )
        window = (
            info.get("speculative_dspark_block_size"),
            info.get("speculative_num_draft_tokens"),
        )
        if window not in ((8, 9), (5, 6)):
            raise ValueError("Expected DSpark block8/draft9 or block5/draft6")
    if nextn and tuple(
        info.get(k)
        for k in (
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
        )
    ) != (4, 1, 5):
        raise ValueError("Expected NEXTN steps4/topk1/draft5")
    graph = mode.endswith("graph")
    backend = (info.get("cuda_graph_config") or {}).get("decode", {}).get("backend")
    if backend not in {"disabled", "full", "breakable", "tc_piecewise"}:
        raise ValueError("Missing/unknown resolved decode graph backend")
    if (backend != "disabled") != graph or info.get("disable_cuda_graph") is not (
        not graph
    ):
        raise ValueError("Requested mode differs from resolved graph configuration")
    if graph and info.get("disable_decode_cuda_graph") is True:
        raise ValueError("Decode graph is disabled")
    return selected_server(info)


def prepare_run(cfg, stage, mode):
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run = (
        Path(cfg["state"])
        / "evidence"
        / "two-node-colocated"
        / f"{stage}-{mode}-{run_id}"
    )
    run.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {run}", flush=True)
    write_json(run / "config.json", cfg)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "UNAVAILABLE"
    write_json(
        run / "client-source.json",
        {
            "git_head": head,
            "files": {p.name: sha(p) for p in HERE.glob("*.py")},
            "scope": "Client checkout identity; retain both server node launch manifests separately",
        },
    )
    return run


def import_benchmark():
    sys.path.insert(0, str(REPO / "python"))
    module = importlib.import_module("sglang.benchmark.serving")
    if Path(module.__file__).resolve() != REPO / "python/sglang/benchmark/serving.py":
        raise ValueError("Benchmark was not imported from this checkout")
    return module


@contextlib.contextmanager
def benchmark_context(bench=None):
    values = {
        "NO_PROXY": "*",
        "no_proxy": "*",
        "SGLANG_IS_IN_CI": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {k: os.environ.get(k) for k in values}
    argv = sys.argv
    os.environ.update(values)
    try:
        yield
    finally:
        sys.argv = argv
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def graph_evidence(before, after, mode):
    result = graph_count_delta(before, after)
    delta = result.get("deltas", {}).get("decode_cuda_graph")
    return {
        "counters": result,
        "requested_graph": mode.endswith("graph"),
        "target_replay_observed": delta > 0
        if result.get("available") and delta is not None
        else None,
        "draft_replay": "NOT_EXPOSED_BY_HTTP_API"
        if not mode.startswith("target")
        else "NOT_APPLICABLE",
        "scope": "Service-wide Target decode/verify counters; run with other clients idle. Not whole DSpark graph proof.",
    }
