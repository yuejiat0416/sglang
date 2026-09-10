#!/usr/bin/env python3
"""Print a two-node candidate launch; --execute explicitly runs this node only.

Filled arguments and launch evidence contain local deployment details. Keep them
on the test machines; only the unfilled templates belong in a shared archive.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import MODES, load_config, node_config

VENDOR_ENV = (
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",
)
REMOVED_ENV = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "ASCEND_LAUNCH_BLOCKING",
    "SGLANG_DSPARK_DEBUG_DUMP",
    "GLM52_CONTEXT_SNAPSHOT_CONFIG",
    "GLM52_PROPOSAL_SNAPSHOT_CONFIG",
    "SGLANG_NPU_GLM_DSPARK_QUAROT",
    "SGLANG_SIMULATE_ACC_LEN",
    "SGLANG_SIMULATE_ACC_METHOD",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE",
    "SGLANG_SIMULATE_UNIFORM_EXPERTS",
    "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
)


def runtime_environment(cfg, rank, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    node = node_config(cfg, rank)
    values = {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "HCCL_BUFFSIZE": "1000",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": node["hccl_socket_ifname"],
        "GLOO_SOCKET_IFNAME": node["gloo_socket_ifname"],
        "TRANSFORMERS_VERBOSITY": "error",
        "SGLANG_NPU_PROFILING": "0",
        "SGLANG_NPU_PROFILING_BS": "16",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if mode.startswith("dspark-"):
        values["SGLANG_NPU_GLM_DSPARK_QUAROT"] = "original"
    return values


def server_arguments(cfg, rank, mode):
    node = node_config(cfg, rank)
    runtime_environment(cfg, rank, mode)  # validate mode for direct callers
    args = [
        "--model-path",
        cfg["target_model"],
        "--tokenizer-path",
        cfg["tokenizer"],
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--tp-size",
        str(cfg["tp_size"]),
        "--nnodes",
        "2",
        "--node-rank",
        str(rank),
        "--dp-size",
        str(cfg["dp_size"]),
        "--dist-init-addr",
        cfg["dist_init_addr"],
        "--context-length",
        str(cfg["context_length"]),
        "--chunked-prefill-size",
        str(cfg["chunked_prefill_size"]),
        "--max-prefill-tokens",
        str(cfg["max_prefill_tokens"]),
        "--trust-remote-code",
        "--mem-fraction-static",
        str(cfg["mem_fraction_static"]),
        "--served-model-name",
        cfg["served_model_name"],
        "--max-running-requests",
        str(cfg["max_running_requests"]),
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--load-balance-method",
        "round_robin",
        "--enable-metrics",
        "--enable-cache-report",
        "--host",
        node["host"],
        "--port",
        str(cfg["port"]),
    ]
    if cfg["dp_size"] > 1:
        args += ["--enable-dp-attention", "--enable-dp-lm-head"]
    if cfg["max_total_tokens"] is not None:
        args += ["--max-total-tokens", str(cfg["max_total_tokens"])]
    if mode.startswith("dspark-"):
        args += [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            cfg["draft_model"],
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-draft-attention-backend",
            "ascend",
            "--speculative-dspark-block-size",
            "8",
            "--speculative-num-draft-tokens",
            "9",
        ]
    elif mode == "nextn-graph":
        args += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-draft-model-quantization",
            "unquant",
        ]
    if mode.endswith("eager"):
        args += ["--disable-cuda-graph"]
    else:
        args += ["--cuda-graph-bs", *(str(x) for x in cfg["graph_batch_sizes"])]
    return args


def build_command(cfg, rank, mode):
    """An argv array, never shell-interpolated configuration."""
    helper = str(Path(cfg["repo"]) / "devtools/glm52_ms1/with_kernel_checkout.py")
    command = ["env"]
    for key in REMOVED_ENV:
        command += ["-u", key]
    command += [f"{k}={v}" for k, v in runtime_environment(cfg, rank, mode).items()]
    command += [
        cfg["python"],
        helper,
        "--kernel-repo",
        cfg["kernel_repo"],
        "--state-dir",
        cfg["state"],
        "--",
        cfg["python"],
        "-m",
        "sglang.launch_server",
        *server_arguments(cfg, rank, mode),
    ]
    # CANN scripts retain their own error handling; only their final status is checked.
    script = (
        'source "$1" || exit 1\nsource "$2" || exit 1\n'
        'export PYTHONPATH="$3${PYTHONPATH:+:$PYTHONPATH}"\nshift 3\nexec "$@"'
    )
    return [
        "bash",
        "-c",
        script,
        "two-node-launch",
        *VENDOR_ENV,
        str(Path(cfg["repo"]) / "python"),
        *command,
    ]


def run_foreground(command, cwd, log_path):
    """Mirror this launch's output; Ctrl+C reaches only its local process group."""
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        log.flush()
                    return process.wait()
                except KeyboardInterrupt:
                    print(
                        "Stopping this node's launch; stop the other node in its own terminal.",
                        flush=True,
                    )
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    # Keep draining shutdown output: waiting with a full pipe
                    # could deadlock workers and lose their final error logs.
        finally:
            process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int, choices=(0, 1))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    command = build_command(cfg, args.rank, args.mode)
    print(
        "Candidate TP32/two-node deployment: not NPU validated. Start each rank manually.",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    from check_node import collect_local

    run = (
        Path(cfg["state"])
        / "evidence"
        / (
            "launch-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-rank{args.rank}-{args.mode}"
        )
    )
    run.mkdir(parents=True, exist_ok=False)
    report = collect_local(cfg, args.rank)
    (run / "node.json").write_text(json.dumps(report, indent=2) + "\n")
    (run / "launch.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "rank": args.rank,
                "config": cfg,
                "argv": command,
                "status": "USER_REQUESTED_EXECUTION",
                "env_overrides": runtime_environment(cfg, args.rank, args.mode),
                "removed_environment_names": list(REMOVED_ENV),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Evidence: {run}", flush=True)
    if not report["ready"]:
        print(
            "Local checks blocked startup: " + "; ".join(report["issues"]), flush=True
        )
        return 2
    print(f"Server log: {run / 'server.log'}", flush=True)
    exit_code = run_foreground(command, cfg["repo"], run / "server.log")
    (run / "exit.json").write_text(
        json.dumps({"exit_code": exit_code}, indent=2) + "\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
