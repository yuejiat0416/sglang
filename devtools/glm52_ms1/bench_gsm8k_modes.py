#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Temporary five-mode GSM8K client using the unchanged serving benchmark.

No server hooks, installed-package edits, automatic restarts, or NPU operations
from the developer machine. The owner runs launch/run inside the test container.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from gsm8k_mode_stats import graph_count_delta, summarize_responses

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODES = ("dspark-eager", "dspark-graph", "target-eager", "target-graph", "nextn-graph")
SOURCE_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def cases():
    bundle = json.loads((HERE / "gsm8k10.json").read_text())
    items = bundle["cases"]
    if bundle["source_sha256"] != SOURCE_SHA or len(items) != 10:
        raise ValueError("Expected the pinned ten-question GSM8K bundle")
    if [c["id"] for c in items] != [f"gsm8k-test-{i:04d}" for i in range(10)]:
        raise ValueError("Question selection/order changed")
    for c in items:
        if not isinstance(c["question"], str) or not c["question"].strip():
            raise ValueError("Missing question")
    return bundle


def request_rows(items, run_id, max_tokens):
    # Gold answers stay in the local bundle, never in the model request.
    return [
        {
            "messages": [{"role": "user", "content": c["question"]}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "rid": f"{run_id}-{c['id']}",
            "cache_salt": f"{run_id}-{c['id']}",
            "return_meta_info": True,
            "return_spec_tokens_details": True,
            "return_token_ids": True,
        }
        for c in items
    ]


def launch_environment(
    mode, state, host, port, target, draft, served_model_name="model"
):
    env = os.environ.copy()
    # A previous observer must not silently turn this into an instrumented run.
    if any(
        env.get(k)
        for k in ("GLM52_CONTEXT_SNAPSHOT_CONFIG", "GLM52_PROPOSAL_SNAPSHOT_CONFIG")
    ):
        raise ValueError("Use a plain terminal outside the snapshot observer launch")
    env.pop("SGLANG_DSPARK_DEBUG_DUMP", None)
    env.pop("SGLANG_NPU_GLM_DSPARK_QUAROT", None)
    env.update(
        MODE="dspark"
        if mode.startswith("dspark")
        else ("nextn" if mode.startswith("nextn") else "target-only"),
        GRAPH="1" if mode.endswith("graph") else "0",
        ENABLE_METRICS="1",
        MS1_STATE=str(state),
        MS1_HOST=host,
        MS1_PORT=str(port),
        TARGET_MODEL=target,
        SERVED_MODEL_NAME=served_model_name,
    )
    env.pop("DRAFT_MODEL", None)
    if draft is not None:
        env["DRAFT_MODEL"] = draft
    if mode.startswith("dspark"):
        env["SGLANG_NPU_GLM_DSPARK_QUAROT"] = "original"
    return env


def selected_config(info):
    keys = (
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
        "nnodes",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "cuda_graph_config",
        "enable_metrics",
        "quantization",
        "served_model_name",
    )
    return {key: info.get(key) for key in keys}


def validate_mode(info, mode):
    cfg = selected_config(info)
    if cfg["device"] != "npu" or (cfg["tp_size"], cfg["dp_size"], cfg["nnodes"]) != (
        16,
        1,
        1,
    ):
        raise ValueError("Expected single-node NPU TP16/DP1 server")
    algo = (cfg["speculative_algorithm"] or "").upper()
    expected = (
        {"DSPARK"}
        if mode.startswith("dspark")
        else ({"NEXTN", "EAGLE"} if mode.startswith("nextn") else {"", "NONE"})
    )
    if algo not in expected:
        raise ValueError(f"Mode {mode} does not match server algorithm {algo!r}")
    if cfg["disable_cuda_graph"] is not (not mode.endswith("graph")):
        raise ValueError("Declared mode does not match server disable_cuda_graph")
    if mode.endswith("graph") and cfg["disable_decode_cuda_graph"] is True:
        raise ValueError("Decode graph is disabled; do not label this a graph run")
    backend = (cfg.get("cuda_graph_config") or {}).get("decode", {}).get("backend")
    if backend not in {"full", "breakable", "tc_piecewise", "disabled"}:
        raise ValueError("Missing/unknown resolved decode graph backend")
    if (backend != "disabled") != mode.endswith("graph"):
        raise ValueError("Mode differs from resolved cuda_graph_config.decode.backend")
    if cfg["enable_metrics"] is not True:
        raise ValueError(
            "Start the mode with this tool's launch command (--enable-metrics)"
        )
    if mode.startswith("dspark") and (
        cfg["speculative_dspark_block_size"],
        cfg["speculative_num_draft_tokens"],
    ) != (8, 9):
        raise ValueError("Expected DSpark block8/verify9")
    if mode.startswith("nextn") and (
        cfg["speculative_num_steps"],
        cfg["speculative_eagle_topk"],
        cfg["speculative_num_draft_tokens"],
    ) != (4, 1, 5):
        raise ValueError("Expected NEXTN steps4/topk1/tokens5")
    return cfg


def fetch_text(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response:
        return response.read().decode("utf-8")


def bench_arguments(args, run):
    return [
        "--backend",
        "sglang-oai-chat",
        "--host",
        args.host,
        "--port",
        str(args.port),
        # The benchmark checks the chat template using --model before loading
        # --tokenizer. Keep both local; only the HTTP model uses the service ID.
        "--model",
        args.target,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.target,
        "--dataset-name",
        "openai",
        "--dataset-path",
        str(run / "requests.jsonl"),
        "--num-prompts",
        "10",
        "--sharegpt-output-len",
        str(args.max_tokens),
        "--request-rate",
        "inf",
        "--max-concurrency",
        "1",
        "--seed",
        "42",
        "--warmup-requests",
        "0",
        "--disable-ignore-eos",
        "--disable-stream",
        "--output-details",
        "--output-file",
        str(run / "benchmark.jsonl"),
    ]


@contextlib.contextmanager
def capture_responses(bench, output, captured):
    """Swap only this client's session factory; preserve raw JSON and return it.

    aiohttp's documented response_class hook, not a server/runtime monkey patch.
    Timeout/read buffer match serving.py's current factory. Restored on exit.
    """
    import aiohttp

    class RecordingResponse(aiohttp.ClientResponse):
        async def json(self, *args, **kwargs):
            value = await super().json(*args, **kwargs)
            if self.url.path.endswith("/chat/completions"):
                captured.append(value)
                with output.open("a") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            return value

    previous = bench._create_bench_client_session
    bench._create_bench_client_session = lambda: aiohttp.ClientSession(
        response_class=RecordingResponse,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
        read_bufsize=10 * 1024**2,
    )
    try:
        yield
    finally:
        bench._create_bench_client_session = previous


def render_answers(summary, items):
    lines = ["# GSM8K 10题：原始响应，未自动评分", "", "不要执行回答中生成的代码。", ""]
    for item, row in zip(items, summary["per_question"]):
        lines += [
            f"## {item['id']}",
            "",
            item["question"],
            "",
            f"结束原因：{row.get('finish_reason')}；输出 token：{row.get('completion_tokens')}",
            "",
            "参考答案：",
            "",
            item["answer"],
            "",
            "模型响应：",
            "",
        ]
        for field in ("reasoning_content", "content"):
            value = row.get(field)
            if value:
                fence = "`" * max(
                    4,
                    max(
                        (len(s) for s in str(value).split() if set(s) == {"`"}),
                        default=0,
                    )
                    + 1,
                )
                lines += [field, "", fence + "text", str(value), fence, ""]
    return "\n".join(lines)


def run_client(args):
    bundle = cases()
    run_id = (
        "gsm8k-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run = args.state / "evidence" / (run_id + "-" + args.mode)
    run.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {run}", flush=True)
    rows = request_rows(bundle["cases"], run_id, args.max_tokens)
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    argv = bench_arguments(args, run)
    write_json(
        run / "protocol.json",
        {
            "mode": args.mode,
            "collector_git_head": git_head(),
            "collector_sha256": sha(__file__),
            "stats_sha256": sha(HERE / "gsm8k_mode_stats.py"),
            "dataset_sha256": sha(HERE / "gsm8k10.json"),
            "source_url": bundle["source_url"],
            "source_sha256": bundle["source_sha256"],
            "case_ids": [c["id"] for c in bundle["cases"]],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "concurrency": 1,
            "warmup_requests": 0,
            "cache_policy": "unique per-question cache_salt; no global cache flush",
            "benchmark_argv": argv,
            "limits": [
                "10 questions, one trial, zero-shot; no accuracy/load qualification or official 6.41 reproduction.",
                "Nonstream benchmark TTFT is response latency, not true TTFT; TPOT is not a streaming performance measurement.",
                "Client source identity does not prove the running server identity; retain its launch log.",
            ],
        },
    )
    base_url = f"http://{args.host}:{args.port}"
    responses, issues = [], []
    before, after, graph = {}, {}, {}
    try:
        before = json.loads(fetch_text(base_url + "/server_info"))
        write_json(run / "server_info.before.json", before)
        validate_mode(before, args.mode)
        if before.get("model_path") != args.target:
            raise ValueError(
                "Server target path differs from the tokenizer/target path"
            )
        if before.get("served_model_name") != args.served_model_name:
            raise ValueError(
                "Server served_model_name differs from --served-model-name"
            )
        if (
            args.mode.startswith("dspark")
            and before.get("speculative_draft_model_path") != args.draft
        ):
            raise ValueError("Server DSpark draft path differs from the selected draft")
        metrics_before = fetch_text(base_url + "/metrics")
        (run / "metrics.before.txt").write_text(metrics_before)
        sys.path.insert(0, str(REPO / "python"))
        bench = importlib.import_module("sglang.benchmark.serving")
        if (
            Path(bench.__file__).resolve()
            != REPO / "python/sglang/benchmark/serving.py"
        ):
            raise ValueError("Benchmark import is not from this checkout")
        write_json(
            run / "benchmark-source.json",
            {"file": bench.__file__, "sha256": sha(bench.__file__)},
        )
        previous_argv = sys.argv
        try:
            sys.argv = ["sglang.benchmark.serving", *argv]
            # Local serving endpoints must bypass a download/Git proxy.
            previous_no_proxy = {
                k: os.environ.get(k)
                for k in ("NO_PROXY", "no_proxy", "SGLANG_IS_IN_CI")
            }
            os.environ.update(NO_PROXY="*", no_proxy="*", SGLANG_IS_IN_CI="false")
            try:
                with capture_responses(bench, run / "responses.jsonl", responses):
                    bench.cli_main()
            finally:
                for key, value in previous_no_proxy.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        finally:
            sys.argv = previous_argv
        after = json.loads(fetch_text(base_url + "/server_info"))
        write_json(run / "server_info.after.json", after)
        if selected_config(before) != selected_config(after):
            raise ValueError("Server configuration changed during the run")
        metrics_after = fetch_text(base_url + "/metrics")
        (run / "metrics.after.txt").write_text(metrics_after)
        graph = graph_count_delta(metrics_before, metrics_after)
        result_lines = (run / "benchmark.jsonl").read_text().splitlines()
        result = json.loads(result_lines[-1])
        if result.get("completed") != 10:
            raise ValueError(
                f"Benchmark completed {result.get('completed')} of 10 requests"
            )
    except (Exception, SystemExit) as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
    summary = summarize_responses([r["rid"] for r in rows], responses, args.mode)
    summary["issues"].extend(issues)
    if issues:
        summary.update(status="GSM8K_RESPONSES_INCOMPLETE", complete=False)
        summary["aggregate"] = None
    summary.update(
        server_configuration=selected_config(before),
        requested_graph=args.mode.endswith("graph"),
        graph_counters=graph,
        target_replay_observed=(graph.get("deltas", {}).get("decode_cuda_graph") or 0)
        > 0
        if graph.get("available")
        else None,
        graph_scope="Target decode/verify only; metric delta is service-wide. Keep other clients idle.",
        draft_replay_evidence="NOT_EXPOSED_BY_CURRENT_HTTP_API"
        if not args.mode.startswith("target")
        else "NOT_APPLICABLE",
        graph_capture=[
            s.get("startup_time", {}).get("cuda_graph")
            for s in before.get("internal_states", [])
        ],
        quality="NOT_SCORED",
        performance="NOT_QUALIFIED",
        length_finishes=sum(
            r.get("finish_reason") == "length" for r in summary["per_question"]
        ),
    )
    (run / "responses.md").write_text(render_answers(summary, bundle["cases"]))
    # Keep the file returned by the owner small; full outputs already live in
    # responses.jsonl / responses.md, including failed or foreign responses.
    for row in summary["per_question"]:
        for key in ("response", "responses", "content", "reasoning_content"):
            row.pop(key, None)
    summary.pop("unexpected_responses", None)
    write_json(run / "summary.json", summary)
    print(summary["status"])
    print("Aggregate:", json.dumps(summary["aggregate"]))
    print("Target graph/eager batch deltas:", json.dumps(graph.get("deltas")))
    print("Target replay observed:", summary["target_replay_observed"])
    print("Length finishes:", summary["length_finishes"])
    for issue in summary["issues"]:
        print("ISSUE:", issue)
    print(
        "Send summary.json; retain responses.jsonl, benchmark.jsonl and server launch log."
    )
    return 0 if summary["complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "run"))
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--target", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--draft", help="Required for DSpark modes; local draft checkpoint path"
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print without starting the server/benchmark",
    )
    args = parser.parse_args(argv)
    if args.mode.startswith("dspark") and not args.draft:
        parser.error("--draft is required for DSpark modes")
    if args.max_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("max-tokens must be positive and port must be 1..65535")
    if args.action == "run":
        if args.print_command:
            print(
                shlex.join(
                    [
                        "python3",
                        "-m",
                        "sglang.benchmark.serving",
                        *bench_arguments(args, Path("RUN_DIRECTORY")),
                    ]
                )
            )
            return 0
        return run_client(args)
    env = launch_environment(
        args.mode,
        args.state,
        args.host,
        args.port,
        args.target,
        args.draft,
        args.served_model_name,
    )
    script = HERE / "single_dspark_static.sh"
    if args.print_command:
        print(f"QuaRot: {env.get('SGLANG_NPU_GLM_DSPARK_QUAROT', 'not set')}")
        return subprocess.call(["bash", str(script), "--print-command"], env=env)
    print(
        f"Starting {args.mode}; TP16/DP1, metrics enabled, DSpark QuaRot={env.get('SGLANG_NPU_GLM_DSPARK_QUAROT', 'not set')}. Stop with Ctrl+C.",
        flush=True,
    )
    os.execvpe("bash", ["bash", str(script)], env)


if __name__ == "__main__":
    raise SystemExit(main())
