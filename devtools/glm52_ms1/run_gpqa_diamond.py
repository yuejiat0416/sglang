#!/usr/bin/env python3
"""One full EvalScope GPQA-D run against the existing A3 two-node service."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request

from gsm8k_mode_stats import graph_count_delta

# 在节点0的客户端终端，用已有 evalscope-venv/bin/python 运行本文件。
BASE_URL = "http://61.47.19.68:8810"
GRAPH = True  # 本轮要求graph；False仅用于另一次eager对照。
MODEL = "GLM-5.2-w8a8"
DATASET = "/home/tyj/glm52-ms1/datasets/gpqa_diamond.csv"
RESULTS = "/home/tyj/glm52-ms1/evidence"
CONCURRENCY = 4
MAX_TOKENS = 65536
TEMPERATURE = 1.0
TIMEOUT = 7200  # 非流式保留服务端原始接受计数，允许长推理等待。
GAMMA = 5
QUESTIONS = 198


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def check_dataset(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )
    if len(rows) != QUESTIONS:
        raise ValueError(f"GPQA-Diamond 必须完整 {QUESTIONS} 题，实际 {len(rows)}")
    if any(not row.get(key, "").strip() for row in rows for key in fields):
        raise ValueError("GPQA CSV 缺题目或答案字段；使用原始 Diamond CSV")
    if len({row["Question"].strip() for row in rows}) != QUESTIONS:
        raise ValueError("GPQA CSV 存在重复题目")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_server(info):
    for key, expected in (
        ("device", "npu"),
        ("nnodes", 2),
        ("tp_size", 32),
        ("dp_size", 4),
        ("served_model_name", MODEL),
        ("speculative_algorithm", "DSPARK"),
        ("speculative_dspark_block_size", GAMMA),
        ("speculative_num_draft_tokens", GAMMA + 1),
    ):
        if info.get(key) != expected:
            raise ValueError(f"服务 {key}={info.get(key)!r}，本轮要求 {expected!r}")
    if info.get("context_length", 0) <= MAX_TOKENS:
        raise ValueError("服务上下文不足以容纳65536输出和GPQA提示；不能静默缩短预算")
    backend = ((info.get("cuda_graph_config") or {}).get("decode") or {}).get("backend")
    if backend not in ("disabled", "full", "breakable", "tc_piecewise"):
        raise ValueError(f"无法核对服务实际graph配置：decode backend={backend!r}")
    enabled = backend != "disabled"
    if enabled != GRAPH or (
        GRAPH
        and (info.get("disable_cuda_graph") or info.get("disable_decode_cuda_graph"))
    ):
        raise ValueError(f"服务graph配置不符：本轮GRAPH={GRAPH}，实际backend={backend}")
    if GRAPH and info.get("enable_metrics") is not True:
        raise ValueError("本轮graph需开启服务--enable-metrics以记录实际回放计数")


def collect_graph_metrics(opener, output, stage):
    """Keep unavailable counters explicit without discarding completed answers."""
    try:
        with opener.open(BASE_URL + "/metrics", timeout=30) as reply:
            content = reply.read().decode("utf-8")
        (output / f"metrics.{stage}.txt").write_text(content)
        return content
    except Exception as exc:
        write_json(output / f"metrics.{stage}.error.json", {"error": str(exc)})
        return None


def add_graph_result(result, before, after):
    counters = graph_count_delta(before, after)
    delta = counters["deltas"].get("decode_cuda_graph")
    observed = delta > 0 if counters["available"] and delta is not None else None
    result["mode"] = "dspark-graph" if GRAPH else "dspark-eager"
    result["graph"] = {
        "requested": GRAPH,
        "target_replay_observed": observed,
        "counters": counters,
        "draft_replay": "NOT_EXPOSED_BY_HTTP_API",
        "scope": "服务级Target decode/verify计数；其他客户端须空闲。不能证明每轮draft均回放。",
    }
    if GRAPH and observed is not True:
        result["evaluation_status"] = result["status"]
        result["status"] = "INCOMPLETE"
        result["graph_issue"] = "未取得Target图回放证据；已完成的精度和接受计数仍保留"
    return result


def response_counts(response):
    details = (response.get("sglext") or {}).get("spec_tokens_details")
    if not isinstance(details, dict):
        raise ValueError("响应缺少 sglext.spec_tokens_details，不能判断接受率")
    a, p, n = (
        details.get(key)
        for key in (
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
            "spec_verify_ct",
        )
    )
    if any(type(v) is not int or v < 0 for v in (a, p, n)):
        raise ValueError("服务端 A/P/N 必须是非负整数")
    if a > p or p != GAMMA * n:
        raise ValueError(f"非本轮gamma5计数：A={a}, P={p}, N={n}")
    histogram = details.get("spec_correct_drafts_histogram")
    if histogram is not None and (
        not isinstance(histogram, list)
        or len(histogram) > GAMMA + 1
        or any(type(v) is not int or v < 0 for v in histogram)
        or sum(histogram) != n
        or sum(i * v for i, v in enumerate(histogram)) != a
    ):
        raise ValueError("接受计数与服务端直方图不一致")
    choices = response.get("choices", [])
    if not response.get("id") or len(choices) != 1:
        raise ValueError("本轮要求每题一个带唯一id的响应")
    finish = choices[0].get("finish_reason")
    if finish not in ("stop", "length"):
        raise ValueError(f"请求未正常完成：finish_reason={finish!r}")
    return {
        "response_id": response["id"],
        "A": a,
        "P": p,
        "N": n,
        "accept_rate": a / p if p else None,
        "accept_length": 1 + a / n if n else None,
        "finish_reason": finish,
        "usage": response.get("usage"),
        "server_spec_tokens_details": details,
    }


def summarize(report, records):
    execution = report.get("execution_summary") or {}
    if any(
        execution.get(k) != v
        for k, v in (
            ("requested", QUESTIONS),
            ("succeeded", QUESTIONS),
            ("errored", 0),
            ("incomplete", False),
        )
    ):
        raise ValueError(f"EvalScope未完成全部198题：{execution}")
    identity = report.get("primary_metric_identity") or {}
    metrics = [m for m in report.get("metrics", []) if m.get("identity") == identity]
    if identity.get("name") != "accuracy" or len(metrics) != 1:
        raise ValueError("EvalScope主指标不是唯一的accuracy")
    metric = metrics[0]
    if report.get("num") != QUESTIONS or metric.get("num") != QUESTIONS:
        raise ValueError("EvalScope评分分母不是198")
    score = metric.get("score")
    if (
        not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("EvalScope accuracy 分数无效")
    # Report may round to four decimals; reconstruct the unique integer count.
    correct = round(score * QUESTIONS)
    if abs(score - correct / QUESTIONS) > 0.000051:
        raise ValueError("分数不是198题单次0/1评分的结果")
    if (
        len(records) != QUESTIONS
        or len({r["response_id"] for r in records}) != QUESTIONS
    ):
        raise ValueError("推理计数日志不是198个唯一响应，不能作完整判定")
    a, p, n = (sum(r[k] for r in records) for k in ("A", "P", "N"))
    if p <= 0 or p != GAMMA * n:
        raise ValueError("完整运行没有有效gamma5接受计数")
    percent = 100 * correct / QUESTIONS
    accuracy_pass = 90.2 <= percent <= 92.2
    acceptance_pass = 2 * a > p  # 严格>0.5，不能比较已四舍五入的日志。
    return {
        "status": "PASS" if accuracy_pass and acceptance_pass else "FAIL",
        "questions": QUESTIONS,
        "correct": correct,
        "evalscope_score": score,
        "accuracy_percent": percent,
        "accuracy_band_percent": [90.2, 92.2],
        "accuracy_pass": accuracy_pass,
        "gamma": GAMMA,
        "A": a,
        "P": p,
        "N": n,
        "accept_rate": a / p,
        "accept_length": 1 + a / n,
        "acceptance_pass": acceptance_pass,
        "length_finished": sum(r["finish_reason"] == "length" for r in records),
        "scope": "单次GPQA-D开发自测；非性能结论，未核对91.2原始评测协议",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="只核验依赖和198题CSV，不发送请求"
    )
    args = parser.parse_args()
    version = importlib.metadata.version("evalscope")
    if version != "1.11.1":
        raise ValueError(
            f"使用已有EvalScope 1.11.1环境，当前{version}；不要更改服务Python"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from evalscope import TaskConfig, run_task
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    dataset = Path(DATASET)
    digest = check_dataset(dataset)
    if args.check_only:
        print(f"DATA_READY: EvalScope {version}, GPQA-D {QUESTIONS}题, SHA256={digest}")
        return 0
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    mode = "graph" if GRAPH else "eager"
    output = Path(tempfile.mkdtemp(prefix=f"gpqa-d-{mode}-gamma5-", dir=RESULTS))
    print(f"本轮结果目录：{output}", flush=True)
    # EvalScope原生loader需要独立数据目录；内部复制，不要求用户另行准备。
    local_data = output / "dataset"
    local_data.mkdir()
    shutil.copyfile(dataset, local_data / "train.csv")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE_URL + "/get_server_info", timeout=30) as reply:
        server = json.load(reply)
    write_json(output / "server-info.json", server)
    check_server(server)
    before_metrics = collect_graph_metrics(opener, output, "before")
    # 本客户端只连接指定内网服务，避免继承终端的外网代理。
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)
    records = []
    lock = threading.Lock()

    class CountingAPI(OpenAICompatibleAPI):
        def on_response(self, response):
            row = response_counts(response)
            with lock:
                if any(r["response_id"] == row["response_id"] for r in records):
                    raise ValueError("重复响应id；本轮不允许重试后择优")
                records.append(row)
                with (output / "inference.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"GPQA推理 {len(records)}/198: A={row['A']} P={row['P']} N={row['N']} "
                    f"accept_rate={row['accept_rate']} finish={row['finish_reason']}",
                    flush=True,
                )

    model = CountingAPI(
        model_name=MODEL,
        base_url=BASE_URL + "/v1",
        api_key="EMPTY",
        max_retries=0,
        timeout=TIMEOUT,
    )
    config = TaskConfig(
        model=model,
        model_id=MODEL,
        eval_type="openai_api",
        datasets=["gpqa_diamond"],
        dataset_args={
            "gpqa_diamond": {
                "local_path": str(local_data),
                "few_shot_num": 0,
                "shuffle": False,
            }
        },
        dataset_dir=str(output / "cache"),
        eval_batch_size=CONCURRENCY,
        repeats=1,
        limit=None,
        seed=42,
        generation_config={
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False,
            "timeout": TIMEOUT,
            "retries": 1,
            "extra_body": {"return_spec_tokens_details": True},
        },
        judge={"strategy": "rule"},
        use_cache=None,
        ignore_errors=False,
        work_dir=str(output),
        no_timestamp=True,
    )
    write_json(
        output / "run-settings.json",
        {
            "evalscope": version,
            "source_csv": str(dataset),
            "sha256": digest,
            "base_url": BASE_URL,
            "model": MODEL,
            "gamma": GAMMA,
            "mode": f"dspark-{mode}",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "concurrency": CONCURRENCY,
            "stream": False,
            "timeout": TIMEOUT,
            "repeats": 1,
            "few_shot_num": 0,
            "seed": 42,
            "accuracy_band_percent": [90.2, 92.2],
        },
    )
    try:
        reports = run_task(task_cfg=config)
        result = summarize(reports["gpqa_diamond"].model_dump(mode="json"), records)
        with opener.open(BASE_URL + "/get_server_info", timeout=30) as reply:
            after_server = json.load(reply)
        write_json(output / "server-info.after.json", after_server)
        check_server(after_server)
        after_metrics = collect_graph_metrics(opener, output, "after")
        add_graph_result(result, before_metrics, after_metrics)
        write_json(output / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"结果与推理日志：{output}", flush=True)
        return 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        write_json(
            output / "summary.json",
            {"status": "INCOMPLETE", "error": str(exc), "responses": len(records)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
