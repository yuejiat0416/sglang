#!/usr/bin/env python3
"""Run small GSM8K and GPQA-Diamond EvalScope evaluations with SGLang A/P/N."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import threading
import urllib.request
import uuid


HERE = Path(__file__).resolve().parent
BASE_URL = "http://61.47.19.68:8810"
MODEL = "GLM-5.2-w8a8"
GSM8K_SOURCE = HERE / "gsm8k-test.jsonl"
GPQA_SOURCE = Path("/home/tyj/glm52-ms1/datasets/gpqa_diamond.csv")
RESULTS = Path("/home/tyj/glm52-ms1/evidence/accuracy-samples-single")
EVALSCOPE_VERSION = "1.11.1"
GSM8K_ROWS = 1319
GPQA_ROWS = 198

DATASETS = (
    {
        "name": "gsm8k",
        "limit": 50,
        "source": GSM8K_SOURCE,
        "local_name": "test.jsonl",
        "subset_list": ["default"],
        "few_shot_num": 0,
        "concurrency": 4,
        "max_tokens": 4096,
        "temperature": 0.0,
        "timeout": 1800,
    },
    {
        "name": "gpqa_diamond",
        "limit": 10,
        "source": GPQA_SOURCE,
        "local_name": "train.csv",
        "subset_list": ["default"],
        "few_shot_num": 0,
        "concurrency": 1,
        "max_tokens": 65536,
        "temperature": 0.0,
        "timeout": 7200,
    },
)

GPQA_FIELDS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gsm8k(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"仓库缺少GSM8K test文件：{path}")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"GSM8K第{number}行不是有效JSON") from exc
        if not all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("question", "answer")
        ):
            raise ValueError(f"GSM8K第{number}行缺question或answer")
        rows.append(row)
    if len(rows) != GSM8K_ROWS:
        raise ValueError(f"GSM8K main/test应为{GSM8K_ROWS}题，实际{len(rows)}")
    if len({row["question"] for row in rows}) != GSM8K_ROWS:
        raise ValueError("GSM8K main/test存在重复题目")
    return {"rows": len(rows), "sha256": digest(path)}


def validate_gpqa(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少本地GPQA-Diamond：{path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != GPQA_ROWS:
        raise ValueError(f"GPQA-Diamond应为{GPQA_ROWS}题，实际{len(rows)}")
    if any(not row.get(key, "").strip() for row in rows for key in GPQA_FIELDS):
        raise ValueError("GPQA-Diamond缺题目或答案字段")
    if len({row["Question"].strip() for row in rows}) != GPQA_ROWS:
        raise ValueError("GPQA-Diamond存在重复题目")
    return {"rows": len(rows), "sha256": digest(path)}


def get_server_info(opener) -> dict:
    with opener.open(BASE_URL + "/get_server_info", timeout=30) as reply:
        return json.load(reply)


def check_server(info: dict) -> dict:
    if info.get("device") != "npu":
        raise ValueError(f"服务device={info.get('device')!r}，本轮要求npu")
    if info.get("nnodes") != 1 or info.get("tp_size") != 16:
        raise ValueError(
            f"本轮使用正式单机TP16服务；实际nnodes={info.get('nnodes')}, "
            f"tp_size={info.get('tp_size')}"
        )
    served = info.get("served_model_name")
    if served != MODEL:
        raise ValueError(f"服务名={served!r}，本轮要求{MODEL!r}")
    if info.get("context_length", 0) < 69632:
        raise ValueError(
            f"服务context_length={info.get('context_length')}，"
            "GPQA的65536输出预算要求69632"
        )
    algorithm = info.get("speculative_algorithm")
    speculative = bool(algorithm and str(algorithm).upper() not in ("NONE", "NULL"))
    draft_tokens = info.get("speculative_num_draft_tokens")
    gamma = draft_tokens - 1 if speculative and isinstance(draft_tokens, int) else None
    if speculative and (not isinstance(gamma, int) or gamma <= 0):
        raise ValueError(f"无法从服务配置核对投机窗口：draft_tokens={draft_tokens!r}")
    backend = ((info.get("cuda_graph_config") or {}).get("decode") or {}).get("backend")
    graph = backend not in (None, "disabled") and not info.get("disable_cuda_graph", False)
    return {
        "algorithm": algorithm,
        "speculative": speculative,
        "gamma": gamma,
        "graph": graph,
        "mode": (
            f"{str(algorithm).lower()}-{'graph' if graph else 'eager'}"
            if speculative
            else f"target-{'graph' if graph else 'eager'}"
        ),
    }


def response_record(response: dict, server_mode: dict) -> dict:
    choices = response.get("choices") or []
    if not response.get("id") or len(choices) != 1:
        raise ValueError("每题必须返回一个带id的OpenAI响应")
    finish = choices[0].get("finish_reason")
    if finish not in ("stop", "length"):
        raise ValueError(f"请求未正常完成：finish_reason={finish!r}")
    row = {
        "response_id": response["id"],
        "finish_reason": finish,
        "usage": response.get("usage"),
    }
    details = (response.get("sglext") or {}).get("spec_tokens_details")
    if not server_mode["speculative"]:
        row.update({"A": None, "P": None, "N": None, "accept_rate": None})
        return row
    if not isinstance(details, dict):
        raise ValueError("投机服务响应缺少sglext.spec_tokens_details")
    a, p, n = (
        details.get(key)
        for key in (
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
            "spec_verify_ct",
        )
    )
    if any(type(value) is not int or value < 0 for value in (a, p, n)):
        raise ValueError("服务端A/P/N必须是非负整数")
    if a > p or p != server_mode["gamma"] * n:
        raise ValueError(
            f"A/P/N与服务窗口不一致：A={a}, P={p}, N={n}, gamma={server_mode['gamma']}"
        )
    row.update(
        {
            "A": a,
            "P": p,
            "N": n,
            "accept_rate": a / p if p else None,
            "mean_accepted_drafts_per_verify": a / n if n else None,
            "server_spec_tokens_details": details,
        }
    )
    return row


def report_dict(report) -> dict:
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    if isinstance(report, dict):
        return report
    raise TypeError(f"未知EvalScope报告类型：{type(report)!r}")


def summarize_report(
    name: str,
    report: dict,
    records: list[dict],
    limit: int,
    server_mode: dict,
) -> dict:
    execution = report.get("execution_summary") or {}
    expected_execution = {
        "requested": limit,
        "succeeded": limit,
        "errored": 0,
        "incomplete": False,
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError(f"{name}没有完成{limit}题：{execution}")
    if report.get("num") != limit:
        raise ValueError(f"{name}评分分母={report.get('num')}，要求{limit}")
    identity = report.get("primary_metric_identity") or {}
    metrics = [item for item in report.get("metrics", []) if item.get("identity") == identity]
    if len(metrics) != 1 or metrics[0].get("num") != limit:
        raise ValueError(f"{name}没有唯一且完整的主指标：{identity}")
    score = metrics[0].get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f"{name}主指标不是有限数值：{score!r}")
    if len(records) != limit or len({row["response_id"] for row in records}) != limit:
        raise ValueError(f"{name}没有{limit}个唯一API响应")
    result = {
        "status": "EVAL_COLLECTED",
        "dataset": name,
        "questions": limit,
        "primary_metric": {"identity": identity, "score": score, "num": limit},
        "approximate_correct": round(score * limit) if 0 <= score <= 1 else None,
        "finish_reasons": {
            "stop": sum(row["finish_reason"] == "stop" for row in records),
            "length": sum(row["finish_reason"] == "length" for row in records),
        },
        "mode": server_mode["mode"],
        "quality_gate": "NOT_ASSIGNED_SMALL_SAMPLE",
    }
    if server_mode["speculative"]:
        a, p, n = (sum(row[key] for row in records) for key in ("A", "P", "N"))
        if p <= 0 or p != server_mode["gamma"] * n:
            raise ValueError(f"{name}没有有效的完整接受计数")
        result["acceptance"] = {
            "applicability": "speculative",
            "algorithm": server_mode["algorithm"],
            "gamma": server_mode["gamma"],
            "accepted_drafts": a,
            "proposed_drafts": p,
            "verify_rounds": n,
            "accept_rate": a / p,
            "mean_accepted_drafts_per_verify": a / n,
            "one_plus_mean_accepted_drafts_per_verify": 1 + a / n,
            "strictly_above_0_5": 2 * a > p,
            "scope": "同一批EvalScope请求的精确A/P/N；不是并发压测准出结论",
        }
    else:
        result["acceptance"] = {
            "applicability": "not_speculative",
            "accept_rate": None,
        }
    return result


def run_dataset(
    TaskConfig,
    run_task,
    OpenAICompatibleAPI,
    spec: dict,
    root: Path,
    server_mode: dict,
) -> dict:
    name = spec["name"]
    output = root / name
    output.mkdir()
    local_data = output / "dataset"
    local_data.mkdir()
    shutil.copyfile(spec["source"], local_data / spec["local_name"])
    records: list[dict] = []
    lock = threading.Lock()

    class CountingAPI(OpenAICompatibleAPI):
        def on_response(self, response):
            row = response_record(response, server_mode)
            with lock:
                if any(item["response_id"] == row["response_id"] for item in records):
                    raise ValueError("出现重复响应id；不允许重试后择优")
                records.append(row)
                with (output / "acceptance.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                acceptance = (
                    f"A/P/N={row['A']}/{row['P']}/{row['N']}"
                    if server_mode["speculative"]
                    else "acceptance=N/A"
                )
                print(
                    f"{name} {len(records)}/{spec['limit']}: {acceptance}, "
                    f"finish={row['finish_reason']}",
                    flush=True,
                )

    model = CountingAPI(
        model_name=MODEL,
        base_url=BASE_URL + "/v1",
        api_key="EMPTY",
        max_retries=0,
        timeout=spec["timeout"],
    )
    config = TaskConfig(
        model=model,
        model_id=MODEL,
        eval_type="openai_api",
        datasets=[name],
        dataset_args={
            name: {
                "local_path": str(local_data),
                "subset_list": spec["subset_list"],
                "few_shot_num": spec["few_shot_num"],
                "few_shot_random": False,
                "shuffle": False,
            }
        },
        dataset_dir=str(output / "cache"),
        eval_batch_size=spec["concurrency"],
        repeats=1,
        limit=spec["limit"],
        seed=42,
        generation_config={
            "max_tokens": spec["max_tokens"],
            "temperature": spec["temperature"],
            "stream": False,
            "timeout": spec["timeout"],
            "retries": 0,
            "extra_body": {"return_spec_tokens_details": True},
        },
        judge={"strategy": "rule"},
        use_cache=None,
        ignore_errors=False,
        work_dir=str(output),
        no_timestamp=True,
    )
    reports = run_task(task_cfg=config)
    report = report_dict(reports[name])
    write_json(output / "evalscope-report.json", report)
    summary = summarize_report(name, report, records, spec["limit"], server_mode)
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    version = importlib.metadata.version("evalscope")
    if version != EVALSCOPE_VERSION:
        raise ValueError(f"要求EvalScope {EVALSCOPE_VERSION}，当前{version}")
    sources = {
        "gsm8k": validate_gsm8k(GSM8K_SOURCE),
        "gpqa_diamond": validate_gpqa(GPQA_SOURCE),
    }
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from evalscope import TaskConfig, run_task
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server_before = get_server_info(opener)
    server_mode = check_server(server_before)
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    root = RESULTS / f"{run_id}-{server_mode['mode']}"
    root.mkdir()
    write_json(root / "server-info.before.json", server_before)
    write_json(
        root / "run-settings.json",
        {
            "evalscope": version,
            "base_url": BASE_URL,
            "model": MODEL,
            "server_mode": server_mode,
            "sources": sources,
            "datasets": [
                {**spec, "source": str(spec["source"])} for spec in DATASETS
            ],
            "selection": (
                "Each EvalScope task uses --limit semantics: first N rows, "
                "seed 42, no shuffle."
            ),
        },
    )
    print(f"Evidence: {root}", flush=True)
    summaries = []
    try:
        for spec in DATASETS:
            print(
                f"开始{name_label(spec['name'])}: {spec['limit']}题, "
                f"并发{spec['concurrency']}, max_tokens={spec['max_tokens']}",
                flush=True,
            )
            summaries.append(
                run_dataset(
                    TaskConfig,
                    run_task,
                    OpenAICompatibleAPI,
                    spec,
                    root,
                    server_mode,
                )
            )
        server_after = get_server_info(opener)
        check_server(server_after)
        write_json(root / "server-info.after.json", server_after)
        overall = {
            "status": "ACCURACY_SAMPLES_COLLECTED",
            "mode": server_mode["mode"],
            "datasets": summaries,
            "comparison": (
                "Run the same command against target-only with unchanged data and "
                "generation settings to assess accuracy regression."
            ),
        }
        write_json(root / "summary.json", overall)
        for item in summaries:
            acceptance = item["acceptance"]
            rate = acceptance.get("accept_rate")
            print(
                f"{item['dataset']}: metric={item['primary_metric']['score']}, "
                f"约{item['approximate_correct']}/{item['questions']}, "
                + (f"accept_rate={rate:.6%}" if rate is not None else "acceptance=N/A"),
                flush=True,
            )
        print("ACCURACY_SAMPLES_COLLECTED", flush=True)
        print(f"结果：{root}", flush=True)
        return 0
    except Exception as exc:
        write_json(
            root / "summary.json",
            {
                "status": "ACCURACY_SAMPLES_INCOMPLETE",
                "mode": server_mode["mode"],
                "completed": summaries,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def name_label(name: str) -> str:
    return "GSM8K" if name == "gsm8k" else "GPQA-Diamond"


if __name__ == "__main__":
    raise SystemExit(main())
