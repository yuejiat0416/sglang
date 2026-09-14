#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download and run fixed samples from the DSpark model-card benchmarks.

``download`` runs on an internet-connected PC. ``run`` runs beside an existing
single-node SGLang service and calls the unchanged serving benchmark.
"""

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench_gsm8k_modes import (
    capture_responses,
    fetch_text,
    selected_config,
    sha,
    validate_mode,
    write_json,
)
from gsm8k_mode_stats import summarize_responses
from two_node_colocated.client_common import (
    benchmark_context,
    graph_evidence,
    import_benchmark,
)

# run时只需让 MODE 和 HOST 与当前单机服务一致。
MODE = "dspark-eager"  # dspark-eager / dspark-graph / target-eager / target-graph
HOST = "61.47.19.69"
PORT = 8810
TARGET_MODEL = "/home/weights/GLM-5.2-w8a8"
DRAFT_MODEL = "/home/weights/GLM-5.2-DSpark-NPU-0805"
SERVED_MODEL_NAME = "GLM-5.2-w8a8"
STATE = Path("/home/tyj/glm52-ms1")
SERVER_BUNDLE = STATE / "datasets/glm52-dspark-modelcard-50.json"
LOCAL_BUNDLE = Path.home() / "Downloads/glm52-dspark-modelcard-50.json"
MAX_TOKENS = 1024
CONCURRENCY = 1
SEED = 42

HERE = Path(__file__).resolve().parent
DATASET_ORDER = (
    "gsm8k",
    "math500",
    "aime2025",
    "mbpp",
    "humaneval",
    "mt_bench",
    "swe_bench",
)
HF_SOURCES = {
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "aime2025": ("math-ai/aime25", "default", "test"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", "default", "train"),
    "swe_bench": ("princeton-nlp/SWE-bench", "default", "test"),
}
HF_SOURCE_ROWS = {
    "math500": 500,
    "aime2025": 30,
    "mbpp": 500,
    "humaneval": 164,
    "mt_bench": 80,
    "swe_bench": 2294,
}
MODEL_CARD = {
    "gsm8k": ([92.36, 84.33, 76.80, 69.59, 63.00, 57.05, 51.58, 46.29], 6.41),
    "math500": ([93.04, 85.32, 77.62, 70.61, 64.12, 58.10, 52.48, 47.02], 6.48),
    "aime2025": ([92.23, 83.18, 74.46, 66.56, 59.47, 53.26, 47.31, 41.70], 6.18),
    "mbpp": ([86.38, 71.80, 58.60, 47.51, 38.57, 31.23, 25.41, 20.59], 4.80),
    "humaneval": ([87.56, 73.77, 61.12, 50.62, 41.83, 34.80, 29.03, 24.37], 5.03),
    "mt_bench": ([78.39, 60.07, 45.76, 35.77, 29.04, 23.94, 20.37, 17.33], 4.11),
    "swe_bench": ([79.57, 61.44, 47.14, 36.20, 28.17, 22.16, 17.70, 14.38], 4.07),
}


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required(row, key, dataset):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dataset}: 缺少文本字段 {key}")
    return value.strip()


def _hf_rows(name, dataset, config, split):
    total = HF_SOURCE_ROWS[name]
    length = min(100, total)
    rng = random.Random(f"{SEED}:{name}")
    offset = rng.randrange(total - length + 1)
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = "https://datasets-server.huggingface.co/rows?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "glm52-dspark-ms1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                page = json.load(response)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    batch = page.get("rows")
    if not isinstance(batch, list) or len(batch) != length:
        raise ValueError(f"{dataset}: 数据服务没有返回预期的{length}行")
    if page.get("num_rows_total") != total:
        raise ValueError(
            f"{dataset}: 公开split行数从{total}变为{page.get('num_rows_total')}，"
            "请审视数据revision后再运行"
        )
    return [item["row"] for item in batch], {
        "source_rows": total,
        "download_window_offset": offset,
        "download_window_length": length,
    }


def _load_download_sources():
    gsm = [
        json.loads(line)
        for line in (HERE / "gsm8k-test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    result = {"gsm8k": gsm}
    windows = {
        "gsm8k": {
            "source_rows": len(gsm),
            "download_window_offset": 0,
            "download_window_length": len(gsm),
        }
    }
    for name in DATASET_ORDER[1:]:
        print(f"下载 {name} ...", flush=True)
        result[name], windows[name] = _hf_rows(name, *HF_SOURCES[name])
    return result, windows


def _prompt(dataset, row):
    if dataset == "gsm8k":
        text = _required(row, "question", dataset)
        return text + "\n\nSolve the problem step by step and give the final answer clearly."
    if dataset in {"math500", "aime2025"}:
        text = _required(row, "problem", dataset)
        return text + "\n\nSolve the problem step by step and give the final answer clearly."
    if dataset == "mbpp":
        text = _required(row, "text", dataset)
        tests = row.get("test_list") or []
        tests = "\n".join(str(item) for item in tests)
        return (
            "Write a correct Python solution for this task.\n\n"
            + text
            + ("\n\nPublic tests:\n" + tests if tests else "")
        )
    if dataset == "humaneval":
        return (
            "Complete the following Python function. Return a correct implementation.\n\n"
            + _required(row, "prompt", dataset)
        )
    if dataset == "mt_bench":
        prompts = row.get("prompt")
        if (
            not isinstance(prompts, list)
            or not prompts
            or not isinstance(prompts[0], str)
        ):
            raise ValueError("mt_bench: prompt必须包含至少一个turn")
        return prompts[0].strip()
    if dataset == "swe_bench":
        repo = _required(row, "repo", dataset)
        issue = _required(row, "problem_statement", dataset)
        hints = row.get("hints_text")
        included_hints = (
            f"\n\nHints:\n{hints.strip()}"
            if isinstance(hints, str) and hints.strip()
            else ""
        )
        return (
            f"Repository: {repo}\n\nIssue:\n{issue}"
            + included_hints
            + "\n\nExplain the code change you would make to resolve this issue."
        )
    raise ValueError(f"未知数据集：{dataset}")


def _case_id(dataset, row, index):
    for key in ("id", "task_id", "prompt_id", "instance_id", "unique_id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value):
            return f"{dataset}-{value}"
    return f"{dataset}-row-{index:05d}"


def build_bundle(source_rows, source_windows=None):
    rng = random.Random(SEED)
    source_windows = source_windows or {
        name: {
            "source_rows": len(rows),
            "download_window_offset": 0,
            "download_window_length": len(rows),
        }
        for name, rows in source_rows.items()
    }
    datasets = {}
    for dataset in DATASET_ORDER:
        rows = source_rows[dataset]
        requested = 50
        count = min(requested, len(rows)) if dataset == "aime2025" else requested
        if len(rows) < count or (dataset != "aime2025" and len(rows) < requested):
            raise ValueError(
                f"{dataset}: 只有{len(rows)}条，无法无放回抽样{requested}条"
            )
        indices = rng.sample(range(len(rows)), count)
        cases, seen = [], set()
        for index in indices:
            content = _prompt(dataset, rows[index])
            rid = _case_id(dataset, rows[index], index)
            if rid in seen:
                rid += f"-row-{index}"
            if rid in seen:
                raise ValueError(f"{dataset}: 重复样本ID {rid}")
            seen.add(rid)
            cases.append(
                {
                    "id": rid,
                    "source_row_index": index,
                    "prompt_sha256": _digest(content),
                    "messages": [{"role": "user", "content": content}],
                }
            )
        source = (
            {
                "dataset": "openai/grade-school-math",
                "split": "test",
                "vendored_file": "devtools/glm52_ms1/gsm8k-test.jsonl",
                "vendored_sha256": sha(HERE / "gsm8k-test.jsonl"),
            }
            if dataset == "gsm8k"
            else dict(zip(("dataset", "config", "split"), HF_SOURCES[dataset]))
        )
        datasets[dataset] = {
            "requested_samples": requested,
            "actual_samples": count,
            **source_windows[dataset],
            "selection": "fixed seed 42, without replacement inside recorded window",
            "source": source,
            "cases": cases,
        }
    return {
        "schema_version": 1,
        "status": "MODELCARD_SAMPLE_PREPARED",
        "model_card": "https://modelscope.cn/models/Eco-Tech/GLM-5.2-DSpark-NPU-0805",
        "seed": SEED,
        "datasets": datasets,
        "limits": [
            "AIME2025 has 30 public problems, so all 30 are used without duplication.",
            "For rate-limit-safe download, each non-GSM dataset uses one recorded "
            "seeded window of up to 100 source rows, then samples without replacement.",
            "The model card does not publish dataset revisions, prompts, sampling or "
            "generation parameters; this is a pinned approximate reproduction.",
            "MT-Bench uses turn 1 only. SWE-bench uses issue text without an agent "
            "repository checkout. These rows measure acceptance, not formal accuracy.",
        ],
    }


def download():
    rows, windows = _load_download_sources()
    bundle = build_bundle(rows, windows)
    LOCAL_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    write_json(LOCAL_BUNDLE, bundle)
    print(f"已生成：{LOCAL_BUNDLE}")
    for name in DATASET_ORDER:
        print(f"{name}: {bundle['datasets'][name]['actual_samples']}条")
    return 0


def _request_rows(cases, run_id):
    rows = []
    for case in cases:
        rid = f"{run_id}-{case['id']}"
        rows.append(
            {
                "messages": case["messages"],
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "top_p": 1,
                "rid": rid,
                "cache_salt": rid,
                "return_meta_info": True,
                "return_spec_tokens_details": True,
                "return_token_ids": True,
            }
        )
    return rows


def _bench_argv(run, count):
    return [
        "--backend",
        "sglang-oai-chat",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--model",
        TARGET_MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tokenizer",
        TARGET_MODEL,
        "--dataset-name",
        "openai",
        "--dataset-path",
        str(run / "requests.jsonl"),
        "--num-prompts",
        str(count),
        "--sharegpt-output-len",
        str(MAX_TOKENS),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(CONCURRENCY),
        "--seed",
        str(SEED),
        "--warmup-requests",
        "0",
        "--disable-ignore-eos",
        "--disable-stream",
        "--output-details",
        "--output-file",
        str(run / "benchmark.jsonl"),
    ]


def _histogram(response):
    choice = (response.get("choices") or [{}])[0]
    meta = choice.get("meta_info") or {}
    details = ((response.get("sglext") or {}).get("spec_tokens_details") or {})
    values = [
        source.get("spec_correct_drafts_histogram")
        for source in (meta, details)
        if source.get("spec_correct_drafts_histogram") is not None
    ]
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError("响应中的接受直方图来源互相冲突")
    value = values[0]
    if not isinstance(value, list) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError("接受直方图格式无效")
    return value


def modelcard_acceptance(responses, aggregate, dataset):
    if MODE.startswith("target-"):
        return {"applicability": "not_applicable_target_only"}
    histograms = [_histogram(response) for response in responses]
    if any(value is None for value in histograms):
        raise ValueError("至少一个响应缺少spec_correct_drafts_histogram")
    size = max(len(value) for value in histograms)
    histogram = [0] * size
    for value in histograms:
        for index, count in enumerate(value):
            histogram[index] += count
    rounds = sum(histogram)
    accepted = sum(index * count for index, count in enumerate(histogram))
    if rounds != aggregate["verify_rounds"] or accepted != aggregate["accepted_drafts"]:
        raise ValueError("接受直方图与A/N原始计数不一致")
    positions = [sum(histogram[index + 1 :]) / rounds for index in range(8)]
    reference_positions, reference_al = MODEL_CARD[dataset]
    actual_al = 1 + accepted / rounds
    return {
        "formula": "accept_length = 1 + accepted_drafts / verify_rounds",
        "accepted_drafts_histogram": histogram,
        "position_acceptance": positions,
        "position_acceptance_percent": [100 * value for value in positions],
        "accept_length": actual_al,
        "model_card_reference": {
            "position_acceptance_percent": reference_positions,
            "accept_length": reference_al,
        },
        "delta_from_model_card": {
            "position_percentage_points": [
                actual - reference
                for actual, reference in zip(
                    [100 * value for value in positions], reference_positions
                )
            ],
            "accept_length": actual_al - reference_al,
        },
    }


def _run_dataset(bench, root, dataset, data):
    run = root / dataset
    run.mkdir()
    rows = _request_rows(data["cases"], root.name + "-" + dataset)
    expected = [row["rid"] for row in rows]
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    argv = _bench_argv(run, len(rows))
    write_json(
        run / "protocol.json",
        {
            "dataset": dataset,
            "sample": {key: value for key, value in data.items() if key != "cases"},
            "case_ids": [case["id"] for case in data["cases"]],
            "mode": MODE,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "top_p": 1,
            "concurrency": CONCURRENCY,
            "eos": "honored",
            "benchmark_argv": argv,
        },
    )
    base_url = f"http://{HOST}:{PORT}"
    responses, issues = [], []
    metrics_before = metrics_after = None
    try:
        metrics_before = fetch_text(base_url + "/metrics")
        (run / "metrics.before.txt").write_text(metrics_before)
        sys.argv = ["sglang.benchmark.serving", *argv]
        with capture_responses(bench, run / "responses.jsonl", responses):
            bench.cli_main()
        result = json.loads((run / "benchmark.jsonl").read_text().splitlines()[-1])
        if result.get("completed") != len(rows):
            raise ValueError(f"只完成{result.get('completed')}/{len(rows)}条请求")
        metrics_after = fetch_text(base_url + "/metrics")
        (run / "metrics.after.txt").write_text(metrics_after)
    except (Exception, SystemExit) as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
    collected = summarize_responses(expected, responses, MODE)
    issues.extend(collected["issues"])
    card = None
    if not issues:
        try:
            card = modelcard_acceptance(responses, collected["aggregate"], dataset)
        except ValueError as exc:
            issues.append(str(exc))
    summary = {
        "status": (
            "MODELCARD_DATASET_COLLECTED"
            if not issues
            else "MODELCARD_DATASET_INCOMPLETE"
        ),
        "complete": not issues,
        "dataset": dataset,
        "samples": len(rows),
        "acceptance": collected["aggregate"] if collected["complete"] else None,
        "partial_valid_counts": collected["partial_valid_counts"],
        "model_card_comparison": card,
        "graph": graph_evidence(metrics_before, metrics_after, MODE),
        "issues": issues,
        "limits": [
            "This is acceptance collection under the recorded approximate protocol, "
            "not formal dataset accuracy.",
            "Nonstream timings are not TTFT/TPOT performance evidence.",
        ],
    }
    write_json(run / "summary.json", summary)
    rate = (summary.get("acceptance") or {}).get("accept_rate")
    length = (card or {}).get("accept_length")
    print(
        f"{dataset}: {summary['status']} "
        f"samples={len(rows)} rate={rate} AL={length}",
        flush=True,
    )
    return summary


def run():
    if MODE not in {"dspark-eager", "dspark-graph", "target-eager", "target-graph"}:
        raise ValueError(f"不支持的MODE：{MODE}")
    bundle = json.loads(SERVER_BUNDLE.read_text())
    if bundle.get("status") != "MODELCARD_SAMPLE_PREPARED":
        raise ValueError("样本文件格式不正确，请重新执行download并上传")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = STATE / "evidence" / f"modelcard-50-{MODE}-{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {root}", flush=True)
    write_json(
        root / "bundle-identity.json",
        {"path": str(SERVER_BUNDLE), "sha256": sha(SERVER_BUNDLE)},
    )
    base_url = f"http://{HOST}:{PORT}"
    before = json.loads(fetch_text(base_url + "/server_info"))
    validate_mode(before, MODE)
    if before.get("model_path") != TARGET_MODEL:
        raise ValueError("服务target路径与脚本不一致")
    if before.get("served_model_name") != SERVED_MODEL_NAME:
        raise ValueError("服务名与脚本不一致")
    if (
        MODE.startswith("dspark")
        and before.get("speculative_draft_model_path") != DRAFT_MODEL
    ):
        raise ValueError("服务draft路径与脚本不一致")
    write_json(root / "server_info.before.json", before)
    with benchmark_context():
        bench = import_benchmark()
        summaries = {
            dataset: _run_dataset(bench, root, dataset, bundle["datasets"][dataset])
            for dataset in DATASET_ORDER
        }
    after = json.loads(fetch_text(base_url + "/server_info"))
    write_json(root / "server_info.after.json", after)
    issues = []
    if selected_config(before) != selected_config(after):
        issues.append("测试期间服务配置发生变化")
    for dataset, summary in summaries.items():
        issues.extend(f"{dataset}: {item}" for item in summary["issues"])
    combined = {
        "status": (
            "MODELCARD_SUITE_COLLECTED"
            if not issues
            else "MODELCARD_SUITE_INCOMPLETE"
        ),
        "complete": not issues,
        "mode": MODE,
        "server_configuration": selected_config(before),
        "dataset_results": {
            name: {
                "samples": value["samples"],
                "acceptance": value["acceptance"],
                "model_card_comparison": value["model_card_comparison"],
            }
            for name, value in summaries.items()
        },
        "issues": issues,
        "limits": bundle["limits"],
    }
    write_json(root / "summary.json", combined)
    print(combined["status"])
    print(f"总结果：{root / 'summary.json'}")
    return 0 if not issues else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "run"))
    args = parser.parse_args()
    return download() if args.action == "download" else run()


if __name__ == "__main__":
    raise SystemExit(main())
