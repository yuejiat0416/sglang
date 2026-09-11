#!/usr/bin/env python3
"""双机客户端：改顶部设置，执行 prepare / accuracy / check / quick / performance / report。

不启动或停止服务。依次切换两台服务的 MODE/GRAPH，再修改本文件 MODE。
命令与结果判读统一见上一层 README.md 第9至12节。
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# 直接编辑这里；不需要另写JSON、Shell变量或长命令。
MODE = "dspark-eager"  # dspark-eager / dspark-graph / target-eager / target-graph / nextn-graph
HOST = "61.47.19.68"  # 节点0的服务地址；客户端不连接节点1
PORT = 8810
TARGET_MODEL = "/home/weights/GLM-5.2-w8a8"
DRAFT_MODEL = "/home/weights/GLM-5.2-DSpark-NPU-0805"
SERVED_MODEL_NAME = "GLM-5.2-w8a8"
DATASETS = "/home/tyj/glm52-ms1/datasets"
RESULTS = "/home/tyj/glm52-ms1/dual-node-validation-20260910"  # 同一轮五组使用相同目录
TP_SIZE = 32
DP_SIZE = 4
ACCURACY_MAX_TOKENS = 4096  # 两数据集、所有模式保持一致；截断不记正确
GSM8K_LIMIT = 10  # 10=当前十题；100=前100题；1319=官方test全量
# 超过10题时填完整路径，如 /home/tyj/glm52-ms1/datasets/gsm8k-test.jsonl
GSM8K_SOURCE = ""
PERFORMANCE_REQUESTS = 64  # 每种共享前缀比例64条；实际缓存命中看原生cache report
PERFORMANCE_CONCURRENCY = 4  # 同时最多4条，按服务本身的路由调度


def settings():
    from config import MODES

    if MODE not in MODES:
        raise ValueError("请修改顶部 MODE 为注释中列出的五个值之一")
    if not HOST or not 1 <= PORT <= 65535:
        raise ValueError("HOST / PORT 无效")
    for name, value in (
        ("TP_SIZE", TP_SIZE),
        ("DP_SIZE", DP_SIZE),
        ("ACCURACY_MAX_TOKENS", ACCURACY_MAX_TOKENS),
        ("PERFORMANCE_REQUESTS", PERFORMANCE_REQUESTS),
        ("PERFORMANCE_CONCURRENCY", PERFORMANCE_CONCURRENCY),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    for value in (TARGET_MODEL, DRAFT_MODEL, DATASETS, RESULTS):
        if not Path(value).is_absolute():
            raise ValueError(f"请使用绝对路径：{value}")
    return {
        "nodes": [{"host": HOST}],
        "base_url": f"http://{HOST}:{PORT}",
        "port": PORT,
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "tokenizer": TARGET_MODEL,
        "served_model_name": SERVED_MODEL_NAME,
        "tp_size": TP_SIZE,
        "dp_size": DP_SIZE,
        "state": RESULTS,
    }


def dataset_settings(dataset):
    directory = Path(DATASETS)
    if dataset == "gsm8k":
        count = GSM8K_LIMIT
        if type(count) is not int or count <= 0:
            raise ValueError("GSM8K_LIMIT 必须是正整数，例如10、100或1319")
        source = Path(GSM8K_SOURCE) if GSM8K_SOURCE else None
        if source is None and count > 10:
            raise ValueError(
                "仓库只自带10题；请将完整test.jsonl传到节点0，"
                "并在顶部GSM8K_SOURCE填写本地文件的绝对路径"
            )
        if source is not None and not source.is_absolute():
            raise ValueError("GSM8K_SOURCE 请填写本地文件的绝对路径")
    else:
        count, source = 10, directory / "gpqa_diamond.csv"
    return source, count, directory / f"{dataset}-{count}.json"


def prepare(datasets=("gsm8k", "gpqa")):
    from client_common import write_json
    from offline_dataset import prepare_dataset

    directory = Path(DATASETS)
    directory.mkdir(parents=True, exist_ok=True)
    failed = False
    for dataset in datasets:
        source, count, destination = dataset_settings(dataset)
        fixture = prepare_dataset(dataset, source, limit=count, seed=42)
        if fixture["status"] != "DATASET_PREPARED":
            print(dataset, fixture["issues"])
            failed = True
            continue
        if destination.exists() and json.loads(destination.read_text()) != fixture:
            raise ValueError(
                f"已有样本与本次来源不同：{destination}；先保留旧文件并使用新的测试目录"
            )
        write_json(destination, fixture)
        print(f"{dataset}: 固定{count}题 -> {destination}")
    return int(failed)


def accuracy(cfg, datasets=("gsm8k", "gpqa")):
    from bench_accuracy import read_fixture, run

    # 自动准备所选样本并核对来源；不要求用户手工生成中间文件。
    if prepare(datasets):
        return 2
    fixtures = [dataset_settings(name) for name in datasets]
    for _, count, path in fixtures:
        if len(read_fixture(path)["cases"]) != count:
            raise ValueError(f"样本数量与本轮要求{count}题不符：{path}")
    for _, count, path in fixtures:
        args = SimpleNamespace(
            mode=MODE,
            fixture=path,
            limit=count,
            max_tokens=ACCURACY_MAX_TOKENS,
            repeats=1,
            concurrency=1,
        )
        result = run(args, cfg=cfg)
        if result:
            return result
    return 0


def native_gsp(cfg, action):
    """Run the unmodified community CLI; no custom prompts, routing or hooks."""
    from client_common import REPO, prepare_run

    run = prepare_run(cfg, "gsp-native", MODE)
    # Two rounds through the configured DP workers, without pinning requests.
    count = 2 * DP_SIZE if action == "quick" else PERFORMANCE_REQUESTS
    concurrency = 1 if action == "quick" else PERFORMANCE_CONCURRENCY
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_PROXY="*",
        no_proxy="*",
        PYTHONUNBUFFERED="1",
    )
    print(
        f"原生GSP：每档{count}条，并发{concurrency}；每档开始会清理服务缓存。",
        flush=True,
    )
    print(
        "0/50/90是共享前缀比例；实际命中看cache_report，Accept length不是A/P接受率。",
        flush=True,
    )
    for percent, prefix in ((0, 0), (50, 65536), (90, 117964)):
        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--model",
            TARGET_MODEL,
            "--tokenizer",
            TARGET_MODEL,
            "--dataset-name",
            "generated-shared-prefix",
            "--gsp-num-groups",
            "1",
            "--gsp-prompts-per-group",
            str(count),
            "--gsp-system-prompt-len",
            str(prefix),
            "--gsp-question-len",
            str(131072 - prefix),
            "--gsp-output-len",
            "1024",
            "--gsp-range-ratio",
            "1",
            "--num-prompts",
            str(count),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            "inf",
            "--seed",
            "42",
            "--temperature",
            "0",
            "--warmup-requests",
            "0",
            "--flush-cache",
            "--cache-report",
            "--output-details",
            "--output-file",
            str(run / f"prefix{percent}.jsonl"),
        ]
        rendered = shlex.join(command)
        (run / f"prefix{percent}.command.txt").write_text(rendered + "\n")
        print(rendered, flush=True)
        with (run / f"prefix{percent}.log").open("w") as log:
            with subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    raise
        if code:
            print(f"原生benchmark退出码{code}；停止后续档位。结果：{run}", flush=True)
            return code
        # Native CLI can exit zero even if requests fail. Do not run the next
        # long case after incomplete generation; retain the original output.
        result = json.loads(
            (run / f"prefix{percent}.jsonl").read_text().splitlines()[-1]
        )
        if (
            result.get("completed") != count
            or len(result.get("output_lens", [])) != count
            or any(n != 1024 for n in result["output_lens"])
            or any(result.get("errors", []))
        ):
            print(
                f"prefix{percent}存在失败或不足1024的输出；停止。结果：{run}",
                flush=True,
            )
            return 1
    print(f"原生GSP三档完成。结果：{run}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "accuracy", "check", "quick", "performance", "report"),
    )
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "gpqa"),
        help="prepare / accuracy只处理指定数据集；不填则分别处理两份",
    )
    args = parser.parse_args(argv)
    if args.dataset and args.action not in ("prepare", "accuracy"):
        parser.error("--dataset 只用于 prepare 或 accuracy")
    datasets = (args.dataset,) if args.dataset else ("gsm8k", "gpqa")
    try:
        cfg = settings()
        if args.action == "prepare":
            return prepare(datasets)
        if args.action == "report":
            from report import campaign_report

            return campaign_report(Path(RESULTS))
        print(f"当前：{MODE}；节点0：{cfg['base_url']}；结果：{RESULTS}", flush=True)
        if args.action == "accuracy":
            return accuracy(cfg, datasets)
        if args.action in ("quick", "performance"):
            return native_gsp(cfg, args.action)
        from bench_prefix import run

        arguments = SimpleNamespace(
            action="check",
            mode=MODE,
            cache_hit="all",
            seed=42,
            duration_seconds=None,
            num_prompts=PERFORMANCE_REQUESTS,
            concurrency=PERFORMANCE_CONCURRENCY,
        )
        return run(arguments, config=cfg)
    except (ValueError, OSError) as exc:
        print(f"未完成：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
