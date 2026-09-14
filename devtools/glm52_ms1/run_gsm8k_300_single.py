#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run 300 pinned GSM8K test rows against an existing single-node server."""

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "two_node_colocated"))

import bench_accuracy  # noqa: E402
from offline_dataset import prepare_dataset  # noqa: E402

# 只需让 MODE 和 HOST 与当前单机服务一致。
MODE = "dspark-eager"  # dspark-eager / dspark-graph / target-eager / target-graph
HOST = "61.47.19.69"
PORT = 8810
TARGET_MODEL = "/home/weights/GLM-5.2-w8a8"
DRAFT_MODEL = "/home/weights/GLM-5.2-DSpark-NPU-0805"
SERVED_MODEL_NAME = "GLM-5.2-w8a8"
STATE = Path("/home/tyj/glm52-ms1")
COUNT = 300
MAX_TOKENS = 4096
CONCURRENCY = 1

SOURCE = HERE / "gsm8k-test.jsonl"


def single_node_config():
    return {
        "state": str(STATE),
        "evidence_scope": "single-node-accuracy",
        "nodes": [{"rank": 0, "host": HOST}],
        "base_url": f"http://{HOST}:{PORT}",
        "port": PORT,
        "nnodes": 1,
        "tp_size": 16,
        "dp_size": 1,
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "tokenizer": TARGET_MODEL,
        "served_model_name": SERVED_MODEL_NAME,
    }


def prepare_fixture():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"仓库缺少完整GSM8K test文件：{SOURCE}")
    fixture = prepare_dataset("gsm8k", SOURCE, limit=COUNT, seed=42)
    if fixture["status"] != "DATASET_PREPARED":
        raise ValueError("GSM8K 300题准备失败：" + "; ".join(fixture["issues"]))
    output = STATE / "datasets" / f"gsm8k-{COUNT}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    bench_accuracy.write_json(output, fixture)
    return output


def main():
    fixture = prepare_fixture()
    print(
        f"GSM8K：固定取官方test前{COUNT}题；样本：{fixture}",
        flush=True,
    )
    args = SimpleNamespace(
        config=None,
        fixture=fixture,
        limit=0,
        mode=MODE,
        max_tokens=MAX_TOKENS,
        concurrency=CONCURRENCY,
        repeats=1,
    )
    return bench_accuracy.run(args, cfg=single_node_config())


if __name__ == "__main__":
    raise SystemExit(main())
