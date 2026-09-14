#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 单机GPQA-Diamond全量精度：连接68上已经启动的SGLang服务。
set -e

VENV='/home/tyj/glm52-ms1/evalscope-venv'
API_URL='http://61.47.19.68:8810/v1'
MODEL='GLM-5.2-w8a8'
RESULTS='/home/tyj/glm52-ms1/evidence/gpqa-diamond-single'
CACHE='/home/tyj/glm52-ms1/cache'
DATASET='/home/tyj/glm52-ms1/datasets/gpqa_diamond.csv'
DATASET_ZIP='/home/tyj/glm52-ms1/datasets/gpqa-dataset.zip'
DATASET_URL='https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip'
DATASET_SHA256='41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305'
MAX_TOKENS=65536
CONCURRENCY=4

mkdir -p "$RESULTS" "$CACHE" "$(dirname "$DATASET")"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import importlib.metadata,sys; sys.exit(importlib.metadata.version("evalscope") != "1.11.1")' 2>/dev/null; then
  "$VENV/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple evalscope==1.11.1
fi

if [ ! -f "$DATASET" ]; then
  echo "本机没有GPQA-Diamond，正在从作者官方仓库下载。"
  if ! curl -fL --retry 3 "$DATASET_URL" -o "$DATASET_ZIP"; then
    echo "内网HTTPS证书校验失败；仅对这个固定的官方数据地址使用curl -k重试。"
    curl -k -fL --retry 3 "$DATASET_URL" -o "$DATASET_ZIP"
  fi
  DATASET_ZIP="$DATASET_ZIP" DATASET="$DATASET" DATASET_SHA256="$DATASET_SHA256" "$VENV/bin/python" - <<'PY'
import csv
import hashlib
import os
import zipfile
from pathlib import Path

archive_path = Path(os.environ["DATASET_ZIP"])
dataset_path = Path(os.environ["DATASET"])
required = {
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
}
with zipfile.ZipFile(archive_path) as archive:
    names = [name for name in archive.namelist() if Path(name).name == "gpqa_diamond.csv"]
    if len(names) != 1:
        raise SystemExit("官方压缩包中没有唯一的gpqa_diamond.csv")
    content = archive.read(names[0], pwd=b"deserted-untie-orchid")
dataset_path.write_bytes(content)
digest = hashlib.sha256(content).hexdigest()
with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
    fields = set(rows[0]) if rows else set()
if digest != os.environ["DATASET_SHA256"] or len(rows) != 198 or not required.issubset(fields):
    dataset_path.unlink(missing_ok=True)
    raise SystemExit(f"GPQA-Diamond数据校验失败：SHA256={digest}, rows={len(rows)}, fields={sorted(fields)}")
print(f"GPQA-Diamond已准备：198题，SHA256={digest}")
PY
fi

LOCAL_DATASET="$CACHE/gpqa-diamond-local"
mkdir -p "$LOCAL_DATASET"
cp "$DATASET" "$LOCAL_DATASET/train.csv"
DATASET="$DATASET" DATASET_SHA256="$DATASET_SHA256" "$VENV/bin/python" - <<'PY'
import csv
import hashlib
import os
from pathlib import Path

path = Path(os.environ["DATASET"])
required = {
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
}
with path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
    fields = set(rows[0]) if rows else set()
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != os.environ["DATASET_SHA256"] or len(rows) != 198 or not required.issubset(fields):
    raise SystemExit(f"本地GPQA-Diamond无效：SHA256={digest}, rows={len(rows)}, fields={sorted(fields)}")
print(f"使用本地GPQA-Diamond：198题，SHA256={digest}")
PY

CONTEXT_LENGTH=$(curl -fsS 'http://61.47.19.68:8810/get_server_info' | "$VENV/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["context_length"])')
if [ "$CONTEXT_LENGTH" -lt 69632 ]; then
  echo "当前服务context_length=$CONTEXT_LENGTH，GPQA的65536输出预算要求至少69632。" >&2
  echo "请停止服务，把single_dspark_static.sh顶部CONTEXT_LENGTH改成69632后重新启动，再运行本脚本。" >&2
  exit 2
fi

export EVALSCOPE_CACHE="$CACHE/evalscope"
export MODELSCOPE_CACHE="$CACHE/modelscope"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_PROXY='61.47.19.68,localhost,127.0.0.1'
export no_proxy="$NO_PROXY"

"$VENV/bin/evalscope" eval \
  --model "$MODEL" \
  --eval-type openai_api \
  --api-url "$API_URL" \
  --api-key EMPTY \
  --datasets gpqa_diamond \
  --dataset-args "{\"gpqa_diamond\":{\"local_path\":\"$LOCAL_DATASET\",\"few_shot_num\":0,\"shuffle\":false}}" \
  --eval-batch-size "$CONCURRENCY" \
  --repeats 1 \
  --seed 42 \
  --generation-config "{\"max_tokens\":$MAX_TOKENS,\"temperature\":1.0,\"timeout\":1200,\"stream\":true}" \
  --work-dir "$RESULTS" \
  --enable-progress-tracker

echo "GPQA-Diamond结果目录：$RESULTS"
