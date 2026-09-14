#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 单机GPQA-Diamond全量精度：连接68上已经启动的SGLang服务。
set -e

VENV='/home/tyj/glm52-ms1/evalscope-venv'
API_URL='http://61.47.19.68:8810/v1'
MODEL='GLM-5.2-w8a8'
RESULTS='/home/tyj/glm52-ms1/evidence/gpqa-diamond-single'
CACHE='/home/tyj/glm52-ms1/cache'
MAX_TOKENS=65536
CONCURRENCY=4

mkdir -p "$RESULTS" "$CACHE"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import importlib.metadata,sys; sys.exit(importlib.metadata.version("evalscope") != "1.11.1")' 2>/dev/null; then
  "$VENV/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple evalscope==1.11.1
fi

CONTEXT_LENGTH=$(curl -fsS 'http://61.47.19.68:8810/get_server_info' | "$VENV/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["context_length"])')
if [ "$CONTEXT_LENGTH" -lt 69632 ]; then
  echo "当前服务context_length=$CONTEXT_LENGTH，GPQA的65536输出预算要求至少69632。" >&2
  echo "请停止服务，把single_dspark_static.sh顶部CONTEXT_LENGTH改成69632后重新启动，再运行本脚本。" >&2
  exit 2
fi

export EVALSCOPE_CACHE="$CACHE/evalscope"
export MODELSCOPE_CACHE="$CACHE/modelscope"
export NO_PROXY='61.47.19.68,localhost,127.0.0.1'
export no_proxy="$NO_PROXY"

"$VENV/bin/evalscope" eval \
  --model "$MODEL" \
  --eval-type openai_api \
  --api-url "$API_URL" \
  --api-key EMPTY \
  --datasets gpqa_diamond \
  --dataset-hub modelscope \
  --eval-batch-size "$CONCURRENCY" \
  --repeats 1 \
  --seed 42 \
  --generation-config "{\"max_tokens\":$MAX_TOKENS,\"temperature\":1.0,\"timeout\":1200,\"stream\":true}" \
  --work-dir "$RESULTS" \
  --enable-progress-tracker

echo "GPQA-Diamond结果目录：$RESULTS"
