#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 单机服务启动后，在另一个容器终端运行本脚本。
set -e

# 只需让 MODE 与正在运行的服务保持一致。
MODE='dspark-eager' # dspark-eager / dspark-graph / target-eager / target-graph / nextn-graph
HOST='61.47.19.69'
PORT=8810
TARGET_MODEL='/home/weights/GLM-5.2-w8a8'
DRAFT_MODEL='/home/weights/GLM-5.2-DSpark-NPU-0805'
SERVED_MODEL_NAME='GLM-5.2-w8a8'
STATE='/home/tyj/glm52-ms1'
MAX_TOKENS=1024
SGLANG_REPO='/home/tyj/glm52/sglang'

COMMAND=(
  python3 devtools/glm52_ms1/bench_gsm8k_modes.py run "$MODE"
  --host "$HOST" --port "$PORT"
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --state "$STATE" --max-tokens "$MAX_TOKENS"
)

if [ "${1:-}" = '--print-command' ]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
elif [ "$#" -eq 0 ]; then
  cd "$SGLANG_REPO"
  exec "${COMMAND[@]}"
else
  printf 'Usage: bash devtools/glm52_ms1/run_gsm8k_single.sh [--print-command]\n' >&2
  exit 2
fi
