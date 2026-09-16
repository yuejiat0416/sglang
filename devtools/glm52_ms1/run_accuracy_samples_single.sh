#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 单机精度抽样：EvalScope运行GSM8K 50题和GPQA-Diamond 10题，并记录同批A/P/N。
set -eo pipefail

cd /home/tyj/glm52/sglang

VENV='/home/tyj/glm52-ms1/evalscope-venv'
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import importlib.metadata,sys; sys.exit(importlib.metadata.version("evalscope") != "1.11.1")' 2>/dev/null; then
  "$VENV/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple evalscope==1.11.1
fi

exec "$VENV/bin/python" devtools/glm52_ms1/run_accuracy_samples_single.py
