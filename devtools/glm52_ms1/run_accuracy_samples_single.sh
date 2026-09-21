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
  "$VENV/bin/python" -m pip install \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    'https://mirrors.aliyun.com/pypi/packages/33/19/4915c3012c2245fe8dfd84d2d5af2378f198f6931d9d9677fd59908d642e/evalscope-1.11.1-py3-none-any.whl#sha256=5058c5112ee0dfff0048a48ee1ff9c6e1ca28bbed39ef5cd34a6a14d3ddf2b66'
fi

exec "$VENV/bin/python" devtools/glm52_ms1/run_accuracy_samples_single.py
