#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Single-node functional recipe based on the team's working single_w8a8.sh.
# Run with Bash inside the existing A3/CANN 9.1 container.
set -e

PRINT_COMMAND=0
case "${1:-}" in
  --print-command) PRINT_COMMAND=1 ;;
  --help|-h)
    cat <<'USAGE'
Usage: bash devtools/glm52_ms1/single_dspark_static.sh [--print-command]

Defaults: MODE=dspark, GRAPH=0 (static/eager), TP16/DP1 on one A3 node.
Set GRAPH=1 to use the original --cuda-graph-bs 16 setting.
Set MODE=target-only for the same recipe without speculative decoding.
Set MODE=nextn for the team's existing NEXTN recipe (4 steps, topk 1, 5 tokens).
GRAPH=0/1 also selects eager/graph for target-only and NEXTN.
Set ENABLE_METRICS=1 to expose runtime metrics (default: 0).
Optional overrides: MS1_HOST, MS1_PORT, TARGET_MODEL, DRAFT_MODEL,
                    KERNEL_REPO, MS1_STATE.
--print-command prints the environment/command without sourcing CANN or starting
the helper/server. Normal execution sources the container's CANN and ATB setup.
USAGE
    exit 0
    ;;
  "") ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  printf 'Expected at most one option. Use --help.\n' >&2
  exit 2
fi

MODE=${MODE:-dspark}
GRAPH=${GRAPH:-0}
ENABLE_METRICS=${ENABLE_METRICS:-0}
case "$MODE" in
  dspark|target-only|nextn) ;;
  *) printf 'MODE must be dspark, target-only or nextn.\n' >&2; exit 2 ;;
esac
case "$GRAPH" in
  0|1) ;;
  *) printf 'GRAPH must be 0 or 1.\n' >&2; exit 2 ;;
esac
case "$ENABLE_METRICS" in
  0|1) ;;
  *) printf 'ENABLE_METRICS must be 0 or 1.\n' >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
KERNEL_REPO=${KERNEL_REPO:-"$SGLANG_REPO/../sgl-kernel-npu"}
MS1_STATE=${MS1_STATE:-/home/tyj/glm52-ms1}
# The current container maps host /home/weights to /workspace/weight.
TARGET_MODEL=${TARGET_MODEL:-/workspace/weight/GLM-5.2-w8a8}
DRAFT_MODEL=${DRAFT_MODEL:-/workspace/weight/GLM-5.2-DSpark-NPU-0805}
MS1_HOST=${MS1_HOST:-61.47.19.71}
MS1_PORT=${MS1_PORT:-8810}

if [ "$PRINT_COMMAND" -eq 0 ]; then
  # Vendor scripts may recover from an internal nonzero command. Check their
  # final status without imposing this launcher's errexit on their internals.
  if ! source /usr/local/Ascend/ascend-toolkit/set_env.sh; then
    printf 'Failed to source /usr/local/Ascend/ascend-toolkit/set_env.sh\n' >&2
    exit 1
  fi
  if ! source /usr/local/Ascend/nnal/atb/set_env.sh; then
    printf 'Failed to source /usr/local/Ascend/nnal/atb/set_env.sh\n' >&2
    exit 1
  fi
fi

# Keep the working service's process environment; no host sysctl/CPU tuning.
# Build PYTHONPATH after sourcing the vendor scripts so their additions survive.
RUNTIME_ENV=(
  SGLANG_SET_CPU_AFFINITY=1
  STREAMS_PER_DEVICE=32
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  SGLANG_ENABLE_SPEC_V2=1
  SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  HCCL_BUFFSIZE=1000
  HCCL_OP_EXPANSION_MODE=AIV
  HCCL_SOCKET_IFNAME=lo
  GLOO_SOCKET_IFNAME=lo
  TRANSFORMERS_VERBOSITY=error
  SGLANG_NPU_PROFILING=0
  SGLANG_NPU_PROFILING_BS=16
  "PYTHONPATH=$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
  DEEPEP_NORMAL_LONG_SEQ_ROUND=72
  DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
  DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
  SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
  SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
  DEEP_NORMAL_MODE_USE_INT8_QUANT=1
  SGLANG_RAGGED_VERIFY_MODE=static
)

SERVER_ARGS=(
  --model-path "$TARGET_MODEL"
  --attention-backend ascend
  --device npu
  --tp-size 16
  --nnodes 1
  --dp-size 1
  --enable-dp-attention
  --chunked-prefill-size -1
  --max-prefill-tokens 69632
  --trust-remote-code
  --mem-fraction-static 0.85
  --served-model-name GLM-5.2-w8a8
  --max-running-requests 8
  --quantization modelslim
  --moe-a2a-backend deepep
  --deepep-mode auto
  --load-balance-method round_robin
  --host "$MS1_HOST"
  --port "$MS1_PORT"
)
if [ "$MODE" = dspark ]; then
  SERVER_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-model-quantization unquant
    --speculative-draft-attention-backend ascend
    --speculative-dspark-block-size 8
    --speculative-num-draft-tokens 9
  )
elif [ "$MODE" = nextn ]; then
  # The existing GLM NEXTN layers come from the target checkpoint.
  # Do not attach the separate DSpark draft or its proposal configuration.
  SERVER_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 4
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 5
    --speculative-draft-model-quantization unquant
  )
fi
if [ "$GRAPH" = 1 ]; then
  SERVER_ARGS+=(--cuda-graph-bs 16)
else
  SERVER_ARGS+=(--disable-cuda-graph)
fi
if [ "$ENABLE_METRICS" = 1 ]; then
  SERVER_ARGS+=(--enable-metrics)
fi

# The helper selects this checkout for this process and its spawned workers.
# It does not replace the container's installed package.
COMMAND=(
  env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY
  -u ASCEND_LAUNCH_BLOCKING
  "${RUNTIME_ENV[@]}"
  python3 "$SCRIPT_DIR/with_kernel_checkout.py"
  --kernel-repo "$KERNEL_REPO" --state-dir "$MS1_STATE"
  -- python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
)
if [ "$PRINT_COMMAND" -eq 1 ]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  exec "${COMMAND[@]}"
fi
