#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 正式双机混部部署入口。基于SGLang官方GLM-5.2 A3双机命令，只增加本项目测试所需配置。
set -eo pipefail

# 两台只让NODE_RANK不同，其余内容保持完全一致。GRAPH=0是eager，GRAPH=1是graph。
MODE='dspark'                         # dspark / target-only / nextn
GRAPH=0                              # 0=eager；1=graph
NODE_RANK=0                          # 61.47.19.68填0；61.47.19.70填1
SGLANG_REPO='/home/tyj/glm52/sglang'
KERNEL_REPO='/home/tyj/glm52/sgl-kernel-npu'
TARGET_MODEL='/home/weights/GLM-5.2-w8a8'
DRAFT_MODEL='/home/weights/GLM-5.2-DSpark-NPU-0805'
NODE0_HOST='61.47.19.68'
NODE1_HOST='61.47.19.70'
DIST_PORT=50000
PORT=8810
HCCL_SOCKET_IFNAME='enp196s0f0'
GLOO_SOCKET_IFNAME='enp196s0f0'
SERVED_MODEL_NAME='GLM-5.2-w8a8'

# 131072输入+1024输出；DP4让四个并发各落到一个DP lane。
CONTEXT_LENGTH=133120
MAX_TOTAL_TOKENS=133120
MAX_RUNNING_REQUESTS=4
MEM_FRACTION_STATIC=0.73

PRINT_COMMAND=0
case "${1:-}" in
  --print-command) PRINT_COMMAND=1 ;;
  --help|-h)
    cat <<'EOF'
Usage: bash devtools/glm52_ms1/two_node_dspark_official.sh [--print-command]
Set NODE_RANK=0 on 61.47.19.68 and NODE_RANK=1 on 61.47.19.70.
Edit MODE and GRAPH at the top; both nodes must use the same values.
--print-command prints the final server command without touching the NPU environment.
EOF
    exit 0 ;;
  "") ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
case "$MODE" in dspark|target-only|nextn) ;; *) echo 'MODE must be dspark, target-only or nextn.' >&2; exit 2 ;; esac
case "$GRAPH" in 0|1) ;; *) echo 'GRAPH must be 0 or 1.' >&2; exit 2 ;; esac
case "$NODE_RANK" in
  0) HOST="$NODE0_HOST" ;;
  1) HOST="$NODE1_HOST" ;;
  *) echo 'NODE_RANK must be 0 or 1.' >&2; exit 2 ;;
esac

SERVER_ARGS=(
  --model-path "$TARGET_MODEL"
  --attention-backend ascend
  --device npu
  --host "$HOST"
  --port "$PORT"
  --dist-init-addr "$NODE0_HOST:$DIST_PORT"
  --tp-size 32
  --nnodes 2
  --node-rank "$NODE_RANK"
  --dp-size 4
  --enable-dp-attention
  --enable-dp-lm-head
  --load-balance-method round_robin
  --context-length "$CONTEXT_LENGTH"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --chunked-prefill-size 16384
  --max-prefill-tokens 131072
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --quantization modelslim
  --moe-a2a-backend deepep
  --deepep-mode auto
  --enable-cache-report
  --enable-metrics
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
  SERVER_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant
  )
fi

if [ "$GRAPH" = 0 ]; then
  SERVER_ARGS+=(--disable-cuda-graph)
else
  SERVER_ARGS+=(--cuda-graph-bs-decode 8)
fi

if [ "$PRINT_COMMAND" = 1 ]; then
  printf 'HCCL_SOCKET_IFNAME=%q GLOO_SOCKET_IFNAME=%q ' "$HCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME"
  if [ "$MODE" = dspark ]; then
    printf 'SGLANG_RAGGED_VERIFY_MODE=static SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT=true '
  fi
  printf 'python3 -m sglang.launch_server '
  printf '%q ' "${SERVER_ARGS[@]}"
  printf '\n'
  exit 0
fi

if [ ! -d "/sys/class/net/$HCCL_SOCKET_IFNAME" ]; then
  echo "HCCL network interface does not exist: $HCCL_SOCKET_IFNAME" >&2
  exit 2
fi
if [ ! -d "/sys/class/net/$GLOO_SOCKET_IFNAME" ]; then
  echo "Gloo network interface does not exist: $GLOO_SOCKET_IFNAME" >&2
  exit 2
fi

# 官方GLM-5.2部署基线：CPU、CANN、通信与DeepEP环境。
if compgen -G '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' >/dev/null; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ASCEND_LAUNCH_BLOCKING PYTHONPATH
unset SGLANG_DSPARK_DEBUG_DUMP GLM52_CONTEXT_SNAPSHOT_CONFIG GLM52_PROPOSAL_SNAPSHOT_CONFIG
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_SIMULATE_UNIFORM_EXPERTS SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
unset SGLANG_ENABLE_SPEC_V2 SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE
unset SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES SGLANG_EXPERIMENTAL_CPP_RADIX_TREE

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export TRANSFORMERS_VERBOSITY=error
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export PYTHONPATH="$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"

if [ "$MODE" = dspark ]; then
  export SGLANG_RAGGED_VERIFY_MODE=static
  # QuaRot target + unrotated draft: use draft-local vocab and fold Q into FC.
  export SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT=true
else
  unset SGLANG_RAGGED_VERIFY_MODE SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT
fi

# 将已审定的192维Python算子注册到当前测试容器；镜像内其余kernel依赖保持不变。
python3 "$SGLANG_REPO/devtools/glm52_ms1/register_glm52_dspark_kernel.py" \
  --kernel-repo "$KERNEL_REPO"

exec python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
