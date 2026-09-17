#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 在已建好的双机 A3/CANN 9.1 容器内运行；先修改下面的设置。
set -e

# 旧GSM8K工具显式启用的兼容段；直接bash运行时忽略旧终端里的部署变量。
_LAUNCH_OVERRIDES=()
if [ "${GLM52_LEGACY_LAUNCH:-0}" = 1 ]; then
  for key in MODE GRAPH NODE_RANK SGLANG_REPO KERNEL_REPO TARGET_MODEL DRAFT_MODEL MS1_STATE MS1_HOST MS1_PORT SERVED_MODEL_NAME CONTEXT_LENGTH MAX_TOTAL_TOKENS ENABLE_METRICS NODE0_HOST NODE1_HOST DIST_PORT HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
    [ "${!key+x}" ] && _LAUNCH_OVERRIDES+=("$key=${!key}")
  done
fi

# 只需编辑以下配置，然后直接 bash 本脚本。
MODE='dspark'                         # dspark / target-only / nextn
GRAPH=0                              # 0=eager；1=graph（所有 MODE 都适用）
NODE_RANK=0                          # 71机器填0，70机器填1；两机MODE/GRAPH必须相同
SGLANG_REPO='/home/tyj/glm52/sglang'
KERNEL_REPO='/home/tyj/glm52/sgl-kernel-npu'
TARGET_MODEL='/home/weights/GLM-5.2-w8a8'
DRAFT_MODEL='/home/weights/GLM-5.2-DSpark-NPU-0805'
MS1_STATE='/home/tyj/glm52-ms1'
NODE0_HOST='61.47.19.71'
NODE1_HOST='61.47.19.70'
DIST_PORT=50000
HCCL_SOCKET_IFNAME='' # 留空：自动取去对端的路由网卡；也可填写真实网卡
GLOO_SOCKET_IFNAME='' # 留空同上；双机不能填lo
MS1_PORT=8810
SERVED_MODEL_NAME='GLM-5.2-w8a8'
CONTEXT_LENGTH=133120 # 128k/1k还需核验实际KV容量；短题可改16384
MAX_TOTAL_TOKENS=''  # 留空让runtime按显存分配；不是context长度
ENABLE_METRICS=1     # 1=记录压测需要的服务指标

for setting in "${_LAUNCH_OVERRIDES[@]}"; do export "$setting"; done
PRINT_COMMAND=0
case "${1:-}" in
  --print-command) PRINT_COMMAND=1 ;;
  --help|-h)
    cat <<'USAGE'
Usage: bash devtools/glm52_ms1/two_node_dspark_static.sh [--print-command]
Edit MODE=dspark / MODE=target-only / MODE=nextn and GRAPH=0/1 at the top.
Edit NODE_RANK=0 on node0 and NODE_RANK=1 on node1; run once in each container.
Defaults: two nodes, TP32/DP8, static eager, CONTEXT_LENGTH=133120.
ENABLE_METRICS=1 exposes service metrics; model and host paths are editable above.
--print-command does not source CANN or start a server; empty NIC settings read the local route.
USAGE
    exit 0 ;;
  "") ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { printf 'Expected at most one option. Use --help.\n' >&2; exit 2; }
case "$MODE" in dspark|target-only|nextn) ;; *) printf 'MODE must be dspark, target-only or nextn.\n' >&2; exit 2 ;; esac
case "$GRAPH" in 0|1) ;; *) printf 'GRAPH must be 0 or 1.\n' >&2; exit 2 ;; esac
case "$ENABLE_METRICS" in 0|1) ;; *) printf 'ENABLE_METRICS must be 0 or 1.\n' >&2; exit 2 ;; esac

# 旧快照sitecustomize会在Python导入时安装observer；不要继承该调试终端。
if [ -n "${GLM52_CONTEXT_SNAPSHOT_CONFIG:-}${GLM52_PROPOSAL_SNAPSHOT_CONFIG:-}" ] || [[ "${PYTHONPATH:-}" == *context-snapshot-* || "${PYTHONPATH:-}" == *proposal-snapshot-* ]]; then
  printf 'Active snapshot environment found. Use a clean terminal without snapshot config/PYTHONPATH before benchmarking.\n' >&2; exit 2
fi

case "$NODE_RANK" in
  0) MS1_HOST=$NODE0_HOST; PEER_HOST=$NODE1_HOST ;;
  1) MS1_HOST=$NODE1_HOST; PEER_HOST=$NODE0_HOST ;;
  *) printf 'NODE_RANK must be 0 or 1.\n' >&2; exit 2 ;;
esac
if [ "$NODE0_HOST" = "$NODE1_HOST" ]; then
  printf 'NODE0_HOST and NODE1_HOST must be different machines.\n' >&2; exit 2
fi
if [ -z "$HCCL_SOCKET_IFNAME" ] || [ -z "$GLOO_SOCKET_IFNAME" ]; then
  if ! ROUTE=$(ip -o route get "$PEER_HOST" 2>/dev/null); then
    printf 'Cannot find route to %s. Set HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME to actual NIC names at the top.\n' "$PEER_HOST" >&2; exit 2
  fi
  ROUTE_NIC=$(printf '%s\n' "$ROUTE" | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')
  HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$ROUTE_NIC}
  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$ROUTE_NIC}
fi
for nic in "$HCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME"; do
  if [[ ! "$nic" =~ ^[[:alnum:]_.:-]+$ ]] || [ "$nic" = lo ]; then
    printf 'Invalid two-node NIC "%s". Inspect ip -o route get %s and edit the NIC settings; lo cannot connect two nodes.\n' "$nic" "$PEER_HOST" >&2; exit 2
  fi
  if [ "$PRINT_COMMAND" -eq 0 ] && [ ! -d "/sys/class/net/$nic" ]; then
    printf 'Configured NIC %s does not exist in this container.\n' "$nic" >&2; exit 2
  fi
done
printf 'Node %s: host=%s peer=%s HCCL=%s GLOO=%s\n' "$NODE_RANK" "$MS1_HOST" "$PEER_HOST" "$HCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME" >&2

if [ "$PRINT_COMMAND" -eq 0 ]; then
  # 只检查vendor脚本最终状态，允许其内部处理可恢复的失败。
  for setup in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    if ! source "$setup"; then
      printf 'Failed to source %s\n' "$setup" >&2
      exit 1
    fi
  done
fi

RUNTIME_ENV=(
  SGLANG_SET_CPU_AFFINITY=1 STREAMS_PER_DEVICE=32
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_ENABLE_SPEC_V2=1
  SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 HCCL_BUFFSIZE=1000 HCCL_OP_EXPANSION_MODE=AIV
  "HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME" "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
  TRANSFORMERS_VERBOSITY=error
  SGLANG_NPU_PROFILING=0 SGLANG_NPU_PROFILING_BS=16
  "PYTHONPATH=$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
  DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
  DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
  SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1 SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
  SGLANG_RAGGED_VERIFY_MODE=static SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=false
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
)
SERVER_ARGS=(
  --model-path "$TARGET_MODEL" --attention-backend ascend --device npu
  --tp-size 32 --nnodes 2 --node-rank "$NODE_RANK" --dp-size 8
  --dist-init-addr "$NODE0_HOST:$DIST_PORT" --enable-dp-attention --enable-dp-lm-head
  --context-length "$CONTEXT_LENGTH" --chunked-prefill-size -1 --max-prefill-tokens 69632
  --trust-remote-code --mem-fraction-static 0.85 --max-running-requests 8
  --served-model-name "$SERVED_MODEL_NAME" --quantization modelslim
  --moe-a2a-backend deepep --deepep-mode auto --load-balance-method round_robin
  --enable-cache-report --host "$MS1_HOST" --port "$MS1_PORT"
)
if [ "$MODE" = dspark ]; then
  RUNTIME_ENV+=(SGLANG_NPU_GLM_DSPARK_QUAROT=original)
  SERVER_ARGS+=(
    --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-model-quantization unquant --speculative-draft-attention-backend ascend
    --speculative-dspark-block-size 4 --speculative-num-draft-tokens 5
  )
elif [ "$MODE" = nextn ]; then
  SERVER_ARGS+=(
    --speculative-algorithm NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 5 --speculative-draft-model-quantization unquant
  )
fi
[ -z "$MAX_TOTAL_TOKENS" ] || SERVER_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
if [ "$GRAPH" = 1 ]; then SERVER_ARGS+=(--cuda-graph-bs 16); else SERVER_ARGS+=(--disable-cuda-graph); fi
[ "$ENABLE_METRICS" = 0 ] || SERVER_ARGS+=(--enable-metrics)

# 只对本次进程选择192维候选Python算子；复用镜像binary，不重装包。
COMMAND=(
  env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u all_proxy
  -u ASCEND_LAUNCH_BLOCKING -u SGLANG_NPU_GLM_DSPARK_QUAROT
  -u SGLANG_DSPARK_DEBUG_DUMP -u GLM52_CONTEXT_SNAPSHOT_CONFIG -u GLM52_PROPOSAL_SNAPSHOT_CONFIG
  -u SGLANG_SIMULATE_ACC_LEN -u SGLANG_SIMULATE_ACC_METHOD -u SGLANG_SIMULATE_ACC_TOKEN_MODE
  -u SGLANG_SIMULATE_UNIFORM_EXPERTS -u SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
  "${RUNTIME_ENV[@]}"
  python3 "$SGLANG_REPO/devtools/glm52_ms1/with_kernel_checkout.py"
  --kernel-repo "$KERNEL_REPO" --state-dir "$MS1_STATE"
  -- python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
)
if [ "$PRINT_COMMAND" -eq 1 ]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; else exec "${COMMAND[@]}"; fi
