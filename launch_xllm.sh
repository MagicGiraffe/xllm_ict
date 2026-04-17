#!/bin/bash
set -eo pipefail

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-5}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-43432}

MODEL_PATH=${MODEL_PATH:-/data/fhb/workspace/models/Qwen3.5-4B}
MASTER_NODE_ADDR=${MASTER_NODE_ADDR:-127.0.0.1:19789}
START_PORT=${START_PORT:-18089}
START_DEVICE=${START_DEVICE:-0}
LOG_DIR=${LOG_DIR:-log}
NNODES=${NNODES:-1}
MAX_MEMORY_UTILIZATION=${MAX_MEMORY_UTILIZATION:-0.59}
MAX_TOKENS_PER_BATCH=${MAX_TOKENS_PER_BATCH:-131072}
MAX_SEQS_PER_BATCH=${MAX_SEQS_PER_BATCH:-1}
MAX_TOKENS_PER_CHUNK_FOR_PREFILL=${MAX_TOKENS_PER_CHUNK_FOR_PREFILL:-4096}
XLLM_BIN=${XLLM_BIN:-$(find build -name xllm -type f | head -n 1)}

mkdir -p "$LOG_DIR"

for (( i=0; i<NNODES; i++ )); do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"

  "$XLLM_BIN" \
    --model "$MODEL_PATH" \
    --devices="npu:$DEVICE" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --max_memory_utilization="$MAX_MEMORY_UTILIZATION" \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --max_tokens_per_chunk_for_prefill="$MAX_TOKENS_PER_CHUNK_FOR_PREFILL" \
    --enable_schedule_overlap=false \
    --max_tokens_per_batch="$MAX_TOKENS_PER_BATCH" \
    --max_seqs_per_batch="$MAX_SEQS_PER_BATCH" \
    --enable_shm=false \
    --node_rank="$i" \
    > "$LOG_FILE" 2>&1 &
done

echo "xllm launched. logs: $LOG_DIR/node_*.log"
