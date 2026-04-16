#!/bin/bash
# 在根目录下执行，确保路径正确
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 调整显卡
export ASCEND_RT_VISIBLE_DEVICES=3
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port

# START_PORT多试几个，确保log/node_0.log已经正确启动
MODEL_PATH="/data/fhb/workspace/models/Qwen3.5-4B"               # Model path
MASTER_NODE_ADDR="127.0.0.1:9749"                  # Master node address (must be globally consistent)
START_PORT=18007                                  # Service starting port
START_DEVICE=0                                     # Starting logical device number
LOG_DIR="log"                                      # Log directory
NNODES=1                                           # Number of nodes (current script launches 1 process)
XLLM_BIN=$(find build -name xllm -type f | head -n 1)

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  $XLLM_BIN \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.59 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --max_tokens_per_batch=131072 \
    --enable_shm=false \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done