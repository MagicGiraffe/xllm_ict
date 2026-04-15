#!/bin/bash
set -eo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=2
export HCCL_IF_BASE_PORT=43500

OUT_DIR=/data/zzy/xllm/test/msprof_32k_official_18000
LOG_DIR=$OUT_DIR/runtime_logs
PROF_DIR=$OUT_DIR/prof_output
RUN_SCRIPT=$OUT_DIR/run_once.sh

rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR" "$PROF_DIR"

cat > "$RUN_SCRIPT" <<'SCRIPT'
#!/bin/bash
set -eo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=2
export HCCL_IF_BASE_PORT=43500

SERVER_LOG=/data/zzy/xllm/test/msprof_32k_official_18000/runtime_logs/server.log
REQUEST_LOG=/data/zzy/xllm/test/msprof_32k_official_18000/runtime_logs/request_32k.log

/data/zzy/xllm/build/xllm/core/server/xllm \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --devices=npu:0 \
  --port 18000 \
  --master_node_addr=127.0.0.1:19770 \
  --nnodes=1 \
  --max_memory_utilization=0.59 \
  --block_size=128 \
  --communication_backend=hccl \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=false \
  --enable_shm=false \
  --max_tokens_per_batch=131072 \
  --node_rank=0 > "$SERVER_LOG" 2>&1 &

server_pid=$!

cleanup() {
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:18000/v1/models >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "server_not_ready" >&2
  exit 2
fi

python /data/zzy/xllm/test_xllm.py \
  --backend xllm \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --random-input-len 32768 \
  --random-output-len 1024 \
  --host 127.0.0.1 \
  --port 18000 \
  --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --disable-stream | tee "$REQUEST_LOG"

kill -TERM "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
SCRIPT

chmod +x "$RUN_SCRIPT"

/usr/local/Ascend/ascend-toolkit/latest/bin/msprof --output="$PROF_DIR" bash "$RUN_SCRIPT"
