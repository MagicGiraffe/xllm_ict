#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 CASE_NAME [extra server args...]" >&2
  exit 2
fi

CASE_NAME="$1"
shift || true
EXTRA_ARGS=("$@")

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# 使用 NPU 1
export ASCEND_RT_VISIBLE_DEVICES=2
export HCCL_IF_BASE_PORT=43505

# 目录结构调整为放在 /data/zzy/xllm/test 中
BASE_DIR="/data/zzy/xllm/test/exp_qwen35_32k/${CASE_NAME}"
mkdir -p "$BASE_DIR"

SERVER_LOG="$BASE_DIR/server.log"
REQUEST_LOG="$BASE_DIR/request.log"
SUMMARY_TXT="$BASE_DIR/summary.txt"
# 将生成的 benchmark JSONL 直接沉淀在 CASE 目录下
OUTPUT_JSONL="$BASE_DIR/result.jsonl"

PORT=18005
MASTER_ADDR=127.0.0.1:19775
MODEL_PATH=/data/fhb/workspace/models/Qwen3.5-4B
# 动态寻找你编译出的可执行文件
BIN=$(find /data/zzy/xllm/build -name xllm -type f | head -n 1)

cleanup() {
  if [ -n "${server_pid:-}" ]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# 启动前先杀死残留，避免端口冲突
pkill -f "xllm --model ${MODEL_PATH}" 2>/dev/null || true
sleep 2

# --enable_prefix_cache=false 是否需要启用？
"$BIN" \
  --model "$MODEL_PATH" \
  --devices=npu:0 \
  --port "$PORT" \
  --master_node_addr="$MASTER_ADDR" \
  --nnodes=1 \
  --max_memory_utilization=0.59 \
  --block_size=128 \
  --communication_backend=hccl \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=false \
  --enable_shm=false \
  --max_tokens_per_batch=131072 \
  --node_rank=0 \
  "${EXTRA_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
server_pid=$!

# 通过轮询 v1/models 接口等待服务端完全 Ready
ready=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "SERVER_START_FAILED" | tee "$SUMMARY_TXT"
  tail -n 80 "$SERVER_LOG" >> "$SUMMARY_TXT" || true
  exit 3
fi

# 开始执行 Benchmark 脚本
# 删除了--disable-stream
cd /data/zzy/xllm
python test_xllm.py \
  --backend xllm \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --random-input-len 32768 \
  --random-output-len 1024 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --output-file "$OUTPUT_JSONL" > "$REQUEST_LOG" 2>&1 || true

# 解析服务器和客户端日志，汇总指标
python - <<'PY' "$BASE_DIR" "$SUMMARY_TXT"
import pathlib, re, sys
base = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
server = (base / 'server.log').read_text(errors='ignore') if (base / 'server.log').exists() else ''
request = (base / 'request.log').read_text(errors='ignore') if (base / 'request.log').exists() else ''
patterns = {
    'benchmark_duration_s': r'Benchmark duration \(s\):\s*([0-9.]+)',
    'output_tps': r'Output token throughput \(tok/s\):\s*([0-9.]+)',
    'total_tps': r'Total token throughput \(tok/s\):\s*([0-9.]+)',
    'mean_e2e_ms': r'Mean E2E Latency \(ms\):\s*([0-9.]+)',
    'mean_ttft_ms': r'Mean TTFT \(ms\):\s*([0-9.]+)',
    'server_ttft_ms': r'ttft:\s*([0-9.]+)ms',
    'server_total_latency_ms': r'total_latency:\s*([0-9.]+)ms',
    'avg_tpot_ms': r'avg tpot:\s*([0-9.]+)ms',
    'generation_speed_tps': r'generation speed:\s*([0-9.]+) tokens/s',
    'prompt_tokens': r'prompt_tokens:\s*([0-9]+)',
    'generated_tokens': r'generated_tokens:\s*([0-9]+)',
    'kv_blocks': r'blocks:\s*([0-9]+)',
    'kv_capacity_gb': r'kv cache capacity:\s*([0-9.]+) GB',
}
items = []
for key, pat in patterns.items():
    text = server if key.startswith('server_') or key in {'avg_tpot_ms','generation_speed_tps','prompt_tokens','generated_tokens','kv_blocks','kv_capacity_gb'} else request
    matches = re.findall(pat, text)
    if matches:
        items.append(f'{key}={matches[-1]}')
status = 'SUCCESS' if 'Serving Benchmark Result' in request and 'generated_tokens: 1024' in server else 'FAILED'
summary.write_text(status + '\n' + '\n'.join(items) + '\n')
print(summary.read_text())
PY

cleanup
trap - EXIT