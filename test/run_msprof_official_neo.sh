#!/bin/bash
# ==============================================================================
# xLLM Qwen3.5-4B 融合算子性能测试脚本 (NEO Fused Ops)
#
# 功能: 使用 xllm_ops_neo 提供的融合算子编译运行 xLLM，并收集 msprof 性能数据
#
# 融合算子替换清单:
#   1. Chunk GDN Attention: Neumann 级数替代逐行 Slice+Mul 循环
#      - 消除 280K 次 Slice (10%) 和 107K 次 Mul (17%)
#   2. Decode GDN: fused_recurrent_gated_delta_rule (triton_npu 自定义 kernel)
#      - 在 NPU UB 中保持隐状态，单次 Kernel 完成时间步递推
#   3. Flash Attention Prefill: x_flash_attention_infer (Catlass 框架)
#      - 优化 UnpadFlashAttentionBF16NdKernel (16次, 4.7s, 16.4%)
#   4. Paged Attention Decode: x_paged_attention / custom_paged_attention
#      - 减少 PagedAttentionMaskNdKernel 调度碎片
#   5. FusedAddRmsNorm: Residual Add + RMSNorm 融合
#      - 消除 BF16→FP32 Cast + Add + Cast 的 4 步链
#   6. MoE GroupedMatmul + SwiGLU: 融合激活减少中间张量 HBM 写回
#
# 基线参考: /data/zzy/xllm/test/xllm_0415_1_32768_1024.jsonl
#   - 85.08s e2e latency, 12.04 tok/s output throughput
# ==============================================================================
set -eo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=2
export HCCL_IF_BASE_PORT=43500

# # # ==============================================================================
# # # 编译步骤 (首次运行或源码更新后执行)
# # # ==============================================================================
# # Step 1: 将 xllm_ops 重定向到 xllm_ops_neo (包含融合算子)
# if [ ! -L /data/zzy/xllm/third_party/xllm_ops ] || \
#    [ "$(readlink -f /data/zzy/xllm/third_party/xllm_ops)" != "/data/zzy/xllm_ops_neo" ]; then
#   echo "==> Linking xllm_ops_neo..."
#   rm -rf /data/zzy/xllm/third_party/xllm_ops
#   ln -s /data/zzy/xllm_ops_neo /data/zzy/xllm/third_party/xllm_ops
# fi

# # Step 2: 编译 xllm_ops_neo 自定义算子 (生成 cust_opapi)
# if [ ! -f /data/zzy/xllm_ops_neo/.build_done ] || \
#    [ "${FORCE_REBUILD_OPS:-0}" = "1" ]; then
#   echo "==> Building xllm_ops_neo custom operators..."
#   pushd /data/zzy/xllm_ops_neo
#   if [ -f build.sh ]; then
#     bash build.sh || echo "WARNING: xllm_ops_neo build.sh returned non-zero"
#   fi
#   touch .build_done
#   popd
# fi

# # Step 3: 编译 xLLM 主程序 (启用 USE_NEO_FUSED_OPS 编译宏)
# # 使用 python setup.py build（官方编译流程），通过 CMAKE_ARGS 传入融合算子宏
# if [ ! -f /data/zzy/xllm/build/.neo_build_done ] || \
#    [ "${FORCE_REBUILD:-0}" = "1" ]; then
#   echo "==> Building xLLM with USE_NEO_FUSED_OPS=ON (via setup.py)..."
#   pushd /data/zzy/xllm
#   export CMAKE_ARGS="-DUSE_NEO_FUSED_OPS=ON"
#   python setup.py build
#   touch build/.neo_build_done
#   popd
# fi

# ==============================================================================
# 运行测试
# ==============================================================================
OUT_DIR=/data/zzy/xllm/test/msprof_32k_official_neo
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

SERVER_LOG=/data/zzy/xllm/test/msprof_32k_official_neo/runtime_logs/server.log
REQUEST_LOG=/data/zzy/xllm/test/msprof_32k_official_neo/runtime_logs/request_32k.log

# 添加 xllm_ops_neo 的编译产物到动态链接路径
export LD_LIBRARY_PATH=/data/zzy/xllm_ops_neo/build:${LD_LIBRARY_PATH:-}

# 启动 xLLM 服务器 (编译时已启用 USE_NEO_FUSED_OPS)
/data/zzy/xllm/build/xllm/core/server/xllm \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --devices=npu:0 \
  --port 18001 \
  --master_node_addr=127.0.0.1:19771 \
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
  if curl -sf http://127.0.0.1:18001/v1/models >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "server_not_ready" >&2
  echo "==> Server startup failed. Check $SERVER_LOG for details."
  tail -50 "$SERVER_LOG" >&2
  exit 2
fi

echo "==> Server ready, running benchmark (32K input, 1024 output)..."

# 使用与基线相同的参数 (random-output-len=1024)
python /data/zzy/xllm/test_xllm.py \
  --backend xllm \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --random-input-len 32768 \
  --random-output-len 1024 \
  --host 127.0.0.1 \
  --port 18001 \
  --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --disable-stream \
  --output-file /data/zzy/xllm/test/msprof_32k_official_neo/runtime_logs/result.jsonl \
  | tee "$REQUEST_LOG"

echo "==> Benchmark complete."

kill -TERM "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
SCRIPT

chmod +x "$RUN_SCRIPT"

echo "================================================================"
echo "  xLLM NEO Fused Ops Performance Test"
echo "  Output: $OUT_DIR"
echo "  Baseline: /data/zzy/xllm/test/xllm_0415_1_32768_1024.jsonl"
echo "================================================================"

# 收集 msprof 性能数据
/usr/local/Ascend/ascend-toolkit/latest/bin/msprof --output="$PROF_DIR" bash "$RUN_SCRIPT"

echo "================================================================"
echo "  Profiling complete!"
echo "  Results: $PROF_DIR"
echo "  Logs:    $LOG_DIR"
echo "================================================================"

# 对比基线
if [ -f "$LOG_DIR/result.jsonl" ]; then
  echo ""
  echo "==> Results (compare with baseline 85.08s / 12.04 tok/s):"
  python3 -c "
import json
with open('$LOG_DIR/result.jsonl') as f:
    for line in f:
        d = json.loads(line.strip())
        print(f'  Duration:          {d.get(\"duration\", \"N/A\"):.2f}s')
        print(f'  Output Throughput: {d.get(\"output_throughput\", \"N/A\"):.2f} tok/s')
        print(f'  Mean E2E Latency: {d.get(\"mean_e2e_latency_ms\", \"N/A\"):.0f}ms')
        print(f'  Mean TTFT:        {d.get(\"mean_ttft_ms\", \"N/A\"):.0f}ms')
        break
" 2>/dev/null || echo "  (Could not parse result)"
fi
