```bash
python test_xllm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 1 \
    --max-concurrency 1 \
    --random-input  65536 \
    --random-output 1024 \
    --host 127.0.0.1 \
    --port 18000 \
    --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /data/fhb/workspace/models/Qwen3.5-4B
```

先启动服务：

```bash
cd /data/fhb/workspace/xllm
./start_server.sh
```

然后二选一：

测性能：
```bash
cd /data/fhb/workspace/xllm
./performance_local_32k.sh
```

测精度：
```bash
cd /data/fhb/workspace/benchmark/ais_bench
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --merge-ds --dump-eval-details
```

# msprof

下面这版就是最短可复制版本。假设你已经：

- `ssh ict`
- `docker exec -it fhb_xllm bash`

并且在容器里执行。

**1. 加载环境并准备目录**
作用：加载 Ascend 环境，指定空卡/空端口，创建 profiling 目录。

```bash
set -eo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_IF_BASE_PORT=43490

OUT_DIR=/data/fhb/workspace/msprof_32k_official_18004
LOG_DIR=$OUT_DIR/runtime_logs
PROF_DIR=$OUT_DIR/prof_output
RUN_SCRIPT=$OUT_DIR/run_once.sh

rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR" "$PROF_DIR"
```

**2. 写一次性采集脚本**
作用：起 xLLM 服务，跑一条 `32k+1024`，然后正常结束服务。

```bash
cat > "$RUN_SCRIPT" <<'SCRIPT'
#!/bin/bash
set -eo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=5
export HCCL_IF_BASE_PORT=43490

SERVER_LOG=/data/fhb/workspace/msprof_32k_official_18004/runtime_logs/server.log
REQUEST_LOG=/data/fhb/workspace/msprof_32k_official_18004/runtime_logs/request_32k.log

/data/fhb/workspace/xllm/build/xllm/core/server/xllm \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --devices=npu:0 \
  --port 18004 \
  --master_node_addr=127.0.0.1:19773 \
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
  if curl -sf http://127.0.0.1:18004/v1/models >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "server_not_ready" >&2
  exit 2
fi

python /data/fhb/workspace/xllm/test_xllm.py \
  --backend xllm \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --random-input-len 32768 \
  --random-output-len 1024 \
  --host 127.0.0.1 \
  --port 18004 \
  --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /data/fhb/workspace/models/Qwen3.5-4B \
  --disable-stream | tee "$REQUEST_LOG"

kill -TERM "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
trap - EXIT
SCRIPT

chmod +x "$RUN_SCRIPT"
```

**3. 启动 msprof 采集**
作用：按官方方式包住一次性脚本，自动采集并导出 timeline。

```bash
/usr/local/Ascend/ascend-toolkit/latest/bin/msprof --output="$PROF_DIR" bash "$RUN_SCRIPT"
```

**4. 查看 benchmark 是否跑完**
作用：看 `32k+1024` 是否成功完成。

```bash
sed -n '1,260p' "$LOG_DIR/request_32k.log"
```

**5. 确认是不是在导出 timeline**
作用：确认这次走的是正确的 `export timeline`，不是 `export db`。

```bash
ps -ef | grep -E 'msprof.py export' | grep -v grep
```

**6. 查看 json 是否已经生成**
作用：找最终的 trace json 和 summary 文件。

```bash
find "$PROF_DIR"/PROF_* -type f | grep mindstudio_profiler_output
```

**7. 如果想直接找 json**
作用：只看 trace json 路径。

```bash
find "$PROF_DIR"/PROF_* -type f | grep 'mindstudio_profiler_output/.*\.json'
```

**8. 如果想把 json 拷到本地**
作用：下载 trace 到你自己的电脑。

先在容器里查到具体路径，再在你本机执行：

```bash
scp ict:/data/fhb/workspace/msprof_32k_official_18004/prof_output/PROF_xxx/mindstudio_profiler_output/msprof_xxx.json .

scp -r ict:/data/fhb/workspace/msprof_32k_official_18004/prof_output/PROF_000001_20260412204420916_DEGIJEFEIPCJJRKA/mindstudio_profiler_output .

```

如果你愿意，我下一条可以再给你一版**“换卡/换端口时只改哪 5 处”**的超短说明。