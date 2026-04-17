#!/usr/bin/env bash
set -euo pipefail

# Memory-focused sweep for Qwen3.5-4B chunked prefill.
# Fixed benchmark shape: 64k + 1024
#
# Usage example:
#   PORT=18089 MODEL_PATH=/data/fhb/workspace/models/Qwen3.5-4B \
#   LAUNCH_SCRIPT=./launch_xllm.sh TPS_SCRIPT=./tps_test.sh \
#   bash tools/qwen35_memopt_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_records}"
mkdir -p "${RESULT_DIR}"

LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${ROOT_DIR}/launch_xllm.sh}"
TPS_SCRIPT="${TPS_SCRIPT:-${ROOT_DIR}/tps_test.sh}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/log/node_0.log}"

START_PORT="${START_PORT:-18011}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:19752}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-5}"
MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-0.59}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-131072}"
MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-1}"
MODEL_PATH="${MODEL_PATH:-/data/fhb/workspace/models/Qwen3.5-4B}"

# Each tier must run 64k+1024.
INPUT_LEN="${INPUT_LEN:-65536}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

# Space separated chunk tiers for memory sweep.
CHUNK_TIERS="${CHUNK_TIERS:-4096 8192 12288 16384}"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary_file="${RESULT_DIR}/qwen35_memopt_sweep_${timestamp}.txt"
touch "${summary_file}"

echo "[memopt] summary file: ${summary_file}"
echo "[memopt] fixed case: ${INPUT_LEN}+${OUTPUT_LEN}"

run_one_tier() {
  local chunk="$1"
  local tier_tag="chunk_${chunk}"
  local out_file="${RESULT_DIR}/${tier_tag}_${timestamp}.txt"
  local status="PASS"
  local tps_line=""
  local err_line=""

  echo "========== ${tier_tag} ==========" | tee -a "${summary_file}"

  pkill -f "/xllm" >/dev/null 2>&1 || true
  sleep 1

  export ASCEND_RT_VISIBLE_DEVICES
  export MASTER_NODE_ADDR
  export START_PORT
  export MAX_MEMORY_UTILIZATION
  export MAX_TOKENS_PER_BATCH
  export MAX_SEQS_PER_BATCH
  export MAX_TOKENS_PER_CHUNK_FOR_PREFILL="${chunk}"
  export ENABLE_CHUNKED_PREFILL=true
  export ENABLE_PREFIX_CACHE=false
  export ENABLE_SCHEDULE_OVERLAP=false
  export MODEL_PATH

  bash "${LAUNCH_SCRIPT}" >/dev/null 2>&1
  sleep 5

  if ! curl -s "http://127.0.0.1:${START_PORT}/v1/models" >/dev/null; then
    status="LAUNCH_FAIL"
  else
    set +e
    python /data/fhb/workspace/otherSpace/xllm_ict/test_xllm.py \
      --backend xllm \
      --dataset-name random \
      --random-range-ratio 1 \
      --num-prompts 1 \
      --max-concurrency 1 \
      --random-input-len "${INPUT_LEN}" \
      --random-output-len "${OUTPUT_LEN}" \
      --host 127.0.0.1 \
      --port "${START_PORT}" \
      --disable-stream \
      --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
      --model "${MODEL_PATH}" >"${out_file}" 2>&1
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      status="TPS_FAIL"
    fi
  fi

  if [[ -f "${out_file}" ]]; then
    tps_line="$(grep -E 'Output TPS|output throughput|tokens/s' "${out_file}" | tail -n 1 || true)"
    err_line="$(grep -E 'Internal Server Error|ServerDisconnectedError|out of memory|Check failed' "${out_file}" | tail -n 1 || true)"
  fi

  if [[ -z "${err_line}" && -f "${LOG_FILE}" ]]; then
    err_line="$(grep -E 'F[0-9]{8}|Check failed|out of memory|OOM|RESOURCE_EXHAUSTED' "${LOG_FILE}" | tail -n 1 || true)"
  fi

  {
    echo "tier=${tier_tag}"
    echo "status=${status}"
    echo "input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"
    echo "max_tokens_per_chunk_for_prefill=${chunk}"
    echo "max_tokens_per_batch=${MAX_TOKENS_PER_BATCH}, max_seqs_per_batch=${MAX_SEQS_PER_BATCH}"
    echo "tps_line=${tps_line}"
    echo "err_line=${err_line}"
    echo "out_file=${out_file}"
    echo
  } | tee -a "${summary_file}"
}

for chunk in ${CHUNK_TIERS}; do
  run_one_tier "${chunk}"
done

echo "[memopt] done. summary: ${summary_file}"
