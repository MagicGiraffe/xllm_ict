#!/bin/bash
set -euo pipefail

python /data/fhb/workspace/otherSpace/xllm_ict/test_xllm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompts 1 \
    --max-concurrency 1 \
    --random-input-len 32768 \
    --random-output-len 1024 \
    --host 127.0.0.1 \
    --port 18011 \
    --disable-stream \
    --dataset-path /data/fhb/workspace/benchmark/ais_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /data/fhb/workspace/models/Qwen3.5-4B/
