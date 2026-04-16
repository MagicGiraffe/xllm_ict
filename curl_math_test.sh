#!/bin/bash
set -euo pipefail

if [[ -n "${XLLM_API_KEY:-}" ]]; then
  AUTH=( -H "Authorization: Bearer ${XLLM_API_KEY}" )
else
  AUTH=()
fi

curl -s "http://127.0.0.1:18011/v1/chat/completions" \
  -H "Content-Type: application/json" \
  "${AUTH[@]}" \
  -d '{
        "model": "Qwen3.5-4B",
        "messages": [
          {"role": "system", "content": "You are a user assistant."},
          {"role": "user", "content": "Loraine makes wax sculptures of animals. Large animals take four sticks of wax and small animals take two sticks. She made three times as many small animals as large animals, and she used 12 sticks of wax for small animals. How many sticks of wax did Loraine use to make all the animals?"}
        ],
        "stream": false
      }'
