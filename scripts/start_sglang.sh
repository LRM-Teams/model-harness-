#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_snapshot="$(.venv-gpu/bin/python -c 'import json; print(json.load(open("artifacts/model-lock.json"))["snapshot_path"])')"
exec .venv-gpu/bin/python -m sglang.launch_server \
  --model-path "$model_snapshot" --host 127.0.0.1 --port 30000 \
  --dtype bfloat16 --context-length 8192 --tp-size 1 --mem-fraction-static 0.70 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --disable-cuda-graph
