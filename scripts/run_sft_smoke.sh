#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-sft/bin/python -m model_harness_g0.sft --mode smoke --output "${1:-artifacts/sft-smoke-001}"
