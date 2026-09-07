#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${G0_ADMIN_KEY:?Set G0_ADMIN_KEY to a random string of at least 24 characters}"
run_dir="${1:-artifacts/run-$(date -u +%Y%m%dT%H%M%SZ)}"
.venv-gpu/bin/mh-g0 smoke --output "$run_dir" --pi "$PWD/vendor/pi/node_modules/.bin/pi"
# With 2 GPUs, use CUDA_VISIBLE_DEVICES=1 for this script and device cuda:0 below.
.venv-gpu/bin/mh-g0 parity --run "$run_dir" --device cuda:0
.venv-gpu/bin/mh-g0 verify --run "$run_dir"
