#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
command -v npm >/dev/null
areal_commit=b5f0820c307e9a02056131a54c6f7f92fa03ec55
mkdir -p vendor artifacts
if [[ ! -d vendor/AReaL/.git ]]; then
  git clone https://github.com/areal-project/AReaL.git vendor/AReaL
fi
if [[ -n "$(git -C vendor/AReaL status --porcelain)" ]]; then
  echo 'vendor/AReaL is dirty; preserve your changes before setup.' >&2
  exit 1
fi
git -C vendor/AReaL checkout --detach "$areal_commit"
repo_dir="$PWD"
# Use upstream's lockfile and CUDA dependency overrides, not a second ad hoc torch install.
UV_PROJECT_ENVIRONMENT="$repo_dir/.venv-gpu" uv sync --project vendor/AReaL --locked --extra sglang --python 3.12
uv pip install --python .venv-gpu/bin/python -e . 'mcp>=1.20,<2' 'flash-linear-attention==0.4.2'
npm install --prefix vendor/pi --save-exact @earendil-works/pi-coding-agent@0.84.3
.venv-gpu/bin/mh-g0 doctor --output artifacts/environment.json
.venv-gpu/bin/python -m pip freeze > artifacts/gpu-pip-freeze.txt 2>/dev/null || uv pip freeze --python .venv-gpu/bin/python > artifacts/gpu-pip-freeze.txt
