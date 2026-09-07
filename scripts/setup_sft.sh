#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-sft/bin/python ]]; then
  uv venv --python 3.12 .venv-sft
fi
uv pip install --python .venv-sft/bin/python 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-sft/bin/python -e . 'transformers==5.3.0' 'jinja2>=3.1,<4' 'jsonschema>=4,<5'
mkdir -p artifacts
uv pip freeze --python .venv-sft/bin/python > artifacts/sft-pip-freeze.txt
