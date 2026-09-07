from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from . import AREAL_COMMIT, PI_VERSION
from .storage import digest, write_json


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_lock(lock):
    path = Path(lock["snapshot_path"])
    if not lock["files"]:
        raise ValueError("empty model lock")
    for name, expected in lock["files"].items():
        if sha_file(path / name) != expected:
            raise ValueError(f"model snapshot changed: {name}")


def prepare(output: Path, revision: str):
    from huggingface_hub import HfApi, snapshot_download

    if output.exists():
        raise ValueError("lock already exists; use a new path for a new experiment")
    model = "Qwen/Qwen3.5-0.8B"
    sha = HfApi().model_info(model, revision=revision).sha
    path = Path(snapshot_download(model, revision=sha))
    files = {str(f.relative_to(path)): sha_file(f) for f in sorted(path.rglob("*")) if f.is_file()}
    config = Path(__file__).parents[2] / "configs/qwen35_08b.json"
    lock = {
        "schema_version": 1,
        "model_id": model,
        "model_revision": sha,
        "tokenizer_revision": sha,
        "snapshot_path": str(path.resolve()),
        "files": files,
        "areal_commit": AREAL_COMMIT,
        "pi_version": PI_VERSION,
        "config_hash": digest(json.loads(config.read_text())) if config.exists() else None,
        "status": "draft-until-g0-passes",
    }
    write_json(output, lock)
    return lock


def doctor():
    versions = {}
    for name in (
        "areal",
        "torch",
        "transformers",
        "sglang",
        "flash-linear-attention",
        "fastapi",
        "httpx",
        "mcp",
        "model-harness-g0",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "gpu": gpu,
        "required_areal_commit": AREAL_COMMIT,
        "optimizer_steps": 0,
    }
