from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from . import AREAL_COMMIT
from .storage import write_json


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 G0 collection-only harness")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument(
        "--revision", default="main", help="resolved to immutable HF commit before download"
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/model-lock.json"))
    p = sub.add_parser("doctor")
    p.add_argument("--output", type=Path, default=Path("artifacts/environment.json"))
    p = sub.add_parser("serve")
    p.add_argument("--backend", choices=["mock", "areal"], default="areal")
    p.add_argument("--model-lock", type=Path, default=Path("artifacts/model-lock.json"))
    p.add_argument("--sglang-address", default="127.0.0.1:30000")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--state", type=Path, default=Path("artifacts/server"))
    p.add_argument("--session-timeout", type=float, default=300)
    p = sub.add_parser("smoke")
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--pi", default="pi")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequential", type=int, default=20)
    p.add_argument("--concurrent", type=int, default=4)
    p = sub.add_parser("parity")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, default=Path("artifacts/model-lock.json"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reference", choices=["fsdp", "hf"], default="fsdp")
    p = sub.add_parser("verify")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare

        lock = prepare(args.output, args.revision)
        print(json.dumps({"lock": str(args.output), "model_revision": lock["model_revision"]}))
    elif args.command == "doctor":
        from .prepare import doctor

        result = doctor()
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "serve":
        import uvicorn

        from .backend import ArealBackend, MockBackend
        from .prepare import check_lock
        from .server import create_app

        if args.backend == "areal":
            import areal

            source = Path(areal.__file__).resolve().parent.parent
            sha = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if sha != AREAL_COMMIT:
                raise ValueError(f"AReaL commit mismatch: {sha}")
            if subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True):
                raise ValueError("AReaL checkout has uncommitted changes")
            check_lock(json.loads(args.model_lock.read_text()))
            backend = ArealBackend(args.model_lock, args.sglang_address, 8192)
        else:
            backend = MockBackend()
        app = create_app(
            backend, args.state.resolve(), os.environ["G0_ADMIN_KEY"], ttl=args.session_timeout
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
    elif args.command == "smoke":
        from .runner import smoke

        result = asyncio.run(
            smoke(
                args.url.rstrip("/"),
                os.environ["G0_ADMIN_KEY"],
                args.output.resolve(),
                args.pi,
                args.sequential,
                args.concurrent,
            )
        )
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
        raise SystemExit(0 if result["collection_checks_passed"] else 1)
    elif args.command == "parity":
        from .parity import parity

        result = parity(args.run, args.model_lock, args.device, args.reference)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    elif args.command == "verify":
        from .parity import verify

        result = verify(args.run)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["g0_passed"] else 1)


if __name__ == "__main__":
    main()
