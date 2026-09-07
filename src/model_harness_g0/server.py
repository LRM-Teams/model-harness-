from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import math
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .storage import Ledger, digest, write_json
from .trace import sse_chunks, validate_trace


def create_app(backend, root: Path, admin_key: str, *, ttl=300, timeout=120, max_calls=8):
    if len(admin_key) < 24:
        raise ValueError("admin key must have at least 24 characters")
    ledger = Ledger(root)
    locks: dict[str, asyncio.Lock] = {}

    def records(sid):
        p = root / "sessions" / sid / "trace.json"
        return json.loads(p.read_text()) if p.exists() else []

    def save(sid, rows):
        write_json(root / "sessions" / sid / "trace.json", rows)

    async def quarantine(sid, reason):
        ledger.state(sid, "QUARANTINED", reason)
        await backend.close_session(sid)

    async def reap():
        while True:
            await asyncio.sleep(min(5, ttl))
            for row in ledger.rows():
                if row["state"] == "ACTIVE" and time.time() - row["touched"] > ttl:
                    lock = locks.setdefault(row["id"], asyncio.Lock())
                    if not lock.locked():
                        async with lock:
                            await quarantine(row["id"], "session_expired")

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(reap())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            for row in ledger.rows():
                if row["state"] == "ACTIVE":
                    ledger.state(row["id"], "QUARANTINED", "server_shutdown")
            await backend.close()
            ledger.close()

    app = FastAPI(lifespan=lifespan)
    app.state.ledger = ledger

    def bearer(req):
        header = req.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        return header[7:]

    def admin(req):
        if not hmac.compare_digest(bearer(req), admin_key):
            raise HTTPException(401, "admin authentication required")

    def session(req, active=False):
        row = ledger.by_key(bearer(req))
        if not row:
            raise HTTPException(401, "invalid session key")
        if active and row["state"] != "ACTIVE":
            raise HTTPException(410, "session is not active")
        return row

    @app.get("/health")
    async def health():
        return {
            "backend": backend.kind,
            "model_revision": backend.revision,
            "optimizer_steps": 0,
            "collection_only": True,
            "parser_check": getattr(backend, "parser_check", {"passed": False, "synthetic": True}),
            "active_sessions": sum(r["state"] == "ACTIVE" for r in ledger.rows()),
        }

    @app.get("/rl/sessions")
    async def sessions(req: Request):
        admin(req)
        return [{k: v for k, v in r.items() if k != "key_hash"} for r in ledger.rows()]

    @app.post("/rl/start_session")
    async def start(req: Request):
        admin(req)
        body = await req.json()
        trial = body.get("trial_id")
        if not isinstance(trial, str) or not trial or len(trial) > 200:
            raise HTTPException(422, "nonempty trial_id required (max 200 characters)")
        sid, key = uuid.uuid4().hex, secrets.token_urlsafe(32)
        try:
            ledger.add(sid, trial, key)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "trial already exists; inspect ledger, do not replay start")
        try:
            backend.open(sid)
        except Exception:
            await quarantine(sid, "backend_open_failed")
            raise
        locks[sid] = asyncio.Lock()
        return {"session_id": sid, "api_key": key}

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        row = session(req, active=True)
        sid = row["id"]
        body = await req.json()
        request_id = req.headers.get("x-request-id")
        if not request_id or len(request_id) > 200:
            raise HTTPException(422, "X-Request-ID required for retry deduplication")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise HTTPException(422, "messages must be nonempty")
        for k, expected in (("temperature", 1.0), ("top_p", 1.0)):
            if body.get(k, expected) != expected:
                raise HTTPException(422, f"G0 requires {k}={expected}")
            body[k] = expected
        if body.get("n", 1) != 1 or body.get("max_tokens", 1024) != 1024:
            raise HTTPException(422, "G0 requires n=1 and max_tokens=1024")
        body["max_tokens"] = 1024
        for k in ("presence_penalty", "frequency_penalty"):
            if body.get(k, 0) != 0:
                raise HTTPException(422, "G0 penalties must be zero")
        template = body.get("chat_template_kwargs", {"enable_thinking": False})
        if template != {"enable_thinking": False}:
            raise HTTPException(422, "G0 requires enable_thinking=false")
        for k, allowed in (
            ("top_k", (None, -1, 0, 100000000)),
            ("min_p", (None, 0)),
            ("repetition_penalty", (None, 1)),
        ):
            if body.get(k) not in allowed:
                raise HTTPException(422, f"unsupported G0 sampling parameter: {k}")
        if body.get("stop") is not None or body.get("tool_choice", "auto") != "auto":
            raise HTTPException(422, "G0 does not override stop or force tool choice")
        body.setdefault("seed", int(digest(request_id)[:8], 16) % (2**31))
        if type(body["seed"]) is not int:
            raise HTTPException(422, "seed must be an integer")
        payload_hash = digest(body)
        async with locks[sid]:
            session(req, active=True)
            rows = records(sid)
            prior = next((x for x in rows if x["request_id"] == request_id), None)
            if prior:
                if prior["request_hash"] != payload_hash:
                    raise HTTPException(409, "request ID reused with different payload")
                response = prior["response"]
            else:
                if len(rows) >= max_calls:
                    raise HTTPException(429, "episode call budget exhausted")
                ledger.touch(sid)
                try:
                    response, trace = await asyncio.wait_for(
                        backend.complete(sid, body), timeout=timeout
                    )
                    validate_trace(trace)
                    trace.update(
                        session_id=sid,
                        trial_id=row["trial_id"],
                        request_id=request_id,
                        request_hash=payload_hash,
                        response=response,
                        consumed=False,
                    )
                    rows.append(trace)
                    save(sid, rows)
                    ledger.touch(sid)
                except BaseException:
                    await quarantine(sid, "generation_or_trace_failed")
                    raise
        headers = {"x-completion-id": response["id"], "x-model-revision": backend.revision}
        if body.get("stream"):

            async def emit():
                for chunk in sse_chunks(response):
                    yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                    await asyncio.sleep(0)
                yield "data: [DONE]\n\n"

            return StreamingResponse(emit(), media_type="text/event-stream", headers=headers)
        return JSONResponse(response, headers=headers)

    @app.post("/rl/ack")
    async def ack(req: Request):
        row = session(req, active=True)
        sid = row["id"]
        body = await req.json()
        ids = body.get("completion_ids")
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise HTTPException(422, "completion_ids list required")
        async with locks[sid]:
            session(req, active=True)
            rows = records(sid)
            if not set(ids) <= {r["completion_id"] for r in rows}:
                raise HTTPException(409, "unknown completion ID")
            for r in rows:
                if r["completion_id"] in ids:
                    r["consumed"] = True
            save(sid, rows)
            ledger.touch(sid)
        return {"acknowledged": ids}

    @app.post("/rl/set_reward")
    async def reward(req: Request):
        row = session(req, active=True)
        sid = row["id"]
        body = await req.json()
        value = body.get("reward")
        if type(value) not in (float, int) or not math.isfinite(value):
            raise HTTPException(422, "finite reward required")
        async with locks[sid]:
            session(req, active=True)
            consumed = [r for r in records(sid) if r["consumed"]]
            if not consumed or body.get("interaction_id") != consumed[-1]["completion_id"]:
                raise HTTPException(409, "reward must target last acknowledged completion")
            try:
                new = ledger.reward(sid, float(value), body["interaction_id"])
            except ValueError as e:
                raise HTTPException(409, str(e))
        return {"accepted": True, "duplicate": not new, "training_admitted": False}

    @app.post("/rl/end_session")
    async def end(req: Request):
        row = session(req)
        sid = row["id"]
        async with locks.setdefault(sid, asyncio.Lock()):
            row = session(req)
            if row["state"] == "QUARANTINED":
                raise HTTPException(410, "session quarantined")
            if row["state"] == "ACTIVE":
                ledger.state(sid, "ENDED")
                await backend.close_session(sid)
        return {"session_id": sid, "state": "ENDED"}

    @app.post("/rl/abort_session")
    async def abort(req: Request):
        row = session(req)
        async with locks.setdefault(row["id"], asyncio.Lock()):
            await quarantine(row["id"], "client_abort")
        return {"state": "QUARANTINED"}

    @app.post("/rl/export")
    async def export(req: Request):
        admin(req)
        sid = (await req.json()).get("session_id")
        row = ledger.get(sid)
        if not row:
            raise HTTPException(404, "unknown session")
        if row["state"] == "ACTIVE":
            raise HTTPException(409, "end session before export")
        return {
            "schema_version": 1,
            "session_id": sid,
            "trial_id": row["trial_id"],
            "state": row["state"],
            "reason": row["reason"],
            "backend": backend.kind,
            "optimizer_steps": 0,
            "training_admitted": False,
            "final_reward": row["reward"],
            "interactions": records(sid),
        }

    return app
