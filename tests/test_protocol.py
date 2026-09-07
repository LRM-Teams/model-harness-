import asyncio
import json
import math

import httpx
import pytest

from model_harness_g0.backend import MockBackend
from model_harness_g0.server import create_app
from model_harness_g0.storage import Ledger
from model_harness_g0.trace import sse_chunks, validate_trace

KEY = "test-admin-key-with-at-least-24-characters"
ADMIN = {"Authorization": "Bearer " + KEY}


@pytest.fixture
async def service(tmp_path):
    backend = MockBackend()
    app = create_app(backend, tmp_path, KEY, ttl=0.05)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            yield client, app, backend


async def start(client, trial="t"):
    r = await client.post("/rl/start_session", headers=ADMIN, json={"trial_id": trial})
    assert r.status_code == 200, r.text
    info = r.json()
    return info["session_id"], {"Authorization": "Bearer " + info["api_key"]}


async def chat(client, auth, rid="r1", text="case=text nonce=hello", stream=False):
    return await client.post(
        "/chat/completions",
        headers={**auth, "X-Request-ID": rid},
        json={
            "model": "g0-qwen",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1024,
            "stream": stream,
        },
    )


async def test_lifecycle_idempotence_export_and_no_optimizer(service):
    c, app, _ = service
    sid, auth = await start(c)
    answer = await chat(c, auth)
    assert answer.status_code == 200, answer.text
    assert (await chat(c, auth)).json() == answer.json()
    assert (await chat(c, auth, text="different")).status_code == 409
    cid = answer.json()["id"]
    assert (await c.post("/rl/export", headers=ADMIN, json={"session_id": sid})).status_code == 409
    assert (
        await c.post("/rl/ack", headers=auth, json={"completion_ids": [cid]})
    ).status_code == 200
    payload = {"reward": 1, "interaction_id": cid}
    a = await c.post("/rl/set_reward", headers=auth, json=payload)
    b = await c.post("/rl/set_reward", headers=auth, json=payload)
    assert a.json()["duplicate"] is False and b.json()["duplicate"] is True
    assert (
        await c.post("/rl/set_reward", headers=auth, json={**payload, "reward": 0})
    ).status_code == 409
    for _ in range(2):
        assert (await c.post("/rl/end_session", headers=auth, json={})).status_code == 200
    r = await c.post("/rl/export", headers=ADMIN, json={"session_id": sid})
    data = r.json()
    assert len(data["interactions"]) == 1
    validate_trace(data["interactions"][0])
    assert data["interactions"][0]["consumed"]
    assert data["optimizer_steps"] == 0 and not data["training_admitted"]
    assert auth["Authorization"][7:] not in r.text
    assert (await c.post("/rl/export", headers=ADMIN, json={"session_id": sid})).json() == data
    assert (await chat(c, auth)).status_code == 410
    assert (await c.get("/health")).json()["active_sessions"] == 0
    assert not app.state.ledger.get(sid)["reason"]


async def test_parallel_isolation_and_auth(service):
    c, _, _ = service

    async def one(i):
        sid, auth = await start(c, f"trial-{i}")
        r = await chat(c, auth, text=f"case=text nonce=nonce{i}")
        assert r.json()["choices"][0]["message"]["content"] == f"nonce{i}"
        await c.post("/rl/end_session", headers=auth, json={})
        trace = (await c.post("/rl/export", headers=ADMIN, json={"session_id": sid})).json()
        assert trace["trial_id"] == f"trial-{i}"
        assert trace["interactions"][0]["session_id"] == sid
        return sid

    ids = await asyncio.gather(*(one(i) for i in range(4)))
    assert len(set(ids)) == 4
    assert (await c.get("/rl/sessions")).status_code == 401
    assert (
        await c.post("/rl/start_session", headers=ADMIN, json={"trial_id": "trial-1"})
    ).status_code == 409
    assert (
        await c.post(
            "/rl/export", headers={"Authorization": "Bearer bad"}, json={"session_id": ids[0]}
        )
    ).status_code == 401


async def test_expiry_and_recovery(service):
    c, app, _ = service
    sid, auth = await start(c)
    await asyncio.sleep(0.16)
    assert app.state.ledger.get(sid)["state"] == "QUARANTINED"
    assert (await chat(c, auth)).status_code == 410
    assert (await c.get("/health")).json()["active_sessions"] == 0


async def test_failed_generation_is_quarantined(service):
    c, app, backend = service

    async def bad(*args):
        raise ValueError("bad engine output")

    backend.complete = bad
    sid, auth = await start(c)
    r = await chat(c, auth)
    assert r.status_code == 500
    assert app.state.ledger.get(sid)["state"] == "QUARANTINED"
    assert (await c.get("/health")).json()["active_sessions"] == 0


async def test_foreign_ack_and_unconsumed_reward(service):
    c, _, _ = service
    _, a = await start(c, "a")
    _, b = await start(c, "b")
    cid = (await chat(c, a)).json()["id"]
    assert (await c.post("/rl/ack", headers=b, json={"completion_ids": [cid]})).status_code == 409
    assert (
        await c.post("/rl/set_reward", headers=a, json={"interaction_id": cid, "reward": 1})
    ).status_code == 409


async def test_sse_roundtrip(service):
    c, _, _ = service
    _, auth = await start(c)
    r = await chat(c, auth, text="case=tool", stream=True)
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert r.text.endswith("data: [DONE]\n\n")
    assert all(x["id"] == r.headers["x-completion-id"] for x in chunks)
    calls = [
        t for x in chunks for choice in x["choices"] for t in choice["delta"].get("tool_calls", [])
    ]
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads("".join(x["function"].get("arguments", "") for x in calls)) == {}


def test_restart_quarantines_active_sessions(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.add("a", "trial-a", "secret")
    ledger.add("b", "trial-b", "secret2")
    ledger.state("b", "ENDED")
    ledger.close()
    restored = Ledger(tmp_path)
    assert restored.get("a")["state"] == "QUARANTINED"
    assert restored.get("b")["state"] == "ENDED"
    assert "secret" not in json.dumps(restored.rows())
    restored.close()


async def test_trace_rejects_bad_alignment():
    backend = MockBackend()
    _, trace = await backend.complete(
        "a", {"model": "m", "messages": [{"role": "user", "content": "case=text nonce=x"}]}
    )
    validate_trace(trace)
    trace["old_logprobs"][0] = math.nan
    with pytest.raises(ValueError):
        validate_trace(trace)
    trace["old_logprobs"][0] = -1
    trace["loss_mask"][0] = 1
    with pytest.raises(ValueError):
        validate_trace(trace)


def test_fragmented_parallel_tool_calls():
    response = {
        "id": "c",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "你好\n",
                    "tool_calls": [
                        {
                            "id": f"t{i}",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps({"text": '汉字\\"\n'}, ensure_ascii=False),
                            },
                        }
                        for i in range(2)
                    ],
                },
            }
        ],
    }
    chunks = list(sse_chunks(response, fragment_size=1))
    reconstructed = {0: "", 1: ""}
    for chunk in chunks:
        for choice in chunk["choices"]:
            for tc in choice["delta"].get("tool_calls", []):
                reconstructed[tc["index"]] += tc["function"].get("arguments", "")
    for value in reconstructed.values():
        assert json.loads(value) == {"text": '汉字\\"\n'}


async def test_generation_timeout_does_not_leave_active_session(tmp_path):
    backend = MockBackend()

    async def stalled(*args):
        await asyncio.sleep(10)

    backend.complete = stalled
    app = create_app(backend, tmp_path, KEY, timeout=0.01)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
        ) as c:
            sid, auth = await start(c)
            assert (await chat(c, auth)).status_code == 500
            assert app.state.ledger.get(sid)["state"] == "QUARANTINED"
            assert (await c.get("/health")).json()["active_sessions"] == 0


async def test_disconnected_response_retries_reuse_completion(service):
    c, _, _ = service
    sid, auth = await start(c)
    # A response exists but caller has not acknowledged consumption.
    first = await chat(c, auth, stream=True)
    second = await chat(c, auth, stream=True)
    assert first.headers["x-completion-id"] == second.headers["x-completion-id"]
    await c.post("/rl/end_session", headers=auth, json={})
    exported = (await c.post("/rl/export", headers=ADMIN, json={"session_id": sid})).json()
    assert len(exported["interactions"]) == 1
    assert exported["interactions"][0]["consumed"] is False
    assert exported["final_reward"] is None


async def test_sampling_controls_are_not_silently_ignored(service):
    c, _, _ = service
    _, auth = await start(c)
    for setting in (
        {"top_p": 0.8},
        {"top_k": 20},
        {"min_p": 0.1},
        {"n": 4},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"stop": ["x"]},
    ):
        r = await c.post(
            "/chat/completions",
            headers={**auth, "X-Request-ID": "r"},
            json={"model": "g0", "messages": [{"role": "user", "content": "hi"}], **setting},
        )
        assert r.status_code == 422
