from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx

from . import PI_VERSION
from .pi import Pi
from .storage import digest, write_json
from .trace import validate_trace

CASES = ("text", "tool", "chain", "multi", "mcp", "recovery")


def prompts(case, nonce):
    return {
        "text": [f"case=text Reply with exactly this nonce={nonce}"],
        "tool": ["case=tool Call lookup, then reply with exactly the returned secret."],
        "chain": [
            "case=chain Call lookup first. Then call advance with the exact returned token. "
            "Reply with only the advance result."
        ],
        "multi": [
            f"case=multi Remember nonce={nonce} and repeat it.",
            "Repeat the nonce from my previous message, exactly.",
        ],
        "mcp": [f"case=mcp Call mcp_echo with text equal to nonce={nonce} then return its result."],
        "recovery": [
            "case=recovery Call fail_once. If TRANSIENT, retry fail_once once. "
            "Return the successful result."
        ],
    }[case]


def read_events(root):
    path = root / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_consumed(event, trace):
    """Compare Pi's completed assistant message with the actual served response."""
    msg = event["message"]
    if msg.get("stopReason") in ("error", "aborted"):
        raise ValueError("Pi did not finish consuming the response")
    visible = msg.get("content", [])
    if not isinstance(visible, list):
        raise ValueError("unexpected Pi assistant content")
    text = "".join(x.get("text", "") for x in visible if x.get("type") == "text")
    served = trace["response"]["choices"][0]["message"]
    if text != (served.get("content") or ""):
        raise ValueError("Pi assistant text differs from served completion")
    pi_tools = [x for x in visible if x.get("type") == "toolCall"]
    expected = served.get("tool_calls") or []
    if len(pi_tools) != len(expected):
        raise ValueError("Pi tool count differs from served completion")
    for a, b in zip(pi_tools, expected, strict=True):
        if (
            a["id"] != b["id"]
            or a["name"] != b["function"]["name"]
            or a["arguments"] != json.loads(b["function"]["arguments"])
        ):
            raise ValueError("Pi tool JSON differs from served completion")


async def run_trial(client, url, admin, output, case, binary):
    trial = uuid.uuid4().hex
    nonce = uuid.uuid4().hex[:12]
    root = (output / trial).resolve()
    root.mkdir(parents=True, mode=0o700)
    started = await client.post(url + "/rl/start_session", headers=admin, json={"trial_id": trial})
    started.raise_for_status()
    session = started.json()
    auth = {"Authorization": "Bearer " + session["api_key"]}
    sid = session["session_id"]
    result = {
        "trial_id": trial,
        "session_id": sid,
        "case": case,
        "nonce": nonce,
        "protocol_ok": False,
        "task_success": False,
    }
    try:
        async with Pi(binary, root, url, session["api_key"], nonce) as pi:
            for prompt in prompts(case, nonce):
                await pi.prompt(prompt)
        events = read_events(root)
        consumed = [e for e in events if e["kind"] == "consumed"]
        ids = [e["completion_id"] for e in consumed]
        if not ids or any(not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("missing/duplicate Pi completion receipts")
        response = await client.post(url + "/rl/ack", headers=auth, json={"completion_ids": ids})
        response.raise_for_status()
        response = await client.post(url + "/rl/end_session", headers=auth, json={})
        response.raise_for_status()
        response = await client.post(url + "/rl/export", headers=admin, json={"session_id": sid})
        response.raise_for_status()
        exported = response.json()
        write_json(root / "episode.json", exported)
        rows = exported["interactions"]
        if {r["completion_id"] for r in rows} != set(ids):
            raise ValueError("orphan or missing completions")
        by_id = {r["completion_id"]: r for r in rows}
        for event in consumed:
            trace = by_id[event["completion_id"]]
            validate_trace(trace)
            check_consumed(event, trace)
            if trace["trial_id"] != trial or trace["session_id"] != sid:
                raise ValueError("cross-trial trace identity")
        # Verify each actual tool result appears verbatim in a later model request.
        observed = [e for e in events if e["kind"] == "tool_result"]
        for tool in observed:
            found = any(
                m.get("role") == "tool" and tool["text"] in str(m.get("content"))
                for r in rows
                for m in r["request"]["messages"]
            )
            if not found:
                raise ValueError("tool result missing from following model input")
        final = "".join(
            p.get("text", "") for p in consumed[-1]["message"]["content"] if p.get("type") == "text"
        )
        expected = {
            "text": nonce,
            "multi": nonce,
            "tool": "stage1:" + nonce,
            "chain": "stage2:" + nonce,
            "mcp": "mcp:" + nonce,
            "recovery": "recovered:" + nonce,
        }[case]
        names = [e["name"] for e in events if e["kind"] == "tool"]
        required = {
            "text": [],
            "multi": [],
            "tool": ["lookup"],
            "chain": ["lookup", "advance"],
            "mcp": ["mcp_echo"],
            "recovery": ["fail_once", "fail_once"],
        }[case]
        ordered = iter(names)
        tool_ok = all(any(actual == wanted for actual in ordered) for wanted in required)
        result.update(
            protocol_ok=True,
            task_success=expected in final and tool_ok,
            completions=len(rows),
            tool_calls=len(names),
            model_revision=rows[0]["model_revision"],
            synthetic=rows[0]["synthetic"],
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            await client.post(url + "/rl/abort_session", headers=auth, json={})
            response = await client.post(
                url + "/rl/export", headers=admin, json={"session_id": sid}
            )
            if response.is_success:
                write_json(root / "episode.json", response.json())
        except Exception:
            result["cleanup_error"] = True
    finally:
        session["api_key"] = ""
    write_json(root / "result.json", result)
    return result


async def smoke(url: str, admin_key: str, output: Path, binary: str, sequential=20, concurrent=4):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "compatibility_report.json").exists():
        raise ValueError("use a fresh output directory; reports are not overwritten")
    proc = await asyncio.create_subprocess_exec(binary, "--version", stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    version = stdout.decode().strip()
    if proc.returncode or version != PI_VERSION:
        raise ValueError(f"Pi {PI_VERSION} required, found {version!r}")
    user_paths = [
        Path.home() / ".pi/agent" / f for f in ("auth.json", "models.json", "settings.json")
    ]
    before = {str(p): digest(p.read_text()) if p.exists() else None for p in user_paths}
    admin = {"Authorization": "Bearer " + admin_key}
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.get(url + "/health")
        response.raise_for_status()
        health = response.json()
        if not health.get("collection_only") or health.get("optimizer_steps") != 0:
            raise ValueError("refusing a service not marked collection-only")
        source = Path(__file__).parent
        manifest = {
            "status": "draft",
            "pi_version": version,
            "backend": health["backend"],
            "model_revision": health["model_revision"],
            "optimizer_steps": 0,
            "source_hashes": {
                p.name: digest(p.read_text())
                for p in source.iterdir()
                if p.suffix in (".py", ".mjs")
            },
            "parameters": {
                "context": 8192,
                "max_tokens": 1024,
                "temperature": 1.0,
                "top_p": 1.0,
                "thinking": False,
            },
            "cases": list(CASES),
            "sequential": sequential,
            "concurrent": concurrent,
        }
        write_json(output / "manifest.json", manifest)
        results = []
        for i in range(sequential):
            results.append(await run_trial(client, url, admin, output, CASES[i % 6], binary))
        results += await asyncio.gather(
            *[
                run_trial(client, url, admin, output, CASES[(sequential + i) % 6], binary)
                for i in range(concurrent)
            ]
        )
        final_health = (await client.get(url + "/health")).json()
    after = {str(p): digest(p.read_text()) if p.exists() else None for p in user_paths}
    user_config_unchanged = before == after
    covered = {r["case"] for r in results if r["task_success"] and r["protocol_ok"]}
    protocol_ok = (
        user_config_unchanged
        and all(r["protocol_ok"] for r in results)
        and set(CASES) <= covered
        and final_health["active_sessions"] == 0
        and sequential >= 20
        and concurrent >= 4
    )
    report = {
        "schema_version": 1,
        "manifest_hash": digest(manifest),
        "user_config_unchanged": user_config_unchanged,
        "parser_check": health.get("parser_check"),
        "backend": health["backend"],
        "pi_version": version,
        "model_revision": health["model_revision"],
        "optimizer_steps": 0,
        "sequential_sessions": sequential,
        "concurrent_sessions": concurrent,
        "active_sessions_after": final_health["active_sessions"],
        "collection_checks_passed": protocol_ok,
        "gpu_parity": "NOT_RUN",
        "g0_passed": False,
        "note": "Run parity and verify. Mock cannot pass the GPU G0 gate.",
        "results": results,
    }
    write_json(output / "compatibility_report.json", report)
    return report
