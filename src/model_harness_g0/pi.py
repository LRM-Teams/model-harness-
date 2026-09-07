from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import uuid
from pathlib import Path

from .storage import write_json

SYSTEM = (
    "You are a protocol test assistant. Follow the user's steps exactly. "
    "Use the named tool when asked. Never guess tool outputs. "
    "After receiving a tool result, answer concisely with that result. "
    "If a tool reports TRANSIENT, retry it once."
)


class Pi:
    def __init__(self, binary: str, root: Path, url: str, key: str, nonce: str):
        self.binary, self.root = binary, root
        self.url, self.key, self.nonce = url, key, nonce
        self.proc = None
        self.stderr = None

    async def __aenter__(self):
        for name in ("agent", "sessions", "workspace"):
            (self.root / name).mkdir(parents=True, mode=0o700)
        provider = {
            "providers": {
                "g0": {
                    "baseUrl": self.url,
                    "api": "openai-completions",
                    "apiKey": "$G0_SESSION_KEY",
                    "models": [
                        {
                            "id": "g0-qwen",
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 8192,
                            "maxTokens": 1024,
                            "samplingParams": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 1024,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                            "compat": {
                                "supportsStore": False,
                                "supportsDeveloperRole": False,
                                "supportsReasoningEffort": False,
                                "maxTokensField": "max_tokens",
                            },
                        }
                    ],
                }
            }
        }
        write_json(self.root / "agent/models.json", provider)
        write_json(
            self.root / "agent/settings.json",
            {
                "compaction": {"enabled": False},
                "retry": {"enabled": False},
            },
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
        }
        env.update(
            PI_CODING_AGENT_DIR=str(self.root / "agent"),
            PI_OFFLINE="1",
            G0_SESSION_KEY=self.key,
            G0_NONCE=self.nonce,
            G0_EVENT_PATH=str(self.root / "events.jsonl"),
            G0_PYTHON=sys.executable,
        )
        extension = Path(__file__).with_name("fixture.mjs")
        self.stderr = open(self.root / "pi.stderr", "wb")
        self.proc = await asyncio.create_subprocess_exec(
            self.binary,
            "--mode",
            "rpc",
            "--provider",
            "g0",
            "--model",
            "g0-qwen",
            "--thinking",
            "off",
            "--session-dir",
            str(self.root / "sessions"),
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-themes",
            "--no-builtin-tools",
            "--offline",
            "--approve",
            "--system-prompt",
            SYSTEM,
            "--extension",
            str(extension),
            cwd=self.root / "workspace",
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=True,
            limit=4 * 1024 * 1024,
        )
        return self

    async def prompt(self, text: str, timeout: float = 180):
        proc = self.proc
        rid = uuid.uuid4().hex
        proc.stdin.write(
            (json.dumps({"id": rid, "type": "prompt", "message": text}) + "\n").encode()
        )
        await proc.stdin.drain()
        events = []
        async with asyncio.timeout(timeout):
            while True:
                line = await proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"Pi exited before agent_end; see {self.root / 'pi.stderr'}")
                event = json.loads(line)
                events.append(event)
                if event.get("type") == "response" and event.get("id") == rid:
                    if not event.get("success"):
                        raise RuntimeError("Pi rejected prompt: " + str(event.get("error")))
                if event.get("type") == "agent_end":
                    break
        return events

    async def __aexit__(self, *args):
        if self.proc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                await self.proc.wait()
        if self.stderr:
            self.stderr.close()
        # No literal credentials on disk, but remove runtime config after use.
        shutil.rmtree(self.root / "agent", ignore_errors=True)
        self.key = ""
