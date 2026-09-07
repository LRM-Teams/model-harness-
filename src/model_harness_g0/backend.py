from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from .storage import digest


class ArealBackend:
    """Pinned AReaL inference engine + per-session ArealOpenAI caches. No trainer."""

    kind = "areal-sglang"

    def __init__(self, model_lock: Path, address: str, context: int):
        import httpx
        from areal.engine.sglang_remote import RemoteSGLangEngine
        from areal.experimental.openai import ArealOpenAI
        from areal.experimental.openai.tool_call_parser import process_tool_calls
        from transformers import AutoTokenizer

        from .parser_check import check_parser

        self.parser_check = check_parser(process_tool_calls)
        self.lock = json.loads(model_lock.read_text())
        self.revision = self.lock["model_revision"]
        self.model_path = str(Path(self.lock["snapshot_path"]).resolve())
        info = httpx.get(f"http://{address}/get_model_info", timeout=30)
        info.raise_for_status()
        server_path = info.json().get("model_path")
        if server_path is None or Path(server_path).resolve() != Path(self.model_path):
            raise ValueError("SGLang model_path does not match the locked snapshot")
        self.engine = RemoteSGLangEngine.from_pretrained(
            tokenizer_path=self.model_path,
            max_concurrent_rollouts=4,
            setup_timeout=60,
        )
        self.engine.initialize(addr=address)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.client_class = ArealOpenAI
        self.context = context
        self.clients = {}

    def open(self, sid):
        self.clients[sid] = self.client_class(
            engine=self.engine,
            tokenizer=self.tokenizer,
            tool_call_parser="qwen3_coder",
            reasoning_parser="qwen3",
            engine_max_tokens=self.context,
            api_key="collection-only",
            base_url="http://unused.invalid",
        )

    async def complete(self, sid, request):
        client = self.clients[sid]
        # No retokenization of responses: input/output IDs come from ModelResponse.
        allowed = ("model", "messages", "tools", "max_tokens", "temperature", "top_p", "seed")
        kwargs = {k: request[k] for k in allowed if k in request}
        result = await client.chat.completions.create(
            **kwargs,
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        interaction = client.get_interaction(result.id)
        resp = interaction.model_response
        if resp is None:
            raise ValueError("AReaL did not retain ModelResponse token data")
        inp, out = list(resp.input_tokens), list(resp.output_tokens)
        trace = {
            "completion_id": result.id,
            "input_ids": inp,
            "output_ids": out,
            "old_logprobs": list(resp.output_logprobs),
            "loss_mask": [0] * len(inp) + [1] * len(out),
            "model_revision": self.revision,
            "policy_version": self.engine.get_version(),
            "input_hash": digest(request["messages"]),
            "request": request,
            "stop_reason": resp.stop_reason,
            "synthetic": False,
        }
        if len(inp) + len(out) > self.context:
            raise ValueError("context budget exceeded")
        return result.model_dump(mode="json"), trace

    async def close_session(self, sid):
        client = self.clients.pop(sid, None)
        if client:
            await client.close()

    async def close(self):
        for sid in list(self.clients):
            await self.close_session(sid)
        self.engine.destroy()


class MockBackend:
    """Deterministic protocol fixture. Every artifact is marked synthetic."""

    kind = "mock"
    revision = "MOCK-NOT-A-MODEL"

    def open(self, sid):
        pass

    async def complete(self, sid, request):
        await asyncio.sleep(0.005)
        messages = request["messages"]
        users = [m["content"] for m in messages if m["role"] == "user"]
        users = [u if isinstance(u, str) else " ".join(x.get("text", "") for x in u) for u in users]
        text = users[-1]
        match = re.search(r"nonce=([\w-]+)", users[0])
        nonce = match.group(1) if match else "READY"
        tools = {t["function"]["name"] for t in request.get("tools", [])}
        results = [m for m in messages if m["role"] == "tool"]
        msg = {"role": "assistant", "content": "READY"}
        name, args = None, {}
        if "case=multi" in str(users[0]):
            msg["content"] = nonce
        elif results:
            result_text = results[-1]["content"]
            if not isinstance(result_text, str):
                result_text = " ".join(x.get("text", "") for x in result_text)
            if "TRANSIENT" in result_text and "fail_once" in tools:
                name = "fail_once"
            elif "case=chain" in str(users[0]) and "stage1:" in result_text:
                name, args = "advance", {"token": result_text.strip()}
            else:
                msg["content"] = result_text
        elif "case=chain" in text or "case=tool" in text:
            name = "lookup"
        elif "case=recovery" in text:
            name = "fail_once"
        elif "case=mcp" in text:
            name, args = "mcp_echo", {"text": nonce}
        else:
            msg["content"] = nonce
        if name:
            msg["content"] = None
            msg["tool_calls"] = [
                {
                    "id": "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        cid = "chatcmpl-" + uuid.uuid4().hex
        inp = list(json.dumps(messages).encode()) or [1]
        out = list(json.dumps(msg).encode())
        response = {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request["model"],
            "choices": [
                {"index": 0, "message": msg, "finish_reason": "tool_calls" if name else "stop"}
            ],
            "usage": {
                "prompt_tokens": len(inp),
                "completion_tokens": len(out),
                "total_tokens": len(inp) + len(out),
            },
        }
        trace = {
            "completion_id": cid,
            "input_ids": inp,
            "output_ids": out,
            "old_logprobs": [-0.1] * len(out),
            "loss_mask": [0] * len(inp) + [1] * len(out),
            "model_revision": self.revision,
            "input_hash": digest(messages),
            "request": request,
            "policy_version": 0,
            "stop_reason": "stop",
            "synthetic": True,
        }
        return response, trace

    async def close_session(self, sid):
        pass

    async def close(self):
        pass
