from __future__ import annotations

import math


def validate_trace(trace: dict) -> None:
    inp, out, logp = (trace[k] for k in ("input_ids", "output_ids", "old_logprobs"))
    if not inp or not out or len(out) != len(logp):
        raise ValueError("missing or misaligned generation tokens/logprobs")
    if any(type(x) is not int or x < 0 for x in inp + out):
        raise ValueError("invalid token ID")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x > 1e-4 for x in logp):
        raise ValueError("invalid behavior logprob")
    if not trace.get("completion_id") or not trace.get("model_revision"):
        raise ValueError("missing completion/model identity")
    expected = [0] * len(inp) + [1] * len(out)
    if trace["loss_mask"] != expected:
        raise ValueError("only newly generated tokens may have loss mask=1")


def sse_chunks(response: dict, fragment_size: int = 7):
    """OpenAI SSE transport from a completed generation; not live token streaming."""
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }

    def chunk(delta, finish=None):
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    msg = response["choices"][0]["message"]
    content = msg.get("content") or ""
    for i in range(0, len(content), fragment_size):
        yield chunk({"content": content[i : i + fragment_size]})
    for idx, call in enumerate(msg.get("tool_calls") or []):
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }
                ]
            }
        )
        args = call["function"]["arguments"]
        for i in range(0, len(args), fragment_size):
            yield chunk(
                {
                    "tool_calls": [
                        {"index": idx, "function": {"arguments": args[i : i + fragment_size]}}
                    ]
                }
            )
    yield chunk({}, response["choices"][0]["finish_reason"])
    yield {**base, "choices": [], "usage": response.get("usage", {})}
