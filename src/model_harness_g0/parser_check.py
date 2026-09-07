"""Fixed legal Qwen XML fixtures; executed by the real AReaL backend at startup."""

import json


def check_parser(parse):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "count": {"type": "integer"},
                        "ok": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    fixtures = [
        ("<parameter=text>你好\nline two</parameter>", {"text": "你好\nline two"}),
        ('<parameter=text>a "quoted" value</parameter>', {"text": 'a "quoted" value'}),
        ("<parameter=count>3</parameter><parameter=ok>true</parameter>", {"count": 3, "ok": True}),
    ]
    for body, expected in fixtures:
        raw = f"<tool_call><function=probe>{body}</function></tool_call>"
        calls, _, finish = parse(raw, tools, "qwen3_coder", "qwen3", "stop")
        if not calls or len(calls) != 1 or finish != "tool_calls":
            raise ValueError("Qwen tool parser failed a legal fixture")
        if calls[0].function.name != "probe" or json.loads(calls[0].function.arguments) != expected:
            raise ValueError("Qwen tool parser changed fixture arguments")
    return {"passed": True, "fixtures": len(fixtures), "parser": "qwen3_coder"}
