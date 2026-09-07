import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "teacher_sft", Path(__file__).parents[1] / "scripts/prepare_teacher_sft.py"
)
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


def fixture():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Find the answer"}]},
        {
            "role": "assistant",
            "model": "deepseek-v4-flash",
            "stopReason": "toolUse",
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {
                    "type": "toolCall",
                    "id": "call1",
                    "name": "lookup",
                    "arguments": {"query": "answer"},
                },
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "call1",
            "toolName": "lookup",
            "content": [{"type": "text", "text": "tool evidence"}],
        },
        {
            "role": "assistant",
            "model": "deepseek-v4-flash",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "final answer"}],
        },
    ]
    return [{"type": "message", "message": m} for m in messages]


def test_conversion_preserves_evidence_and_excludes_thinking():
    messages, stats = sft.convert(fixture())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"query": "answer"}
    assert messages[3]["tool_call_id"] == "call1"
    assert "private reasoning" not in str(messages)
    assert stats["removed_thinking_blocks"] == 1


@pytest.mark.parametrize("mutation", ["missing_result", "wrong_name", "error", "compacted"])
def test_broken_sessions_rejected(mutation):
    rows = fixture()
    if mutation == "missing_result":
        rows.pop(2)
    elif mutation == "wrong_name":
        rows[2]["message"]["toolName"] = "other"
    elif mutation == "error":
        rows[-1]["message"]["stopReason"] = "error"
    else:
        rows.insert(2, {"type": "compaction"})
    with pytest.raises(ValueError):
        sft.convert(rows)


def test_bilingual_family_never_crosses_splits():
    assert sft.family("T001zh_email_triage") == sft.family("T002_email_triage")
    assert sft.split_for("T001zh_email_triage") == sft.split_for("T002_email_triage")


def test_export_full_prefix_only_one_target_and_no_grader(tmp_path, monkeypatch):
    import json

    source = tmp_path / "source"
    folder = source / "dsv4_flash"
    folder.mkdir(parents=True)
    session = tmp_path / "trial" / "sessions"
    session.mkdir(parents=True)
    (session / "one.jsonl").write_text("\n".join(json.dumps(r) for r in fixture()))
    cache = session.parent / "agent" / "mcp-cache.json"
    cache.parent.mkdir()
    cache.write_text("{}")
    monkeypatch.setattr(sft, "tools_from_cache", lambda *args: [])
    trial = {
        "trial_id": "trial1",
        "task_id": "T002_email_triage",
        "grading_results": [{"passed": True, "feedback": "SECRET GRADER ANSWER"}],
        "transcript": {
            "model_name": "deepseek-v4-flash",
            "metadata": {"claw_eval": {"pi_session_dir": str(session)}},
        },
    }
    (folder / "task.json").write_text(json.dumps({"trials": [trial]}))
    output = tmp_path / "output"
    report = sft.build(source, output)
    assert report["counts"]["episodes"] == 1
    rows = [
        json.loads(line)
        for p in output.glob("*.unlengthchecked.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(rows) == 2
    assert len(rows[-1]["completion"]) == 1
    assert rows[-1]["prompt"][-1]["role"] == "tool"
    assert "final answer" not in str(rows[-1]["prompt"])
    assert "SECRET GRADER ANSWER" not in str(rows)
    assert not report["length_checked"]

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["return_dict"] is False
            return list(range(10 if len(messages) == 3 else 20))

    bounded = tmp_path / "bounded"
    report2 = sft.build(source, bounded, FakeTokenizer(), {"revision": "fixture"}, 15)
    assert report2["counts"]["over_length_steps"] == 1
    assert report2["counts"]["train_samples"] + report2["counts"]["validation_samples"] == 1
    assert json.loads((bounded / "overlength.jsonl").read_text())["num_tokens"] == 20
    with pytest.raises(ValueError, match="already exists"):
        sft.build(source, output)


def test_cached_schema_required_and_arguments_validated(tmp_path):
    import json

    path = tmp_path / "mcp-cache.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "fixture": {
                        "tools": [
                            {
                                "name": "lookup",
                                "description": "Find evidence",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                    "additionalProperties": False,
                                },
                            }
                        ]
                    }
                }
            }
        )
    )
    messages, _ = sft.convert(fixture())
    assert len(sft.tools_from_cache(path, messages)) == 1
    messages[2]["tool_calls"][0]["function"]["arguments"]["query"] = 42
    with pytest.raises(ValueError, match="invalid_tool_arguments"):
        sft.tools_from_cache(path, messages)
    messages[2]["tool_calls"][0]["function"]["name"] = "unknown"
    with pytest.raises(ValueError, match="missing_tool_schema"):
        sft.tools_from_cache(path, messages)
