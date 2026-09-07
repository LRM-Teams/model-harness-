import pytest

from model_harness_g0.sft import encode_sample, stratified


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        assert kwargs["enable_thinking"] is False
        return [10, 20, 30] if kwargs["add_generation_prompt"] else [10, 20, 30, 40, 50]


def row():
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "evidence"},
        ],
        "completion": [{"role": "assistant", "content": "answer"}],
        "tools": [],
        "sample_id": "s1",
        "task_id": "t1",
        "family": "f1",
    }


def test_masks_entire_history_and_keeps_target():
    encoded = encode_sample(Tokenizer(), row(), 5)
    assert encoded["labels"] == [-100, -100, -100, 40, 50]
    assert encoded["target_tokens"] == 2
    assert encode_sample(Tokenizer(), row(), 4) is None


def test_rejects_template_that_changes_prompt():
    class Bad(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [10, 20] if kwargs["add_generation_prompt"] else [10, 21, 40]

    with pytest.raises(ValueError, match="Non-prefix"):
        encode_sample(Bad(), row(), 100)


def test_rejects_supervising_tool_result():
    r = row()
    r["completion"][0]["role"] = "tool"
    with pytest.raises(ValueError, match="assistant"):
        encode_sample(Tokenizer(), r, 100)


def test_stratified_selection_covers_tasks_before_repeats():
    rows = [{"sample_id": str(i), "task_id": t} for i, t in enumerate(["a", "a", "a", "b", "c"])]
    assert {r["task_id"] for r in stratified(rows, 3)} == {"a", "b", "c"}
    assert len(stratified(rows, 0)) == 5
