import json

import pytest

from order_support import replay


def test_inference_set_is_current():
    assert replay.is_current(), "run: uv run order-support-evals replay build"


def test_rows_never_leak_labels():
    for case in replay.load_cases():
        text = json.dumps(replay.case_to_row(case))
        assert "labels" not in text and "human_pass" not in text
        for note_key in ("notes", "rationale"):
            if isinstance(case.get(note_key), str) and len(case[note_key]) > 20:
                assert case[note_key] not in text


def test_row_shape():
    rows = [replay.case_to_row(c) for c in replay.load_cases()]
    assert len(rows) == 30
    for row in rows:
        assert row["type"] == "prompt" and row["stop_reason"] == "completed"
        events = row["events"]
        assert events[0]["edit"]["type"] == "set_system_message"
        assert events[-1]["actor"] == "target"
        assert events[-1]["edit"]["message"]["role"] == "assistant"


def test_tool_call_placement_after_message():
    case = next(c for c in replay.load_cases() if c["id"] == "c26")
    events = replay.case_to_row(case)["events"]
    kinds = [e["edit"]["type"] if e["edit"]["type"] != "add_message" else e["edit"]["message"]["role"]
             for e in events]
    # lookup_order runs after the first user turn, before the earlier assistant reply.
    assert kinds[:4] == ["set_system_message", "user", "tool_call", "assistant"]


def test_bad_after_message_rejected():
    case = {"id": "x", "tags": ["t"], "observable": {
        "agent_policy": "p", "final_response": "ok",
        "conversation": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}],
        "tool_calls": [{"name": "lookup_order", "arguments": {}, "result": {}, "after_message": 1}]}}
    with pytest.raises(ValueError):
        replay.case_to_row(case)
