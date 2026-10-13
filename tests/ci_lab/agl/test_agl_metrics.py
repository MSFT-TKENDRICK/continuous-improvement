from __future__ import annotations

from pathlib import Path

from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.agl.metrics import rollout_metrics
from ci_lab.agl.scope import RolloutScope
from ci_lab.contracts import RolloutKey, op_id


def test_rollout_metrics_is_pure_and_summary_events_override() -> None:
    events = [
        {"event_type": "model_request", "data": {
            "latency_ms": 10.5, "usage": {"prompt_tokens": 7, "completion_tokens": 3}}},
        {"event_type": "model_request", "data": {
            "latency_ms": 2, "usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"event_type": "ci.tool_call", "data": {"wall_ms": 4}},
        {"event_type": "ci.runtime", "data": {
            "llm_calls": 2, "tool_calls": 3, "tokens_in": 11, "tokens_out": 6, "wall_ms": 8}},
        {"event_type": "ci.metric", "data": {
            "llm_calls": 99, "tool_calls": 99, "tokens_in": 99, "tokens_out": 99, "wall_ms": 99}},
        {"event_type": "ci.score", "data": {"wall_ms": 500, "llm_calls": 20}},
    ]
    assert rollout_metrics(events) == {
        "wall_ms": 500.0, "llm_calls": 20, "tool_calls": 99, "tokens_in": 99, "tokens_out": 99}
    assert events[0]["data"]["latency_ms"] == 10.5


def test_scope_metric_is_one_logical_event_on_replay(tmp_path: Path) -> None:
    journal = FileRolloutJournal(tmp_path, fsync=False)
    key = RolloutKey("camp-r00", "base", "case-1")
    for _ in range(3):
        with RolloutScope(journal, key) as scope:
            scope.record_model_request(
                {"latency_ms": 5, "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
                name="request")
            scope.record_tool_call("read_component", name="tool")
    metrics = [e for e in journal.events(key) if e["event_type"] == "ci.metric"]
    assert len(metrics) == 1
    assert metrics[0]["event_id"] == op_id(key.rollout_id, "0", "ci.metric", "finish")
    assert metrics[0]["data"]["llm_calls"] == 1
    assert metrics[0]["data"]["tool_calls"] == 1
