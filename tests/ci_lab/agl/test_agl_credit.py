from __future__ import annotations

import asyncio
import json

import pytest

from ci_lab.agl.credit import Credit, RolloutDigest, assign_credit, journal_credits
from ci_lab.contracts import RolloutKey
from ci_lab.testing import FakeChatClient, MemoryJournal


def digest(case: str, *, score: float = 0.2, rules: tuple[str, ...] = (),
           metrics: dict[str, float] | None = None) -> RolloutDigest:
    return RolloutDigest(case, "harness_triage", score, rules, {"agent": 2}, metrics or {})


def test_digest_is_bounded_and_contains_only_typed_fields() -> None:
    d = digest("c1", rules=("workflow.bad_step",), metrics={"llm_calls": 3})
    assert set(d.as_dict()) == {
        "case", "suite", "score", "violation_rule_ids", "component_touches", "metrics"}
    with pytest.raises(ValueError, match="case"):
        digest("x" * 129)
    with pytest.raises(ValueError, match="metric"):
        RolloutDigest("c", "s", 0.5, metrics={"raw_tool_output": 1})


def test_assign_credit_strict_json_one_call() -> None:
    reply = json.dumps({"credits": [{
        "component": "workflow", "weight": 0.8, "reason_code": "violation.rule",
        "evidence_ids": ["c1:workflow.bad_step"]}]})
    client = FakeChatClient([reply])
    credits = asyncio.run(assign_credit([digest("c1")], client=client, components=("agent", "workflow")))
    assert credits == [Credit("workflow", 0.8, "violation.rule", ("c1:workflow.bad_step",))]
    assert len(client.requests) == 1
    assert client.requests[0][1]["response_format"]["json_schema"]["strict"] is True


def test_malformed_response_retries_then_falls_back() -> None:
    client = FakeChatClient(["not json", '{"credits":[{"component":"prompt"}]}'])
    digests = [
        digest("normal", rules=("workflow.invalid",), metrics={"llm_calls": 2, "tool_calls": 1, "tokens_in": 10}),
        digest("costly", metrics={"llm_calls": 20, "tool_calls": 10, "tokens_in": 1000}),
    ]
    credits = asyncio.run(assign_credit(digests, client=client, components=("agent", "loop", "workflow")))
    assert len(client.requests) == 2
    assert Credit("workflow", 1.0, "violation.rule", ("normal:workflow.invalid",)) in credits
    assert any(c.component == "loop" and c.reason_code == "cost.calls" for c in credits)
    assert any(c.component == "agent" and c.reason_code == "cost.tokens" for c in credits)


def test_credit_journal_is_idempotent() -> None:
    journal, key = MemoryJournal(), RolloutKey("camp-r00", "base", "case-1")
    journal.start(key, {})
    credit = Credit("agent", 0.7, "score.failure", ("case-1",))
    journal_credits(journal, key, [credit])
    journal_credits(journal, key, [credit])
    events = journal.events(key)
    assert len(events) == 1 and events[0]["event_type"] == "ci.credit"
    assert events[0]["data"] == {
        "component": "agent", "weight": 0.7, "reason_code": "score.failure",
        "evidence_ids": ["case-1"]}
