"""Recorder, GuardView closure (B2), decision sink, domain pack and seed rules."""

from __future__ import annotations

import json

import pytest
from agent_framework import Content

from ci_lab import rules
from ci_lab.guards import (
    STATE_KEY,
    JsonlDecisionSink,
    TrajectoryRecorder,
    read_decisions,
)
from ci_lab.guards.domains.order_support import (
    EXTRACTORS,
    SEED_RULES,
    TOOL_POLICIES,
    verify_identity,
)
from ci_lab.guards.recorder import structured_result
from ci_lab.rulespec import ExtractorFile, GuardDecision, GuardView, RuleSpec


def test_structured_result_parses_json_only() -> None:
    assert structured_result('{"a": 1}') == {"a": 1}
    assert structured_result([Content.from_text('{"a":'), Content.from_text(" 2}")]) == {"a": 2}
    assert structured_result({"a": 3}) == {"a": 3}
    assert structured_result("refund issued: RF-1") is None
    assert structured_result("[1, 2]") is None
    assert structured_result(None) is None


def test_recorder_steps_and_view_are_closed() -> None:
    rec = TrajectoryRecorder("c1")
    rec.record_user("hi, my email is a@b.co")
    rec.record_call("lookup_order", "k1", {"order_id": "NW-10001"})
    rec.record_result("lookup_order", "k1", '{"order_id": "NW-10001", "total": 5}')
    rec.record_call("issue_refund", "k2", {"order_id": "NW-10001", "amount": 1}, blocked=True)
    rec.record_result("search_kb", "k3", '{"error": "boom"}')
    rec.record_response("Your email is a@b.co")
    kinds = [(s.kind, s.status) for s in rec.steps]
    assert kinds == [("user", None), ("tool_call", None), ("tool_result", "ok"), ("tool_call", "blocked"),
                     ("tool_result", "error"), ("response", None)]
    assert [s.i for s in rec.steps] == list(range(6))
    assert rec.steps[0].text is None and rec.steps[5].text is None  # digests only (B3, no PII at rest)
    assert rec.blocks_total == 1
    view = rec.view(rec.pending_call("issue_refund", "k4", {"order_id": "x"}))
    assert set(GuardView.model_fields) == {"steps", "pending"}
    dumped = view.model_dump_json()
    for leak in ("suite", "split", "case", "env", "a@b.co"):
        assert leak not in dumped
    assert view.pending is not None and view.pending.i == 6


def test_recorder_mirrors_and_resumes_session_state() -> None:
    state: dict = {}
    rec = TrajectoryRecorder("c1", state=state)
    rec.record_call("verify_identity", "k1", {"order_id": "NW-10001"})
    rec.record_result("verify_identity", "k1", '{"verified": true, "order_id": "NW-10001"}')
    snapshot = json.loads(json.dumps(state))
    resumed = TrajectoryRecorder("c1", state=snapshot)
    assert resumed.steps == rec.steps
    assert snapshot[STATE_KEY]["version"] == 1


def test_jsonl_sink_roundtrip(tmp_path) -> None:
    sink = JsonlDecisionSink(tmp_path / "guards" / "decisions.jsonl")
    d = GuardDecision(rule_id="r.one", rule_version=1, mode="shadow", action="block", enforced=False,
                      step_index=3, target="issue_refund", attempt_digest="sha256:x")
    sink(d)
    sink(d.model_copy(update={"step_index": 4}))
    assert [x.step_index for x in read_decisions(sink.path)] == [3, 4]


@pytest.mark.parametrize(("name", "contact", "ok"), [
    ("Alex Rivera", "alex.rivera@example.com", True),
    ("  alex   RIVERA ", "ALEX.Rivera@Example.com", True),
    ("Alex Rivera", "0141", True),
    ("Alex Rivera", "(206) 555-0141", True),
    ("Alex Rivera", "+1 206 555 0141", True),
    ("Alex Rivera", "0142", False),
    ("Alex Rivera", "141", False),
    ("Alex Rivera", "alex@example.com", False),
    ("Alex Rivers", "0141", False),
    ("Alex Rivera", "", False),
])
def test_verify_identity(name: str, contact: str, ok: bool) -> None:
    got = verify_identity(" nw-10001 ", name, contact)
    assert got == {"verified": ok, "order_id": "NW-10001"}


def test_verify_identity_never_echoes_pii() -> None:
    got = verify_identity("NW-10001", "Alex Rivera", "0141")
    assert set(got) == {"verified", "order_id"}
    assert verify_identity("NW-99999", "Alex Rivera", "0141") == {"verified": False, "order_id": "NW-99999"}


def test_seed_files_validate_and_ship_shadow() -> None:
    import yaml

    ExtractorFile.model_validate(yaml.safe_load(EXTRACTORS.read_text(encoding="utf-8")))
    bundle = rules.load_bundle([SEED_RULES], [EXTRACTORS])  # RuleFile validation (YAML-1.1-safe `on:`)
    assert {r.id for r in bundle.rules} == {"refund.requires_verified_identity", "verify.before_lookup",
                                            "refund.order_eligible", "refund.amount_within_total",
                                            "pii.before_verification"}
    assert all(r.mode == "shadow" for r in bundle.rules)
    assert all(isinstance(r, RuleSpec) for r in bundle.rules)
    assert bundle.extractors[0].ttl_steps == 50
    assert TOOL_POLICIES["issue_refund"] and TOOL_POLICIES["escalate_to_human"]
    assert not any(TOOL_POLICIES[t] for t in ("lookup_order", "search_kb", "verify_identity"))


def test_seed_pii_patterns() -> None:
    from ci_lab.rulespec import TrajectoryStep

    bundle = rules.load_bundle([SEED_RULES], [EXTRACTORS])
    text = ("Reach alex.rivera@example.com or +1-206-555-0141 / (503) 555-0177; ships to 418 Alder St, "
            "Seattle and 21 Harbor Way. Delivered 2026-09-10, total 89.50, order NW-10001.")
    view = GuardView(pending=TrajectoryStep(i=0, kind="response", text=text))
    matches = rules.evaluate(bundle, view, on="response")
    assert [m.rule.id for m in matches] == ["pii.before_verification"]
    out = rules.redact(text, matches, bundle)
    for leak in ("alex.rivera", "555-0141", "555-0177", "Alder", "Harbor"):
        assert leak not in out
    for keep in ("2026-09-10", "89.50", "NW-10001", "Seattle"):
        assert keep in out
