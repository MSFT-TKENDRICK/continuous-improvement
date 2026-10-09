"""Recorder, GuardView closure (B2), decision sink, domain pack and seed rules."""

from __future__ import annotations

import json

import pytest
from agent_framework import Content
from guards_support import (
    EXTRACTORS,
    SEED_RULES,
    TOOL_POLICIES,
    verify_access,
)

from ci_lab import rules
from ci_lab.guards import (
    STATE_KEY,
    JsonlDecisionSink,
    TrajectoryRecorder,
    read_decisions,
)
from ci_lab.guards.recorder import structured_result
from ci_lab.rulespec import ExtractorFile, GuardDecision, GuardView, RuleSpec


def test_structured_result_parses_json_only() -> None:
    assert structured_result('{"a": 1}') == {"a": 1}
    assert structured_result([Content.from_text('{"a":'), Content.from_text(" 2}")]) == {"a": 2}
    assert structured_result({"a": 3}) == {"a": 3}
    assert structured_result("change issued: RF-1") is None
    assert structured_result("[1, 2]") is None
    assert structured_result(None) is None


def test_recorder_steps_and_view_are_closed() -> None:
    rec = TrajectoryRecorder("c1")
    rec.record_user("hi, my email is a@b.co")
    rec.record_call("inspect_item", "k1", {"item_id": "item-a"})
    rec.record_result("inspect_item", "k1", '{"item_id": "item-a", "total": 5}')
    rec.record_call("apply_change", "k2", {"item_id": "item-a", "amount": 1}, blocked=True)
    rec.record_result("search_docs", "k3", '{"error": "boom"}')
    rec.record_response("Your email is a@b.co")
    kinds = [(s.kind, s.status) for s in rec.steps]
    assert kinds == [("user", None), ("tool_call", None), ("tool_result", "ok"), ("tool_call", "blocked"),
                     ("tool_result", "error"), ("response", None)]
    assert [s.i for s in rec.steps] == list(range(6))
    assert rec.steps[0].text is None and rec.steps[5].text is None  # digests only (B3, no PII at rest)
    assert rec.blocks_total == 1
    view = rec.view(rec.pending_call("apply_change", "k4", {"item_id": "x"}))
    assert set(GuardView.model_fields) == {"steps", "pending"}
    dumped = view.model_dump_json()
    for leak in ("suite", "split", "case", "env", "a@b.co"):
        assert leak not in dumped
    assert view.pending is not None and view.pending.i == 6


def test_recorder_mirrors_and_resumes_session_state() -> None:
    state: dict = {}
    rec = TrajectoryRecorder("c1", state=state)
    rec.record_call("verify_access", "k1", {"item_id": "item-a"})
    rec.record_result("verify_access", "k1", '{"verified": true, "item_id": "item-a"}')
    snapshot = json.loads(json.dumps(state))
    resumed = TrajectoryRecorder("c1", state=snapshot)
    assert resumed.steps == rec.steps
    assert snapshot[STATE_KEY]["version"] == 1


def test_jsonl_sink_roundtrip(tmp_path) -> None:
    sink = JsonlDecisionSink(tmp_path / "guards" / "decisions.jsonl")
    d = GuardDecision(rule_id="r.one", rule_version=1, mode="shadow", action="block", enforced=False,
                      step_index=3, target="apply_change", attempt_digest="sha256:x")
    sink(d)
    sink(d.model_copy(update={"step_index": 4}))
    assert [x.step_index for x in read_decisions(sink.path)] == [3, 4]


@pytest.mark.parametrize(("principal", "ok"), [
    ("reviewer", True),
    (" REVIEWER ", True),
    ("author", False),
    ("", False),
])
def test_verify_access(principal: str, ok: bool) -> None:
    assert verify_access("item-a", principal) == {"verified": ok, "item_id": "item-a"}


def test_verify_access_never_echoes_pii() -> None:
    got = verify_access("item-a", "reviewer")
    assert set(got) == {"verified", "item_id"}
    assert verify_access("item-z", "reviewer") == {"verified": False, "item_id": "item-z"}


def test_seed_files_validate_and_ship_shadow() -> None:
    import yaml

    ExtractorFile.model_validate(yaml.safe_load(EXTRACTORS.read_text(encoding="utf-8")))
    bundle = rules.load_bundle([SEED_RULES], [EXTRACTORS])  # RuleFile validation (YAML-1.1-safe `on:`)
    assert {r.id for r in bundle.rules} == {"action.requires_verified_access", "verify.before_inspect",
                                            "action.item_approved", "action.amount_within_limit",
                                            "sensitive.before_access"}
    assert all(r.mode == "shadow" for r in bundle.rules)
    assert all(isinstance(r, RuleSpec) for r in bundle.rules)
    assert bundle.extractors[0].ttl_steps == 50
    assert TOOL_POLICIES["apply_change"] and TOOL_POLICIES["escalate"]
    assert not any(TOOL_POLICIES[t] for t in ("inspect_item", "search_docs", "verify_access"))


def test_seed_pii_patterns() -> None:
    from ci_lab.rulespec import TrajectoryStep

    bundle = rules.load_bundle([SEED_RULES], [EXTRACTORS])
    text = ("Reach reviewer@example.com or +1-206-555-0141 / (503) 555-0177; located at 418 Alder St, "
            "Seattle and 21 Harbor Way. Updated 2026-09-10, total 89.50, item item-a.")
    view = GuardView(pending=TrajectoryStep(i=0, kind="response", text=text))
    matches = rules.evaluate(bundle, view, on="response")
    assert [m.rule.id for m in matches] == ["sensitive.before_access"]
    out = rules.redact(text, matches, bundle)
    for leak in ("reviewer@", "555-0141", "555-0177", "Alder", "Harbor"):
        assert leak not in out
    for keep in ("2026-09-10", "89.50", "item-a", "Seattle"):
        assert keep in out
