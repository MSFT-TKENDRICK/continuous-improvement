"""R3 output lint (non-streaming + GuardedStream), anti-gaming, telemetry and LKG loading."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agent_framework import AgentSession
from guards_support import (
    EXTRACTORS,
    SEED_RULES,
    SpyEngine,
    make_agent,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import rules
from ci_lab.contracts import (
    ATTR_GUARD_ACTION,
    ATTR_GUARD_BUNDLE,
    ATTR_GUARD_ENFORCED,
    ATTR_GUARD_MODE,
    ATTR_GUARD_RULE,
    ATTR_GUARD_VERSION,
    SPAN_GUARD,
)
from ci_lab.guards import GuardStreamingError, load_guard_bundle
from ci_lab.rules import Match
from ci_lab.rulespec import RuleSpec
from ci_lab.testing import Call

ACCESS = {"item_id": "item-a", "principal": "reviewer"}
LEAKY = "Contact reviewer@example.com or +1-206-555-0141; item-a is pending."


def run(agent: Any, text: str = "hi", **kw: Any) -> Any:
    return asyncio.run(agent.run(text, **kw))


def stream(agent: Any, text: str = "hi", **kw: Any) -> tuple[list[Any], Any]:
    async def go() -> tuple[list[Any], Any]:
        s = agent.run(text, stream=True, **kw)
        updates = [u async for u in s]
        return updates, await s.get_final_response()
    return asyncio.run(go())


def test_redacts_pii_before_verification() -> None:
    agent, _, _, rt = make_agent([LEAKY])
    out = run(agent).text
    assert "reviewer@" not in out and "555-0141" not in out and "item-a" in out
    assert [(d.rule_id, d.action, d.enforced) for d in rt.decisions] == [("sensitive.before_access", "redact", True)]


def test_no_redaction_after_verification() -> None:
    agent, _, _, rt = make_agent([[Call("verify_access", ACCESS)], LEAKY])
    assert run(agent).text == LEAKY
    assert rt.decisions == []


def test_shadow_redact_records_but_delivers_original() -> None:
    agent, _, _, rt = make_agent([LEAKY], mode="shadow")
    assert run(agent).text == LEAKY
    assert rt.decisions[0].rule_id == "sensitive.before_access" and not rt.decisions[0].enforced


class _ResponseBlockEngine(SpyEngine):
    """The contract forbids `target: "*"` blocks (B2), so response blocks are unreachable from
    validated rules today; this injects one to exercise the middleware path defensively."""

    def evaluate(self, bundle: Any, view: Any, *, on: str) -> list[Any]:
        out = super().evaluate(bundle, view, on=on)
        if on == "response" and "INTERNAL-" in (view.pending.text or ""):
            rule = RuleSpec.model_construct(id="response.no_internal_codes", version=1, rung="R3", on="response",
                                            target="*", action="block", mode="enforce", template="response.blocked")
            out.append(Match(rule=rule, message="m", fix="f", see="", step_index=view.pending.i))
        return out


def test_response_block_replaces_text_and_later_rules_see_original() -> None:
    agent, _, _, rt = make_agent(["code INTERNAL-42 for reviewer@example.com"], engine=_ResponseBlockEngine())
    out = run(agent).text
    assert "INTERNAL-42" not in out and "reviewer@" not in out
    assert out == rt.template("response.blocked").message
    # both rules evaluated the ORIGINAL text and were recorded (N6)
    assert {d.rule_id for d in rt.decisions} == {"sensitive.before_access", "response.no_internal_codes"}


def test_streaming_is_fully_buffered_and_redacted() -> None:
    agent, _, tools, rt = make_agent([[Call("inspect_item", {"item_id": "item-a"})], LEAKY], streaming=True)
    updates, final = stream(agent)
    streamed = "".join(u.text or "" for u in updates)
    assert "reviewer@" not in streamed and "[redacted]" in streamed
    assert final.text == streamed.strip() or "reviewer@" not in final.text
    assert tools.count("inspect_item") == 1
    assert [d.rule_id for d in rt.decisions] == ["sensitive.before_access"]


def test_streaming_first_update_only_after_full_run() -> None:
    agent, _, tools, _ = make_agent([[Call("inspect_item", {"item_id": "item-a"})], "done"], streaming=True)

    async def go() -> int:
        s = agent.run("hi", stream=True)
        async for _ in s:
            return tools.count("inspect_item")  # tools already ran when the first update arrives
        return -1
    assert asyncio.run(go()) == 1


def test_streaming_guard_blocks_side_effect() -> None:
    script = [[Call("apply_change", {"item_id": "item-a", "amount": 5.0})], "done"]
    agent, _, tools, _ = make_agent(script, streaming=True)
    _, final = stream(agent)
    assert tools.count("apply_change") == 0 and final.text == "done"


def test_streaming_without_buffering_raises() -> None:
    agent, *_ = make_agent(["hello"], streaming=True, buffer_streams=False)
    with pytest.raises(GuardStreamingError):
        stream(agent)


def test_anti_gaming_verify_after_lookup_is_blocked() -> None:
    script = [[Call("inspect_item", {"item_id": "item-a"})], [Call("verify_access", ACCESS)],
              [Call("apply_change", {"item_id": "item-a", "amount": 5.0})], "done"]
    agent, _, tools, rt = make_agent(script)
    run(agent)
    assert tools.count("verify_access") == 0 and tools.count("apply_change") == 0
    assert "verify.before_inspect" in [d.rule_id for d in rt.decisions if d.enforced]


def test_span_events_and_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_framework.observability as maf_obs

    exporter = InMemorySpanExporter()
    provider = TracerProvider()  # local provider: never set globally
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    # MAF opens its own agent/tool spans from the global (no-op here) provider; route them locally.
    monkeypatch.setattr(maf_obs, "get_tracer", lambda *a, **k: provider.get_tracer("maf"))
    agent, _, _, rt = make_agent([[Call("apply_change", {"item_id": "item-a", "amount": 5.0})], "done"])
    with tracer.start_as_current_span("turn"):
        run(agent)
    spans = exporter.get_finished_spans()
    events = [e for s in spans for e in s.events if e.name == SPAN_GUARD]
    assert len(events) == len(rt.decisions) == 3
    attrs = events[0].attributes
    assert attrs[ATTR_GUARD_MODE] == "enforce" and attrs[ATTR_GUARD_ENFORCED] is True
    assert attrs[ATTR_GUARD_BUNDLE] == rt.bundle.digest and attrs[ATTR_GUARD_VERSION] == 1
    assert {e.attributes[ATTR_GUARD_RULE] for e in events} == {d.rule_id for d in rt.decisions}
    assert any(s.attributes.get(ATTR_GUARD_ACTION) == "block" for s in spans)


def test_load_bundle_lkg_and_unavailable(tmp_path) -> None:
    gdir = tmp_path / "guards"
    gdir.mkdir()
    (gdir / "rules.yaml").write_bytes(SEED_RULES.read_bytes())
    bundle, degraded = load_guard_bundle(gdir, extractor_paths=(EXTRACTORS,))
    assert bundle is not None and not degraded
    rules.write_lkg(gdir, bundle)
    (gdir / "rules.yaml").write_text("schema_version: 1\nrules: [{id: broken}]\n", encoding="utf-8")
    lkg, degraded = load_guard_bundle(gdir, extractor_paths=(EXTRACTORS,))
    assert degraded and lkg is not None and lkg.digest == bundle.digest
    (gdir / "BUNDLE.lock").unlink()
    assert load_guard_bundle(gdir, extractor_paths=(EXTRACTORS,)) == (None, True)
    agent, _, _, rt = make_agent([[Call("search_docs", {"query": "x"})], "done"], bundle=lkg, degraded=True)
    run(agent)
    assert all(d.degraded for d in rt.decisions)


def test_state_flags_require_extractors_and_problems_are_logged(tmp_path, caplog) -> None:
    gdir = tmp_path / "guards"
    gdir.mkdir()
    (gdir / "rules.yaml").write_bytes(SEED_RULES.read_bytes())
    with caplog.at_level("ERROR", logger="ci_lab.guards.engine"):
        assert load_guard_bundle(gdir) == (None, True)  # access_verified has no extractor
    assert "[GUARDS][ERROR]" in caplog.text and "access_verified" in caplog.text


def test_session_state_is_json_and_pii_free() -> None:
    agent, *_ = make_agent([[Call("inspect_item", {"item_id": "item-a"})], LEAKY])
    session = AgentSession(session_id="pii")
    run(agent, "my email is reviewer@example.com", session=session)
    blob = json.dumps(session.state["ci_lab.guards"])
    assert "Your email" not in blob and "my email" not in blob
