"""Guards inside REAL MAF tool loops (FakeChatClient drives agent_framework's function invocation)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agent_framework import AgentSession, MiddlewareFailure
from guards_support import SpyEngine, Tools, make_agent, seed_bundle

from ci_lab.guards import install_guards, read_decisions, resolve_mode
from ci_lab.guards.domains.order_support import TOOL_POLICIES
from ci_lab.guards.runtime import DEGRADED_RULE_ID
from ci_lab.rulespec import attempt_digest, guard_result_json
from ci_lab.testing import Call

ALEX = {"order_id": "NW-10001", "full_name": "Alex Rivera", "email_or_phone": "0141"}
REFUND = {"order_id": "NW-10001", "amount": 10.0}


def run(agent: Any, text: str = "hi", **kw: Any) -> Any:
    return asyncio.run(agent.run(text, **kw))


def results(client: Any, request: int) -> list[str]:
    """function_result payloads (exact strings) answering the latest tool batch in request ``request``."""
    msgs = client.requests[request][0]
    last = max(k for k, m in enumerate(msgs) if any(c.type == "function_call" for c in m.contents))
    return [c.result for m in msgs[last + 1:] for c in m.contents if c.type == "function_result"]


def test_block_short_circuits_and_exact_guard_json_reaches_model() -> None:
    agent, client, tools, rt = make_agent([[Call("issue_refund", REFUND, "c1")], "done"])
    assert run(agent).text == "done"
    assert tools.count("issue_refund") == 0
    first = rt.decisions[0]
    match = next(m for m in rt.engine.evaluate(rt.bundle, rt.engine.views[0], on="tool_call")
                 if m.rule.id == first.rule_id)
    assert results(client, 1) == [guard_result_json(match.rule.id, match.message, match.fix, match.see)]
    payload = json.loads(results(client, 1)[0])["guard"]
    assert payload["rule"] == "refund.amount_within_total" and "terminal" not in payload
    assert {d.rule_id for d in rt.decisions} == {"refund.requires_verified_identity", "refund.order_eligible",
                                                "refund.amount_within_total"}
    assert all(d.enforced and d.mode == "enforce" for d in rt.decisions)
    conv = next(iter(rt._convs.values()))
    blocked = [s for s in conv.recorder.steps if s.status == "blocked"]
    assert len(blocked) == 1 and blocked[0].tool == "issue_refund"
    assert first.attempt_digest == attempt_digest(rt.engine.views[0].pending)  # recorded pre-enforcement (B1)


def test_model_retries_after_guard_and_succeeds() -> None:
    script = [[Call("issue_refund", REFUND)], [Call("verify_identity", ALEX)],
              [Call("lookup_order", {"order_id": "NW-10001"})], [Call("issue_refund", REFUND)], "refunded"]
    agent, client, tools, rt = make_agent(script)
    assert run(agent).text == "refunded"
    assert tools.count("issue_refund") == 1
    assert '"refund_id"' in results(client, 4)[0]
    assert sum(d.enforced for d in rt.decisions) == 3  # only the first attempt matched


def test_shadow_records_without_changing_behavior(tmp_path) -> None:
    sink = tmp_path / "guards" / "decisions.jsonl"
    agent, client, tools, _ = make_agent([[Call("issue_refund", REFUND)], "done"], mode=None, sink=sink)
    run(agent)
    assert tools.count("issue_refund") == 1
    assert '"refund_id"' in results(client, 1)[0]
    got = read_decisions(sink)
    assert len(got) == 3 and all(d.mode == "shadow" and not d.enforced for d in got)


def test_off_still_records_attempts() -> None:
    agent, _, tools, rt = make_agent([[Call("issue_refund", REFUND)], "done"], mode="off")
    run(agent)
    assert tools.count("issue_refund") == 1
    assert len(rt.decisions) == 3 and not any(d.enforced for d in rt.decisions)
    assert all(d.mode == "off" for d in rt.decisions)


def test_ci_guards_env_read_by_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI_GUARDS", "enforce")
    assert resolve_mode(None) == "enforce"
    assert resolve_mode("shadow") == "shadow"  # explicit override wins
    agent, _, tools, _ = make_agent([[Call("issue_refund", REFUND)], "done"], mode=None)
    run(agent)
    assert tools.count("issue_refund") == 0
    monkeypatch.setenv("CI_GUARDS", "bogus")
    with pytest.raises(ValueError, match="guard mode"):
        install_guards(bundle=seed_bundle(), tool_policies=TOOL_POLICIES)


def test_terminal_after_max_blocks_stops_loop() -> None:
    script = [[Call("issue_refund", REFUND)], [Call("issue_refund", REFUND)], [Call("issue_refund", REFUND)],
              "never reached"]
    agent, client, tools, rt = make_agent(script)
    response = run(agent)
    assert tools.count("issue_refund") == 0
    assert len(client.requests) == 3  # loop stopped after the terminal block (B4)
    assert "never reached" not in response.text
    assert rt.template("guard.terminal").message in response.text
    conv = next(iter(rt._convs.values()))
    assert conv.turn.blocks == 3 and conv.turn.terminal and conv.recorder.blocks_total == 3


def test_terminal_json_payload_when_loop_continues() -> None:
    """Sibling calls in a terminal turn get the terminal guard JSON without running."""
    script = [[Call("issue_refund", REFUND)], [Call("issue_refund", REFUND)],
              [Call("issue_refund", REFUND, "t1"), Call("search_kb", {"query": "x"}, "t2")], "never"]
    agent, _, tools, rt = make_agent(script)
    run(agent)
    assert tools.count("issue_refund") == 0
    conv = next(iter(rt._convs.values()))
    assert conv.turn.terminal


def test_batch_preflight_two_refunds_only_authorized_one_runs() -> None:
    tools = Tools(refund_delay=0.02)
    script = [[Call("verify_identity", ALEX)], [Call("lookup_order", {"order_id": "NW-10001"})],
              [Call("issue_refund", REFUND, "r1"), Call("issue_refund", {"order_id": "NW-10002", "amount": 10.0}, "r2")],
              "done"]
    agent, client, tools, _ = make_agent(script, tools=tools)
    run(agent)
    assert [a["order_id"] for n, a in tools.calls if n == "issue_refund"] == ["NW-10001"]
    assert tools.max_parallel_side_effects == 1  # serialized per conversation (B5)
    out = results(client, 3)
    assert '"refund_id"' in out[0] and json.loads(out[1])["guard"]["rule"].startswith("refund.")


def test_batch_preflight_uses_pre_batch_snapshot() -> None:
    """verify_identity + issue_refund in ONE batch: the refund is judged on the pre-batch snapshot,
    so it is blocked regardless of scheduling (deterministic)."""
    script = [[Call("lookup_order", {"order_id": "NW-10002"})],
              [Call("verify_identity", {**ALEX, "order_id": "NW-10002", "full_name": "Priya Nair",
                                        "email_or_phone": "0177"}),
               Call("issue_refund", {"order_id": "NW-10002", "amount": 5.0})], "done"]
    agent, _, tools, rt = make_agent(script, bundle=None)
    run(agent)
    assert tools.count("issue_refund") == 0
    assert any(d.rule_id == "refund.requires_verified_identity" and d.enforced for d in rt.decisions)


def test_preflight_snapshot_includes_earlier_allowed_batch_calls() -> None:
    """Two refunds for the same order in one batch see each other (count rules stay exact)."""
    from ci_lab import rules
    from ci_lab.rulespec import RuleSpec

    once = RuleSpec.model_validate({
        "id": "refund.once", "version": 1, "rung": "R2", "on": "tool_call", "target": "issue_refund",
        "require": {"kind": "count", "tool": "issue_refund", "op": "eq", "n": 0}, "action": "block",
        "template": "count.exceeded", "slots": {"tool": "issue_refund"}})
    bundle = rules.build_bundle([once])
    script = [[Call("issue_refund", REFUND, "a"), Call("issue_refund", REFUND, "b")], "done"]
    agent, _, tools, _ = make_agent(script, bundle=bundle)
    run(agent)
    assert tools.count("issue_refund") == 1


def test_fail_closed_on_side_effect_tool() -> None:
    agent, _, tools, rt = make_agent([[Call("issue_refund", REFUND)], "done"], engine=SpyEngine(fail=True))
    with pytest.raises(MiddlewareFailure):
        run(agent)
    assert tools.count("issue_refund") == 0
    assert rt.decisions[-1].rule_id == DEGRADED_RULE_ID and rt.decisions[-1].degraded


def test_degraded_warn_only_on_read_only_tool() -> None:
    agent, _, tools, rt = make_agent([[Call("search_kb", {"query": "returns"})], "done"],
                                     engine=SpyEngine(fail=True))
    assert run(agent).text == "done"
    assert tools.count("search_kb") == 1
    assert rt.decisions and all(d.rule_id == DEGRADED_RULE_ID and d.degraded and not d.enforced
                                for d in rt.decisions)  # tool call + response lint both degraded
    assert rt.decisions[0].target == "search_kb"


def test_unavailable_bundle_fails_closed_for_side_effects() -> None:
    mw = install_guards(bundle=None, tool_policies=TOOL_POLICIES)
    assert mw.runtime.degraded
    from agent_framework import Agent

    from ci_lab.testing import FakeChatClient
    tools = Tools()
    agent = Agent(client=FakeChatClient([[Call("lookup_order", {"order_id": "NW-10001"})],
                                         [Call("escalate_to_human", {"reason": "x"})], "done"]),
                  tools=tools.all(), middleware=mw)
    with pytest.raises(MiddlewareFailure):
        run(agent)
    assert tools.count("lookup_order") == 1 and tools.count("escalate_to_human") == 0


def test_unknown_tools_default_to_side_effecting() -> None:
    mw = install_guards(bundle=seed_bundle(), tool_policies={"search_kb": False})
    assert mw.runtime.is_side_effect("brand_new_tool") and not mw.runtime.is_side_effect("search_kb")
    mw = install_guards(bundle=seed_bundle(), tool_policies={"x": {"side_effect": True}}, default_side_effect=False)
    assert mw.runtime.is_side_effect("x") and not mw.runtime.is_side_effect("y")


def test_views_never_carry_run_metadata() -> None:
    agent, _, _, rt = make_agent([[Call("lookup_order", {"order_id": "NW-10001"})], "done"])
    meta = {"suite": "s1", "split": "heldout", "case": "c9", "env": "prod"}
    run(agent, options={"metadata": meta}, function_invocation_kwargs=meta)
    assert rt.engine.views
    for view in rt.engine.views:
        dumped = view.model_dump_json()
        assert not any(k in dumped for k in ('"suite"', '"split"', '"case"', '"env"'))


def test_session_checkpoint_resume_preserves_guard_state() -> None:
    """Guard state mirrors into AgentSession.state, so to_dict/from_dict resumes it (flags included)."""
    script1 = [[Call("verify_identity", ALEX)], [Call("lookup_order", {"order_id": "NW-10001"})], "verified"]
    agent, _, _, _ = make_agent(script1)
    session = AgentSession(session_id="conv-1")
    run(agent, session=session)
    saved = json.loads(json.dumps(session.to_dict()))
    restored = AgentSession.from_dict(saved)
    agent2, _, tools2, rt2 = make_agent([[Call("issue_refund", REFUND)], "refunded"])  # fresh process
    run(agent2, session=restored)
    assert tools2.count("issue_refund") == 1
    assert not any(d.enforced for d in rt2.decisions)
    # without the restored state the same call is blocked
    agent3, _, tools3, _ = make_agent([[Call("issue_refund", REFUND)], "refunded"])
    run(agent3, session=AgentSession(session_id="conv-2"))
    assert tools3.count("issue_refund") == 0


def test_blocks_per_turn_reset_but_conversation_ceiling_holds() -> None:
    script = [[Call("issue_refund", REFUND)], [Call("issue_refund", REFUND)], "t1",
              [Call("issue_refund", REFUND)], [Call("issue_refund", REFUND)], "t2",
              [Call("issue_refund", REFUND)], "t3"]
    agent, _, _, rt = make_agent(script, max_blocks_per_conversation=4)
    session = AgentSession(session_id="conv-3")
    assert run(agent, session=session).text == "t1"
    assert run(agent, session=session).text == "t2"
    third = run(agent, session=session)
    assert rt.template("guard.terminal").message in third.text  # 5th block overall => terminal
