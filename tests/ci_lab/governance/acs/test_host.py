"""ACS host obligations: enforcement, approval path and audit records (spec §5, §17, §19)."""

from __future__ import annotations

import asyncio

import pytest

from ci_lab.governance.acs import (
    AcsRuntime,
    AgentControl,
    AgentControlBlocked,
    AgentControlSuspended,
    ApprovalOutcome,
    ApprovalResolution,
)

MANIFEST = """\
agent_control_specification_version: 0.4.0-alpha.1
policies:
  p: {type: custom, adapter: fixture}
intervention_points:
  input: {policy: {id: p}, policy_target: $snap.input}
  output: {policy: {id: p}, policy_target: $snap.output.text}
approval:
  default_resolver: human
  on_timeout: %s
  fatigue_threshold: 2
  fatigue_window_seconds: 60
"""
ALLOW = {"decision": "allow"}
ESCALATE = {"decision": "escalate", "reason": "review"}


def _control(responses, resolver=None, mode="enforce", on_timeout="deny", **kwargs):
    log: list[dict] = []
    rt = AcsRuntime(
        MANIFEST % on_timeout,
        dispatcher={
            "fixture": lambda inv: responses[inv["input"]["intervention_point"]]
        },
    )
    ctl = AgentControl(rt, resolver, mode, decision_log=log.append, **kwargs)
    return ctl, log


def _run(ctl, value="hi"):
    calls = []

    def action(v):
        calls.append(v)
        return {"text": f"echo {v}", "n": 1}

    return asyncio.run(ctl.run(value, action)), calls


def _blocked(ctl, exc=AgentControlBlocked):
    with pytest.raises(exc) as info:
        _run(ctl)
    return info.value


def test_transforms_are_applied_in_enforce_and_logged_without_content():
    ctl, log = _control(
        {
            "input": {
                "decision": "transform",
                "transform": {"path": "$target", "value": "HI"},
            },
            "output": {
                "decision": "transform",
                "transform": {"path": "$target", "value": "x"},
            },
        }
    )
    out, calls = _run(ctl)
    assert calls == ["HI"] and out == {"text": "x", "n": 1}
    assert [r["decision"] for r in log] == ["transform", "transform"]
    assert set(log[0]) == {
        "point", "decision", "reason", "approval", "input_identity", "enforced_identity", "mode", "ts",
    }  # fmt: skip
    assert log[0]["input_identity"] != log[0]["enforced_identity"]


def test_final_deny_blocks_before_the_action():
    ctl, log = _control({"input": {"decision": "deny", "reason": "nope"}})
    err = _blocked(ctl)
    assert err.intervention_point == "input" and err.result.verdict.reason == "nope"
    assert log[0]["decision"] == "deny" and log[0]["approval"] is None


def test_evaluate_only_records_and_proceeds_untransformed():
    seen = []
    ctl, log = _control(
        {"input": {"decision": "deny"}, "output": ESCALATE},
        resolver=lambda p, r: seen.append(p),
        mode="evaluate_only",
    )
    out, calls = _run(ctl)
    assert calls == ["hi"] and out == {"text": "echo hi", "n": 1} and seen == []
    assert [(r["decision"], r["mode"]) for r in log] == [("deny", "evaluate_only")] * 2


@pytest.mark.parametrize(
    ("resolver", "reason"),
    [
        (None, "host_error:approval_unresolved"),
        ({"other": lambda p, r: ApprovalOutcome.ALLOW}, "host_error:approval_unresolved"),
        (lambda p, r: 1 / 0, "host_error:approval_resolver_failed"),
        (lambda p, r: "yes", "host_error:approval_resolver_failed"),
        (lambda p, r: ApprovalResolution.allow("sha256:other"), "host_error:approval_identity_mismatch"),
        (lambda p, r: ApprovalResolution.deny("not today"), "review"),
    ],
)  # fmt: skip
def test_liftable_deny_fails_closed_unless_approved(resolver, reason):
    ctl, _ = _control({"input": ESCALATE}, resolver)
    assert _blocked(ctl).result.verdict.reason == reason


def test_approval_binds_to_the_rederived_enforced_identity():
    async def mutate(point, result):
        approved = result.action_identity
        result.policy_input["snapshot"]["input"] = "mutated"
        return ApprovalResolution.allow(approved)

    ctl, _ = _control({"input": ESCALATE}, mutate)
    assert (
        _blocked(ctl).result.verdict.reason == "host_error:approval_identity_mismatch"
    )


def test_approved_allow_and_suspend():
    ctl, log = _control(
        {"input": ESCALATE, "output": ALLOW},
        {"human": lambda p, r: ApprovalResolution.allow(r.action_identity)},
    )
    assert _run(ctl)[1] == ["hi"] and log[0]["approval"] == "allow"
    ctl, _ = _control(
        {"input": ESCALATE, "output": ALLOW}, lambda p, r: ApprovalOutcome.ALLOW
    )
    assert _run(ctl)[1] == ["hi"]
    ctl, log = _control(
        {"input": ESCALATE},
        lambda p, r: ApprovalResolution.suspend("h1", r.action_identity),
    )
    assert _blocked(ctl, AgentControlSuspended).handle == "h1"
    assert log[0]["approval"] == "suspend"


@pytest.mark.parametrize("on_timeout", ["deny", "allow", "suspend"])
def test_approval_timeout_policy(on_timeout):
    async def slow(point, result):
        await asyncio.sleep(1)

    responses = {"input": ESCALATE, "output": ALLOW}
    ctl, _ = _control(
        responses, slow, on_timeout=on_timeout, approval_timeout_seconds=0.01
    )
    if on_timeout == "allow":
        assert _run(ctl)[1] == ["hi"]
    elif on_timeout == "suspend":
        _blocked(ctl, AgentControlSuspended)
    else:
        assert _blocked(ctl).result.verdict.reason == "host_error:approval_unresolved"


def test_approval_fatigue_fails_closed_within_the_window():
    now = [0.0]
    ctl, _ = _control(
        {"input": ESCALATE, "output": ALLOW},
        lambda p, r: ApprovalOutcome.ALLOW,
        clock=lambda: now[0],
    )
    _run(ctl)
    _run(ctl)
    assert _blocked(ctl).result.verdict.reason == "host_error:approval_unresolved"
    now[0] = 61.0
    assert _run(ctl)[1] == ["hi"]


def test_guard_returns_result_and_rejects_bad_mode():
    ctl, _ = _control({"input": ALLOW})
    result = asyncio.run(ctl.guard("input", {"input": "x"}))
    assert (
        result.verdict.decision == "allow" and result.transformed_policy_target is None
    )
    with pytest.raises(ValueError, match="mode"):
        AgentControl(ctl.runtime, mode="shadow")
