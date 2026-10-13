"""Governed-factory wiring: the order-support target and the ``agt.governed-agent-factory`` rule."""

from __future__ import annotations

import json

import pytest

from ci_lab.governance.audit import AuditTrail
from ci_lab.testing import Call, FakeChatClient
from order_support import agent as os_agent
from order_support import data, guarding
from order_support import tools as os_tools


def _records(trail):
    out = []
    for line in trail.path.read_text("utf-8").splitlines():
        e = json.loads(line)
        out.append({**e["data"], "decision": e["policy_decision"], "agent_did": e["agent_did"]})
    return out


@pytest.fixture
def order_agent(monkeypatch):
    executed = []
    real = os_tools.execute

    def spy(name, args):
        executed.append(name)
        return real(name, args)

    monkeypatch.setattr(os_tools, "execute", spy)
    for name in (guarding.DECISIONS_ENV, guarding.GUARDS_ENV, os_agent.TIMEOUT_ENV, os_agent.HARNESS_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(os_agent.PROFILE_ENV, "fake")
    guarding.reset()
    yield executed
    os_agent.set_client_override(None)
    guarding.reset()


def test_order_support_refund_without_verification_is_blocked(order_agent, monkeypatch):
    monkeypatch.setenv("CI_GOVERNANCE_MODE", "enforce")  # guards stay shadow: only ACS blocks
    refund = Call("issue_refund", {"order_id": "NW-10001", "amount": 10.0}, "r1")
    os_agent.set_client_override(FakeChatClient([[refund], "Refunded!"]))
    os_agent.chat("refund NW-10001 please")
    assert "issue_refund" not in order_agent
    rec = _records(AuditTrail())
    assert any(r["reason"] == "identity_not_verified" and r["decision"] == "deny"
               and r["agent_did"].startswith("did:ci-lab:agent:") for r in rec)


def test_order_support_refund_after_verification_runs(order_agent):
    customer = data.ORDERS["NW-10001"]["customer"]
    verify = Call("verify_identity", {"order_id": "NW-10001", "full_name": customer["name"],
                                      "email_or_phone": customer["email"]}, "v1")
    refund = Call("issue_refund", {"order_id": "NW-10001", "amount": 10.0}, "r1")
    os_agent.set_client_override(FakeChatClient([[verify], [refund], "Refunded!"]))
    assert os_agent.chat("refund NW-10001 please") == "Refunded!"
    assert order_agent == ["verify_identity", "issue_refund"]


def test_target_mode_is_evaluate_only_unless_governance_mode_is_set():
    from ci_lab.governance.policies import target_mode

    assert target_mode({}) == "evaluate_only" and target_mode({"CI_GUARDS": "enforce"}) == "evaluate_only"
    assert target_mode({"CI_GOVERNANCE_MODE": "enforce"}) == "enforce"
    with pytest.raises(RuntimeError):
        target_mode({"CI_GOVERNANCE_MODE": "bogus"})


def test_lint_rule_flags_ungoverned_construction_outside_the_factory():
    from pathlib import Path

    from ci_lab.lint.engine import lint
    from ci_lab.lint.spec import load_rules

    root = Path(__file__).resolve().parents[3]
    rules = load_rules([root / "lint" / "rules" / "governance.yaml"])
    files = {
        "src/ci_lab/x.py": "from agent_framework import Agent as A\na = A(client=None)\n",
        "src/ci_lab/y.py": "agent = factory.create_agent_from_dict({})\n",
        "src/ci_lab/z.py": "from ci_lab.governance.maf import governed_agent\ngoverned_agent(client=None)\n",
        "src/ci_lab/governance/maf.py": "from agent_framework import Agent\nAgent()\n",
    }
    res = lint(root, rules, list(files), reader=files.get)
    assert sorted((f.path, f.rule) for f in res.findings) == [
        ("src/ci_lab/x.py", "agt.governed-agent-factory"), ("src/ci_lab/y.py", "agt.governed-agent-factory")]
