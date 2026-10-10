"""MAF host (ci_lab.governance.maf): real Agent + FakeChatClient runs through the governed factory."""

from __future__ import annotations

import asyncio
import json

import pytest

from ci_lab.governance import maf as gmaf
from ci_lab.governance.audit import AuditTrail
from ci_lab.governance.hypervisor import KillSwitchAdapter
from ci_lab.testing import Call, FakeChatClient

SECRET = "sk-" + "A1b2C3d4" * 3
KILL_OFF = KillSwitchAdapter(path="no-such-kill-file", env={})


@pytest.fixture
def audit(tmp_path):
    return AuditTrail(tmp_path / "audit.jsonl")


def _records(trail):
    out = []
    for line in trail.path.read_text("utf-8").splitlines():
        e = json.loads(line)
        out.append({**e, **e["data"], "intervention_point": e["action"], "decision": e["policy_decision"]})
    return out


def _agent(client, audit, *, tools=(), **gov):
    gov.setdefault("kill_switch", KILL_OFF)
    return gmaf.governed_agent(client=client, name="writer", instructions="x", tools=list(tools),
                               governance={"audit": audit, **gov})


def _write_file(log):
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        log.append(path)
        return "written"
    return write_file


def test_protected_write_is_blocked_before_the_tool_runs(audit):
    log = []
    client = FakeChatClient([[Call("write_file", {"path": "src/ci_lab/rules/x.py", "content": "c"})], "done"])
    asyncio.run(_agent(client, audit, tools=[_write_file(log)]).run("go"))
    assert log == []
    rec = [r for r in _records(audit) if r["intervention_point"] == "pre_tool_call"]
    assert (rec[-1]["decision"], rec[-1]["reason"], rec[-1]["mode"]) == ("deny", "protected_path", "enforce")
    assert rec[-1]["agent_did"] == "did:ci-lab:agent:writer"


def test_evaluate_only_records_the_deny_but_lets_the_tool_run(audit):
    log = []
    client = FakeChatClient([[Call("write_file", {"path": "evals/x.yaml", "content": "c"})], "done"])
    res = asyncio.run(_agent(client, audit, tools=[_write_file(log)], mode="evaluate_only").run("go"))
    assert log == ["evals/x.yaml"] and res.text == "done"
    assert {(r["decision"], r["mode"]) for r in _records(audit) if r["reason"] == "protected_path"} == {
        ("deny", "evaluate_only")}


def test_disallowed_model_never_reaches_the_client(audit):
    client = FakeChatClient(default="hello")
    res = asyncio.run(_agent(client, audit, model="evil-model-9").run("hi"))
    assert client.requests == [] and "model_not_allowed" in res.text


def test_kill_switch_blocks_tools(audit):
    log = []
    client = FakeChatClient([[Call("write_file", {"path": "notes.md", "content": "c"})], "done"])
    ks = KillSwitchAdapter(path="no-such-kill-file", env={"CI_KILL_SWITCH": "1"})
    asyncio.run(_agent(client, audit, tools=[_write_file(log)], kill_switch=ks).run("go"))
    assert log == [] and any(r["reason"] == "kill_switch_engaged" for r in _records(audit))


def test_output_secret_is_redacted_and_audit_is_content_free(audit):
    client = FakeChatClient(default=f"the key is {SECRET}")
    res = asyncio.run(_agent(client, audit).run("hi"))
    assert SECRET not in res.text and "[redacted:secret]" in res.text
    assert SECRET not in audit.path.read_text("utf-8")
    assert any(r["intervention_point"] == "output" and r["decision"] == "transform" for r in _records(audit))


def test_unconfigured_point_is_skipped_and_govern_is_idempotent(audit):
    a = _agent(FakeChatClient(), audit)
    n = len(a.middleware)
    assert gmaf.govern(a) is a and len(a.middleware) == n
    gov = gmaf.Governance("meta_agents", audit=audit, kill_switch=KILL_OFF)
    assert "input" not in gov.points and asyncio.run(gov.check("input", {"input": {"text": "x"}})) is None


def test_invalid_mode_env_fails_at_startup(monkeypatch, audit):
    monkeypatch.setenv("CI_GOVERNANCE_MODE", "yolo")
    with pytest.raises(RuntimeError, match="CI_GOVERNANCE_MODE"):
        gmaf.Governance("meta_agents", audit=audit)




def test_harness_mcp_hook_allows_introspection_and_fails_closed(monkeypatch):
    hook = gmaf.harness_mcp_before_call(
        agent_name="CiFailureAnalyst", allowed_tools={"list_components"})
    assert asyncio.run(hook("harness", "list_components", {})) is None
    assert "frozen case exposure" in asyncio.run(hook("harness", "read_component", {}))

    async def broken(*_args, **_kwargs):
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(gmaf.Governance, "check", broken)
    hook = gmaf.harness_mcp_before_call(agent_name="CiFailureAnalyst")
    assert asyncio.run(hook("harness", "list_components", {})) == (
        "harness governance adapter failure")
