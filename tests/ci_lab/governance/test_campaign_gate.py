"""Campaign launch gate (ci_lab.governance.campaign + ``ci-lab campaign`` exit 3)."""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from ci_lab.campaign import cli as campaign_cli
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import StubDomain, fake_deps
from ci_lab.cli import main
from ci_lab.contracts import Profile
from ci_lab.domain import harness as harness_domain
from ci_lab.governance import maf as gmaf
from ci_lab.governance.approvals import FileApprovalQueue
from ci_lab.governance.audit import AuditTrail
from ci_lab.governance.campaign import LaunchRefused, arm_scopes, check_launch
from ci_lab.governance.hypervisor import KillSwitchAdapter
from ci_lab.oes.validate import validate_envelope

FAKE = StubDomain
SCOPES = arm_scopes(FAKE.component_globs, FAKE.surface_globs)


def _gov(tmp_path, *, killed=False, mode=None):
    return gmaf.Governance("campaign", agent_name="campaign-c", role="orchestrator", mode=mode,
                           audit=AuditTrail(tmp_path / "audit.jsonl"), approvals=FileApprovalQueue(tmp_path / "q"),
                           kill_switch=KillSwitchAdapter(path=tmp_path / "kill", env={"CI_KILL_SWITCH": "1"}
                                                         if killed else {}))


def _launch(gov, **kw):
    kw = {"arms": SCOPES, "publish": False, "dry_run": True, **kw}
    return asyncio.run(check_launch("c", governance=gov, **kw))


def test_dry_run_launch_is_allowed_for_real_and_fake_domains(tmp_path):
    assert _launch(_gov(tmp_path))["decision"] == "allow"
    real = arm_scopes(harness_domain.COMPONENT_GLOBS, harness_domain.SURFACE_GLOBS)
    assert _launch(_gov(tmp_path), arms=real, publish=True)["decision"] == "allow"


@pytest.mark.parametrize(("kw", "reason"), [
    ({"killed": True}, "kill_switch_engaged"),
    ({"budget_exhausted": True}, "budget_exhausted"),
    ({"arms": [{"id": "x", "edit_scope": ["src/ci_lab/rules/**"]}]}, "protected_scope"),
    ({"arms": [{"id": "x", "edit_scope": ["harness/**/sealed/**"]}]}, "protected_scope"),
])
def test_launch_denied(tmp_path, kw, reason):
    gov = _gov(tmp_path, killed=kw.pop("killed", False))
    with pytest.raises(LaunchRefused) as exc:
        _launch(gov, **kw)
    assert exc.value.reason == reason and not exc.value.held
    rec = json.loads(gov.audit.path.read_text("utf-8").splitlines()[-1])
    assert (rec["action"], rec["policy_decision"]) == ("agent_startup", "deny")


def test_evaluate_only_records_but_does_not_refuse(tmp_path):
    gov = _gov(tmp_path, killed=True, mode="evaluate_only")
    assert _launch(gov)["decision"] == "deny"


def test_publish_is_held_until_approved_then_consumed(tmp_path):
    gov = _gov(tmp_path)
    with pytest.raises(LaunchRefused) as exc:
        _launch(gov, publish=True, dry_run=False)
    held = exc.value
    assert held.held and held.reason == "publish_requires_approval"
    assert held.to_dict()["approve"] == f"ci-lab governance approve {held.identity}"
    queue = FileApprovalQueue(tmp_path / "q")
    assert [p["enforced_identity"] for p in queue.pending()] == [held.identity]
    queue.decide(held.identity, "approve", by="test")
    assert _launch(gov, publish=True, dry_run=False)["approved"]
    with pytest.raises(LaunchRefused):  # one-shot approval
        _launch(gov, publish=True, dry_run=False)


def test_cli_gate_refuses_with_exit_3(tmp_path, capsys, monkeypatch):
    common = ["--profile", "fake", "--run-dir", str(tmp_path), "--dry-run-publish"]
    assert main(["campaign", "new", "gate-camp", *common]) == 0
    capsys.readouterr()
    monkeypatch.setenv("CI_KILL_SWITCH", "1")
    assert main(["campaign", "run", "gate-camp", *common, "--rounds", "1"]) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "governance_refused" and out["reason"] == "kill_switch_engaged"
    assert main(["campaign", "status", "gate-camp", *common]) == 0  # read-only commands are not gated


def test_cli_gate_holds_non_dry_run_publish(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CI_GOVERNANCE_APPROVALS", str(tmp_path / "q"))
    deps = fake_deps(tmp_path / "s", repo="o/r", ledger_root=tmp_path / "l")
    camp = Campaign.new("held-camp", Profile.FAKE, deps=deps, run_root=tmp_path)
    args = argparse.Namespace(defer_publish=None, dry_run_publish=False)
    assert campaign_cli._launch_gate(camp, "run", args, Profile.FAKE) is None  # fake never publishes
    assert campaign_cli._launch_gate(camp, "run", args, Profile.COPILOT) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["held"] and out["approve"].startswith("ci-lab governance approve ")
    FileApprovalQueue().decide(out["identity"], "approve")
    assert campaign_cli._launch_gate(camp, "run", args, Profile.COPILOT) is None
    deferred = argparse.Namespace(defer_publish="reqs", dry_run_publish=False)
    assert campaign_cli._launch_gate(camp, "run", deferred, Profile.COPILOT) is None


def test_sre_exhausted_when_every_strategy_is_vetoed(tmp_path):
    deps = fake_deps(tmp_path / "s", repo="o/r", ledger_root=tmp_path / "l")
    camp = Campaign.new("sre-camp", Profile.FAKE, {"strategies": ["gepa"]}, deps=deps, run_root=tmp_path)
    assert not camp.sre_exhausted()
    for i in (1, 2, 3):
        assert not camp.sre_exhausted()
        deps.ledger.append_jsonl(camp.env.rel("history.jsonl"), {
            "eid": f"sre-camp-r{i:02d}", "round": i, "arms": [{"arm": "v1", "status": "failed", "strategy": "gepa"}]},
            key="eid")
    assert camp.sre_exhausted()


def _round_envelope(tmp_path, cid):
    deps = fake_deps(tmp_path / "s", repo="o/r", ledger_root=tmp_path / "l")
    camp = Campaign.new(cid, Profile.FAKE, {"aa_repeats": 2}, deps=deps, run_root=tmp_path)
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(1))
    (path,) = (tmp_path / "l").rglob("rounds/*/envelope.json")
    return json.loads(path.read_text("utf-8"))


def test_round_envelope_carries_the_governance_extension(tmp_path, monkeypatch):
    from ci_lab.governance.audit import AuditError, envelope_extension

    assert envelope_extension(tmp_path / "none.jsonl") == {}
    trail = AuditTrail(tmp_path / "audit.jsonl")
    trail.append({"agent_did": "did:ci-lab:agent:x", "intervention_point": "pre_tool_call", "decision": "deny",
                  "reason": "protected_path"})
    monkeypatch.setenv("CI_GOVERNANCE_AUDIT", str(trail.path))
    env = _round_envelope(tmp_path / "b", "audit-camp")
    validate_envelope(env)
    ext = env["extensions"]["x-ci-governance"]  # governed fake agents audited more decisions this round
    assert ext == trail.oes_extension()["x-ci-governance"] and ext["decisions"] > 1 and ext["denies"] == 1
    trail.path.write_text(trail.path.read_text("utf-8").replace("protected_path", "tampered_path"), "utf-8")
    with pytest.raises(AuditError):
        envelope_extension(trail.path)
