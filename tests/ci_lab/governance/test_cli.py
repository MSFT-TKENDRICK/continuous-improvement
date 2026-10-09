"""``ci-lab governance`` CLI: doctor, eval, audit-verify, approve/deny (offline)."""

from __future__ import annotations

import io
import json

import pytest

from ci_lab.cli import main
from ci_lab.governance.approvals import FileApprovalQueue
from ci_lab.governance.audit import AuditTrail

HEX = "ab" * 32


def _run(capsys, *argv: str) -> tuple[int, dict]:
    code = main(["governance", *argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # default kill file artifacts/governance/KILL lives under cwd
    monkeypatch.setenv("CI_GOVERNANCE_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("CI_GOVERNANCE_APPROVALS", str(tmp_path / "approvals"))
    return tmp_path


def _decision(i: int, decision: str = "allow") -> dict:
    return {"agent_did": "did:mesh:t", "intervention_point": "input", "decision": decision,
            "reason": "ok", "mode": "enforce", "input_identity": f"{i:064x}",
            "enforced_identity": f"{i:064x}"}


def test_doctor_healthy_then_kill_switch_and_bad_mode(iso, capsys, monkeypatch):
    code, rep = _run(capsys, "doctor")
    assert code == 0 and rep["ok"], rep
    assert set(rep["policies"]) == {"meta_agents", "campaign", "harness"}
    assert all(p["points"] and not p["missing_policies"] for p in rep["policies"].values())
    assert rep["mode"]["value"] == "enforce" and all(rep["agt_modules"].values())
    assert rep["audit"]["ok"] and rep["audit"]["entries"] == 0 and not rep["kill_switch"]["engaged"]
    monkeypatch.setenv("CI_KILL_SWITCH", "1")
    code, rep = _run(capsys, "doctor")
    assert code == 1 and rep["kill_switch"]["engaged"]
    monkeypatch.delenv("CI_KILL_SWITCH")
    monkeypatch.setenv("CI_GOVERNANCE_MODE", "audit")
    code, rep = _run(capsys, "doctor")
    assert code == 1 and not rep["mode"]["ok"] and "invalid" in rep["mode"]["error"]


def test_audit_verify_detects_tamper(iso, capsys):
    trail = AuditTrail()
    for i in range(3):
        trail.append(_decision(i, "deny" if i == 1 else "allow"))
    code, res = _run(capsys, "audit-verify")
    assert code == 0 and res["ok"] and res["entries"] == 3 and res["denies"] == 1
    lines = trail.path.read_text("utf-8").splitlines()
    lines[1] = lines[1].replace('"deny"', '"allow"', 1)
    trail.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code, res = _run(capsys, "audit-verify", "--path", str(trail.path))
    assert code == 1 and not res["ok"] and res["error"]
    code, rep = _run(capsys, "doctor")
    assert code == 1 and not rep["audit"]["ok"]


def _snapshot(iso, **campaign) -> str:
    snap = {"campaign": {"id": "c1", "publish": False, "dry_run": True, "budget_exhausted": False,
                         "arms": [{"id": "prompt", "edit_scope": ["harness/prompts/**"]}], **campaign}}
    path = iso / "snap.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    return str(path)


def test_eval_allow_deny_and_liftable(iso, capsys, monkeypatch):
    args = ("eval", "--policy", "campaign", "--point", "agent_startup", "--snapshot")
    code, out = _run(capsys, *args, _snapshot(iso))
    assert code == 0 and out["decision"] == "allow"
    code, out = _run(capsys, *args, _snapshot(iso, budget_exhausted=True))
    assert code == 2 and (out["decision"], out["reason"]) == ("deny", "budget_exhausted")
    assert not out["liftable"]
    code, out = _run(capsys, *args, _snapshot(iso, arms=[{"id": "x", "edit_scope": ["evals/**"]}]))
    assert code == 2 and out["reason"] == "protected_scope"
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"campaign": {"id": "c1", "publish": True, "dry_run": False, "budget_exhausted": False, "arms": []}})))
    code, out = _run(capsys, *args, "-")
    assert code == 2 and out["reason"] == "publish_requires_approval" and out["liftable"]
    assert len(out["enforced_identity"].removeprefix("sha256:")) == 64
    assert not AuditTrail().path.exists()  # eval is a dry check: nothing audited, nothing queued
    assert FileApprovalQueue().pending() == []


def test_approve_and_deny_write_the_queue(iso, capsys):
    code, data = _run(capsys, "approve", f"sha256:{HEX}", "--by", "alice", "--reason", "reviewed")
    assert code == 0 and (data["status"], data["by"], data["enforced_identity"]) == ("approved", "alice", HEX)
    code, data = _run(capsys, "deny", HEX)
    assert code == 0 and FileApprovalQueue().read(HEX)["status"] == "denied"
    assert main(["governance", "approve", "../../etc"]) == 2
    assert "64-hex" in capsys.readouterr().err
