from __future__ import annotations

import pytest

from ci_lab.governance.hypervisor import (
    ActionGuard,
    KillSwitchAdapter,
    RingDecision,
    check_action,
)

DID = "did:mesh:" + "a" * 32


@pytest.fixture
def guard(tmp_path):
    return ActionGuard("run-1", kill_switch=KillSwitchAdapter(tmp_path / "KILL", env={}))


@pytest.mark.parametrize(("action", "ring"), [
    ("read_file", 3), ("list_files", 3), ("git_commit", 2), ("draft_pr", 2),
    ("write_file", 2), ("publish_pr", 1), ("push", 1), ("totally_unknown_tool", 1),
])
def test_table_rings(guard, action, ring):
    assert int(guard.classify(action)) == ring


def test_hints_classify_unknown_but_never_widen_known(guard):
    assert int(guard.classify("custom", read_only=True)) == 3
    assert int(guard.classify("custom", reversible=True)) == 2
    assert int(guard.classify("custom", reversible=False, read_only=True)) == 1
    assert int(guard.classify("write_file", read_only=True, reversible=True)) == 2
    assert int(guard.classify("read_file", read_only=False)) == 2
    assert int(guard.classify("git_commit", reversible=False)) == 1


def test_check_action_requires_approval_for_ring1(guard):
    assert guard.check_action(DID, "read_file") == RingDecision(3, True, "ok", False)
    assert guard.check_action(DID, "publish_pr") == RingDecision(1, False, "approval_required", True)
    assert guard.check_action(DID, "publish_pr", approved=True) == RingDecision(1, True, "approved", True)


def test_kill_switch_file_engage_release(tmp_path):
    ks = KillSwitchAdapter(tmp_path / "gov" / "KILL", env={})
    g = ActionGuard(kill_switch=ks)
    assert not ks.engaged()
    result = ks.engage("incident 42")
    assert ks.engaged() and result.details == "incident 42" and ks.switch.total_kills == 1
    assert (tmp_path / "gov" / "KILL").read_text(encoding="utf-8").strip() == "incident 42"
    d = g.check_action(DID, "read_file")
    assert not d.allowed and d.reason == "kill_switch_engaged"
    assert ks.release() and not ks.engaged()
    assert g.check_action(DID, "read_file").allowed


def test_kill_switch_env_cannot_be_released(tmp_path):
    ks = KillSwitchAdapter(tmp_path / "KILL", env={"CI_KILL_SWITCH": "1"})
    assert ks.engaged() and not ks.release()
    assert not ActionGuard(kill_switch=ks).check_action(DID, "read_file", approved=True).allowed


def test_per_ring_rate_limits_are_independent(tmp_path):
    ks = KillSwitchAdapter(tmp_path / "KILL", env={})
    g = ActionGuard("r", kill_switch=ks, ring_limits={r: (0.0, 2.0) for r in range(4)})
    assert [g.check_action(DID, "read_file").allowed for _ in range(3)] == [True, True, False]
    assert g.check_action(DID, "read_file").reason == "rate_limited"
    assert g.check_action(DID, "git_commit").allowed
    assert g.check_action("did:mesh:other", "read_file").allowed


def test_module_check_action_uses_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI_KILL_SWITCH", "1")
    assert check_action(DID, "read_file").reason == "kill_switch_engaged"
    monkeypatch.delenv("CI_KILL_SWITCH")
    assert check_action(DID, "read_file").allowed
