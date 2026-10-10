"""Experiment-designer tools: validation, estimate, persistence and (dry-run) launches."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ci_lab.campaign.defaults import DEFAULT_HYPER
from ci_lab.chat import launch as launch_mod
from ci_lab.chat import tools as tools_mod
from ci_lab.chat.tools import (
    ChatConfig,
    ChatTools,
    case_counts,
    hyper_descriptions,
    maf_tools,
)
from ci_lab.contracts import STRATEGIES

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def cfg(tmp_path):
    return ChatConfig(run_root=tmp_path / "runs", chat_dir=tmp_path / "chat", campaign_profile="fake",
                      repo_root=REPO_ROOT)


@pytest.fixture
def t(cfg):
    return ChatTools(cfg)


def evolve_cases():
    return sum(s["evolve"] for s in case_counts(str(REPO_ROOT)).values())


@pytest.mark.parametrize("cid", ["../x", "..", "a/b", "x", "AB-test", "-abc", "abc/../../etc", "c:\\x\\y",
                                 "a" * 42, "", None, 7])
def test_bad_campaign_ids_are_rejected_and_nothing_is_written(t, cfg, cid):
    r = t.draft_campaign(cid, {}, 1, "local", "why")
    assert not r["ok"] and "bad campaign id" in r["errors"][0]
    assert not cfg.drafts_dir.exists()


def test_good_draft_is_normalized_estimated_and_persisted(t, cfg):
    r = t.draft_campaign("exp-one", {"arms": 3, "k": 2, "strategies": "gepa,agent"}, 2, "local", "  test  ")
    assert r["ok"], r
    d = r["draft"]
    assert d["hyper"]["arms"] == 3 and d["hyper"]["strategies"] == ["gepa", "agent"]
    assert d["overrides"] == {"arms": 3, "k": 2, "strategies": ["gepa", "agent"]}
    assert d["rationale"] == "test" and d["profile"] == "fake" and d["domain"] == "harness"
    path = Path(r["path"])
    assert path == cfg.drafts_dir / "exp-one.json"
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(d))
    new, cal, run = d["commands"]
    assert new[:6] == [sys.executable, "-m", "ci_lab.cli", "campaign", "new", "exp-one"]
    assert new[new.index("--domain") + 1] == "harness"
    assert new[new.index("--hyper") + 1] == "arms=3"
    assert 'strategies=["gepa","agent"]' in new
    assert "--dry-run-publish" in cal and run[-2:] == ["--rounds", "2"]


def test_estimate_math(t):
    n = evolve_cases()
    assert n == 15  # frozen harness sets: 50 cases = 15 evolve + 15 heldout + 20 ood
    e = t.draft_campaign("est-one", {"arms": 2, "aa_repeats": 5, "k": 3}, 4, "local", "x")["draft"]["estimate"]
    assert e["evaluations"] == (5 + 4 * 2) * n * 3
    assert e["breakdown"] == {"aa_calibration": 5 * n * 3, "arms": 4 * 2 * n * 3, "incumbent_reevaluation": 4 * n * 3}
    assert e["evaluations_with_incumbent"] == e["evaluations"] + 4 * n * 3
    assert e["heldout_confirm_per_look"] == 2 * 15 * 3


@pytest.mark.parametrize(("hyper", "rounds", "needle"), [
    ({"nope": 1}, 1, "unknown hyperparameters"),
    ({"arms": 9}, 1, "arms must be in 1..8"),
    ({"arms": 0}, 1, "arms must be >= 1"),
    ({"arms": "2"}, 1, "arms must be an integer"),
    ({"arms": True}, 1, "arms must be an integer"),
    ({"k": 0}, 1, "k must be >= 1"),
    ({"aa_repeats": 1}, 1, "aa_repeats must be >= 2"),
    ({"strategies": []}, 1, "strategies must be a non-empty subset"),
    ({"strategies": ["agent", "evil"]}, 1, "strategies must be a non-empty subset"),
    ({"budget_tokens": 0}, 1, "budget_tokens must be null or an integer >= 1"),
    ({"heartbeat_s": 120}, 1, "heartbeat_s must be in (0, 60]"),
    ({"b_min": 5, "b_max": 2}, 1, "b_min must be <= b_max"),
    ({}, 0, "rounds must be in 1..max_rounds"),
    ({"max_rounds": 2}, 3, "rounds must be in 1..max_rounds (2)"),
    ({}, 1.5, "rounds must be an integer"),
])
def test_invalid_hyper_and_rounds(t, cfg, hyper, rounds, needle):
    r = t.draft_campaign("bad-one", hyper, rounds, "local", "why")
    assert not r["ok"]
    assert any(needle in e for e in r["errors"]), r["errors"]
    assert not (cfg.drafts_dir / "bad-one.json").exists()


def test_target_and_rationale_are_validated(t):
    r = t.draft_campaign("tgt-one", {}, 1, "cloud", " ")
    assert any("target must be one of" in e for e in r["errors"])
    assert any("rationale" in e for e in r["errors"])


def test_existing_campaign_is_rejected(t, cfg):
    meta = cfg.ledger_root / "campaigns" / "old-one" / "campaign.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"profile": "fake", "hyper": DEFAULT_HYPER}), encoding="utf-8")
    r = t.draft_campaign("old-one", {}, 1, "local", "why")
    assert not r["ok"] and "already exists" in r["errors"][0]
    listed = {c["cid"]: c for c in t.list_campaigns()["campaigns"]}
    assert listed["old-one"]["source"] == "ledger"


def test_workflow_target_ignores_overrides_and_caps_rounds(t):
    r = t.draft_campaign("wf-one", {"arms": 3}, 2, "workflow", "nightly")
    assert r["ok"]
    d = r["draft"]
    assert d["hyper"] == DEFAULT_HYPER and d["overrides"] == {} and d["ignored_overrides"] == {"arms": 3}
    assert d["profile"] == "copilot"
    assert "IGNORED" in r["warnings"][0]
    assert d["estimate"]["arms"] == DEFAULT_HYPER["arms"]
    assert not t.draft_campaign("wf-two", {}, 10, "workflow", "x")["ok"]


def test_dry_run_launch_local_records_argv(cfg):
    cfg.dry_run_launch = True
    t = ChatTools(cfg)
    assert t.draft_campaign("dry-local", {"arms": 2}, 1, "local", "why")["ok"]
    r = t.launch_campaign("dry-local")
    assert r["ok"] and r["dry_run"] and r["launched"] is False and r["status"] == "dry_run"
    assert r["argv"] == [sys.executable, "-m", "ci_lab.chat.launch", str(cfg.drafts_dir / "dry-local.json")]
    assert [c[4] for c in r["commands"]] == ["new", "calibrate", "run"]
    assert r["launches"] == [{"cid": "dry-local", "target": "local", "status": "dry_run"}]
    assert not (cfg.launches_dir / "dry-local.log").exists()
    assert t.draft_campaign("dry-local", {}, 1, "local", "again")["ok"]  # a dry run does not consume the id


def test_dry_run_launch_workflow_records_gh_argv(cfg):
    cfg.dry_run_launch, cfg.repo = True, "octo/harness"
    t = ChatTools(cfg)
    assert t.draft_campaign("dry-wf", {"arms": 4}, 3, "workflow", "why")["ok"]
    r = t.launch_campaign("dry-wf")
    assert r["argv"] == ["gh", "workflow", "run", "campaign-scheduled.yml", "-f", "cid=dry-wf", "-f", "rounds=3",
                         "--repo", "octo/harness"]
    assert r["ignored_overrides"] == {"arms": 4} and "accepts only cid and rounds" in r["note"]
    assert r["launches"][0]["status"] == "dry_run"


def test_launch_requires_a_valid_draft(t, cfg):
    assert "call draft_campaign first" in t.launch_campaign("no-draft")["errors"][0]
    assert not t.launch_campaign("../etc")["ok"]
    assert t.draft_campaign("late-one", {}, 1, "local", "why")["ok"]
    meta = cfg.ledger_root / "campaigns" / "late-one" / "campaign.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("{}", encoding="utf-8")
    r = t.launch_campaign("late-one")
    assert not r["ok"] and not r["launched"] and "already exists" in r["errors"][0]


def test_tampered_draft_is_revalidated(t, cfg):
    assert t.draft_campaign("tamper-one", {}, 1, "local", "why")["ok"]
    path = cfg.drafts_dir / "tamper-one.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["overrides"] = {"arms": 50}
    path.write_text(json.dumps(d), encoding="utf-8")
    r = t.launch_campaign("tamper-one")
    assert not r["ok"] and "arms must be in 1..8" in r["errors"][0]


def test_workflow_launch_runs_gh_without_shell(cfg, monkeypatch):
    cfg.repo = "octo/harness"
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, "https://github.com/octo/harness/actions/runs/42\n", "")

    monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("CI_CHAT_TOKEN", "secret-token-123456")
    t = ChatTools(cfg)
    assert t.draft_campaign("wf-live", {}, 1, "workflow", "why")["ok"]
    r = t.launch_campaign("wf-live")
    assert r["launched"] and r["status"] == "dispatched"
    assert r["run_url"] == "https://github.com/octo/harness/actions/runs/42"
    (argv, kw), = calls
    assert isinstance(argv, list) and not kw.get("shell")
    assert "CI_CHAT_TOKEN" not in kw["env"]
    assert not t.draft_campaign("wf-live", {}, 1, "workflow", "again")["ok"]  # launched ids are consumed


def test_workflow_launch_failure_is_reported(cfg, monkeypatch):
    monkeypatch.setattr(tools_mod.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "HTTP 404"))
    t = ChatTools(cfg)
    t.draft_campaign("wf-fail", {}, 1, "workflow", "why")
    r = t.launch_campaign("wf-fail")
    assert not r["ok"] and not r["launched"] and r["status"] == "failed" and "HTTP 404" in r["errors"][0]


def test_local_launch_spawns_detached_launcher_that_runs_the_chain(cfg):
    """Real end-to-end launch on the offline ``fake`` campaign profile (~15 s)."""
    t = ChatTools(cfg)
    assert t.draft_campaign("e2e-one", {"aa_repeats": 2}, 1, "local", "e2e")["ok"]
    r = t.launch_campaign("e2e-one")
    assert r["launched"] and r["status"] == "started" and r["pid"] > 0
    deadline = time.time() + 120
    while time.time() < deadline:
        status = t.campaign_status("e2e-one")
        if status["launch"]["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    log = Path(r["log"]).read_text(encoding="utf-8", errors="replace")
    assert status["launch"]["status"] == "succeeded", log
    assert status["campaign"]["calibrated"] and len(status["campaign"]["rounds"]) == 1
    assert "campaign chain finished" in log
    assert t.launches()[0]["status"] == "succeeded"


def test_launcher_rejects_foreign_commands(tmp_path):
    good = {"cid": "abc", "target": "local", "commands": [
        [sys.executable, "-m", "ci_lab.cli", "campaign", s, "abc"] for s in ("new", "calibrate", "run")]}
    assert launch_mod.validate_commands(good)
    for bad in ({**good, "cid": "../x"}, {**good, "target": "workflow"},
                {**good, "commands": [["cmd", "/c", "calc"]] * 3},
                {**good, "commands": [[sys.executable, "-m", "ci_lab.cli", "campaign", "land", "abc"]] * 3}):
        with pytest.raises(ValueError):
            launch_mod.validate_commands(bad)


def test_read_only_tools(t, cfg):
    assert t.list_strategies()["strategies"] == list(STRATEGIES)
    suites = t.list_suites()
    assert {s["suite"] for s in suites["suites"]} == {
        "injection", "proposal", "taskgraph", "tool_use", "triage"}
    assert suites["totals"] == {"evolve": 15, "heldout": 15, "ood": 20, "all": 50}
    hp = t.get_default_hyperparameters()["hyperparameters"]
    assert set(hp) == set(DEFAULT_HYPER)
    assert hp["arms"]["default"] == 2 and "arms per round" in hp["arms"]["description"]
    assert all(hyper_descriptions().values())
    assert t.list_campaigns()["campaigns"] == []  # missing dirs are fine
    assert not t.campaign_status("ghost-one")["ok"]
    assert not t.campaign_status("../x")["ok"]


def test_maf_tools_gate_only_launch(t):
    fts = {f.name: f for f in maf_tools(t)}
    assert set(fts) == {"list_strategies", "list_suites", "get_default_hyperparameters", "list_campaigns",
                        "campaign_status", "draft_campaign", "launch_campaign"}
    assert fts["launch_campaign"].approval_mode == "always_require"
    assert {n for n, f in fts.items() if f.approval_mode == "always_require"} == {"launch_campaign"}
    props = fts["draft_campaign"].parameters()["properties"]
    assert set(props) == {"cid", "hyper", "rounds", "target", "rationale"}
    assert props["target"]["enum"] == ["local", "workflow"]
