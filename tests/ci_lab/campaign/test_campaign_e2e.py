from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import fake_deps
from ci_lab.campaign.local import FileOutbox
from ci_lab.workflows.runtime import StepFailed

CID = "demo-camp"


def _new(tmp_path: Path, **kw):
    deps = fake_deps(tmp_path, **kw)
    camp = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 3, "max_rounds": 4}, deps=deps,
                        run_root=tmp_path / "runs")
    return camp, deps


def _publish_argv(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "publish-calls.jsonl"
    return [json.loads(line)["argv"] for line in path.read_text().splitlines()] if path.exists() else []


def _outbox_ops(tmp_path: Path) -> list[str]:
    return [json.loads(line)["op"] for line in (tmp_path / "outbox.jsonl").read_text().splitlines()]


def test_end_to_end_fake_campaign(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path, reject_first={"v2"})
    delta = asyncio.run(camp.calibrate())
    assert delta == pytest.approx(0.25)  # identical A/A runs -> 1/M floor with 4 cases

    out = asyncio.run(camp.run(rounds=2))
    assert [r["winner"] for r in out["rounds"]] == ["v1", "v1"]
    assert [r["decision"] for r in out["rounds"]] == ["ship", "ship"]

    ledger = tmp_path / "experiments" / "campaigns" / CID
    for name in ("campaign.json", "frontier.json", "calibration.json", "calibration/envelope.json",
                 "history.jsonl", "stack.json"):
        assert (ledger / name).exists(), name
    for eid in ("demo-camp-r01", "demo-camp-r02"):
        for name in ("envelope.json", "decisions.json", "evals.json"):
            assert (ledger / "rounds" / eid / name).exists()
        env = json.loads((ledger / "rounds" / eid / "envelope.json").read_text())
        assert env["id"] == eid and env["decision"] == "ship"
    frontier = json.loads((ledger / "frontier.json").read_text())
    assert frontier["round"] == 2 and frontier["score"] == pytest.approx(0.7)
    assert len((ledger / "history.jsonl").read_text().splitlines()) == 2

    # Critic rejected v2's first attempt -> repair re-invoked the proposer once per round.
    assert deps.critique.calls.count(("demo-camp-r01", "v2", 2)) == 1
    run_dir = tmp_path / "runs" / "demo-camp-r01"
    assert json.loads((run_dir / "v2" / "repair_1.json").read_text())["repaired"] is True
    assert json.loads((run_dir / "v1" / "repair_1.json").read_text())["repaired"] is False
    order = json.loads((run_dir / "run_order.json").read_text())
    assert sorted(order) == ["inc", "v1", "v2"]
    # Incumbent re-evaluated every round (fake profile may reuse the tree cache, so check markers).
    assert (run_dir / "inc" / "arm.done").exists()

    argv = _publish_argv(tmp_path)
    kinds = [(a[0], a[1], a[2] if len(a) > 2 else "") for a in argv]
    assert kinds == [
        ("git", "push", "origin"),        # r01 winner branch
        ("gh", "pr", "create"),           # r01 PR (base main)
        ("git", "push", "origin"),        # r01 loser archive tag
        ("git", "push", "origin"),        # r02 winner branch
        ("gh", "pr", "create"),           # r02 PR (base = r01 layer)
        ("gh", "api", "-X"),              # create native stack (2 PRs)
        ("git", "push", "origin"),        # r02 loser archive tag
    ]
    assert argv[0][3].endswith(":refs/heads/exp/demo-camp-r01/v1")
    assert argv[1][argv[1].index("--base") + 1] == "main"
    assert argv[2][3].endswith(":refs/tags/exp-archive/demo-camp-r01/v2")
    assert argv[4][argv[4].index("--base") + 1] == "exp/demo-camp-r01/v1"
    assert argv[5][4] == "repos/example/harness/stacks"
    stack = json.loads((ledger / "stack.json").read_text())
    assert [layer["eid"] for layer in stack["layers"]] == ["demo-camp-r01", "demo-camp-r02"]
    assert stack["stack_number"] is not None

    # Third round: append to the existing stack.
    asyncio.run(camp.run(rounds=3))
    assert [a[:3] for a in _publish_argv(tmp_path)][-2] == ["gh", "api", "-X"]
    assert _publish_argv(tmp_path)[-2][4].endswith(f"/stacks/{stack['stack_number']}/add")

    status = camp.status()
    assert status["calibrated"] and len(status["rounds"]) == 3 and status["in_flight"] == []
    assert camp.readjudicate("demo-camp-r01")["consistent"] is True

    conf = asyncio.run(camp.confirm())
    assert conf["decision"] == "ship"
    landed = camp.land()
    assert landed["layers"] == [layer["pr"] for layer in json.loads((ledger / "stack.json").read_text())["layers"]]
    assert _publish_argv(tmp_path)[-1][:3] == ["gh", "pr", "merge"]
    assert Counter(_outbox_ops(tmp_path)).most_common(1)[0][1] == 1


def test_no_ship_when_no_arm_beats_delta(tmp_path: Path) -> None:
    camp, _ = _new(tmp_path, hypothesis_fn=lambda eid, arm, d: f"tweak {arm} {eid}")
    asyncio.run(camp.calibrate())
    out = asyncio.run(camp.run(rounds=1))
    assert out["rounds"][0]["decision"] == "do_not_ship" and out["rounds"][0]["winner"] is None
    kinds = [a[:2] for a in _publish_argv(tmp_path)]
    assert kinds == [["git", "push"], ["git", "push"]]  # both losers archived, no PR
    assert not (tmp_path / "experiments" / "campaigns" / CID / "stack.json").exists()


def test_kill_and_resume_has_no_duplicate_effects(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path)
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))

    # Round 2: crash inside an arm's evaluate (after its proposal) ...
    crash = {"arm": True, "publish": True}
    deps.domain.fail_on = lambda eid, variant, split: crash["arm"] and eid.endswith("r02") and variant == "v1"
    with pytest.raises(StepFailed):
        asyncio.run(camp.run(rounds=2))
    assert not (tmp_path / "runs" / "demo-camp-r02" / "round.done").exists()
    assert (tmp_path / "runs" / "demo-camp-r02" / "v1" / "proposal.json").exists()

    # ... then a "new process" (fresh deps over the same durable state) crashes in publish.
    crash["arm"] = False
    deps2 = fake_deps(tmp_path, outbox=FileOutbox(tmp_path / "outbox.jsonl"))
    from ci_lab.publish.github import GitHubPublisher

    class CrashingPublisher(GitHubPublisher):
        def open_pr(self, *a, **kw):
            if crash["publish"]:
                crash["publish"] = False
                raise RuntimeError("network down")
            return super().open_pr(*a, **kw)

    deps2.publisher = CrashingPublisher("example/harness", outbox=deps2.outbox, dry_run=True,
                                        journal=tmp_path / "publish-calls.jsonl")
    camp2 = Campaign.load(CID, deps=deps2, run_root=tmp_path / "runs")
    with pytest.raises(StepFailed):
        asyncio.run(camp2.run(rounds=2))
    proposer_runs_before = list(deps2.make_agent.runs)

    deps3 = fake_deps(tmp_path, outbox=FileOutbox(tmp_path / "outbox.jsonl"))
    camp3 = Campaign.load(CID, deps=deps3, run_root=tmp_path / "runs")
    out = asyncio.run(camp3.run(rounds=2))
    assert [r["winner"] for r in out["rounds"]] == ["v1", "v1"]
    # Proposals/analysis were not redone after the crash (GatedAgent + markers).
    assert deps3.make_agent.runs == [] and ("demo-camp-r02", "v1") not in proposer_runs_before

    ops = _outbox_ops(tmp_path)
    assert len(ops) == len(set(ops)), "duplicate outbox effect"
    argv = [tuple(a) for a in _publish_argv(tmp_path)]
    assert len(argv) == len(set(argv)), "duplicate publish mutation"
    assert sum(a[:3] == ("gh", "pr", "create") for a in argv) == 2
    assert sum(1 for line in (tmp_path / "experiments" / "campaigns" / CID / "history.jsonl")
               .read_text().splitlines()) == 2


def test_fresh_run_dir_skips_rounds_the_ledger_records(tmp_path: Path) -> None:
    # CI: every scheduled run starts on a fresh runner (empty run dir) over the checked-out ledger
    camp, _ = _new(tmp_path)
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    calls = len(_publish_argv(tmp_path))
    deps2 = fake_deps(tmp_path, outbox=FileOutbox(tmp_path / "outbox.jsonl"))
    camp2 = Campaign.load(CID, deps=deps2, run_root=tmp_path / "runs2")
    out = asyncio.run(camp2.run(rounds=2))
    assert out["rounds"][0] == {"eid": "demo-camp-r01", "round": 1, "decision": "ship", "winner": "v1",
                                "pr": None, "recorded": True}
    assert out["rounds"][1]["eid"] == "demo-camp-r02" and "recorded" not in out["rounds"][1]
    assert not (tmp_path / "runs2" / "demo-camp-r01").exists()  # round 1 was not re-run
    assert ("demo-camp-r01", "v1") not in [tuple(r[:2]) for r in deps2.make_agent.runs]
    assert len((tmp_path / "experiments" / "campaigns" / CID / "history.jsonl").read_text().splitlines()) == 2
    assert len(_publish_argv(tmp_path)) > calls

def test_stop_file_and_budget(tmp_path: Path) -> None:
    camp, _ = _new(tmp_path)
    asyncio.run(camp.calibrate())
    stop = tmp_path / "STOP"
    stop.write_text("")
    assert asyncio.run(camp.run(stop_file=stop))["stopped"] == "stop_file"
    stop.unlink()

    deps = fake_deps(tmp_path / "b")
    camp_b = Campaign.new("budget-camp", "fake", {"aa_repeats": 2, "budget_tokens": 100}, deps=deps,
                          run_root=tmp_path / "b" / "runs")
    asyncio.run(camp_b.calibrate())  # 2 x 4 cases x 10 tokens = 80 spent
    out = asyncio.run(camp_b.run())
    assert out["stopped"] == "budget" and len(out["rounds"]) == 1


def test_new_is_idempotent_and_rejects_bad_input(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path)
    again = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 3, "max_rounds": 4}, deps=deps,
                         run_root=tmp_path / "runs")
    assert again.cid == CID
    with pytest.raises(ValueError):
        Campaign.new(CID, "fake", {"arms": 3}, deps=deps, run_root=tmp_path / "runs")
    with pytest.raises(ValueError):
        Campaign.new("Bad_ID", "fake", deps=deps, run_root=tmp_path / "runs")
    with pytest.raises(ValueError):
        Campaign.new("other-camp", "fake", {"nope": 1}, deps=deps, run_root=tmp_path / "runs")
    with pytest.raises(RuntimeError):
        asyncio.run(camp.run(rounds=1))  # not calibrated


def test_confirm_uses_one_global_look(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path)
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    assert asyncio.run(camp.confirm())["decision"] == "ship"
    assert asyncio.run(camp.confirm())["decision"] == "ship"  # resume: same look, no re-evaluation
    other = Campaign.new("other-camp", "fake", {"aa_repeats": 2}, deps=deps, run_root=tmp_path / "runs")
    asyncio.run(other.calibrate())
    with pytest.raises(StepFailed, match="HoldoutExhausted"):
        asyncio.run(other.confirm())
    looks = (tmp_path / "experiments" / "holdout-looks.jsonl").read_text().splitlines()
    assert len(looks) == 1
    with pytest.raises(RuntimeError, match="confirm"):
        other.land()
