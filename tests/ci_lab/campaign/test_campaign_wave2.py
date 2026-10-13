"""Wave 2: per-strategy arm workflows, per-round traces, live status markers, telemetry."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import fake_deps
from ci_lab.contracts import (
    ATTR_CAMPAIGN,
    ATTR_DECISION,
    ATTR_EXPERIMENT,
    ATTR_PHASE,
    ATTR_PROFILE,
    ATTR_ROUND,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_ARM,
    SPAN_CALIBRATE,
    SPAN_CAMPAIGN_ROUND,
    SPAN_CONFIRM,
    SPAN_STEP,
    ArmContext,
    Edit,
)
from ci_lab.workflows.progress import Progress
from ci_lab.workflows.runtime import StepFailed

CID = "wave-camp"


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


def _new(tmp_path: Path, hyper: dict[str, Any] | None = None, **kw: Any):
    deps = fake_deps(tmp_path, **kw)
    camp = Campaign.new(CID, "fake", {"arms": 3, "aa_repeats": 2, "max_rounds": 4, **(hyper or {})},
                        deps=deps, run_root=tmp_path / "runs")
    return camp, deps


def test_strategy_arms_dispatch_and_fill_arm_result_strategy(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path, {"strategies": ["gepa", "agent", "skillopt"]}, reject_first={"v3"})
    asyncio.run(camp.calibrate())
    out = asyncio.run(camp.run(rounds=1))
    assert out["rounds"][0]["winner"] == "v1"  # gepa arm proposed the "boost" edit

    run_dir = tmp_path / "runs" / f"{CID}-r01"
    begin = json.loads((run_dir / "begin.json").read_text())
    assert [d["strategy"] for d in begin["directives"]] == ["gepa", "agent", "skillopt"]
    for arm, strategy in (("v1", "gepa"), ("v2", "agent"), ("v3", "skillopt")):
        done = json.loads((run_dir / arm / "arm.done").read_text())
        assert done["result"]["strategy"] == strategy
    evals = json.loads((tmp_path / "experiments" / "campaigns" / CID / "rounds" / f"{CID}-r01"
                        / "evals.json").read_text())
    text = json.dumps(evals)
    assert '"gepa"' in text and '"skillopt"' in text

    strategies = deps.get_strategy.instances
    assert set(strategies) == {"gepa", "skillopt"}  # the agent arm runs the MAF Proposer instead
    gepa = strategies["gepa"].calls
    assert len(gepa) == 1 and isinstance(gepa[0], ArmContext)
    assert gepa[0].directive.strategy == "gepa" and gepa[0].run_dir == run_dir / "v1"
    # skillopt arm was rejected once: repair re-ran the strategy with critic feedback.
    sk = strategies["skillopt"].calls
    assert len(sk) == 2
    assert not any(f.category == "critic_rejected" for f in sk[0].failures)
    assert [f.excerpt for f in sk[1].failures if f.category == "critic_rejected"] == [
        "hypothesis lacks a mechanism"]
    proposal = json.loads((run_dir / "v3" / "proposal.json").read_text())
    assert proposal["strategy"] == "skillopt" and proposal["edits"][0]["hypothesis"].endswith("(repaired)")
    assert json.loads((run_dir / "v3" / "repair_1.json").read_text())["repaired"] is True
    # Proposer agent only ran for the agent-strategy arm.
    assert [arm for _, arm in deps.make_agent.runs if arm != "analyst"] == ["v2"]


def test_strategy_resume_does_not_repropose(tmp_path: Path) -> None:
    camp, deps = _new(tmp_path, {"arms": 2, "strategies": ["gepa"]})
    asyncio.run(camp.calibrate())
    real = deps.domain.evaluate
    state = {"crash": True}

    async def flaky(worktree: Any, split: str, k: int, **kw: Any):
        if state["crash"] and kw.get("variant") == "v2":
            raise RuntimeError("evaluator crashed")
        return await real(worktree, split, k, **kw)

    deps.domain.evaluate = flaky
    with pytest.raises((StepFailed, RuntimeError)):
        asyncio.run(camp.run(rounds=1))
    state["crash"] = False
    out = asyncio.run(Campaign.load(CID, deps=deps, run_root=tmp_path / "runs").run(rounds=1))
    assert out["rounds"][0]["decision"] == "ship"
    calls = [c.directive.arm for c in deps.get_strategy.instances["gepa"].calls]
    assert sorted(calls) == ["v1", "v2"]  # proposal.json marker -> no second propose on resume


def test_unknown_strategy_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategies"):
        _new(tmp_path, {"strategies": ["agent", "dspy-magic"]})
    with pytest.raises(ValueError, match="heartbeat_s"):
        _new(tmp_path / "b", {"heartbeat_s": 120})


def test_lazy_get_strategy_imports_m10(monkeypatch: pytest.MonkeyPatch) -> None:
    from ci_lab.campaign.deps import lazy_get_strategy

    seen: list[tuple[str, dict[str, Any]]] = []

    class S:
        name = "gepa"

        async def propose(self, ctx: ArmContext) -> list[Edit]:
            return []

    mod = types.ModuleType("ci_lab.strategies")
    mod.get_strategy = lambda name, **kw: seen.append((name, kw)) or S()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.strategies", mod)
    assert isinstance(lazy_get_strategy("gepa", model="m"), S)
    assert seen == [("gepa", {"model": "m"})]


def _by_trace(finished: list[Any]) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    for s in finished:
        out.setdefault(s.context.trace_id, []).append(s)
    return out


def test_each_round_is_its_own_trace_with_arm_and_step_spans(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    camp, _ = _new(tmp_path, {"arms": 2, "strategies": ["agent", "gepa"]})
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=2))
    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.parent is None]
    assert [s.name for s in roots] == [SPAN_CALIBRATE, SPAN_CAMPAIGN_ROUND, SPAN_CAMPAIGN_ROUND]
    assert len({s.context.trace_id for s in roots}) == 3

    for i, root in enumerate(roots[1:], start=1):
        eid = f"{CID}-r{i:02d}"
        assert root.attributes[ATTR_CAMPAIGN] == CID
        assert root.attributes[ATTR_EXPERIMENT] == eid
        assert root.attributes[ATTR_ROUND] == i
        assert root.attributes[ATTR_PROFILE] == "fake"
        assert root.attributes[ATTR_DECISION] == "ship"
        members = _by_trace(finished)[root.context.trace_id]
        arms = {s.attributes[ATTR_VARIANT]: s.attributes[ATTR_STRATEGY] for s in members if s.name == SPAN_ARM}
        assert arms == {"v1": "agent", "v2": "gepa", "inc": "incumbent"}
        phases = {s.attributes[ATTR_PHASE] for s in members if s.name == SPAN_STEP}
        assert {"begin_round", "analyst", "run_arms", "select", "record", "publish", "provision_slot",
                "propose", "critique", "repair", "evaluate", "finalize_arm"} <= phases
        v1_steps = [s for s in members if s.name == SPAN_STEP and s.attributes.get(ATTR_VARIANT) == "v1"]
        assert v1_steps and all(s.attributes[ATTR_STRATEGY] == "agent" for s in v1_steps)
    cal = roots[0]
    assert cal.attributes[ATTR_EXPERIMENT] == f"{CID}-cal"


def test_resumed_round_links_to_interrupted_trace(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    camp, deps = _new(tmp_path, {"arms": 2})
    asyncio.run(camp.calibrate())
    real = deps.domain.evaluate
    state = {"crash": True}

    async def flaky(worktree: Any, split: str, k: int, **kw: Any):
        if state["crash"] and kw.get("variant") == "v2":
            raise RuntimeError("evaluator crashed")
        return await real(worktree, split, k, **kw)

    deps.domain.evaluate = flaky
    with pytest.raises((StepFailed, RuntimeError)):
        asyncio.run(camp.run(rounds=1))
    status = obs.read_status(tmp_path / "runs", f"{CID}-r01")
    assert status["state"] == "failed" and status["error"]
    live = camp.status()["live"][f"{CID}-r01"]
    assert live["state"] == "failed" and live["arms"]["v2"]["state"] == "error"
    first = [s for s in spans.get_finished_spans() if s.name == SPAN_CAMPAIGN_ROUND]
    assert len(first) == 1 and not first[0].links
    state["crash"] = False
    asyncio.run(Campaign.load(CID, deps=deps, run_root=tmp_path / "runs").run(rounds=1))
    rounds = [s for s in spans.get_finished_spans() if s.name == SPAN_CAMPAIGN_ROUND]
    assert len(rounds) == 2 and rounds[1].parent is None
    assert rounds[1].context.trace_id != rounds[0].context.trace_id
    assert [link.context.trace_id for link in rounds[1].links] == [rounds[0].context.trace_id]


def test_confirm_is_its_own_trace(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    camp, _ = _new(tmp_path, {"arms": 2})
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    asyncio.run(camp.confirm())
    roots = [s for s in spans.get_finished_spans() if s.parent is None]
    assert roots[-1].name == SPAN_CONFIRM
    assert roots[-1].attributes[ATTR_EXPERIMENT] == f"{CID}-confirm"
    status = obs.read_status(tmp_path / "runs", f"{CID}-confirm")
    assert status["phase"] == "done" and "campaign" in status["writers"]


def test_status_markers_per_writer(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    camp, _ = _new(tmp_path, {"arms": 2, "strategies": ["agent", "skillopt"], "heartbeat_s": 15})
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    runs = tmp_path / "runs"
    eid = f"{CID}-r01"
    status = obs.read_status(runs, eid)
    assert status["phase"] == "done" and status["state"] == "done"
    assert status["round"] == 1 and status["campaign_id"] == CID and status["heartbeat_s"] == 15
    assert status["decision"] == "ship" and status["winner"] == "v1"
    assert set(status["writers"]) == {"round", "v1", "v2", "inc"}
    arms = status["arms"]
    assert {a: (v["strategy"], v["state"], v["phase"]) for a, v in arms.items()} == {
        "v1": ("agent", "evaluated", "done"), "v2": ("skillopt", "evaluated", "done"),
        "inc": ("incumbent", "evaluated", "done")}
    assert set(status["trace"]) == {"trace_id", "span_id"}
    files = sorted(p.name for p in (runs / eid / "status.d").glob("*.json"))
    assert files == ["inc.json", "round.json", "v1.json", "v2.json"]
    # Each arm worker only writes its own arm.
    v1 = json.loads((runs / eid / "status.d" / "v1.json").read_text())
    assert set(v1["arms"]) == {"v1"}
    cal = obs.read_status(runs, f"{CID}-cal")
    assert cal["phase"] == "done" and cal["writers"] == ["campaign"] and "delta" in cal


def test_progress_heartbeat_rewrites_marker(tmp_path: Path) -> None:
    progress = Progress(tmp_path, "e1", writer="v1", heartbeat_s=0.02, campaign_id="c")

    async def main() -> int:
        async with progress.heartbeat(lambda: {"phase": "evaluate"}):
            await asyncio.sleep(0.5)
        return json.loads((tmp_path / "e1" / "status.d" / "v1.json").read_text())["seq"]

    seq = asyncio.run(main())
    assert seq >= 2  # periodic rewrites while the block runs (Windows timers are coarse)
    status = progress.read()
    assert status["phase"] == "evaluate" and status["heartbeat_s"] == 0.02 and status["campaign_id"] == "c"


def test_progress_write_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise PermissionError("locked by reader")

    monkeypatch.setattr(obs, "write_status", boom)
    Progress(tmp_path, "e1", writer="round").write(phase="x")


def _cli_args(tmp_path: Path, cmd: str, **extra: Any) -> argparse.Namespace:
    base = dict(campaign_command=cmd, cid=CID, profile="fake", run_dir=str(tmp_path / "runs"),
                ledger_dir=None, repo="example/harness", dry_run_publish=True, hyper=[], rounds=1,
                stop_file=None, eid=None)
    base.update(extra)
    return argparse.Namespace(**base)


def test_cli_sets_up_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    from ci_lab.campaign import cli

    calls: list[Any] = []
    mod = types.ModuleType("ci_lab.telemetry")
    mod.setup = lambda component, **kw: calls.append(("setup", component, kw))  # type: ignore[attr-defined]
    mod.shutdown = lambda: calls.append(("shutdown",))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.telemetry", mod)
    import ci_lab

    monkeypatch.setattr(ci_lab, "telemetry", mod, raising=False)
    assert cli._main(_cli_args(tmp_path, "new")) == 0
    assert calls[0][0:2] == ("setup", "campaign")
    assert calls[0][2]["profile"].value == "fake" and calls[0][2]["run_dir"] == tmp_path / "runs"
    assert calls[-1] == ("shutdown",)


def test_cli_tolerates_missing_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    import ci_lab
    from ci_lab.campaign import cli

    monkeypatch.delattr(ci_lab, "telemetry", raising=False)
    monkeypatch.setitem(sys.modules, "ci_lab.telemetry", None)
    assert cli._main(_cli_args(tmp_path, "new")) == 0
    assert json.loads(capsys.readouterr().out)["campaign"] == CID
