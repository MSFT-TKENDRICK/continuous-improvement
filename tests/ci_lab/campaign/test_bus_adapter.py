from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ci_lab.campaign import bus_adapter
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import fake_deps
from ci_lab.workflows.steps import RoundContext, arm_tools

CID = "demo-camp"
EID = "demo-camp-r01"


def _round(tmp_path: Path, **hyper: Any) -> tuple[Campaign, Any, list[tuple[str, Any, list[str]]]]:
    deps = fake_deps(tmp_path, reject_first={"v2"})
    made: list[tuple[str, Any, list[str]]] = []
    make = deps.make_agent

    def spy(role: str, ctx: Any) -> Any:
        agent = make(role, ctx)
        if role == "proposer":
            seen: list[str] = []
            made.append((ctx.arm, agent, seen))
            run = agent.run

            async def recording(messages: Any = None, **kw: Any) -> Any:
                seen.append(str(messages))
                return await run(messages, **kw)

            agent.run = recording
        return agent

    deps.make_agent = spy
    camp = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 3, "max_rounds": 2, "strategies": ["agent"],
                                      **hyper}, deps=deps, run_root=tmp_path / "runs")
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    return camp, deps, made


def _envelope(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "experiments" / "campaigns" / CID / "rounds" / EID / "envelope.json"
    return json.loads(path.read_text())


def test_bus_entries_per_arm_and_envelope_heads(tmp_path: Path) -> None:
    _camp, deps, _ = _round(tmp_path, challenger="off")  # lane entries: test_challenger_lane
    bus = bus_adapter.campaign_bus(tmp_path / "runs")
    assert bus.topics(f"{EID}/") == [f"{EID}/_run", f"{EID}/v1", f"{EID}/v2"]
    manifest = bus.state(f"{EID}/_run").manifest
    assert manifest is not None and manifest.voters == ("critic", "critic-checks") and manifest.quorum == 2
    kinds = [e.kind for e in bus.read(f"{EID}/v2")]
    assert kinds == ["proposal", "vote", "vote", "verdict", "proposal", "vote", "vote", "verdict", "intent",
                     "outcome"]
    verdicts = [e.body.decision for e in bus.read(f"{EID}/v2") if e.kind == "verdict"]
    assert verdicts == ["revise", "commit"]
    assert [e.kind for e in bus.read(f"{EID}/v1")] == ["proposal", "vote", "vote", "verdict", "intent", "outcome"]
    assert deps.critique.calls.count((EID, "v2", 1)) == 1  # the campaign critic votes once per attempt
    ext = _envelope(tmp_path)["extensions"]["x-ci-bus"]
    assert ext == {"run": EID, "topics": sorted(bus.heads(EID)), "heads": bus.heads(EID)}


def test_repair_is_a_fresh_agent_on_the_student_projection(tmp_path: Path) -> None:
    _, _, made = _round(tmp_path)
    v2 = [(agent, seen) for arm, agent, seen in made if arm == "v2"]
    assert len(v2) == 2 and v2[0][0] is not v2[1][0]  # succession, not continuation
    (message,) = v2[1][1]
    assert message.startswith("## Task") and "## Correction" in message and "mechanism" in message
    assert "attempt 2 of 3" in message
    for leaked in ("c-critic", "critic-checks", "ballot", "score", "vote", "verdict", "suite"):
        assert leaked not in message.lower(), leaked
    st = bus_adapter.campaign_bus(tmp_path / "runs").state(f"{EID}/v2")
    assert st.latest_correction() is not None


def test_legacy_flag_keeps_reinvoke_and_writes_no_bus(tmp_path: Path) -> None:
    _, deps, made = _round(tmp_path, bus=False)
    v2 = [(agent, seen) for arm, agent, seen in made if arm == "v2"]
    assert len(v2) == 1 and "Repair the edits" in v2[0][1][-1]
    assert deps.critique.calls.count((EID, "v2", 2)) == 1
    assert bus_adapter.campaign_bus(tmp_path / "runs").topics() == []
    assert "x-ci-bus" not in (_envelope(tmp_path).get("extensions") or {})
    repaired = json.loads((tmp_path / "runs" / EID / "v2" / "repair_1.json").read_text())
    assert repaired["repaired"] is True


def test_evaluate_rerun_is_idempotent(tmp_path: Path) -> None:
    camp, deps, _ = _round(tmp_path)
    calls = list(deps.domain.calls)
    arm = RoundContext(camp.env, 1).arm("v1")
    first = json.loads((arm.dir / "eval.json").read_text())
    (arm.dir / "eval.json").unlink()
    asyncio.run(arm_tools(arm)["evaluate"](split="evolve"))
    assert deps.domain.calls == calls
    assert json.loads((arm.dir / "eval.json").read_text()) == first
    kinds = [e.kind for e in bus_adapter.campaign_bus(tmp_path / "runs").read(f"{EID}/v1")]
    assert kinds.count("intent") == kinds.count("outcome") == 1
