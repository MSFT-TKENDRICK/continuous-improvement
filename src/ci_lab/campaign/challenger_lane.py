"""Out-of-band challenger lane (bus contract v2 §13, P13): per round, after the arms ran and before
selection, deterministic gamers and/or an LLM adversary attack each arm's final student proposal.

Adversary proposals and the student's are voted by :func:`bus_adapter.lane_voters` (the arm's voters
minus the replayed campaign critic, which only judges the arm's own worktree, plus the ``proposal-shape``
validity oracle and the quality voters of ``CampaignDeps.lane_voters``) and dueled; an exploit (soft
judges prefer or pass the attack while an independent validity oracle fails it) becomes a bus
``exploit`` entry and feeds an OES evaluator-experiment proposal
(``<round>/challenger/adversary/evaluator_proposal.json``). Without a soft (quality) voter, or for
``llm`` without ``CampaignDeps.adversary_complete``, the lane is inert and says so in a bus ``note``.
The lane only writes bus entries and that directory: never arm results, directives, selection or
history, and its rows are filtered out of ``SelectionInputs``/history (:func:`out_of_band`), so
strategies, allocation and selection are identical with the lane on or off. The adversary is never a
campaign strategy. ``hyper["challenger"]``: ``off | det | llm | both`` (default ``det``). Best effort:
lane failures are logged, never raised.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ci_lab.adversary.challenger import (
    AdversaryRubricView,
    Challenger,
    DeterministicChallenger,
    LLMAdversary,
)
from ci_lab.adversary.harden import Corpus, CorpusItem, Hardener
from ci_lab.bus import ids
from ci_lab.bus.judge import duel
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.types import Author, Entry, ExploitBody, NoteBody
from ci_lab.bus.voters.local import Voter, run_voters
from ci_lab.bus.wal import AgentBus
from ci_lab.campaign import bus_adapter
from ci_lab.taskgraph.model import Rubric, StudentSpec
from ci_lab.taskgraph.vault import RubricVault

if TYPE_CHECKING:
    from ci_lab.workflows.steps import RoundContext

__all__ = ["INERT_LLM", "INERT_QUALITY", "LANE_DIR", "LANE_ROLES", "MODES", "challengers", "in_band", "lane_mode",
           "out_of_band", "run_lane"]

LANE_DIR = "challenger"
LANE_ROLES = frozenset({"adversary", "challenger"})
MODES = ("off", "det", "llm", "both")
INERT_QUALITY = ("challenger lane inert: no quality voter configured (no soft voter judges attacks; "
                 "set $CI_S1_LLAMA_URL for the System-1 judge)")
INERT_LLM = "challenger lane: llm adversary inert: no CampaignDeps.adversary_complete (offline profile)"
_log = logging.getLogger(__name__)


def lane_mode(hyper: Mapping[str, Any]) -> str:
    mode = str(hyper.get("challenger", "det"))
    if mode not in MODES:
        raise ValueError(f"hyper.challenger must be one of {MODES}, got {mode!r}")
    return mode


def out_of_band(row: Any) -> bool:
    """A challenger-lane row/arm (by ``strategy`` or ``role``): never selection or history input."""
    get = row.get if isinstance(row, Mapping) else (lambda k: getattr(row, k, None))
    return any(get(k) in LANE_ROLES for k in ("strategy", "role"))


def in_band(arms: Mapping[str, Any]) -> dict[str, Any]:
    return {k: a for k, a in arms.items() if not out_of_band(a)}


def challengers(mode: str, seed: int, complete: Any = None) -> list[Challenger]:
    out: list[Challenger] = []
    if mode in ("det", "both"):
        out.append(DeterministicChallenger(seed=seed))
    if mode in ("llm", "both") and complete is not None:
        out.append(LLMAdversary(complete))
    return out


async def run_lane(rctx: RoundContext) -> int:
    """Attack every arm of the round; returns the number of exploits found (0 when off or on error)."""
    env = rctx.env
    mode = lane_mode(env.hyper)
    if mode == "off" or not bus_adapter.bus_enabled(env.hyper):
        return 0
    complete = getattr(env.deps, "adversary_complete", None)
    lanes = challengers(mode, int(env.hyper.get("seed", 0)), complete)
    try:
        return await _run(rctx, lanes, [INERT_LLM] if mode in ("llm", "both") and complete is None else [])
    except Exception:  # noqa: BLE001 - the lane is out of band: it never sinks a round
        _log.warning("challenger lane failed for %s", rctx.eid, exc_info=True)
        return 0


async def _run(rctx: RoundContext, lanes: Sequence[Challenger], inert: Sequence[str]) -> int:
    bus = bus_adapter.campaign_bus(rctx.env.run_root)
    corpus, rubric = Corpus(), None
    for d in rctx.begin()["directives"]:
        arm = rctx.arm(d["arm"])
        topic = bus_adapter.arm_topic(arm)
        student = _last_student(bus, topic)
        if student is None:
            continue
        voters = bus_adapter.lane_voters(arm)
        rubric = bus_adapter.arm_rubric(voters)
        for text in [*inert, *([] if rubric.soft() else [INERT_QUALITY])]:
            await _note(bus, topic, text)
        if not lanes or not rubric.soft():
            continue
        spec, aid = bus_adapter.arm_spec(arm, rubric), student.body.attempt  # type: ignore[union-attr]
        s_votes = await _votes(bus, topic, student, voters, rubric, spec)
        for adv in await _adversaries(bus, topic, aid, spec, rubric, lanes):
            r = duel(rubric, s_votes, await _votes(bus, topic, adv, voters, rubric, spec))  # type: ignore[arg-type]
            if not r.exploit:
                continue
            body: Any = adv.body
            text = bus.read_artifact(topic, body.artifact).decode("utf-8", "replace")
            corpus = corpus.add(CorpusItem(body.proposal, "exploit", text, adv.author.name, aid,
                                           r.oracle_invalid_adversary))
            if not any(e.kind == "exploit" and e.body.adversary_proposal == body.proposal  # type: ignore[union-attr]
                       for e in bus.state(topic).entries):
                await bus.append(topic, "exploit", bus_adapter.ORCHESTRATOR, ExploitBody(
                    attempt=aid, adversary_proposal=body.proposal, student_proposal=student.body.proposal,  # type: ignore[union-attr]
                    gamer=adv.author.name, soft_pref=r.soft_pref, soft_pass_adversary=r.soft_pass_adversary,
                    oracle_invalid=r.oracle_invalid_adversary))
    if corpus.exploits and rubric is not None:
        out = rctx.dir / LANE_DIR
        corpus.save(out)
        Hardener(RubricVault.for_run(out), out).emit_proposal(rubric, corpus)
    return len(corpus.exploits)


async def _note(bus: AgentBus, topic: str, text: str) -> None:
    if not any(e.kind == "note" and e.body.text == text for e in bus.state(topic).entries):  # type: ignore[union-attr]
        _log.warning("%s: %s", topic, text)
        await bus.append(topic, "note", bus_adapter.ORCHESTRATOR, NoteBody(text=text))


def _last_student(bus: AgentBus, topic: str) -> Entry | None:
    props = [e for e in bus.state(topic).proposals.values() if e.author.role == "student"]
    return max(props, key=lambda e: e.seq) if props else None


async def _adversaries(bus: AgentBus, topic: str, aid: str, spec: StudentSpec, rubric: Rubric,
                       lanes: Sequence[Challenger]) -> list[Entry]:
    """Adversary proposals on the student's final attempt (reused on rerun; a failing challenger adds none)."""
    have = [e for e in bus.state(topic).proposals.values()
            if e.author.role == "adversary" and e.body.attempt == aid]  # type: ignore[union-attr]
    if have:
        return have
    out = []
    for lane in lanes:
        try:
            made = await lane.propose(spec, AdversaryRubricView.of(rubric), aid)
        except Exception:  # noqa: BLE001
            _log.warning("challenger %s failed on %s", type(lane).__name__, aid, exc_info=True)
            continue
        for cp in made:
            ref = await bus.put_artifact(topic, cp.artifact)
            name = ids.parse_proposal(cp.body.proposal)[2]
            out.append(await bus.append(topic, "proposal", Author("adversary", name),
                                        replace(cp.body, artifact=ref)))
    return out


async def _votes(bus: AgentBus, topic: str, prop: Entry, voters: Sequence[Voter], rubric: Rubric,
                 spec: StudentSpec) -> list[Any]:
    have = {v.body.voter for v in bus.state(topic).votes_for(prop.seq)}  # type: ignore[union-attr]
    todo = [v for v in voters if v.name not in have]
    if todo:
        body: Any = prop.body
        art = bus.read_artifact(topic, body.artifact)
        for vb in await run_voters(todo, body, art, rubric, spec, pools=ResourcePools({"llm": 4, "s1": 1, "cpu": 8}),
                                   timeout_s=600.0):
            await bus.append(topic, "vote", Author("voter", vb.voter), vb, ref=prop.seq)
    return [v.body for v in bus.state(topic).votes_for(prop.seq)]
