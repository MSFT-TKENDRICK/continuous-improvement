"""Campaign bus adapter (bus contract v2 §13, P13): the arm steps of :mod:`ci_lab.workflows.steps`
mirrored onto the agent bus, an observer around the existing harness rather than a rewrite.

Each round is a bus run (``eid``) with its manifest on ``<eid>/_run``; each arm is the task topic
``<eid>/<arm>``. ``critique`` records the arm's ``proposal.json`` as a student proposal, collects
votes (the injected campaign critic as an ``llm`` voter, structural critic checks, optional extra
voters from ``CampaignDeps.bus_voters``) and appends the judge's ``aggregate`` verdict. ``repair``
replaces the proposer continuation with succession: a fresh proposer whose only input is the
student projection of the topic (task, attempt, own outputs, sanitized correction). ``evaluate``
runs as a ``bus.effect`` so a rerun skips completed evaluations. ``hyper["bus"] = False`` keeps the
legacy path (critic call, ``reinvoke_proposer``, direct evaluate) for one release.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ci_lab.bus import ids
from ci_lab.bus.judge import aggregate
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.project import succeed
from ci_lab.bus.state import BusInvariantError
from ci_lab.bus.types import Author, ManifestBody, NoteBody, ProposalBody
from ci_lab.bus.voters.local import (
    Ballot,
    BaseVoter,
    CallableVoter,
    CriticChecksVoter,
    Voter,
    run_voters,
)
from ci_lab.bus.wal import AgentBus
from ci_lab.campaign import records
from ci_lab.contracts import CriticVerdict, EvalResult
from ci_lab.meta.brief import failure_corpus
from ci_lab.taskgraph.firewall import LeakScreen, sanitize_correction
from ci_lab.taskgraph.model import (
    Budget,
    Criterion,
    OutputSpec,
    Rubric,
    StudentSpec,
    canonical_json,
)
from ci_lab.telemetry.core import git_ref
from ci_lab.tools.critic_checks import CriticConfig

if TYPE_CHECKING:
    from ci_lab.workflows.steps import ArmRun, RoundContext

__all__ = ["BUS_DIR", "CHECKS", "CRITIC", "SHAPE", "arm_rubric", "arm_spec", "arm_topic", "arm_voters", "bus_enabled",
           "bus_extension", "campaign_bus", "criterion_id", "critique", "evaluate", "lane_voters", "proposal_shape",
           "repair"]

BUS_DIR = "bus"
CRITIC, CHECKS, SHAPE = "critic", "critic-checks", "proposal-shape"
PROPOSAL = "proposal.json"
ORCHESTRATOR = Author("orchestrator", "campaign")
JUDGE = Author("judge", "campaign")
STUDENT = Author("student", "proposer")
_BUSES: dict[Path, AgentBus] = {}


def bus_enabled(hyper: Mapping[str, Any]) -> bool:
    return bool(hyper.get("bus", True))


def campaign_bus(run_root: Path) -> AgentBus:
    """One ``AgentBus`` per ``<run_root>/bus`` in this process (shared caches and append locks)."""
    root = (Path(run_root) / BUS_DIR).resolve()
    if root not in _BUSES:
        _BUSES[root] = AgentBus(root)
    return _BUSES[root]


def arm_topic(arm: ArmRun) -> str:
    return ids.task_topic(arm.eid, arm.arm)


def criterion_id(voter: str) -> str:
    return f"c-{voter}"


def arm_voters(arm: ArmRun, critic: CriticVerdict | None = None) -> list[Voter]:
    """Critic (``llm``; replays the injected critic's ``verdict``), critic checks and ``deps.bus_voters``.
    Every voter votes on its own required criterion ``c-<name>``."""
    def critic_ballot(*_: Any) -> Ballot:
        if critic is None:
            raise RuntimeError("critic verdict not computed")
        return Ballot(critic.passed, 1.0 if critic.passed else 0.0, 1.0, tuple(critic.reasons))

    cfg = CriticConfig(surface_globs=(PROPOSAL,), component_globs={}, denylist=())
    extra = arm.round.env.deps.bus_voters
    voters: list[Voter] = [CallableVoter(CRITIC, critic_ballot, measure="llm", criteria=[criterion_id(CRITIC)]),
                           CriticChecksVoter(cfg, name=CHECKS, criteria=[criterion_id(CHECKS)]),
                           *(extra(arm) if extra is not None else ())]
    return _own_criteria(voters)


def lane_voters(arm: ArmRun) -> list[Voter]:
    """The challenger lane's voters: :func:`arm_voters` minus the replayed critic (it only judges the arm's
    own worktree), the :func:`proposal_shape` validity oracle and ``deps.lane_voters`` (quality voters,
    e.g. the System-1 judge; lane only, so arm critiques and verdicts are unchanged)."""
    extra = getattr(arm.round.env.deps, "lane_voters", None)
    voters = [v for v in arm_voters(arm) if v.name != CRITIC]
    for v in [CallableVoter(SHAPE, proposal_shape), *(extra(arm) if extra is not None else ())]:
        if all(v.name != w.name for w in voters):
            voters.append(v)
    return _own_criteria(voters)


def proposal_shape(_proposal: Any, artifact: bytes, *_: Any) -> Ballot:
    """Independent validity oracle: the artifact is an arm ``proposal.json`` (non-empty ``edits``, each
    with a ``component``), as the arm steps write it."""
    try:
        doc = json.loads(artifact)
    except ValueError:
        doc = None
    edits = doc.get("edits") if isinstance(doc, dict) else None
    ok = isinstance(edits, list) and bool(edits) and all(
        isinstance(e, dict) and isinstance(e.get("component"), str) and bool(e["component"]) for e in edits)
    return Ballot(ok, float(ok), 1.0, () if ok else ("not an arm proposal: no edits with a component",))


def _own_criteria(voters: list[Voter]) -> list[Voter]:
    for v in voters:
        if isinstance(v, BaseVoter) and v.criteria is None:
            v.criteria = frozenset({criterion_id(v.name)})
    return voters


def arm_rubric(voters: Sequence[Voter]) -> Rubric:
    """One required criterion per voter; deterministic ones are independent validity oracles."""
    def check(v: Voter) -> dict[str, Any]:
        if v.measure not in ("llm", "s1"):
            return {"independent": True}
        return {"question": "Is the proposed harness edit sound?", **({"type": "noul"} if v.measure == "s1" else {})}

    crit = tuple(Criterion(criterion_id(v.name), f"{v.name} ballot", v.measure, check(v), 0.5, required=True)
                 for v in voters)
    return Rubric("campaign-arm", 1, "arm", crit, 0.5, "ci-bus-canary-" + _sha(sorted(v.name for v in voters))[:12])


def arm_spec(arm: ArmRun, rubric: Rubric) -> StudentSpec:
    from ci_lab.workflows.steps import FINAL_CRITIQUE

    text = (f"Propose one harness edit to the {arm.directive.get('component', 'prompt')} component: apply it with "
            "your tools, then call submit_proposal with the component and a one-line hypothesis.")
    return StudentSpec(arm.arm, f"{arm.eid} {arm.arm} proposal", text, OutputSpec("json", PROPOSAL), (), (),
                       Budget(max_attempts=FINAL_CRITIQUE), rubric.commitment())


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _screen(arm: ArmRun, rubric: Rubric) -> tuple[LeakScreen, list[str]]:
    corpus = list(failure_corpus(records.failure_from_dict(f) for f in arm.round.begin()["failures"]))
    return LeakScreen([rubric], corpus), corpus


async def _manifest(bus: AgentBus, eid: str, voters: Sequence[Voter]) -> ManifestBody:
    topic = ids.run_topic(eid)
    if (m := bus.state(topic).manifest) is not None:
        return m
    names = sorted(v.name for v in voters)
    body = ManifestBody(run=eid, created=datetime.now(UTC).isoformat(timespec="seconds"),
                        code_rev=await asyncio.to_thread(git_ref) or "unknown",
                        config_sha256=_sha({"adapter": "campaign/1", "voters": names}), graph_sha256=None,
                        max_parallel=1, voters=tuple(names), quorum=len(names))
    try:
        return (await bus.append(topic, "manifest", ORCHESTRATOR, body)).body  # type: ignore[return-value]
    except BusInvariantError:  # a concurrent arm wrote it first
        return bus.state(topic).manifest  # type: ignore[return-value]


async def critique(arm: ArmRun, attempt: int) -> CriticVerdict:
    """Votes + judge verdict for ``proposal.json`` at ``attempt`` (resumes from whatever is recorded)."""
    deps = arm.round.env.deps
    if not bus_enabled(arm.round.env.hyper):
        return await deps.critique(arm, attempt)
    from ci_lab.workflows.steps import FINAL_CRITIQUE

    bus, topic = campaign_bus(arm.round.env.run_root), arm_topic(arm)
    aid = ids.attempt_id(arm.arm, attempt)
    pid = ids.proposal_id(aid, "student", STUDENT.name)
    rubric = arm_rubric(arm_voters(arm))
    manifest = await _manifest(bus, arm.eid, arm_voters(arm))
    prop = bus.state(topic).proposal_by_id(pid)
    if prop is None:
        ref = await bus.put_artifact(topic, arm.proposal_path.read_bytes())
        prop = await bus.append(topic, "proposal", STUDENT, ProposalBody(
            proposal=pid, attempt=aid, rubric_version=rubric.version_id, artifact=ref, summary=f"arm {aid}"))
    have = {v.body.voter for v in bus.state(topic).votes_for(prop.seq)}  # type: ignore[union-attr]
    legacy = await deps.critique(arm, attempt) if CRITIC not in have else None
    todo = [v for v in arm_voters(arm, legacy) if v.name not in have]
    if todo:
        body: Any = prop.body
        art = bus.read_artifact(topic, body.artifact)
        for vb in await run_voters(todo, body, art, rubric, arm_spec(arm, rubric),
                                   pools=ResourcePools({"llm": 4, "cpu": 8}), timeout_s=600.0):
            await bus.append(topic, "vote", Author("voter", vb.voter), vb, ref=prop.seq)
    votes = {v.seq: v.body for v in bus.state(topic).votes_for(prop.seq)}
    draft = aggregate(rubric, list(votes.values()), manifest.quorum, attempts_left=FINAL_CRITIQUE - attempt,
                      proposal=pid)
    failing = [r for v in votes.values() if v.passed is False for r in v.reasons]  # type: ignore[attr-defined]
    if not any(e.ref == prop.seq for e in bus.state(topic).verdicts.values()):
        correction = None
        if draft.decision != "commit":
            correction = sanitize_correction(failing or list(draft.reasons), rubric, attempt=aid,
                                             extra_corpus=_screen(arm, rubric)[1])
        await bus.append(topic, "verdict", JUDGE, draft.to_body(votes, correction=correction), ref=prop.seq)
    passed = draft.decision == "commit"
    return CriticVerdict(passed, [] if passed else (failing or list(draft.reasons)))


async def repair(arm: ArmRun, attempt: int) -> dict[str, Any] | None:
    """Succession for an agent arm: a fresh proposer runs on the student projection (``None``: legacy)."""
    env = arm.round.env
    if not bus_enabled(env.hyper) or arm.strategy != "agent":
        return None
    bus, topic = campaign_bus(env.run_root), arm_topic(arm)
    rubric = arm_rubric(arm_voters(arm))
    async with arm.tracker.phase("propose"):
        res = await succeed(bus, topic, "student", lambda _mw: env.deps.make_agent("proposer", arm),
                            arm_spec(arm, rubric), screen=_screen(arm, rubric)[0])
    if res.leak:  # fail closed: the projection is withheld and the next critique sees the old proposal
        await bus.append(topic, "note", ORCHESTRATOR,
                         NoteBody(text=f"repair {attempt}: projection withheld ({len(res.leak_hits)} leak hits)"))
        return {"repaired": False, "reason": "projection_leak"}
    return {"repaired": True}


async def evaluate(arm: ArmRun, split: str, head: str, run: Callable[[], Awaitable[EvalResult]]) -> EvalResult:
    """``run()`` as the ``evaluate`` effect keyed by (arm, split, head): a rerun reuses the recorded result."""
    if not bus_enabled(arm.round.env.hyper):
        return await run()
    bus = campaign_bus(arm.round.env.run_root)
    key = f"evaluate:{arm.arm}:{split}:{head}"
    async with bus.effect(arm_topic(arm), "evaluate", key, ORCHESTRATOR, {"split": split, "head": head}) as eff:
        if eff.skipped:
            return records.eval_from_dict(dict(eff.result))
        result = await run()
        eff.set_result(records.eval_to_dict(result))
    return result


def bus_extension(rctx: RoundContext) -> dict[str, Any]:
    """``{"x-ci-bus": {run, topics, heads}}`` for the round envelope (empty when the bus is off/unused)."""
    if not bus_enabled(rctx.env.hyper):
        return {}
    heads = campaign_bus(rctx.env.run_root).heads(rctx.eid)
    if not heads:
        return {}
    return {"x-ci-bus": {"run": rctx.eid, "topics": sorted(heads), "heads": dict(sorted(heads.items()))}}
