"""Asyncio task-graph scheduler over the agent bus (bus contract v2 §10).

``run_graph`` writes the run manifest first (I0), then runs one :func:`run_deliverable` coroutine per
deliverable in a ``TaskGroup``; independent deliverables run concurrently (bounded by
``max_parallel``), dependents wait for their dependencies and are aborted ``dependency_blocked``
unless every dependency committed. Each attempt is one ``effect("attempt", key="task@n")`` that
freezes the rubric version at its start, runs a fresh student successor (``project.succeed``),
snapshots its artifact, collects votes and appends the judge's verdict; commit/reject/abort follow
the effect (I4). With a challenger, adversary proposals are made concurrently with the student's,
voted alongside it and dueled against it; each exploit is recorded and handed to the hardener as a
background task whose patch applies from the next attempt (the current verdict is never re-judged).
A patch that lands after its task is terminal goes to the run topic; the next run in the same bus root
adopts it (and any earlier run's accepted hardening) as its baseline, see :func:`adopt_hardened`.
Everything is re-derived from the bus, so a rerun after a crash skips terminal
tasks and reuses the proposal/votes of an in-flight attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from ci_lab.adversary.challenger import AdversaryRubricView, Challenger, artifact_ref
from ci_lab.adversary.harden import Corpus, CorpusItem, Hardener, Scorer
from ci_lab.bus import ids
from ci_lab.bus.effects import effect
from ci_lab.bus.judge import Escalator, aggregate, duel, judge
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.project import succeed
from ci_lab.bus.state import BusInvariantError
from ci_lab.bus.types import (
    AbortBody,
    Author,
    CommitBody,
    Entry,
    ExploitBody,
    ManifestBody,
    NoteBody,
    ProposalBody,
    RejectBody,
    RubricPatchBody,
)
from ci_lab.bus.voters.local import Voter, run_voters
from ci_lab.bus.wal import AgentBus, BusCorrupt
from ci_lab.taskgraph.firewall import LeakScreen, sanitize_correction
from ci_lab.taskgraph.model import (
    Deliverable,
    Rubric,
    StudentSpec,
    TaskGraph,
    canonical_json,
)
from ci_lab.taskgraph.vault import RubricVault
from ci_lab.telemetry.core import git_ref

__all__ = ["GraphResult", "GraphRun", "StudentFactory", "TaskResult", "VotersFor", "adopt_hardened", "default_scorer",
           "rubric_for", "run_deliverable", "run_graph", "write_manifest"]

ORCHESTRATOR = Author("orchestrator", "scheduler")
JUDGE = Author("judge", "judge")
STUDENT = Author("student", "student")
ADOPTER = Author("hardener", "epoch-adopt")  # re-issues an earlier run's accepted hardening
BLOCKED = "dependency_blocked"
_tracer = trace.get_tracer("ci_lab.taskgraph.scheduler")

StudentFactory = Callable[[StudentSpec, Sequence[Any]], Any]
"""``student_factory(spec, middleware) -> agent`` with ``async run(message)``; one fresh agent per attempt."""
VotersFor = Callable[[Deliverable], Sequence[Voter]]


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class GraphRun:
    """Shared state of one ``run_graph`` call (also the unit an alternative engine drives)."""

    graph: TaskGraph
    bus: AgentBus
    vault: RubricVault
    voters_for: VotersFor
    student_factory: StudentFactory
    run_id: str
    pools: ResourcePools
    max_parallel: int
    quorum: int = 1
    escalator: Escalator | None = None
    timeout_s: float = 120.0
    challenger: Challenger | None = None
    hardener: Hardener | None = None
    hardener_scorer: Scorer | None = None
    hardening: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    baseline: dict[str, Rubric] = field(default_factory=dict)  # rubric id -> epoch-adopted version
    done: dict[str, asyncio.Event] = field(init=False)
    sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.done = {t: asyncio.Event() for t in self.graph.topo_order()}
        self.sem = asyncio.Semaphore(self.max_parallel)

    def topic(self, task: str) -> str:
        return ids.task_topic(self.run_id, task)


async def write_manifest(run: GraphRun) -> ManifestBody:
    """Seq 0 of ``<run>/_run``; on resume the existing manifest must describe the same graph."""
    topic, graph_sha = ids.run_topic(run.run_id), _sha(run.graph.to_json())
    voters = sorted({v.name for d in run.graph.deliverables for v in run.voters_for(d)})
    if (m := run.bus.state(topic).manifest) is not None:
        if m.graph_sha256 != graph_sha:
            raise ValueError(f"run {run.run_id!r} was started with a different task graph")
        return m
    config = {"max_parallel": run.max_parallel, "quorum": run.quorum, "timeout_s": run.timeout_s, "voters": voters}
    body = ManifestBody(run=run.run_id, created=datetime.now(UTC).isoformat(timespec="seconds"),
                        code_rev=await asyncio.to_thread(git_ref) or "unknown", config_sha256=_sha(config),
                        graph_sha256=graph_sha, max_parallel=run.max_parallel, voters=tuple(voters),
                        quorum=run.quorum)
    return (await run.bus.append(topic, "manifest", ORCHESTRATOR, body)).body  # type: ignore[return-value]


def _version(run: GraphRun, rubric_id: str, version_id: str) -> Rubric:
    return next(r for r in run.vault.versions(rubric_id) if r.version_id == version_id)


def _vnum(version_id: str) -> int:
    return ids.parse_rubric_version(version_id)[1]


def rubric_for(run: GraphRun, topic: str, initial: Rubric, n: int) -> Rubric:
    """The version frozen for attempt ``n``: latest accepted patch with ``applies_from_attempt <= n`` on top
    of the run's epoch-adopted baseline (:func:`adopt_hardened`), else ``initial``."""
    base = run.baseline.get(initial.id, initial)
    applied = [b for e in run.bus.read(topic) if e.kind == "rubric_patch" and (b := e.body).accepted  # type: ignore[union-attr]
               and b.rubric_id == initial.id and b.applies_from_attempt is not None and b.applies_from_attempt <= n]
    to = max((b.to_version for b in applied), key=_vnum, default=None)
    return base if to is None or _vnum(to) <= base.version else _version(run, initial.id, to)


def _prior_patches(run: GraphRun) -> list[RubricPatchBody]:
    """Accepted ``rubric_patch`` bodies of every OTHER run in the bus root (unreadable topics are skipped)."""
    out: list[RubricPatchBody] = []
    for topic in run.bus.topics():
        try:
            if ids.topic_run(topic) == run.run_id:
                continue
            entries = run.bus.read(topic)
        except (BusCorrupt, ids.IdError):
            continue
        out += [e.body for e in entries if e.kind == "rubric_patch" and e.body.accepted]  # type: ignore[misc, union-attr]
    return out


def _hardened(run: GraphRun, pinned: Rubric, patches: Sequence[RubricPatchBody]) -> Rubric | None:
    """Newest sealed version reachable from ``pinned`` through a chain of accepted patches, if newer."""
    reach, grew = {pinned.version_id}, True
    while grew:
        new = {p.to_version for p in patches if p.rubric_id == pinned.id and p.from_version in reach} - reach
        reach, grew = reach | new, bool(new)
    sealed = {r.version_id for r in run.vault.versions(pinned.id)}
    newer = [v for v in reach & sealed if _vnum(v) > pinned.version]
    return _version(run, pinned.id, max(newer, key=_vnum)) if newer else None


def _adopted(run: GraphRun) -> list[RubricPatchBody]:
    return [b for e in run.bus.read(ids.run_topic(run.run_id)) if e.kind == "rubric_patch"
            and (b := e.body).applies_from_epoch == 0]  # type: ignore[union-attr]


async def adopt_hardened(run: GraphRun, enabled: bool = True) -> dict[str, Rubric]:
    """Epoch adoption, decided once per run right after the manifest and re-read on resume.

    For each pinned rubric, the newest accepted hardening of an earlier run in the same bus root (task-
    or run-topic patch) that descends from it and is sealed in this vault becomes the baseline: recorded
    as a ``rubric_patch`` with ``applies_from_epoch = 0`` on ``<run>/_run``, then an ``epoch adoption``
    note marks the decision complete (also when nothing, or ``enabled=False``, was adopted)."""
    topic = ids.run_topic(run.run_id)
    if not any(e.kind == "note" and "epoch_adoption" in e.body.data for e in run.bus.read(topic)):  # type: ignore[union-attr]
        done = {b.rubric_id for b in _adopted(run)}
        pinned = {r.id: r for r in (run.vault.open(d.rubric_commitment, role="orchestrator")
                                    for d in run.graph.deliverables)}
        prior = _prior_patches(run) if enabled else []
        for rid, old in sorted(pinned.items()):
            if rid not in done and (new := _hardened(run, old, prior)) is not None:
                diff = _sha({"from": old.commitment(), "to": new.commitment()})
                await run.bus.append(topic, "rubric_patch", ADOPTER, RubricPatchBody(
                    rubric_id=rid, from_version=old.version_id, to_version=new.version_id, applies_from_attempt=None,
                    applies_from_epoch=0, diff_sha256=diff, metrics={}, accepted=True))
        got = {b.rubric_id: b.to_version for b in _adopted(run)}
        await run.bus.append(topic, "note", ORCHESTRATOR, NoteBody(
            text=f"epoch adoption: {', '.join(sorted(got.values())) or 'none'}"[:400],
            data={"epoch_adoption": got, "enabled": enabled}))
    run.baseline = {b.rubric_id: _version(run, b.rubric_id, b.to_version) for b in _adopted(run)}
    return run.baseline


def _decision(verdict: Entry) -> dict[str, Any]:
    b: Any = verdict.body
    return {"decision": b.decision, "verdict": verdict.seq, "score": b.score}


async def _student(run: GraphRun, d: Deliverable, topic: str, aid: str, rubric: Rubric,
                   screen: LeakScreen) -> Entry | tuple[str, ...]:
    pid = ids.proposal_id(aid, "student", STUDENT.name)
    if (p := run.bus.state(topic).proposal_by_id(pid)) is not None:
        return p
    spec = StudentSpec.of(d)
    async with asyncio.timeout(d.budget.timeout_s):
        res = await succeed(run.bus, topic, "student", lambda mw: run.student_factory(spec, mw), spec,
                            d.depends_on, screen=screen)
    if res.leak:
        return res.leak_hits
    ref = await run.bus.put_artifact(topic, res.text)
    body = ProposalBody(proposal=pid, attempt=aid, rubric_version=rubric.version_id, artifact=ref,
                        summary=f"student attempt {aid}")
    return await run.bus.append(topic, "proposal", STUDENT, body)


async def _votes(run: GraphRun, d: Deliverable, topic: str, prop: Entry, rubric: Rubric) -> dict[int, Any]:
    """Votes on ``prop``; on resume only voters without a recorded vote run."""
    have = {v.body.voter for v in run.bus.state(topic).votes_for(prop.seq)}  # type: ignore[union-attr]
    todo = [v for v in run.voters_for(d) if v.name not in have]
    if todo:
        body: Any = prop.body
        art = run.bus.read_artifact(topic, body.artifact)
        for vb in await run_voters(todo, body, art, rubric, StudentSpec.of(d), pools=run.pools,
                                   timeout_s=run.timeout_s):
            await run.bus.append(topic, "vote", Author("voter", vb.voter), vb, ref=prop.seq)
    return {v.seq: v.body for v in run.bus.state(topic).votes_for(prop.seq)}


async def _adversaries(run: GraphRun, d: Deliverable, topic: str, aid: str, rubric: Rubric) -> list[Entry]:
    """Challenger proposals for ``aid`` (adversary role; reused on resume). A failing challenger yields none."""
    if run.challenger is None:
        return []
    have = [e for e in run.bus.state(topic).proposals.values()
            if e.author.role == "adversary" and e.body.attempt == aid]  # type: ignore[union-attr]
    if have:
        return have
    try:
        made = await run.challenger.propose(StudentSpec.of(d), AdversaryRubricView.of(rubric), aid)
    except Exception:  # noqa: BLE001 - the challenger lane never sinks the student attempt
        return []
    out = []
    for cp in made:
        ref = await run.bus.put_artifact(topic, cp.artifact)
        name = ids.parse_proposal(cp.body.proposal)[2]
        out.append(await run.bus.append(topic, "proposal", Author("adversary", name), replace(cp.body, artifact=ref)))
    return out


def _corpus(run: GraphRun, topic: str) -> Corpus:
    """Exploits found so far + honest student proposals (no oracle veto in their verdict)."""
    st = run.bus.state(topic)

    def text(p: Entry) -> str:
        return run.bus.read_artifact(topic, p.body.artifact).decode("utf-8", "replace")  # type: ignore[union-attr]

    exploits = [CorpusItem(b.adversary_proposal, "exploit", text(st.proposal_by_id(b.adversary_proposal)),  # type: ignore[arg-type]
                           b.gamer, b.attempt, b.oracle_invalid)
                for e in st.entries if e.kind == "exploit" and (b := e.body)]  # type: ignore[union-attr]
    honest = [CorpusItem(st.entries[v.ref].body.proposal, "honest", text(st.entries[v.ref]))  # type: ignore[index, union-attr]
              for v in st.verdicts.values() if st.entries[v.ref].author.role == "student"  # type: ignore[index]
              and not any(c.oracle and c.passed is False for c in v.body.criteria.values())]  # type: ignore[union-attr]
    return Corpus(tuple(exploits), tuple(honest))


def default_scorer(run: GraphRun, d: Deliverable) -> Scorer:
    """Would the gameable part of a rubric (all but ``check.independent`` validity oracles) accept an item?"""
    async def score(rubric: Rubric, items: Sequence[CorpusItem]) -> list[bool]:
        sub = replace(rubric, criteria=tuple(c for c in rubric.criteria if c.check.get("independent") is not True))
        pid, out = ids.proposal_id(ids.attempt_id(d.id, 1), "student", "corpus"), []
        for item in items:
            data = item.text.encode("utf-8")
            body = ProposalBody(proposal=pid, attempt=ids.attempt_id(d.id, 1), rubric_version=sub.version_id,
                                artifact=artifact_ref(data), summary="corpus item")
            votes = await run_voters(run.voters_for(d), body, data, sub, StudentSpec.of(d), pools=run.pools,
                                     timeout_s=run.timeout_s)
            out.append(aggregate(sub, votes, run.quorum, attempts_left=1, proposal=pid).decision == "commit")
        return out
    return score


async def _harden(run: GraphRun, d: Deliverable, topic: str, rubric: Rubric) -> None:
    # own span: the attempt span may have ended by the time hardening appends (ci.bus.append events)
    with _tracer.start_as_current_span("ci.taskgraph.harden", attributes={"ci.run": run.run_id, "ci.task": d.id}):
        await _harden_body(run, d, topic, rubric)


async def _harden_body(run: GraphRun, d: Deliverable, topic: str, rubric: Rubric) -> None:
    """Off the attempt's critical path: gate a patch of ``rubric``; it applies from the next attempt, or
    (task already terminal) from the next epoch via the run topic. Never re-judges a recorded verdict."""
    assert run.hardener is not None
    hardener = Author("hardener", "hardener")
    try:
        decision, patch = await run.hardener.harden(rubric, _corpus(run, topic), run.hardener_scorer
                                                    or default_scorer(run, d),
                                                    current_max_attempt=run.bus.state(topic).max_proposed_attempt())
        if patch is not None:
            try:
                await run.bus.append(topic, "rubric_patch", hardener, patch)
            except BusInvariantError:  # the task went terminal meanwhile: patch the next epoch instead
                await run.bus.append(ids.run_topic(run.run_id), "rubric_patch", hardener,
                                     replace(patch, applies_from_attempt=None, applies_from_epoch=1))
        else:
            why = "; ".join(decision.reasons) if decision else "no candidate"
            await run.bus.append(topic, "note", ORCHESTRATOR, NoteBody(text=f"hardener: no patch ({why})"[:400]))
    except Exception as exc:  # noqa: BLE001 - hardening is advisory
        await run.bus.append(topic, "note", ORCHESTRATOR, NoteBody(text=f"hardener error: {type(exc).__name__}"))


async def _attempt(run: GraphRun, d: Deliverable, topic: str, n: int, initial: Rubric) -> Mapping[str, Any]:
    aid, rubric = ids.attempt_id(d.id, n), rubric_for(run, topic, initial, n)
    pid = ids.proposal_id(aid, "student", STUDENT.name)

    async def reconcile(_key: str) -> Mapping[str, Any] | None:
        st = run.bus.state(topic)
        p = st.proposal_by_id(pid)
        v = next((e for e in st.verdicts.values() if p is not None and e.ref == p.seq), None)
        return None if v is None else _decision(v)

    attrs = {"ci.run": run.run_id, "ci.task": d.id, "ci.attempt": aid, "ci.rubric_version": rubric.version_id}
    with _tracer.start_as_current_span("ci.taskgraph.attempt", attributes=attrs):
        async with effect(run.bus, topic, "attempt", aid, ORCHESTRATOR, {"rubric_version": rubric.version_id},
                          attempt=aid, reconcile=reconcile) as eff:
            if eff.skipped:
                return eff.result
            got = await asyncio.gather(_student(run, d, topic, aid, rubric, LeakScreen([initial, rubric])),
                                       _adversaries(run, d, topic, aid, rubric), return_exceptions=True)
            if err := next((g for g in got if isinstance(g, BaseException)), None):
                raise err
            prop, advs = got
            if not isinstance(prop, Entry):
                eff.set_result({"leak": list(prop)})
                return eff.result
            votes, *adv_votes = await asyncio.gather(*(_votes(run, d, topic, p, rubric) for p in (prop, *advs)))
            draft = await judge(rubric, list(votes.values()), run.quorum, d.budget.max_attempts - n,
                                run.escalator, proposal=pid)
            correction = None
            if draft.decision != "commit":
                failing = [r for v in votes.values() if v.passed is False for r in v.reasons]
                correction = sanitize_correction([*draft.reasons, *failing], rubric, attempt=aid)
            verdict = await run.bus.append(topic, "verdict", JUDGE, draft.to_body(votes, correction=correction),
                                           ref=prop.seq)
            exploits = 0
            for adv, av in zip(advs, adv_votes, strict=True):
                r = duel(rubric, list(votes.values()), list(av.values()))
                if r.exploit:
                    exploits += 1
                    await run.bus.append(topic, "exploit", ORCHESTRATOR, ExploitBody(
                        attempt=aid, adversary_proposal=adv.body.proposal, student_proposal=pid,  # type: ignore[union-attr]
                        gamer=adv.author.name, soft_pref=r.soft_pref, soft_pass_adversary=r.soft_pass_adversary,
                        oracle_invalid=r.oracle_invalid_adversary))
            if exploits and run.hardener is not None:  # applies from a later attempt; this verdict stands
                run.hardening[d.id] = asyncio.create_task(_harden(run, d, topic, rubric))
            eff.set_result({**_decision(verdict), "exploits": exploits})
    return eff.result


async def _finish(run: GraphRun, topic: str, kind: str, body: Any, author: Author = ORCHESTRATOR,
                  ref: int | None = None) -> None:
    if run.bus.state(topic).terminal is None:
        await run.bus.append(topic, kind, author, body, ref=ref)  # type: ignore[arg-type]


async def _attempts(run: GraphRun, d: Deliverable, topic: str) -> None:
    for n in range(1, d.budget.max_attempts + 1):
        if (pending := run.hardening.pop(d.id, None)) is not None:
            await pending  # a patch gated on attempt n-1's exploits applies from attempt n
        if run.bus.state(topic).terminal is not None:
            return
        aid = ids.attempt_id(d.id, n)
        try:
            result = await _attempt(run, d, topic, n, run.vault.open(d.rubric_commitment, role="orchestrator"))
        except TimeoutError:
            return await _finish(run, topic, "abort", AbortBody(attempt=aid, reason="timeout"))
        except Exception as exc:  # noqa: BLE001 - a broken attempt aborts its task, never the graph
            return await _finish(run, topic, "abort", AbortBody(attempt=aid, reason=f"error: {type(exc).__name__}"))
        if "leak" in result:
            return await _finish(run, topic, "abort", AbortBody(attempt=aid, reason="context_leak"))
        if result["decision"] == "commit":
            verdict = run.bus.state(topic).entries[result["verdict"]]
            prop: Any = run.bus.state(topic).entries[verdict.ref].body  # type: ignore[index]
            body = CommitBody(proposal=prop.proposal, verdict_seq=verdict.seq, artifact=prop.artifact)
            return await _finish(run, topic, "commit", body, JUDGE, verdict.seq)
        if result["decision"] == "reject":
            return await _finish(run, topic, "reject", RejectBody(attempt=aid, reason="attempts exhausted"), JUDGE)


async def run_deliverable(run: GraphRun, task: str) -> None:
    """Wait for dependencies, then attempt ``task`` (or abort it ``dependency_blocked``)."""
    d, topic = run.graph.deliverable(task), run.topic(task)
    for dep in d.depends_on:
        await run.done[dep].wait()
    if run.bus.state(topic).terminal is None:
        if any(run.bus.state(run.topic(dep)).commit is None for dep in d.depends_on):
            await _finish(run, topic, "abort", AbortBody(attempt=None, reason=BLOCKED))
        else:
            async with run.sem:
                with _tracer.start_as_current_span("ci.taskgraph.deliverable",
                                                   attributes={"ci.run": run.run_id, "ci.task": task}):
                    await _attempts(run, d, topic)
    run.done[task].set()


@dataclass(frozen=True)
class TaskResult:
    task: str
    status: str  # committed | rejected | aborted | blocked | pending
    score: float | None
    attempts: int
    exploits: int
    rubric_versions: tuple[str, ...]
    critical_path: tuple[str, ...]  # longest (by wall time) dependency chain ending here
    wall_s: float
    reason: str | None = None


def _wall(entries: Sequence[Entry]) -> float:
    if len(entries) < 2:
        return 0.0
    return (datetime.fromisoformat(entries[-1].ts) - datetime.fromisoformat(entries[0].ts)).total_seconds()


@dataclass(frozen=True)
class GraphResult:
    run: str
    tasks: Mapping[str, TaskResult]
    critical_path: tuple[str, ...]
    wall_s: float

    @property
    def ok(self) -> bool:
        return all(t.status == "committed" for t in self.tasks.values())

    def to_json(self) -> dict[str, Any]:
        return {"run": self.run, "ok": self.ok, "wall_s": self.wall_s, "critical_path": list(self.critical_path),
                "tasks": {k: asdict(t) for k, t in self.tasks.items()}}

    @classmethod
    def from_bus(cls, bus: AgentBus, graph: TaskGraph, run_id: str, *, wall_s: float = 0.0) -> GraphResult:
        tasks: dict[str, TaskResult] = {}
        finish: dict[str, float] = {}
        for t in graph.topo_order():
            st = bus.state(ids.task_topic(run_id, t))
            own = [e for e in st.proposals.values() if e.author.role == "student"]
            scores = [v.body.score for v in st.verdicts.values() if st.entries[v.ref].author.role == "student"]  # type: ignore[index, union-attr]
            term: Any = st.terminal
            reason = getattr(term.body, "reason", None) if term is not None else None
            status = ("pending" if term is None else "blocked" if reason == BLOCKED else
                      {"commit": "committed", "reject": "rejected", "abort": "aborted"}[term.kind])
            prev = max(graph.deliverable(t).depends_on, key=lambda x: finish[x], default=None)
            wall = _wall(st.entries)
            finish[t] = wall + (finish[prev] if prev else 0.0)
            path = (*(tasks[prev].critical_path if prev else ()), t)
            tasks[t] = TaskResult(t, status, scores[-1] if scores else None, len(own),
                                  sum(e.kind == "exploit" for e in st.entries),
                                  tuple(e.body.rubric_version for e in own), path, wall, reason)  # type: ignore[union-attr]
        end = max(finish, key=lambda k: finish[k], default=None)
        return cls(run_id, tasks, tasks[end].critical_path if end else (), wall_s)


async def run_graph(graph: TaskGraph, *, bus: AgentBus, vault: RubricVault, voters_for: VotersFor,
                    student_factory: StudentFactory, run_id: str, pools: ResourcePools,
                    max_parallel: int | None = None, challenger: Challenger | None = None,
                    hardener: Hardener | None = None, quorum: int = 1, escalator: Escalator | None = None,
                    timeout_s: float = 120.0, hardener_scorer: Scorer | None = None,
                    adopt_epoch_patches: bool = True) -> GraphResult:
    """Run (or resume) ``graph`` as run ``run_id``. ``max_parallel=None`` runs every ready deliverable at
    once; ``timeout_s`` bounds each voter, ``Budget.timeout_s`` each student attempt. With a
    ``challenger``, adversary proposals are voted and dueled each attempt; exploits go to ``hardener``
    (scored by ``hardener_scorer``, default :func:`default_scorer`) off the attempt's critical path.
    ``adopt_epoch_patches`` starts each rubric from earlier runs' accepted hardening (:func:`adopt_hardened`)."""
    start = time.perf_counter()
    run = GraphRun(graph, bus, vault, voters_for, student_factory, run_id, pools,
                   max_parallel or len(graph.deliverables), quorum, escalator, timeout_s, challenger, hardener,
                   hardener_scorer)
    run.quorum = (await write_manifest(run)).quorum
    await adopt_hardened(run, adopt_epoch_patches)
    with _tracer.start_as_current_span("ci.taskgraph.run", attributes={"ci.run": run_id, "ci.graph": graph.id}):
        async with asyncio.TaskGroup() as tg:
            for t in graph.topo_order():
                tg.create_task(run_deliverable(run, t))
        await asyncio.gather(*run.hardening.values())
    return GraphResult.from_bus(bus, graph, run_id, wall_s=time.perf_counter() - start)
