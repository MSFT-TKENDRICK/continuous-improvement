"""Local voters and the concurrent voter runner (bus contract v2 §9, P8a).

A voter turns one proposal artifact into ``VoteBody`` entries: one per (voter, criterion) it
covers, or a single overall vote (``criterion=None``) for whole-artifact voters. Votes always
carry the proposal's rubric version. :func:`run_voters` runs voters concurrently: each waits for
its resource pool (and optional host admission) first, *then* its timeout starts; errors and
timeouts become abstentions (``passed=None``) with a reason, never exceptions.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jsonschema import Draft202012Validator

from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.types import Measure, ProposalBody, VoteBody
from ci_lab.rules import Bundle, evaluate
from ci_lab.rulespec import GuardView, TrajectoryStep
from ci_lab.taskgraph.model import Criterion, Rubric, StudentSpec
from ci_lab.taskgraph.validate import COMMAND_ALLOWLIST, PYTHON_TARGET_RE
from ci_lab.tools.critic_checks import ArmDiff, CriticConfig, FileChange, run_checks
from ci_lab.tools.paths import normalize_rel

__all__ = ["Ballot", "BaseVoter", "CallableVoter", "CriticChecksVoter", "DeterministicCheckVoter",
           "RulesVoter", "Voter", "abstain", "artifact_path", "run_voters", "targets_of"]


@runtime_checkable
class Voter(Protocol):
    """Optional extras read by :func:`run_voters`: ``pool: str | None``, ``weight: int``,
    ``targets(rubric) -> list[str | None]`` and ``admission() -> async context manager``."""

    name: str
    measure: Measure
    criteria: frozenset[str] | None

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]: ...


@dataclass(frozen=True)
class Ballot:
    passed: bool | None
    score: float | None = None
    confidence: float | None = None
    reasons: tuple[str, ...] = ()

    @classmethod
    def of(cls, value: Ballot | bool | None) -> Ballot:
        if isinstance(value, Ballot):
            return value
        if value is None:
            return cls(None, reasons=("abstained",))
        if isinstance(value, bool):
            return cls(value, 1.0 if value else 0.0, 1.0)
        raise TypeError(f"expected Ballot | bool | None, got {type(value).__name__}")


def _vote(voter: Voter, proposal: ProposalBody, criterion: str | None, b: Ballot) -> VoteBody:
    return VoteBody(proposal=proposal.proposal, rubric_version=proposal.rubric_version, voter=voter.name,
                    measure=voter.measure, criterion=criterion, passed=b.passed, score=b.score,
                    confidence=b.confidence, reasons=tuple(b.reasons))


def abstain(voter: Voter, proposal: ProposalBody, criterion: str | None, reason: str) -> VoteBody:
    return _vote(voter, proposal, criterion, Ballot(None, reasons=(reason,)))


def targets_of(voter: Voter, rubric: Rubric) -> list[str | None]:
    """Criteria ids ``voter`` votes on for ``rubric`` (``[None]`` = one overall vote)."""
    fn = getattr(voter, "targets", None)
    if fn is not None:
        return list(fn(rubric))
    if voter.criteria is None:
        return [None]
    return [c.id for c in rubric.criteria if c.id in voter.criteria]


class BaseVoter:
    """Shared plumbing. ``per_criterion`` voters cover the rubric criteria of their measure
    (optionally restricted to ``criteria``); others give one overall vote unless ``criteria``."""

    per_criterion = False

    def __init__(self, name: str, measure: Measure, *, criteria: Sequence[str] | None = None,
                 pool: str | None = None, weight: int = 1) -> None:
        self.name, self.measure, self.pool, self.weight = name, measure, pool, weight
        self.criteria = None if criteria is None else frozenset(criteria)

    def covered(self, rubric: Rubric) -> list[Criterion]:
        return [c for c in rubric.criteria if c.measure == self.measure
                and (self.criteria is None or c.id in self.criteria)]

    def targets(self, rubric: Rubric) -> list[str | None]:
        if self.per_criterion:
            return [c.id for c in self.covered(rubric)]
        if self.criteria is None:
            return [None]
        return [c.id for c in rubric.criteria if c.id in self.criteria]

    def ballots(self, proposal: ProposalBody, targets: Sequence[str | None],
                got: Mapping[str | None, Ballot | bool | None] | Ballot | bool | None) -> list[VoteBody]:
        if not isinstance(got, Mapping):
            return [_vote(self, proposal, t, Ballot.of(got)) for t in targets]
        return [_vote(self, proposal, t, Ballot.of(got[t])) if t in got
                else abstain(self, proposal, t, "no ballot") for t in targets]


Verdictish = Mapping[str | None, Ballot | bool | None] | Ballot | bool | None
VoteFn = Callable[[ProposalBody, bytes, Rubric, StudentSpec], Verdictish | Awaitable[Verdictish]]


class CallableVoter(BaseVoter):
    """Wraps ``fn(proposal, artifact, rubric, spec)`` (sync runs in a thread) returning a
    ``Ballot``/``bool``/``None`` for every target or a mapping ``criterion -> Ballot|bool|None``."""

    def __init__(self, name: str, fn: VoteFn, *, measure: Measure = "deterministic",
                 criteria: Sequence[str] | None = None, pool: str | None = None, weight: int = 1) -> None:
        super().__init__(name, measure, criteria=criteria, pool=pool, weight=weight)
        self.fn = fn

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        if inspect.iscoroutinefunction(self.fn):
            got = await self.fn(proposal, artifact, rubric, spec)
        else:
            got = await asyncio.to_thread(self.fn, proposal, artifact, rubric, spec)
            if inspect.isawaitable(got):
                got = await got
        return self.ballots(proposal, self.targets(rubric), got)  # type: ignore[arg-type]


def artifact_path(spec: StudentSpec) -> str:
    """Snapshot-relative path of the artifact: ``spec.output.path`` or ``artifact.txt``."""
    return normalize_rel(spec.output.path) if spec.output.path else "artifact.txt"


def _verdict(ok: bool, *reasons: str) -> Ballot:
    return Ballot(ok, 1.0 if ok else 0.0, 1.0, tuple(reasons))


class DeterministicCheckVoter(BaseVoter):
    """Every ``deterministic`` criterion's ``check`` against an isolated temp snapshot holding only
    the artifact. ``command`` runs an allowlisted argv (never a shell) with a timeout; ``python``
    imports a ``ci_lab.*`` callable ``fn(artifact: bytes, spec)`` returning bool, ``(bool, reason)``
    or a [0, 1] score (pass iff ``>= threshold``). Unknown kinds and check errors abstain."""

    per_criterion = True

    def __init__(self, name: str = "deterministic", *, criteria: Sequence[str] | None = None,
                 pool: str | None = None, weight: int = 1, command_timeout_s: float = 60.0,
                 allowlist: Collection[str] = COMMAND_ALLOWLIST) -> None:
        super().__init__(name, "deterministic", criteria=criteria, pool=pool, weight=weight)
        self.command_timeout_s, self.allowlist = command_timeout_s, frozenset(allowlist)

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        crits = self.covered(rubric)
        rel = artifact_path(spec)
        out: list[VoteBody] = []
        with tempfile.TemporaryDirectory(prefix="ci-vote-", ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(artifact)
            for c in crits:
                try:
                    b = await self.check(c, root, artifact, spec)
                except Exception as exc:  # noqa: BLE001 - one broken check abstains alone
                    b = Ballot(None, reasons=(f"check error: {type(exc).__name__}: {exc}"[:500],))
                out.append(_vote(self, proposal, c.id, b))
        return out

    async def check(self, c: Criterion, root: Path, artifact: bytes, spec: StudentSpec) -> Ballot:
        chk = c.to_json()["check"]
        kind = chk.get("kind")
        if kind == "regex":
            found = re.search(chk["pattern"], artifact.decode("utf-8", "replace"), re.MULTILINE) is not None
            return _verdict(found != bool(chk.get("negate", False)), f"pattern {'found' if found else 'absent'}")
        if kind == "json_schema":
            try:
                doc = json.loads(artifact)
            except ValueError as exc:
                return _verdict(False, f"invalid JSON: {exc}")
            errs = sorted(e.message for e in Draft202012Validator(chk["schema"]).iter_errors(doc))
            return _verdict(not errs, *errs[:5])
        if kind == "file_exists":
            ok = (root / normalize_rel(chk["path"])).is_file()
            return _verdict(ok, f"{chk['path']} {'exists' if ok else 'missing'}")
        if kind == "command":
            return await self._command(chk, root)
        if kind == "python":
            return await self._python(chk["callable"], c, artifact, spec)
        return Ballot(None, reasons=(f"unknown check kind {kind!r}",))

    async def _command(self, chk: Mapping[str, Any], root: Path) -> Ballot:
        argv = [str(a) for a in chk["argv"]]
        if not argv or argv[0] not in self.allowlist:
            return Ballot(None, reasons=(f"command {argv[:1]} not in allowlist {sorted(self.allowlist)}",))
        exe = sys.executable if argv[0] == "python" else shutil.which(argv[0])
        if exe is None:
            return Ballot(None, reasons=(f"executable {argv[0]!r} not found",))
        timeout = float(chk.get("timeout_s", self.command_timeout_s))
        proc = await asyncio.create_subprocess_exec(
            exe, *argv[1:], cwd=root, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            return _verdict(False, f"command timed out after {timeout:g}s")
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        tail = stdout.decode("utf-8", "replace").strip()[-400:]
        return _verdict(proc.returncode == 0, f"exit {proc.returncode}" + (f": {tail}" if tail else ""))

    async def _python(self, target: str, c: Criterion, artifact: bytes, spec: StudentSpec) -> Ballot:
        if not PYTHON_TARGET_RE.match(target):
            return Ballot(None, reasons=(f"python callable {target!r} not under ci_lab.",))
        module, _, attr = target.rpartition(".")
        fn = getattr(importlib.import_module(module), attr)
        got = await fn(artifact, spec) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, artifact, spec)
        if isinstance(got, tuple):
            return _verdict(bool(got[0]), *map(str, got[1:]))
        if isinstance(got, bool):
            return _verdict(got)
        if isinstance(got, (int, float)) and 0.0 <= got <= 1.0:
            return Ballot(got >= c.threshold, float(got), 1.0, (f"score {got:g} vs threshold {c.threshold:g}",))
        return Ballot(None, reasons=(f"python check returned unsupported {got!r}"[:200],))


class CriticChecksVoter(BaseVoter):
    """``tools.critic_checks.run_checks`` over the artifact as a one-file added diff."""

    def __init__(self, cfg: CriticConfig, *, name: str = "critic-checks", criteria: Sequence[str] | None = None,
                 pool: str | None = None, weight: int = 1) -> None:
        super().__init__(name, "deterministic", criteria=criteria, pool=pool, weight=weight)
        self.cfg = cfg

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        change = FileChange(artifact_path(spec), "A", None, artifact.decode("utf-8", "replace"))
        diff = ArmDiff(base="empty", head=proposal.artifact.sha256, files=[change])
        reasons = await asyncio.to_thread(run_checks, diff, self.cfg)
        return self.ballots(proposal, self.targets(rubric), _verdict(not reasons, *reasons))


class RulesVoter(BaseVoter):
    """``ci_lab.rules`` response rules over the artifact text; any match whose severity is in
    ``fail_severities`` fails the vote (all matches are listed as reasons)."""

    def __init__(self, bundle: Bundle, *, name: str = "rules", criteria: Sequence[str] | None = None,
                 fail_severities: Collection[str] = ("critical", "major"), pool: str | None = None,
                 weight: int = 1) -> None:
        super().__init__(name, "deterministic", criteria=criteria, pool=pool, weight=weight)
        self.bundle, self.fail_severities = bundle, frozenset(fail_severities)

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        step = TrajectoryStep(i=0, kind="response", text=artifact.decode("utf-8", "replace"))
        matches = evaluate(self.bundle, GuardView(pending=step), on="response")
        bad = [m for m in matches if m.rule.severity in self.fail_severities]
        b = _verdict(not bad, *(f"{m.rule.id} [{m.rule.severity}]: {m.message}" for m in matches))
        return self.ballots(proposal, self.targets(rubric), b)


def _normalize(voter: Voter, proposal: ProposalBody, targets: list[str | None],
               votes: Sequence[Any]) -> list[VoteBody]:
    first: dict[str | None, VoteBody] = {}
    for v in votes:
        if (isinstance(v, VoteBody) and v.voter == voter.name and v.proposal == proposal.proposal
                and v.rubric_version == proposal.rubric_version and v.criterion in targets):
            first.setdefault(v.criterion, v)
    return [first.get(t) or abstain(voter, proposal, t, "no vote") for t in targets]


async def _run_one(voter: Voter, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec, pools: ResourcePools, timeout_s: float) -> list[VoteBody]:
    targets = targets_of(voter, rubric)
    if not targets:
        return []
    pool: str | None = getattr(voter, "pool", None)
    admission: Callable[[], AbstractAsyncContextManager[Any]] | None = getattr(voter, "admission", None)
    try:
        async with AsyncExitStack() as stack:
            if pool is not None:
                await stack.enter_async_context(pools.acquire(pool, getattr(voter, "weight", 1)))
            if admission is not None:
                await stack.enter_async_context(admission())
            votes = await asyncio.wait_for(voter.vote(proposal, artifact, rubric, spec), timeout_s)
    except TimeoutError:
        return [abstain(voter, proposal, t, f"timeout after {timeout_s:g}s") for t in targets]
    except Exception as exc:  # noqa: BLE001 - a broken voter abstains, it never sinks the round
        return [abstain(voter, proposal, t, f"error: {type(exc).__name__}: {exc}"[:500]) for t in targets]
    return _normalize(voter, proposal, targets, votes)


async def run_voters(voters: Sequence[Voter], proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                     spec: StudentSpec, *, pools: ResourcePools, timeout_s: float) -> list[VoteBody]:
    """All voters concurrently; votes in voter order, then target order."""
    if rubric.version_id != proposal.rubric_version:
        raise ValueError(f"rubric {rubric.version_id!r} != proposal rubric {proposal.rubric_version!r}")
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
    names = [v.name for v in voters]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate voter names: {sorted(names)}")
    for v in voters:
        pool = getattr(v, "pool", None)
        if pool is not None and pool not in pools:
            raise ValueError(f"voter {v.name!r}: unknown resource pool {pool!r}")
    results = await asyncio.gather(*(_run_one(v, proposal, artifact, rubric, spec, pools, timeout_s)
                                     for v in voters))
    return [vote for r in results for vote in r]
