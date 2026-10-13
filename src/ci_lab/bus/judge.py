"""Judge core (bus contract v2 §8): pure vote aggregation, student/adversary duels, escalation.

``aggregate`` folds the votes on one proposal into a :class:`VerdictDraft` and fails closed: a
failing oracle vote vetoes, a missing/abstained required criterion fails, quorum counts distinct
voters with answered votes. ``judge`` may consult an :class:`Escalator` (the MAF judge agent,
``meta/specs/judge.yaml``) on close soft-criterion calls; it can flip commit<->revise only when
the decision rests on soft margins, never over a veto, a missing required criterion or quorum.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Protocol

from ci_lab.bus.types import (
    CriterionResult,
    Decision,
    SoftPref,
    StudentCorrection,
    VerdictBody,
    VoteBody,
)
from ci_lab.taskgraph.model import Criterion, Rubric

__all__ = ["ESCALATE_MARGIN", "ESCALATE_STDEV", "JUDGE_SPEC", "DuelResult", "Escalator", "VerdictDraft",
           "aggregate", "duel", "judge"]

ESCALATE_STDEV = 0.25
ESCALATE_MARGIN = 0.05
JUDGE_SPEC = Path(__file__).resolve().parents[1] / "meta" / "specs" / "judge.yaml"


def _value(v: VoteBody) -> float:
    return v.score if v.score is not None else (1.0 if v.passed else 0.0)


def _vote_passes(v: VoteBody, c: Criterion) -> bool:
    return v.passed if v.passed is not None else _value(v) >= c.threshold


@dataclass(frozen=True)
class _Fold:
    result: CriterionResult
    values: tuple[float, ...]


def _fold(c: Criterion, votes: Sequence[VoteBody]) -> _Fold:
    answered = [v for v in votes if v.answered]
    if not answered:
        return _Fold(CriterionResult(None, None, c.required, c.oracle, 0, c.resource), ())
    values = tuple(_value(v) for v in answered)
    score = fmean(values)
    passed = all(_vote_passes(v, c) for v in answered) if c.oracle else score >= c.threshold
    return _Fold(CriterionResult(passed, score, c.required, c.oracle, len(answered), c.resource), values)


@dataclass(frozen=True)
class VerdictDraft:
    """Deterministic verdict on one proposal; ``to_body`` turns it into a ``VerdictBody``."""

    proposal: str
    rubric_version: str
    decision: Decision
    score: float
    pass_score: float
    criteria: Mapping[str, CriterionResult]
    quorum: int
    voters: tuple[str, ...]
    vetoed: tuple[str, ...]
    missing_required: tuple[str, ...]
    failed_required: tuple[str, ...]
    soft_stdev: float
    escalate: bool
    escalated: bool = False
    reasons: tuple[str, ...] = ()
    subscores: Mapping[str, float] = MappingProxyType({})

    @property
    def attempt(self) -> str:
        return self.proposal.split("/", 1)[0]

    @property
    def quorum_met(self) -> bool:
        return len(self.voters) >= self.quorum

    @property
    def hard_blocked(self) -> bool:
        """Veto, missing required criterion, lost quorum or failed required oracle: no override."""
        return bool(self.vetoed or self.missing_required or not self.quorum_met
                    or any(self.criteria[c].oracle for c in self.failed_required))

    @property
    def soft_only(self) -> bool:
        """True when the decision rests only on soft-criterion margins (escalation may flip it)."""
        return not self.hard_blocked

    def to_body(self, votes: Mapping[int, VoteBody], *,
                correction: StudentCorrection | None = None) -> VerdictBody:
        """``votes``: bus seq -> vote body. Only votes on this proposal and rubric version are cited."""
        cited = tuple(sorted(s for s, v in votes.items()
                             if v.proposal == self.proposal and v.rubric_version == self.rubric_version))
        if self.decision == "commit":
            distinct = {votes[s].voter for s in cited if votes[s].answered}
            if len(distinct) < self.quorum:
                raise ValueError(f"commit needs {self.quorum} distinct cited voters, got {len(distinct)}")
        return VerdictBody(proposal=self.proposal, attempt=self.attempt, rubric_version=self.rubric_version,
                           decision=self.decision, score=self.score, criteria=self.criteria, votes=cited,
                           correction=correction, escalated=self.escalated, subscores=self.subscores)


def _not_commit(attempts_left: int) -> Decision:
    return "reject" if attempts_left <= 0 else "revise"


def _single(values: Iterable[str], what: str, given: str | None) -> str:
    found = set(values) | ({given} if given else set())
    if len(found) != 1:
        raise ValueError(f"votes must be on exactly one {what}, got {sorted(found)}")
    return found.pop()


def aggregate(rubric: Rubric, votes: Sequence[VoteBody], manifest_quorum: int, *, attempts_left: int,
              proposal: str | None = None) -> VerdictDraft:
    """Fold ``votes`` (all on one proposal, under ``rubric``) into a fail-closed draft verdict."""
    prop = _single((v.proposal for v in votes), "proposal", proposal)
    _single((v.rubric_version for v in votes), "rubric version", rubric.version_id)
    by_id = {c.id: c for c in rubric.criteria}
    known = [v for v in votes if v.criterion is None or v.criterion in by_id]
    folds = {c.id: _fold(c, [v for v in known if v.criterion == c.id]) for c in rubric.criteria}
    criteria = MappingProxyType({k: f.result for k, f in folds.items()})
    voters = tuple(sorted({v.voter for v in known if v.answered}))
    # Resource (cost/simplicity) criteria never veto and never enter the quality score; a *required*
    # resource criterion that fails still blocks via failed_required.
    vetoed = tuple(k for k, r in criteria.items() if r.oracle and not r.resource and r.passed is False)
    missing = tuple(k for k, r in criteria.items() if r.required and r.passed is None)
    failed = tuple(k for k, r in criteria.items() if r.required and r.passed is False)
    score = rubric.quality_score({k: r.score for k, r in criteria.items()})
    subscores = MappingProxyType(rubric.resource_subscores({k: r.score for k, r in criteria.items()}))
    stdev = max((pstdev(f.values) for k, f in folds.items() if not by_id[k].oracle and len(f.values) > 1),
                default=0.0)
    reasons = [*(f"oracle veto: {k}" for k in vetoed), *(f"missing required: {k}" for k in missing),
               *(f"failed required: {k}" for k in failed if k not in vetoed)]
    if len(voters) < manifest_quorum:
        reasons.append(f"quorum: {len(voters)} < {manifest_quorum}")
    if score < rubric.pass_score:
        reasons.append("score below pass")
    commit = not reasons
    return VerdictDraft(
        proposal=prop, rubric_version=rubric.version_id,
        decision="commit" if commit else _not_commit(attempts_left), score=min(1.0, max(0.0, score)),
        pass_score=rubric.pass_score, criteria=criteria, quorum=manifest_quorum, voters=voters, vetoed=vetoed,
        missing_required=missing, failed_required=failed, soft_stdev=stdev,
        escalate=stdev > ESCALATE_STDEV or abs(score - rubric.pass_score) <= ESCALATE_MARGIN + 1e-9,
        reasons=tuple(reasons), subscores=subscores)


class Escalator(Protocol):
    """Judge agent consulted on close calls; returns ``"commit"``/``"revise"`` or ``None`` (keep)."""

    async def review(self, rubric: Rubric, votes: Sequence[VoteBody], draft: VerdictDraft) -> Decision | None: ...


async def judge(rubric: Rubric, votes: Sequence[VoteBody], quorum: int, attempts_left: int,
                escalator: Escalator | None = None, *, proposal: str | None = None) -> VerdictDraft:
    """:func:`aggregate`, then (if flagged and soft-only) let ``escalator`` flip commit<->revise."""
    draft = aggregate(rubric, votes, quorum, attempts_left=attempts_left, proposal=proposal)
    if escalator is None or not draft.escalate or not draft.soft_only:
        return draft
    try:
        wanted = await escalator.review(rubric, votes, draft)
    except Exception as exc:  # noqa: BLE001 - a failing escalator keeps the deterministic verdict
        return replace(draft, reasons=(*draft.reasons, f"escalator error: {type(exc).__name__}"))
    if wanted not in ("commit", "revise"):
        return replace(draft, escalated=True)
    decision: Decision = "commit" if wanted == "commit" else _not_commit(attempts_left)
    note = () if decision == draft.decision else (f"escalated: {draft.decision} -> {decision}",)
    return replace(draft, decision=decision, escalated=True, reasons=(*draft.reasons, *note))


@dataclass(frozen=True)
class DuelResult:
    soft_pref: SoftPref
    soft_pass_adversary: bool
    oracle_invalid_adversary: tuple[str, ...]
    exploit: bool


def _soft_score(rubric: Rubric, votes: Sequence[VoteBody]) -> tuple[float | None, bool]:
    folds = [(c, _fold(c, [v for v in votes if v.criterion == c.id]).result) for c in rubric.soft()]
    scored = [(c.weight, r.score) for c, r in folds if r.score is not None]
    if not scored:
        return None, False
    score = sum(w * s for w, s in scored) / sum(w for w, _ in scored)
    failing = any(r.passed is False or (c.required and r.passed is None) for c, r in folds)
    return score, not failing and score >= rubric.pass_score


def duel(rubric: Rubric, student_votes: Sequence[VoteBody], adversary_votes: Sequence[VoteBody], *,
         validity_oracles: Sequence[Criterion] = ()) -> DuelResult:
    """Compare soft scores; the adversary is an exploit iff the soft judges prefer or pass it while an
    *independent* validity oracle (``check.independent`` or the deliverable's frozen
    ``validity_oracles``) fails it."""
    s_score, _ = _soft_score(rubric, student_votes)
    a_score, a_pass = _soft_score(rubric, adversary_votes)
    pref: SoftPref
    if a_score is None:
        pref = "tie" if s_score is None else "student"
    elif s_score is None:
        pref = "adversary"
    else:
        pref = "tie" if math.isclose(a_score, s_score) else "adversary" if a_score > s_score else "student"
    validity = {c.id: c for c in rubric.oracles() if c.check.get("independent") is True}
    validity.update({c.id: c for c in validity_oracles})
    invalid = tuple(sorted(k for k, c in validity.items()
                           if _fold(c, [v for v in adversary_votes if v.criterion == k]).result.passed is False))
    return DuelResult(pref, a_pass, invalid, (pref == "adversary" or a_pass) and bool(invalid))
