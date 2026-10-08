"""Evolve-split scoring for optimizer strategies (C15, C18, C20).

Optimizers (GEPA, SkillOpt) never evaluate on their own: they call an injected
:class:`EvolveScorer` with a *candidate* (``{target_id: text}``) and evolve case ids.
Wrappers here enforce that only evolve cases are ever scored (:class:`EvolveGuard`),
hard-cap the number of metric calls (:class:`MetricBudget`, C18) and bridge async
scorers into the optimizers' synchronous worker threads. Reflection text is rendered
from typed :class:`~ci_lab.contracts.FailureRecord` s only (C12/C20).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import shutil
import threading
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ci_lab.contracts import Domain, FailureRecord

EVOLVE = "evolve"
EXCERPT_CHARS = 400


@dataclass(frozen=True)
class CaseOutcome:
    """Score of one evolve case under one candidate (mean over trials; missing = 0)."""

    case_id: str
    score: float
    failure: FailureRecord | None = None
    tokens: int = 0


class EvolveScorer(Protocol):
    """``candidate`` maps target ids (worktree-relative paths, optionally ``#key``) to
    full replacement texts; returns one outcome per requested case (same order)."""

    def __call__(self, candidate: Mapping[str, str], case_ids: Sequence[str]
                 ) -> Sequence[CaseOutcome] | Awaitable[Sequence[CaseOutcome]]: ...


class HeldOutAccessError(PermissionError):
    """An optimizer asked to score a case outside the evolve split (C15)."""


class BudgetExhausted(RuntimeError):
    """The arm's metric-call budget (C18) is spent."""


def candidate_hash(candidate: Mapping[str, str]) -> str:
    raw = json.dumps(dict(sorted(candidate.items())), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def render_failure(f: FailureRecord | None, score: float | None = None) -> str:
    """Typed, bounded feedback for reflection prompts — never raw tool output (C12)."""
    if f is None:
        return "passed" if score is None else f"score={score:.3f}; no failure recorded"
    parts = [f"suite={f.suite}", f"category={f.category}"]
    if f.rule_ids:
        parts.append("violated_rules=" + ",".join(f.rule_ids))
    if f.rubric_scores:
        parts.append("rubric=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(f.rubric_scores.items())))
    if score is not None:
        parts.insert(0, f"score={score:.3f}")
    text = "; ".join(parts)
    if f.excerpt:
        text += "\nexcerpt: " + f.excerpt[:EXCERPT_CHARS]
    return text


def stable_split(case_ids: Sequence[str], val_fraction: float = 0.34, *, salt: str = ""
                 ) -> tuple[list[str], list[str]]:
    """Deterministic (train, val) carve of the evolve split for an optimizer's own
    train/val (C18). With <2 cases both halves are the same list."""
    ids = sorted(dict.fromkeys(case_ids), key=lambda c: hashlib.sha256(f"{salt}|{c}".encode()).hexdigest())
    if len(ids) < 2:
        return list(ids), list(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    return ids[n_val:], ids[:n_val]


def subsample(case_ids: Sequence[str], n: int | None, *, salt: str = "") -> list[str]:
    """Deterministic evolve *sub*-split of at most ``n`` cases (stable-hash order)."""
    ids = list(dict.fromkeys(case_ids))
    if n is None or n >= len(ids):
        return ids
    keep = set(sorted(ids, key=lambda c: hashlib.sha256(f"sub|{salt}|{c}".encode()).hexdigest())[:max(0, n)])
    return [c for c in ids if c in keep]


class MetricBudget:
    """Thread-safe hard cap on per-case metric calls (C18)."""

    def __init__(self, max_calls: int) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        self.max_calls = max_calls
        self.used = 0
        self.refused = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls

    def charge(self, n: int = 1) -> None:
        with self._lock:
            if self.used + n > self.max_calls:
                self.refused += n
                raise BudgetExhausted(f"metric budget {self.max_calls} exhausted")
            self.used += n


def resolve_scorer_result(result: Any, loop: asyncio.AbstractEventLoop | None = None) -> Any:
    """Return ``result`` or, when awaitable, its value. From a worker thread the
    coroutine runs on ``loop`` (the strategy's event loop, free while the optimizer
    runs in ``asyncio.to_thread``); without a usable loop it gets its own."""
    if not inspect.isawaitable(result):
        return result
    if loop is not None and loop.is_running():
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not loop:
            async def _await() -> Any:
                return await result
            return asyncio.run_coroutine_threadsafe(_await(), loop).result()
    async def _await_own() -> Any:
        return await result
    return asyncio.run(_await_own())


class EvolveGuard:
    """Sync facade over an :class:`EvolveScorer` used inside optimizer threads:
    rejects non-evolve cases, charges the budget *before* scoring, records cost and
    the outcomes seen (for reflection fallbacks and the cost report)."""

    def __init__(self, scorer: EvolveScorer, evolve_cases: Iterable[str], budget: MetricBudget, *,
                 loop: asyncio.AbstractEventLoop | None = None,
                 incumbent_failures: Iterable[FailureRecord] = ()) -> None:
        self.scorer = scorer
        self.evolve = frozenset(evolve_cases)
        if not self.evolve:
            raise ValueError("no evolve cases to optimise on")
        self.budget = budget
        self.loop = loop
        self.tokens = 0
        self.calls = 0
        self.candidates: set[str] = set()
        self._lock = threading.Lock()
        # incumbent failures restricted to evolve cases (C15): fallback reflection data
        self.incumbent = {f.case_id: f for f in incumbent_failures if f.case_id in self.evolve}

    def check(self, case_ids: Iterable[str]) -> None:
        bad = sorted(set(case_ids) - self.evolve)
        if bad:
            raise HeldOutAccessError(f"non-evolve cases requested: {bad[:5]}")

    def score(self, candidate: Mapping[str, str], case_ids: Sequence[str]) -> list[CaseOutcome]:
        self.check(case_ids)
        self.budget.charge(len(case_ids))
        out = list(resolve_scorer_result(self.scorer(dict(candidate), list(case_ids)), self.loop))
        if len(out) != len(case_ids):
            raise ValueError("scorer returned a different number of outcomes than cases")
        with self._lock:
            self.calls += len(case_ids)
            self.tokens += sum(o.tokens for o in out)
            self.candidates.add(candidate_hash(candidate))
        return [o if o.failure is not None or o.score >= 1.0 or o.case_id not in self.incumbent
                else replace(o, failure=self.incumbent[o.case_id]) for o in out]

    def score_one(self, candidate: Mapping[str, str], case_id: str) -> CaseOutcome:
        return self.score(candidate, [case_id])[0]

    @property
    def scorer_tokens(self) -> int:
        spent = getattr(self.scorer, "tokens_spent", None)
        return int(spent) if isinstance(spent, int) else self.tokens


# ---------------------------------------------------------------- Domain adapter

def _write_target(root: Path, target_id: str, text: str) -> None:
    from ci_lab.optim.targets import TextTarget

    TextTarget.parse(target_id).write(root, text)


class DomainEvolveScorer:
    """:class:`EvolveScorer` over a contracts :class:`~ci_lab.contracts.Domain`.

    Each distinct candidate is materialised once into a scratch copy of the arm
    worktree (under ``scratch_root``) and evaluated with
    ``domain.evaluate(dir, "evolve", k, ...)`` — the split is hard-coded, so held-out
    data is unreachable. Results are cached per candidate; per-case lookups after the
    first are free. ``Domain.evaluate`` has no case filter in contracts v2.3, so one
    candidate evaluation costs the whole evolve split (see docs/optim.md)."""

    def __init__(self, domain: Domain, worktree: Path, scratch_root: Path, *, experiment_id: str,
                 variant: str, k: int = 1) -> None:
        self.domain = domain
        self.worktree = Path(worktree)
        self.scratch_root = Path(scratch_root)
        self.experiment_id = experiment_id
        self.variant = variant
        self.k = k
        self.tokens_spent = 0
        self.evaluations = 0
        self._cache: dict[str, dict[str, CaseOutcome]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def evolve_cases(self) -> list[str]:
        return list(self.domain.splits()[EVOLVE])

    async def __call__(self, candidate: Mapping[str, str], case_ids: Sequence[str]) -> list[CaseOutcome]:
        evolve = set(self.evolve_cases())
        if bad := sorted(set(case_ids) - evolve):
            raise HeldOutAccessError(f"non-evolve cases requested: {bad[:5]}")
        h = candidate_hash(candidate)
        lock = self._locks.setdefault(h, asyncio.Lock())
        async with lock:
            if h not in self._cache:
                self._cache[h] = await self._evaluate(h, candidate)
        res = self._cache[h]
        return [res.get(c, CaseOutcome(c, 0.0)) for c in case_ids]

    async def _evaluate(self, h: str, candidate: Mapping[str, str]) -> dict[str, CaseOutcome]:
        d = self.scratch_root / f"cand-{h}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.worktree, d, ignore=shutil.ignore_patterns(
            ".git", ".venv", "node_modules", "__pycache__"))
        for tid, text in candidate.items():
            _write_target(d, tid, text)
        result = await self.domain.evaluate(d, EVOLVE, self.k, experiment_id=self.experiment_id,
                                            variant=f"{self.variant}~opt-{h[:8]}")
        if result.split != EVOLVE:
            raise HeldOutAccessError(f"domain returned split {result.split!r}")
        self.evaluations += 1
        failures = {f.case_id: f for f in self.domain.failures(result)}
        per: dict[str, list[float]] = {}
        tokens: dict[str, int] = {}
        for s in result.scores:
            per.setdefault(s.case_id, []).append(0.0 if s.score is None else float(s.score))
            tokens[s.case_id] = tokens.get(s.case_id, 0) + s.tokens_in + s.tokens_out
        self.tokens_spent += sum(tokens.values())
        shutil.rmtree(d, ignore_errors=True)
        return {c: CaseOutcome(c, sum(v) / max(len(v), self.k), failures.get(c), tokens.get(c, 0))
                for c, v in per.items()}
