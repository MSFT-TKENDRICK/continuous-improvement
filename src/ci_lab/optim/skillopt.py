"""SkillOpt-Sleep text-space optimization of skill markdown inside an arm (design §11.2
``skillopt`` strategy; C3, C18, C20).

``dream_consolidate`` (skillopt-sleep 0.2.x) runs against a ``CliBackend`` subclass:

* ``attempt`` = injected :class:`~ci_lab.optim.scoring.EvolveScorer` on the candidate
  ``{skill_target: skill, memory_target: memory}`` for one **evolve** case (guarded,
  budget-charged); the "response" is an opaque outcome token;
* ``judge`` = that outcome (hard = score ≥ threshold, soft = score, rationale = typed
  :class:`~ci_lab.contracts.FailureRecord` rendering);
* ``reflect`` = SkillOpt's own bounded-edit reflection prompt, whose LLM call goes to a
  ``dspy.LM`` from :func:`ci_lab.optim.lm.make_lm`. Task intents are built from case
  ids/suite/category only, so reflection never sees raw transcripts (C20).

SkillOpt's train/val gate (strict val improvement) is applied, but like GEPA's it is
diagnostic: accepted text becomes an ordinary Edit that still goes through critic +
ASSERT + RRSI/OES. The evolve sub-split is sized so a full consolidation fits the
metric budget (C18).
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ci_lab.contracts import FailureRecord
from ci_lab.optim.gepa import (
    DspyReflectionLM,
    OptimizerCost,
    TextOptimization,
    metric_cap,
)
from ci_lab.optim.scoring import (
    BudgetExhausted,
    CaseOutcome,
    EvolveGuard,
    EvolveScorer,
    MetricBudget,
    candidate_hash,
    render_failure,
    stable_split,
    subsample,
)


@dataclass(frozen=True)
class SkillOptConfig:
    max_metric_calls: int = 60
    tokens_per_metric_call: int = 4000
    val_fraction: float = 0.34
    pass_threshold: float = 1.0
    gate_metric: str = "mixed"
    max_cases: int | None = None  # None = largest sub-split that fits the budget
    project: str = "harness"
    seed: int = 0


def calls_needed(n_train: int, n_val: int, memory: bool) -> int:
    """Per-case scorer calls of one gated ``dream_consolidate`` epoch: baseline val,
    train replay, skill gate val, final val (+ memory: train replay, memory gate val)."""
    return n_train + 3 * n_val + ((n_train + n_val) if memory else 0)


def fit_cases(evolve: Sequence[str], cap: int, *, val_fraction: float, memory: bool, salt: str,
              max_cases: int | None = None) -> tuple[list[str], list[str]]:
    """Largest deterministic evolve sub-split whose consolidation fits ``cap``."""
    ids = subsample(evolve, max_cases, salt=salt)
    for n in range(len(ids), 1, -1):
        tr, va = stable_split(subsample(ids, n, salt=salt), val_fraction, salt=salt)
        if calls_needed(len(tr), len(va), memory) <= cap:
            return tr, va
    return [], []


def _backend(guard: EvolveGuard, skill_id: str, memory_id: str | None, refl: DspyReflectionLM,
             threshold: float) -> Any:
    from skillopt_sleep.backend import CliBackend

    class GuardedScorerBackend(CliBackend):
        name = "ci-lab-evolve"

        def __init__(self) -> None:
            super().__init__(model=str(getattr(refl.lm, "model", "")), timeout=0)
            self._outcomes: dict[tuple[str, str], CaseOutcome | None] = {}
            self._lock = threading.Lock()

        def attempt(self, task: Any, skill: str, memory: str, sample_id: int = 0) -> str:
            cand = {skill_id: skill, **({memory_id: memory} if memory_id else {})}
            try:
                outcome: CaseOutcome | None = guard.score_one(cand, task.id)
                resp = f"[{candidate_hash(cand)[:8]}] score={outcome.score:.3f}"
            except BudgetExhausted:
                outcome, resp = None, f"[{candidate_hash(cand)[:8]}] budget-exhausted"
            with self._lock:
                self._outcomes[(task.id, resp)] = outcome
            return resp

        def attempt_with_tools(self, task: Any, skill: str, memory: str, tools: Any) -> tuple[str, list[str]]:
            return self.attempt(task, skill, memory), []

        def judge(self, task: Any, response: str) -> tuple[float, float, str]:
            with self._lock:
                o = self._outcomes.get((task.id, response))
            if o is None:
                return 0.0, 0.0, "budget exhausted"
            f = o.failure if o.score < threshold else None
            return (1.0 if o.score >= threshold else 0.0), o.score, render_failure(f, o.score)

        def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
            return refl(prompt)

    return GuardedScorerBackend()


def _task_records(train: Sequence[str], val: Sequence[str], failures: dict[str, FailureRecord],
                  project: str) -> list[Any]:
    from skillopt_sleep.types import TaskRecord

    def intent(c: str) -> str:
        f = failures.get(c)
        return f"evolve case {c}" + (f" ({f.suite}/{f.category})" if f else "")

    return [TaskRecord(id=c, project=project, intent=intent(c), split=split, origin="ci-lab")
            for split, ids in (("train", train), ("val", val)) for c in ids]


async def optimize_skill(skill: tuple[str, str], scorer: EvolveScorer, evolve_cases: Sequence[str], *,
                         reflection_lm: Any, memory: tuple[str, str] | None = None, edit_budget: int = 1,
                         budget_tokens: int | None = None, config: SkillOptConfig | None = None,
                         incumbent_failures: Iterable[FailureRecord] = ()) -> TextOptimization:
    """One gated SkillOpt consolidation epoch over ``skill`` (and optional ``memory``),
    each given as ``(target_id, text)``. ``edit_budget`` bounds SkillOpt's edit records
    per document."""
    config = config or SkillOptConfig()
    skill_id, skill_text = skill
    memory_id, memory_text = memory if memory else (None, "")
    seed = {skill_id: skill_text, **({memory_id: memory_text} if memory_id else {})}
    cost = OptimizerCost("skillopt")
    t0 = time.monotonic()
    cap = metric_cap(config.max_metric_calls, budget_tokens, config.tokens_per_metric_call)
    cost.metric_budget = cap
    salt = f"skillopt{config.seed}"
    train, val = fit_cases(evolve_cases, cap, val_fraction=config.val_fraction, memory=memory_id is not None,
                           salt=salt, max_cases=config.max_cases)
    if not train or edit_budget < 1:
        return TextOptimization(seed, dict(seed), cost, note=f"metric budget {cap} too small for a gated epoch"
                                if edit_budget >= 1 else "edit_budget < 1")

    incumbent = list(incumbent_failures)
    budget = MetricBudget(cap)
    guard = EvolveGuard(scorer, [*train, *val], budget, loop=asyncio.get_running_loop(),
                        incumbent_failures=incumbent)
    refl = reflection_lm if isinstance(reflection_lm, DspyReflectionLM) else DspyReflectionLM(reflection_lm)
    backend = _backend(guard, skill_id, memory_id, refl, config.pass_threshold)
    tasks = _task_records(train, val, guard.incumbent, config.project)

    def run() -> Any:
        from skillopt_sleep.dream import dream_consolidate

        return dream_consolidate(backend, tasks, skill_text, memory_text, edit_budget=edit_budget,
                                 gate_metric=config.gate_metric, gate_mode="on", evolve_skill=True,
                                 evolve_memory=memory_id is not None)

    try:
        res = await asyncio.to_thread(run)
    finally:
        cost.metric_calls = budget.used
        cost.refused_calls = budget.refused
        cost.scorer_tokens = guard.scorer_tokens
        cost.reflection_calls = refl.calls
        cost.reflection_tokens = refl.tokens
        cost.candidates = len(guard.candidates)
        cost.wall_s = round(time.monotonic() - t0, 3)
    best = dict(seed)
    if res.accepted:
        best[skill_id] = res.new_skill
        if memory_id:
            best[memory_id] = res.new_memory
    return TextOptimization(
        seed, best, cost, seed_score=res.baseline_score, best_score=res.candidate_score,
        note="" if res.accepted else f"gate {res.gate_action}",
        diagnostics={"accepted": res.accepted, "gate_action": res.gate_action,
                     "applied_edits": len(res.applied_edits), "rejected_edits": len(res.rejected_edits),
                     "rationales": [e.rationale[:200] for e in res.applied_edits],
                     "train_cases": len(train), "val_cases": len(val),
                     "reflect_raw_chars": len(res.reflect_raw or "")},
    )
