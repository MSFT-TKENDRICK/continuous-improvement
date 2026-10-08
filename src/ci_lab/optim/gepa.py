"""GEPA over harness text components (design §11.2 ``gepa`` strategy; C18, C20).

GEPA's engine (``gepa`` 0.1.4, the optimizer behind ``dspy.GEPA``) runs with a custom
``GEPAAdapter`` whose candidate is ``{target_id: text}``:

* ``evaluate`` = injected :class:`~ci_lab.optim.scoring.EvolveScorer` on a stable
  train/val carve of the **evolve split only**, behind an
  :class:`~ci_lab.optim.scoring.EvolveGuard` (held-out ids raise) and a hard
  :class:`~ci_lab.optim.scoring.MetricBudget` (C18);
* ``make_reflective_dataset`` = typed :class:`~ci_lab.contracts.FailureRecord` s only
  (rendered by :func:`~ci_lab.optim.scoring.render_failure`), never raw outputs (C20);
* reflection LM = a ``dspy.LM`` from :func:`ci_lab.optim.lm.make_lm` (purpose
  ``reflector``), per-call W3C trace headers for the AGL proxy / copilot-serve.

GEPA's own acceptance and scores are **diagnostic only** — the critic + ASSERT/RRSI/OES
gates decide. Cost (metric calls, scorer and reflection tokens) is returned so the
caller can add it to ΔC.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ci_lab import obs
from ci_lab.contracts import FailureRecord
from ci_lab.optim.lm import lm_usage_tokens
from ci_lab.optim.scoring import (
    BudgetExhausted,
    CaseOutcome,
    EvolveGuard,
    EvolveScorer,
    MetricBudget,
    render_failure,
    stable_split,
    subsample,
)

log = logging.getLogger(__name__)
MAX_REFLECTION_RECORDS = 8


@dataclass(frozen=True)
class GepaConfig:
    max_metric_calls: int = 60          # per-case scorer calls (gepa semantics)
    tokens_per_metric_call: int = 4000  # converts ArmContext.budget_tokens to a call cap
    reflection_minibatch_size: int = 3
    val_fraction: float = 0.34
    candidate_selection_strategy: str = "pareto"
    module_selector: str = "round_robin"
    skip_perfect_score: bool = True
    max_cases: int | None = 24          # evolve sub-split size (stable hash)
    seed: int = 0


@dataclass
class OptimizerCost:
    """What the optimizer itself spent — add to the arm's ΔC (C18)."""

    strategy: str
    metric_budget: int = 0
    metric_calls: int = 0
    refused_calls: int = 0
    scorer_tokens: int = 0
    reflection_calls: int = 0
    reflection_tokens: int = 0
    candidates: int = 0
    wall_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.scorer_tokens + self.reflection_tokens

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_tokens": self.total_tokens}

    def span_attrs(self) -> dict[str, int | float]:
        p = "ci.optimizer."
        return {p + "metric_calls": self.metric_calls, p + "metric_budget": self.metric_budget,
                p + "scorer_tokens": self.scorer_tokens, p + "reflection_tokens": self.reflection_tokens,
                p + "total_tokens": self.total_tokens, p + "candidates": self.candidates}


@dataclass
class TextOptimization:
    """Outcome of one optimizer run. ``changed`` holds only texts that differ from seed."""

    seed: dict[str, str]
    best: dict[str, str]
    cost: OptimizerCost
    seed_score: float | None = None
    best_score: float | None = None
    note: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> dict[str, str]:
        return {k: v for k, v in self.best.items() if self.seed.get(k) != v}


def metric_cap(config_cap: int, budget_tokens: int | None, tokens_per_call: int) -> int:
    """C18: the arm's token budget bounds GEPA's ``max_metric_calls``."""
    cap = max(0, int(config_cap))
    if budget_tokens is not None:
        cap = min(cap, max(0, int(budget_tokens)) // max(1, tokens_per_call))
    return cap


class DspyReflectionLM:
    """gepa ``LanguageModel`` (``prompt -> str``) over a ``dspy.LM`` with cost tracking."""

    def __init__(self, lm: Any) -> None:
        self.lm = lm
        self.calls = 0
        self.est_tokens = 0
        self._hist0 = len(getattr(lm, "history", []) or [])

    def __call__(self, prompt: str | list[dict[str, Any]], **call_kwargs: Any) -> str:
        kw: dict[str, Any] = {"messages": prompt} if isinstance(prompt, list) else {"prompt": prompt}
        kw.update(call_kwargs)
        if (headers := obs.carrier()) and isinstance(getattr(self.lm, "_engine_spec", None), str):
            kw["extra_headers"] = headers  # per-request trace propagation (§12.5); HTTP engines only
        out = self.lm(**kw)
        self.calls += 1
        first = out[0] if out else ""
        text = first if isinstance(first, str) else str(first.get("text") or "")
        self.est_tokens += (len(str(prompt)) + len(text)) // 4
        return text

    @property
    def tokens(self) -> int:
        hist = (getattr(self.lm, "history", None) or [])[self._hist0:]
        used = lm_usage_tokens(type("H", (), {"history": hist})())
        return used or self.est_tokens


class ComponentAdapter:
    """``gepa.GEPAAdapter`` over ``{target_id: text}`` candidates (see module doc)."""

    propose_new_texts = None

    def __init__(self, guard: EvolveGuard) -> None:
        self.guard = guard

    def evaluate(self, batch: list[Mapping[str, str]], candidate: dict[str, str],
                 capture_traces: bool = False) -> Any:
        from gepa.core.adapter import EvaluationBatch

        case_ids = [b["case_id"] for b in batch]
        try:
            outs = self.guard.score(candidate, case_ids)
            refused = False
        except BudgetExhausted:
            outs, refused = [CaseOutcome(c, 0.0) for c in case_ids], True
        scores = [o.score for o in outs]
        traj = [{"case_id": o.case_id, "score": o.score, "failure": o.failure, "refused": refused}
                for o in outs] if capture_traces else None
        return EvaluationBatch(outputs=scores, scores=scores, trajectories=traj,
                               num_metric_calls=0 if refused else len(case_ids))

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: Any,
                                components_to_update: list[str]) -> dict[str, list[dict[str, Any]]]:
        records = []
        for t in sorted(eval_batch.trajectories or [], key=lambda t: (t["score"], t["case_id"])):
            if t["refused"]:
                continue
            f: FailureRecord | None = t["failure"]
            if f is None and t["score"] < 1.0:
                f = self.guard.incumbent.get(t["case_id"])
            records.append({
                "Inputs": {"case_id": t["case_id"], **({"suite": f.suite, "category": f.category} if f else {})},
                "Generated Outputs": f"score={t['score']:.3f}",
                "Feedback": render_failure(f, t["score"]),
            })
        records = records[:MAX_REFLECTION_RECORDS]
        return {c: records for c in components_to_update}


class _QuietLogger:
    def log(self, message: str) -> None:
        log.debug("gepa: %s", message)


async def optimize_texts(seed: Mapping[str, str], scorer: EvolveScorer, evolve_cases: Sequence[str], *,
                         reflection_lm: Any, budget_tokens: int | None = None,
                         config: GepaConfig = GepaConfig(),
                         incumbent_failures: Iterable[FailureRecord] = ()) -> TextOptimization:
    """Run GEPA on ``seed`` (``{target_id: text}``) over ``evolve_cases``. Must be awaited
    from the event loop the (async) scorer belongs to; GEPA itself runs in a worker
    thread (``asyncio.to_thread`` carries the OTel context)."""
    cost = OptimizerCost("gepa")
    seed = dict(seed)
    t0 = time.monotonic()
    evolve = subsample(evolve_cases, config.max_cases, salt=f"gepa{config.seed}")
    train, val = stable_split(evolve, config.val_fraction, salt=f"gepa{config.seed}")
    mb = max(1, min(config.reflection_minibatch_size, len(train)))
    cap = metric_cap(config.max_metric_calls, budget_tokens, config.tokens_per_metric_call)
    cost.metric_budget = cap
    if not seed or not evolve:
        return TextOptimization(seed, dict(seed), cost, note="nothing to optimise")
    per_iteration = 2 * mb + len(val)  # parent + child minibatch, then val of an accepted child
    if cap < len(val) + per_iteration:
        return TextOptimization(seed, dict(seed), cost,
                                note=f"metric budget {cap} < one GEPA iteration ({len(val) + per_iteration})")

    budget = MetricBudget(cap)
    guard = EvolveGuard(scorer, evolve, budget, loop=asyncio.get_running_loop(),
                        incumbent_failures=incumbent_failures)
    refl = reflection_lm if isinstance(reflection_lm, DspyReflectionLM) else DspyReflectionLM(reflection_lm)

    def stop(_state: Any) -> bool:
        return budget.remaining < per_iteration

    def run() -> Any:
        import gepa

        return gepa.optimize(
            seed_candidate=dict(seed),
            trainset=[{"case_id": c} for c in train],
            valset=[{"case_id": c} for c in val],
            adapter=ComponentAdapter(guard),
            reflection_lm=refl,
            candidate_selection_strategy=config.candidate_selection_strategy,
            module_selector=config.module_selector,
            reflection_minibatch_size=mb,
            skip_perfect_score=config.skip_perfect_score,
            max_metric_calls=cap,
            stop_callbacks=[stop],
            use_merge=False,
            logger=_QuietLogger(),
            display_progress_bar=False,
            seed=config.seed,
        )

    try:
        result = await asyncio.to_thread(run)
    finally:
        cost.metric_calls = budget.used
        cost.refused_calls = budget.refused
        cost.scorer_tokens = guard.scorer_tokens
        cost.reflection_calls = refl.calls
        cost.reflection_tokens = refl.tokens
        cost.candidates = len(guard.candidates)
        cost.wall_s = round(time.monotonic() - t0, 3)
    best = result.best_candidate if isinstance(result.best_candidate, dict) else dict(seed)
    scores = list(result.val_aggregate_scores or [])
    return TextOptimization(
        seed, {k: best.get(k, v) for k, v in seed.items()}, cost,
        seed_score=scores[0] if scores else None,
        best_score=scores[result.best_idx] if scores else None,
        diagnostics={"num_candidates": result.num_candidates, "val_scores": scores,
                     "train_cases": len(train), "val_cases": len(val)},
    )
