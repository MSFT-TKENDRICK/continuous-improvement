"""``gepa`` arm strategy: GEPA reflective evolution of text components (design §11.2)."""
from __future__ import annotations

from typing import Any

from ci_lab.contracts import ArmContext
from ci_lab.optim.gepa import GepaConfig, TextOptimization, optimize_texts
from ci_lab.optim.scoring import EvolveScorer
from ci_lab.optim.targets import TextTarget
from ci_lab.strategies.text import TextOptimizerStrategy


class GepaStrategy(TextOptimizerStrategy):
    """Optimises up to ``edit_budget`` components jointly (one GEPA candidate =
    ``{target_id: text}``); each changed component becomes one Edit."""

    name = "gepa"
    default_focus = ("prompt",)

    def __init__(self, *, config: GepaConfig | None = None, **kw: Any) -> None:
        super().__init__(config=config or GepaConfig(), **kw)

    async def _optimize(self, ctx: ArmContext, targets: list[tuple[str, TextTarget]], scorer: EvolveScorer,
                        evolve: list[str], lm: Any) -> list[TextOptimization]:
        chosen = targets[:ctx.directive.edit_budget]
        seed = {t.id: t.read(ctx.worktree) for _, t in chosen}
        return [await optimize_texts(seed, scorer, evolve, reflection_lm=lm, budget_tokens=ctx.budget_tokens,
                                     config=self.config, incumbent_failures=ctx.failures)]

    def _hypothesis(self, target: TextTarget, opt: TextOptimization) -> str:
        s0 = "n/a" if opt.seed_score is None else f"{opt.seed_score:.3f}"
        s1 = "n/a" if opt.best_score is None else f"{opt.best_score:.3f}"
        return (f"gepa: reflective rewrite of {target.id} from evolve FailureRecords "
                f"(diagnostic evolve-val {s0}->{s1}, {opt.cost.metric_calls} metric calls)")
