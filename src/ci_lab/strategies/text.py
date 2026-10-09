"""Common driver for optimizer strategies that rewrite text components (gepa, skillopt).

Flow: resolve focus → targets (evolvable surface only, ≤ ``edit_budget``) → optimizer on
the evolve split → write changed texts → one git commit (= one :class:`Edit`) per
component → cost report + span attributes. Optimizer text is returned as ordinary
Edits so the critic applies the same leak/denylist/size/binding checks (C20).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.contracts import ArmContext, Domain, Edit
from ci_lab.optim.gepa import TextOptimization
from ci_lab.optim.scoring import DomainEvolveScorer, EvolveScorer
from ci_lab.optim.targets import TextTarget, resolve_targets
from ci_lab.strategies.base import (
    GUARD_GLOBS,
    Committer,
    check_edit_budget,
    evolve_cases_for,
    git_commit,
    optimizer_commit_message,
    optimizer_dir,
    optimizer_span,
    write_report,
)


class TextOptimizerStrategy:
    name = "text"
    default_focus: tuple[str, ...] = ("prompt",)

    def __init__(self, *, scorer: EvolveScorer | None = None, domain: Domain | None = None, lm: Any = None,
                 evolve_cases: Sequence[str] | None = None, committer: Committer | None = None, k: int = 1,
                 config: Any = None, component_globs: Mapping[str, Sequence[str]] | None = None,
                 surface_globs: Sequence[str] | None = None, frozen_globs: Sequence[str] | None = None) -> None:
        if scorer is None and domain is None:
            raise TypeError(f"{type(self).__name__} needs a scorer or a domain")
        self.scorer, self.domain, self.lm = scorer, domain, lm
        self.evolve_cases, self.k, self.config = evolve_cases, k, config
        self.committer: Committer = committer or git_commit
        self.component_globs = component_globs if component_globs is not None else \
            getattr(domain, "component_globs", None)
        self.surface_globs = surface_globs if surface_globs is not None else getattr(domain, "surface_globs", None)
        self.frozen_globs = (*(frozen_globs if frozen_globs is not None else getattr(domain, "frozen_globs", ())),
                             *GUARD_GLOBS)
        self.last: list[TextOptimization] = []

    # -- hooks -------------------------------------------------------------------------
    async def _optimize(self, ctx: ArmContext, targets: list[tuple[str, TextTarget]], scorer: EvolveScorer,
                        evolve: list[str], lm: Any) -> list[TextOptimization]:
        raise NotImplementedError

    def _hypothesis(self, target: TextTarget, opt: TextOptimization) -> str:
        return f"{self.name}: rewrite {target.id}"

    def _artifacts(self, ctx: ArmContext, opt: TextOptimization) -> None:
        """Optional extra run-dir artifacts (e.g. SkillOpt's best_skill.md)."""

    # -- driver ------------------------------------------------------------------------
    def _targets(self, ctx: ArmContext) -> list[tuple[str, TextTarget]]:
        return resolve_targets(ctx.worktree, ctx.directive.component_focus, default_focus=self.default_focus,
                               component_globs=self.component_globs, surface_globs=self.surface_globs,
                               frozen_globs=tuple(self.frozen_globs or ()))

    def _scorer(self, ctx: ArmContext) -> EvolveScorer:
        if self.scorer is not None:
            return self.scorer
        assert self.domain is not None
        return DomainEvolveScorer(self.domain, Path(ctx.worktree),
                                  optimizer_dir(ctx) / f"{ctx.directive.arm}-{self.name}-scratch",
                                  experiment_id=ctx.experiment_id, variant=ctx.directive.arm, k=self.k)

    def _lm(self, ctx: ArmContext) -> Any:
        if self.lm is not None:
            return self.lm
        from ci_lab.optim.lm import make_lm

        return make_lm(ctx.profile, "optimizer")

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        with optimizer_span(self.name, ctx):
            self.last = []
            if ctx.directive.edit_budget < 1:
                return []
            targets = self._targets(ctx)
            if not targets:
                obs.annotate({"ci.optimizer.note": "no text targets in focus"})
                return []
            scorer = self._scorer(ctx)
            evolve = evolve_cases_for(ctx, explicit=self.evolve_cases, domain=self.domain, scorer=scorer)
            opts = await self._optimize(ctx, targets, scorer, evolve, self._lm(ctx))
            self.last = opts
            edits = self._commit(ctx, targets, opts)
            self._report(ctx, opts, edits)
            return check_edit_budget(edits, ctx)

    def _commit(self, ctx: ArmContext, targets: list[tuple[str, TextTarget]],
                opts: list[TextOptimization]) -> list[Edit]:
        edits: list[Edit] = []
        for comp, t in targets:
            opt = next((o for o in opts if t.id in o.changed), None)
            if opt is None or len(edits) >= ctx.directive.edit_budget:
                continue
            t.write(ctx.worktree, opt.changed[t.id])
            hyp = self._hypothesis(t, opt)
            message = optimizer_commit_message(ctx.directive.arm, self.name, comp, hyp)
            sha = self.committer(Path(ctx.worktree), [t.path], message)
            edits.append(Edit(component=comp, hypothesis=hyp, files=(t.path,), commit=sha))
        for opt in opts:
            if opt.changed:
                self._artifacts(ctx, opt)
        return edits

    def _report(self, ctx: ArmContext, opts: list[TextOptimization], edits: list[Edit]) -> None:
        total: dict[str, int | float] = {}
        cost: dict[str, int | float] = {}
        for o in opts:
            for k, v in o.cost.span_attrs().items():
                total[k] = total.get(k, 0) + v
            for k, v in o.cost.as_dict().items():
                if isinstance(v, (int, float)):
                    cost[k] = round(cost.get(k, 0) + v, 3)
        obs.annotate({**total, "ci.edits": len(edits)})
        write_report(ctx, self.name, {
            "cost": cost,
            "runs": [{"targets": sorted(o.seed), "changed": sorted(o.changed), "note": o.note,
                      "seed_score": o.seed_score, "best_score": o.best_score, "cost": o.cost.as_dict(),
                      "diagnostics": o.diagnostics} for o in opts],
            "edits": [{"component": e.component, "files": list(e.files), "commit": e.commit} for e in edits],
            "acceptance": "diagnostic-only",
        })
