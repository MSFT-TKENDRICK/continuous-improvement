"""``skillopt`` arm strategy: SkillOpt-Sleep gated consolidation of skill markdown
(and its sibling ``memory.md`` when ``memory`` is in focus) on the evolve split."""
from __future__ import annotations

import shutil
from pathlib import PurePosixPath
from typing import Any

from ci_lab.contracts import ArmContext
from ci_lab.optim.gepa import TextOptimization
from ci_lab.optim.scoring import EvolveScorer
from ci_lab.optim.skillopt import SkillOptConfig, optimize_skill
from ci_lab.optim.targets import TextTarget
from ci_lab.strategies.base import optimizer_dir
from ci_lab.strategies.text import TextOptimizerStrategy


class SkillOptStrategy(TextOptimizerStrategy):
    name = "skillopt"
    default_focus = ("skill",)

    def __init__(self, *, config: SkillOptConfig | None = None, **kw: Any) -> None:
        super().__init__(config=config or SkillOptConfig(), **kw)

    def _plan(self, ctx: ArmContext, targets: list[tuple[str, TextTarget]]
              ) -> list[tuple[TextTarget, TextTarget | None]]:
        skills = [t for c, t in targets if c == "skill" and t.key is None]
        memories = {PurePosixPath(t.path).parent: t for c, t in targets if c == "memory" and t.key is None}
        plan, left = [], ctx.directive.edit_budget
        for s in skills:
            if left < 1:
                break
            mem = memories.get(PurePosixPath(s.path).parent) if left >= 2 else None
            plan.append((s, mem))
            left -= 1 + (mem is not None)
        return plan

    async def _optimize(self, ctx: ArmContext, targets: list[tuple[str, TextTarget]], scorer: EvolveScorer,
                        evolve: list[str], lm: Any) -> list[TextOptimization]:
        plan = self._plan(ctx, targets)
        share = None if ctx.budget_tokens is None or not plan else ctx.budget_tokens // len(plan)
        out = []
        for skill, mem in plan:
            out.append(await optimize_skill(
                (skill.id, skill.read(ctx.worktree)), scorer, evolve, reflection_lm=lm,
                memory=(mem.id, mem.read(ctx.worktree)) if mem else None,
                edit_budget=ctx.directive.edit_budget, budget_tokens=share, config=self.config,
                incumbent_failures=ctx.failures))
        return out

    def _hypothesis(self, target: TextTarget, opt: TextOptimization) -> str:
        d = opt.diagnostics
        return (f"skillopt: {d.get('applied_edits', 0)} gated rule edit(s) to {target.id} from evolve "
                f"failures (diagnostic gate {d.get('gate_action', '?')}: "
                f"{(opt.seed_score or 0):.3f}->{(opt.best_score or 0):.3f})")

    def _artifacts(self, ctx: ArmContext, opt: TextOptimization) -> None:
        """Keep SkillOpt's conventional ``best_skill.md`` / ``best_memory.md`` next to the report."""
        skill_path = PurePosixPath(TextTarget.parse(next(iter(opt.seed))).path)
        d = optimizer_dir(ctx) / f"{ctx.directive.arm}-{self.name}" / skill_path.parent.name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for tid, text in opt.best.items():
            name = "best_memory.md" if PurePosixPath(TextTarget.parse(tid).path).name == "memory.md" \
                else "best_skill.md"
            (d / name).write_text(text, encoding="utf-8", newline="\n")
