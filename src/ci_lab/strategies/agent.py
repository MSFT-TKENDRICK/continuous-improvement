"""``agent`` arm strategy: the MAF meta-agent proposer (built by M8a) behind the common
strategy surface (span, edit budget)."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from ci_lab import obs
from ci_lab.contracts import ArmContext, Edit
from ci_lab.strategies.base import check_edit_budget, optimizer_span

Proposer = Callable[[ArmContext], Awaitable[list[Edit]]]


class AgentStrategy:
    name = "agent"

    def __init__(self, proposer: Proposer) -> None:
        if not callable(proposer):
            raise TypeError("AgentStrategy needs a proposer callable")
        self.proposer = proposer

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        with optimizer_span(self.name, ctx):
            if ctx.directive.edit_budget < 1:
                return []
            edits = check_edit_budget(await self.proposer(ctx), ctx)
            obs.annotate({"ci.edits": len(edits)})
            return edits
