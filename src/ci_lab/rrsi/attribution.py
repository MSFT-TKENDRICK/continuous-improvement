"""Edit attribution (Alg. 2 line 18), structural novelty (Eq. 15-16) and per-component
success statistics that feed the next round's schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ci_lab.contracts import COMPONENTS, ArmResult, Edit

from .history import HistoryRecord, accepted_counts, edit_events

if TYPE_CHECKING:
    from .selection import SelectionDecision

# Paper K_str = {client_tool, skill, memory, subagent}; subagent is not on our data-only surface.
STRUCTURAL_COMPONENTS: tuple[str, ...] = tuple(c for c in ("client_tool", "skill", "memory", "subagent")
                                               if c in COMPONENTS)


def novelty(edits: Iterable[Edit], accepted: Mapping[str, int] | Iterable[HistoryRecord]) -> int:
    """nu_t(H') = number of structural components touched by the candidate that have
    never been *accepted* before (tried-but-rejected still counts as novel)."""
    counts = accepted if isinstance(accepted, Mapping) else accepted_counts(accepted)
    comps = {e.component for e in edits}
    return sum(1 for c in STRUCTURAL_COMPONENTS if c in comps and counts.get(c, 0) == 0)


@dataclass(frozen=True)
class ComponentStats:
    component: str
    tried: int            # measured edits tagged with this component
    accepted: int         # of which accepted (a = 1)
    mean_delta_s: float   # mean arm dS over measured edits (0 if untried)
    best_delta_s: float   # max dS (-inf if untried)

    @property
    def success_rate(self) -> float:
        return self.accepted / self.tried if self.tried else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["best_delta_s"] = self.best_delta_s if math.isfinite(self.best_delta_s) else None
        d["success_rate"] = self.success_rate
        return d


def component_stats(records: Iterable[HistoryRecord], components: Sequence[str] = COMPONENTS) -> dict[str, ComponentStats]:
    acc: dict[str, list[tuple[float, bool]]] = {c: [] for c in components}
    for ev in edit_events(records):
        if ev.delta_s is not None:
            acc.setdefault(ev.component, []).append((float(ev.delta_s), ev.accepted))
    out = {}
    for c, rows in acc.items():
        ds = [d for d, _ in rows]
        out[c] = ComponentStats(component=c, tried=len(rows), accepted=sum(1 for _, a in rows if a),
                                mean_delta_s=sum(ds) / len(ds) if ds else 0.0,
                                best_delta_s=max(ds) if ds else -math.inf)
    return out


def attribute(decision: SelectionDecision, arms: Sequence[ArmResult]) -> list[HistoryRecord]:
    """History records for every *measured* arm of the round; ``accepted`` (a) is True only
    for the winner's edits. A ``rerun`` decision records nothing (the round is repeated)."""
    if decision.decision == "rerun":
        return []
    by_arm = {a.arm: a for a in arms}
    out = []
    for tr in decision.arms:
        if not tr.evaluated:
            continue
        arm = by_arm[tr.arm]
        out.append(HistoryRecord(round=decision.round, arm=tr.arm, edits=tuple(arm.edits), score=tr.score,
                                 cost=tr.cost, delta_s=tr.delta_s, delta_c=tr.delta_c,
                                 accepted=decision.winner == tr.arm, novelty=tr.novelty,
                                 admissible=tr.admissible, reasons=tuple(tr.reasons)))
    return out
