"""Re-adjudication: re-run Algorithm 2 on a round's *stored* inputs with the *stored* delta.

No new evaluation is spent. Because bootstrap seeds are derived from (hp.seed, round,
arm), re-running with unchanged hyper-parameters reproduces the stored decision exactly;
changing acceptance weights (beta, w) via ``hp`` re-decides the round. delta is
immutable after calibration and is always taken from the stored inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .attribution import attribute
from .history import HistoryRecord
from .params import Hyperparams
from .selection import DomainGuard, SelectionDecision, SelectionInputs, select


def round_record(inputs: SelectionInputs, decision: SelectionDecision) -> dict[str, Any]:
    if decision.round != inputs.round:
        raise ValueError("decision and inputs are for different rounds")
    return {"schema": "ci_lab.rrsi.round/1", "inputs": inputs.to_dict(), "decision": decision.to_dict()}


def save_round(path: str | Path, inputs: SelectionInputs, decision: SelectionDecision) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(round_record(inputs, decision), indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(p)


def load_round(path: str | Path) -> tuple[SelectionInputs, dict[str, Any]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return SelectionInputs.from_dict(doc["inputs"]), doc["decision"]


@dataclass(frozen=True)
class Readjudication:
    decision: SelectionDecision
    records: tuple[HistoryRecord, ...]
    changed: bool                      # differs from the stored decision (None stored -> False)


def readjudicate(source: SelectionInputs | Mapping[str, Any] | str | Path, *, hp: Hyperparams | None = None,
                 guard: DomainGuard | None = None) -> Readjudication:
    """Re-run selection. ``source`` is a ``SelectionInputs``, a stored round record (dict)
    or a path to one. ``hp`` overrides acceptance knobs; delta always comes from storage."""
    stored: Mapping[str, Any] | None = None
    if isinstance(source, SelectionInputs):
        inputs = source
    elif isinstance(source, Mapping):
        inputs, stored = SelectionInputs.from_dict(source["inputs"]), source.get("decision")
    else:
        inputs, stored = load_round(source)
    if hp is not None:
        inputs = replace(inputs, hp=hp)
    decision = select(inputs, guard=guard)
    changed = stored is not None and decision.to_dict() != dict(stored)
    return Readjudication(decision=decision, records=tuple(attribute(decision, inputs.arms)), changed=changed)


def verify(source: Mapping[str, Any] | str | Path, *, guard: DomainGuard | None = None) -> bool:
    """True iff re-running the stored inputs reproduces the stored decision exactly."""
    return not readjudicate(source, guard=guard).changed
