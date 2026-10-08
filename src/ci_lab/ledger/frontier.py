"""Campaign frontier (incumbent pointer) with compare-and-swap updates (design §3)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.contracts import ATTR_EXPERIMENT, ATTR_PHASE, ATTR_ROUND, ATTR_SCORE, SPAN_STEP
from ci_lab.ledger.atomic import atomic_write_json, read_json
from ci_lab.ledger.lock import DEFAULT_TIMEOUT, FileLock, lock_for

_KNOWN = ("incumbent", "harness_tree", "score", "round")


class FrontierConflict(RuntimeError):
    def __init__(self, expected: str | None, actual: Frontier | None) -> None:
        self.expected, self.actual = expected, actual
        found = actual.incumbent if actual else "<absent>"
        super().__init__(f"frontier moved: expected incumbent {expected or '<absent>'}, found {found}")


@dataclass(frozen=True)
class Frontier:
    incumbent: str          # incumbent commit (top of the campaign stack)
    harness_tree: str       # harness tree hash = incumbent identity
    score: float            # S*
    round: int              # round that produced the incumbent (0 = H0)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {**dict(self.extra), "incumbent": self.incumbent, "harness_tree": self.harness_tree,
                "score": self.score, "round": self.round}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Frontier:
        missing = [k for k in _KNOWN if k not in data]
        if missing:
            raise ValueError(f"frontier missing keys {missing}")
        return cls(incumbent=str(data["incumbent"]), harness_tree=str(data["harness_tree"]),
                   score=float(data["score"]), round=int(data["round"]),
                   extra={k: v for k, v in data.items() if k not in _KNOWN})


def read_frontier(path: str | os.PathLike[str]) -> Frontier | None:
    data = read_json(path, default=None)
    return None if data is None else Frontier.from_json(data)


def cas_frontier(path: str | os.PathLike[str], expected_incumbent: str | None,
                 new: Frontier | Mapping[str, Any], *, lock: FileLock | None = None,
                 timeout: float = DEFAULT_TIMEOUT, experiment_id: str | None = None) -> Frontier:
    """Write ``new`` iff the current incumbent equals ``expected_incumbent`` (``None`` = the
    frontier must not exist yet). Raises :class:`FrontierConflict` otherwise. Idempotent:
    re-applying the same swap after it already happened returns the stored frontier.
    Emits a ``ci.step{ci.phase=record}`` span (``oes.experiment_id`` when given)."""
    target = new if isinstance(new, Frontier) else Frontier.from_json(new)
    with obs.span(SPAN_STEP, {ATTR_PHASE: "record", ATTR_EXPERIMENT: experiment_id, ATTR_ROUND: target.round,
                              ATTR_SCORE: target.score}), lock or lock_for(path, timeout=timeout):
        current = read_frontier(path)
        if current is not None and current.to_json() == target.to_json():
            return current
        actual = current.incumbent if current else None
        if actual != expected_incumbent:
            raise FrontierConflict(expected_incumbent, current)
        atomic_write_json(Path(path), target.to_json())
        return target
