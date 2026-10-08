"""Edit history L_t (paper Eq. 10) and JSONL helpers.

One record per *measured* arm per round: ``(round, arm, edits, S', C', dS, dC, novelty, a)``.
Every edit of an arm shares the arm's measured deltas (the arm is the unit of evaluation);
``accepted`` (paper ``a``) is 1 only for the winning arm's edits (Alg. 2 line 18).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ci_lab.contracts import Edit

from .codec import edit_from_dict, edit_to_dict, finite


@dataclass(frozen=True)
class HistoryRecord:
    round: int
    arm: str
    edits: tuple[Edit, ...]
    score: float | None          # S' of the arm (missing = 0)
    cost: float | None           # C' of the arm (mean tokens per completed trial)
    delta_s: float | None        # S' - S_t vs contemporaneous incumbent
    delta_c: float | None        # (C' - C_t) / C_t
    accepted: bool               # paper a_i
    novelty: int = 0
    admissible: bool = False
    reasons: tuple[str, ...] = field(default=())

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.component for e in self.edits))

    @property
    def measured(self) -> bool:
        return self.delta_s is not None

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "arm": self.arm, "edits": [edit_to_dict(e) for e in self.edits],
                "score": finite(self.score), "cost": finite(self.cost), "delta_s": finite(self.delta_s),
                "delta_c": finite(self.delta_c), "accepted": self.accepted, "novelty": self.novelty,
                "admissible": self.admissible, "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HistoryRecord:
        return cls(round=int(d["round"]), arm=d["arm"], edits=tuple(edit_from_dict(e) for e in d.get("edits", ())),
                   score=d.get("score"), cost=d.get("cost"), delta_s=d.get("delta_s"), delta_c=d.get("delta_c"),
                   accepted=bool(d.get("accepted", False)), novelty=int(d.get("novelty", 0)),
                   admissible=bool(d.get("admissible", False)), reasons=tuple(d.get("reasons", ())))


@dataclass(frozen=True)
class EditEvent:
    """Flattened per-edit view ``(t_i, l_i, h_i, dS_i, dC_i, a_i)``."""

    round: int
    arm: str
    component: str
    hypothesis: str
    delta_s: float | None
    delta_c: float | None
    accepted: bool


def edit_events(records: Iterable[HistoryRecord]) -> Iterator[EditEvent]:
    for r in records:
        for e in r.edits:
            yield EditEvent(r.round, r.arm, e.component, e.hypothesis, r.delta_s, r.delta_c, r.accepted)


def tried_components(records: Iterable[HistoryRecord]) -> set[str]:
    """T_t: components of every *measured* edit."""
    return {ev.component for ev in edit_events(records) if ev.delta_s is not None}


def accepted_counts(records: Iterable[HistoryRecord]) -> dict[str, int]:
    """N_t(l): number of previously accepted edits tagged l."""
    out: dict[str, int] = {}
    for ev in edit_events(records):
        if ev.accepted:
            out[ev.component] = out.get(ev.component, 0) + 1
    return out


def has(records: Iterable[HistoryRecord], round_no: int, arm: str) -> bool:
    return any(r.round == round_no and r.arm == arm for r in records)


def before(records: Iterable[HistoryRecord], round_no: int) -> list[HistoryRecord]:
    return [r for r in records if r.round < round_no]


def replace_round(records: Sequence[HistoryRecord], round_no: int,
                  new: Iterable[HistoryRecord]) -> list[HistoryRecord]:
    """Return history with round ``round_no`` replaced (used by readjudicate)."""
    new = list(new)
    if any(r.round != round_no for r in new):
        raise ValueError("replacement records must all belong to the replaced round")
    return [r for r in records if r.round != round_no] + new


# ---------------------------------------------------------------- JSONL (the only I/O in rrsi)

def read_jsonl(path: str | Path) -> list[HistoryRecord]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(HistoryRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{p}:{n}: bad history record: {exc}") from exc
    return out


def append_jsonl(path: str | Path, records: Iterable[HistoryRecord]) -> int:
    """Append records (one JSON object per line); refuses duplicate (round, arm) keys."""
    p = Path(path)
    existing = {(r.round, r.arm) for r in read_jsonl(p)}
    lines = []
    for r in records:
        if (r.round, r.arm) in existing:
            raise ValueError(f"history already has round {r.round} arm {r.arm!r}")
        existing.add((r.round, r.arm))
        lines.append(json.dumps(r.to_dict(), sort_keys=True, allow_nan=False))
    if lines:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(line + "\n" for line in lines))
    return len(lines)


def write_jsonl(path: str | Path, records: Iterable[HistoryRecord]) -> None:
    """Rewrite the whole history (readjudicate); written via temp file + replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r.to_dict(), sort_keys=True, allow_nan=False) + "\n" for r in records),
                   encoding="utf-8", newline="\n")
    tmp.replace(p)
