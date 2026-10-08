"""Global held-out look ledger (design §6, C15): ``experiments/holdout-looks.jsonl``.

Every evaluation on a held-out dataset is a "look", counted across campaigns by the
dataset's content hash. The number of planned looks is fixed by the first record for a
dataset (it cannot be raised post hoc); a look beyond the plan is refused. Recording is
idempotent per ``look_id`` (default: dataset hash + experiment id), so a resumed
confirmation run does not consume a second look.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from ci_lab.contracts import op_id
from ci_lab.ledger.atomic import append_jsonl, read_jsonl
from ci_lab.ledger.lock import DEFAULT_TIMEOUT, lock_for

HASH_RE = re.compile(r"^[A-Za-z0-9:_-]{8,128}$")


class LookBudgetExceeded(RuntimeError):
    pass


def _check_hash(dataset_hash: str) -> str:
    if not isinstance(dataset_hash, str) or not HASH_RE.match(dataset_hash):
        raise ValueError(f"bad dataset hash {dataset_hash!r}")
    return dataset_hash


def looks(path: str | os.PathLike[str], dataset_hash: str | None = None) -> list[dict[str, Any]]:
    recs = [r for r in read_jsonl(path) if isinstance(r.get("dataset_hash"), str)]
    return recs if dataset_hash is None else [r for r in recs if r["dataset_hash"] == dataset_hash]


def count_looks(path: str | os.PathLike[str], dataset_hash: str) -> int:
    return len(looks(path, _check_hash(dataset_hash)))


def planned_looks(path: str | os.PathLike[str], dataset_hash: str) -> int | None:
    recs = looks(path, _check_hash(dataset_hash))
    return int(recs[0]["planned_looks"]) if recs else None


def record_look(path: str | os.PathLike[str], dataset_hash: str, *, experiment_id: str, planned: int = 1,
                campaign_id: str | None = None, split: str = "heldout", look_id: str | None = None,
                timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Record one look; returns the stored record. Raises :class:`LookBudgetExceeded` when
    the dataset already had ``planned`` looks, ``ValueError`` if ``planned`` disagrees with
    the plan recorded by the first look."""
    _check_hash(dataset_hash)
    if planned < 1:
        raise ValueError("planned looks must be >= 1")
    look_id = look_id or op_id("holdout-look", dataset_hash, experiment_id)
    with lock_for(path, timeout=timeout):
        prior = looks(path, dataset_hash)
        for rec in prior:
            if rec.get("look_id") == look_id:
                return rec
        if prior:
            plan = int(prior[0]["planned_looks"])
            if plan != planned:
                raise ValueError(f"dataset {dataset_hash} was pre-registered with {plan} look(s), not {planned}")
            if len(prior) >= plan:
                raise LookBudgetExceeded(
                    f"held-out dataset {dataset_hash} already used {len(prior)}/{plan} planned look(s)")
        rec = {"dataset_hash": dataset_hash, "look_id": look_id, "look_no": len(prior) + 1,
               "planned_looks": planned, "experiment_id": experiment_id, "campaign_id": campaign_id,
               "split": split, "ts": time.time()}
        append_jsonl(Path(path), rec)
        return rec
