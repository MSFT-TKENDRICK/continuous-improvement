"""Global held-out look ledger (design §6, C15): ``experiments/holdout-looks.jsonl``.

Every evaluation on a held-out dataset is a "look", counted across campaigns by the
dataset's content hash. The number of planned looks is fixed by the first record for a
dataset (it cannot be raised post hoc); a look beyond the plan is refused. Recording is
idempotent per ``look_id`` (default: dataset hash + experiment id), so a resumed
confirmation run does not consume a second look.

Dataset hashes are compared in normalized form (:func:`normalize_hash`: ``sha256:`` prefix
stripped, lowercased), so records written before the OES ``sha256:<hex>`` form (bare ``<hex>``,
keyed ``"<campaign>|<hex>"`` without ``planned_looks``/``look_id``) still count against the
budget and still make a resumed confirmation idempotent.
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


def normalize_hash(dataset_hash: str) -> str:
    """Canonical comparison form: ``sha256:<hex>`` and bare ``<hex>`` are the same dataset."""
    h = dataset_hash.strip().lower()
    return h.removeprefix("sha256:")


def hash_forms(dataset_hash: str) -> tuple[str, str]:
    """``(sha256:<hex>, <hex>)`` spellings of one dataset hash (current and legacy ledger forms)."""
    bare = normalize_hash(dataset_hash)
    return f"sha256:{bare}", bare


def is_same_look(rec: dict[str, Any], dataset_hash: str, experiment_id: str, campaign_id: str | None = None,
                 look_id: str | None = None) -> bool:
    """Whether ``rec`` already is experiment ``experiment_id``'s look at ``dataset_hash``: same
    ``look_id`` (either hash spelling), same experiment id, or — for records without one — a legacy
    ``"<eid|campaign>|<hash>"`` key."""
    forms = hash_forms(dataset_hash)
    if look_id is not None:
        return rec.get("look_id") == look_id
    if rec.get("look_id") in {op_id("holdout-look", h, experiment_id) for h in forms}:
        return True
    if (owner := rec.get("experiment_id") or rec.get("eid")) is not None:
        return owner == experiment_id
    owners = [experiment_id] + ([campaign_id] if campaign_id else [])
    return rec.get("key") in {f"{o}|{h}" for o in owners for h in forms}


def looks(path: str | os.PathLike[str], dataset_hash: str | None = None) -> list[dict[str, Any]]:
    recs = [r for r in read_jsonl(path) if isinstance(r.get("dataset_hash"), str)]
    if dataset_hash is None:
        return recs
    want = normalize_hash(dataset_hash)
    return [r for r in recs if normalize_hash(r["dataset_hash"]) == want]


def _plan(recs: list[dict[str, Any]]) -> int | None:
    """Planned looks fixed by the first record that carries one (legacy records do not)."""
    return next((int(r["planned_looks"]) for r in recs if r.get("planned_looks") is not None), None)


def count_looks(path: str | os.PathLike[str], dataset_hash: str) -> int:
    return len(looks(path, _check_hash(dataset_hash)))


def planned_looks(path: str | os.PathLike[str], dataset_hash: str) -> int | None:
    return _plan(looks(path, _check_hash(dataset_hash)))


def record_look(path: str | os.PathLike[str], dataset_hash: str, *, experiment_id: str, planned: int = 1,
                campaign_id: str | None = None, split: str = "heldout", look_id: str | None = None,
                timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Record one look; returns the stored record. Raises :class:`LookBudgetExceeded` when
    the dataset already had ``planned`` looks, ``ValueError`` if ``planned`` disagrees with
    the plan recorded by the first look."""
    _check_hash(dataset_hash)
    if planned < 1:
        raise ValueError("planned looks must be >= 1")
    custom_id = look_id
    look_id = look_id or op_id("holdout-look", dataset_hash, experiment_id)
    with lock_for(path, timeout=timeout):
        prior = looks(path, dataset_hash)
        for idx, rec in enumerate(prior):
            if is_same_look(rec, dataset_hash, experiment_id, campaign_id, custom_id):
                return {**rec, "look_no": rec.get("look_no", idx + 1)}
        if prior:
            plan = _plan(prior)
            if plan is None:
                plan = planned
            elif plan != planned:
                raise ValueError(f"dataset {dataset_hash} was pre-registered with {plan} look(s), not {planned}")
            if len(prior) >= plan:
                raise LookBudgetExceeded(
                    f"held-out dataset {dataset_hash} already used {len(prior)}/{plan} planned look(s)")
        rec = {"dataset_hash": dataset_hash, "look_id": look_id, "look_no": len(prior) + 1,
               "planned_looks": planned, "experiment_id": experiment_id, "campaign_id": campaign_id,
               "split": split, "ts": time.time()}
        append_jsonl(Path(path), rec)
        return rec
