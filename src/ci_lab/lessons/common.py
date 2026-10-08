"""Shared helpers: split policy (B2/C15), digests, JSONL I/O for trajectories and clusters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from ci_lab.rulespec import LessonCluster, Trajectory, canonical_json

log = logging.getLogger("ci_lab.lessons")

ALLOWED_SPLITS = ("evolve", "usage")
SEALED_SPLITS = frozenset({"heldout", "ood", "aa", "confirm", "sealed", "test", "holdout"})
INJECTION_SUITES = ("indirect_prompt_injection",)
TRAJECTORIES_FILE = "trajectories.jsonl"
HARVEST_REPORT = "harvest_report.json"
LESSONS_DIR = "lessons"
CANDIDATES_FILE = "candidates.jsonl"
ROUTES_FILE = "routes.jsonl"
CONFIRMATIONS_FILE = "confirmations.jsonl"
SALT_ENV = "CI_LESSONS_SALT"

_RULE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


class SealedSplitError(ValueError):
    """A heldout/ood/aa/confirm (or unknown) split reached ``lessons`` (B2, C15)."""


def check_split(split: Any, *, source: str, default: str | None = None) -> Literal["evolve", "usage"]:
    """Return the harvestable split or raise :class:`SealedSplitError` (fail closed on unknown)."""
    raw = split if split not in (None, "") else default
    s = str(raw or "").strip().lower()
    if source == "usage" and s in ("", "usage"):
        return "usage"
    if s == "evolve":
        return "evolve"
    if s in SEALED_SPLITS:
        raise SealedSplitError(f"refusing sealed split {s!r} from {source} (B2/C15)")
    raise SealedSplitError(f"refusing unknown split {s or '<missing>'!r} from {source}; "
                           "pass an explicit evolve split")


def digest(text: str, n: int = 16) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def stable_id(prefix: str, *parts: Any) -> str:
    return prefix + hashlib.sha256(canonical_json([str(p) for p in parts]).encode()).hexdigest()[:20]


def salt() -> bytes:
    return (os.environ.get(SALT_ENV) or "ci-lab-lessons-v1").encode()


def keyed_hash(value: str, n: int = 12) -> str:
    return hashlib.blake2b(value.encode("utf-8", "surrogatepass"), key=salt()[:64], digest_size=16).hexdigest()[:n]


def clean_rule_ids(ids: Iterable[Any]) -> tuple[str, ...]:
    """Rule/rubric ids from data files: only well-formed identifiers survive (no free text)."""
    return tuple(sorted({str(i).strip() for i in ids if isinstance(i, str) and _RULE_ID_RE.match(i.strip())}))


def clean_ident(value: Any) -> str | None:
    if isinstance(value, str) and _IDENT_RE.match(value.strip()):
        return value.strip()
    return None


def injection_suspect(oracle_rules: Iterable[str], *labels: Any) -> bool:
    if any(r.startswith("injection") for r in oracle_rules):
        return True
    return any(isinstance(lab, str) and "injection" in lab.lower() for lab in labels)


# ---------------------------------------------------------------- JSONL io

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                log.warning("skipping unparsable line %s:%d", path.name, n)
                continue
            if isinstance(rec, dict):
                yield rec


def write_jsonl_atomic(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(row + "\n" for row in rows)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_trajectories(path: Path) -> list[Trajectory]:
    """Trajectories from a file or a harvest dir (``trajectories.jsonl``). Re-checks splits."""
    p = path / TRAJECTORIES_FILE if path.is_dir() else path
    out = [Trajectory.model_validate(rec) for rec in read_jsonl(p)]
    for t in out:
        check_split(t.split, source=t.source if t.source == "usage" else "trajectory")
    return out


def save_trajectories(out_dir: Path, trajectories: Iterable[Trajectory], *, merge: bool = True) -> int:
    """Write ``<out_dir>/trajectories.jsonl`` (merged with existing, deduped by id, sorted)."""
    path = out_dir / TRAJECTORIES_FILE
    by_id: dict[str, Trajectory] = {}
    if merge and path.exists():
        by_id.update({t.id: t for t in load_trajectories(path)})
    by_id.update({t.id: t for t in trajectories})
    write_jsonl_atomic(path, (canonical_json(by_id[k].model_dump(mode="json")) for k in sorted(by_id)))
    return len(by_id)


def lessons_dir(run_dir: Path) -> Path:
    return run_dir / LESSONS_DIR


def load_clusters(path: Path) -> list[LessonCluster]:
    """Clusters from ``candidates.jsonl`` (plain LessonCluster lines or ``{"cluster": ..., ...}``)."""
    p = lessons_dir(path) / CANDIDATES_FILE if path.is_dir() else path
    if not p.exists():
        return []
    return [LessonCluster.model_validate(r.get("cluster", r)) for r in read_jsonl(p)]
