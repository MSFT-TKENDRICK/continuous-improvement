"""Holdout split by family + fingerprint clustering (design §13.3 step 2, N2, C15).

The holdout split is decided per *family* by a stable hash **before** clustering, so paraphrases /
near-duplicates of one case can never sit on both sides (mined vs. replay-recall). Clusters must reach
``min_support`` members from ≥ ``min_families`` families and ≥ ``min_slices`` distinct slices, with no slice
holding more than ``max_slice_share`` of the members and members in both the early and late half of the
corpus' slices (stability across time slices). Status is ``candidate`` when trusted support suffices,
else ``backlog`` (usage-only evidence waits for a human label, B3); ``confirmed`` / ``human_confirmed`` are
only ever set from human confirmations (``ci-lab lessons confirm``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ci_lab.lessons.common import CONFIRMATIONS_FILE, lessons_dir, read_jsonl, salt
from ci_lab.lessons.fingerprint import (
    DEFAULT_WINDOW,
    cluster_id,
    fingerprint,
    is_failure,
)
from ci_lab.rulespec import Fingerprint, LessonCluster, Trajectory

HOLDOUT_FRACTION = 0.5
MIN_SUPPORT = 3


def holdout_score(family: str) -> float:
    h = hashlib.sha256(salt() + b"|holdout|" + family.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def is_holdout(family: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    return holdout_score(family) < fraction


def split_by_family(trajs: Iterable[Trajectory], fraction: float = HOLDOUT_FRACTION
                    ) -> tuple[list[Trajectory], list[Trajectory]]:
    """(mine, holdout) — whole families on one side."""
    mine, hold = [], []
    for t in trajs:
        (hold if is_holdout(t.family, fraction) else mine).append(t)
    return mine, hold


@dataclass(frozen=True)
class MineConfig:
    min_support: int = MIN_SUPPORT
    min_slices: int = 2
    min_families: int = 2
    max_slice_share: float = 0.8
    holdout_fraction: float = HOLDOUT_FRACTION
    window: int = DEFAULT_WINDOW
    require_stability: bool = True


@dataclass
class MineResult:
    clusters: list[LessonCluster]
    holdout_members: dict[str, list[str]]          # cluster id -> holdout trajectory ids (same fingerprint)
    dropped: dict[str, list[str]] = field(default_factory=dict)  # fingerprint digest -> reasons
    n_failures: int = 0
    n_mined: int = 0
    n_holdout: int = 0
    members: dict[str, list[Trajectory]] = field(default_factory=dict)  # cluster id -> mined members
    fingerprints: dict[str, Fingerprint] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {"failures": self.n_failures, "mined": self.n_mined, "holdout": self.n_holdout,
                "clusters": len(self.clusters),
                "by_status": {s: sum(c.status == s for c in self.clusters)
                              for s in ("candidate", "confirmed", "rejected", "backlog")},
                "dropped": {k: v for k, v in sorted(self.dropped.items())}}


def _halves(slices: Sequence[str]) -> tuple[set[str], set[str]]:
    s = sorted(set(slices))
    k = math.ceil(len(s) / 2)
    return set(s[:k]), set(s[k:])


def load_confirmations(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Latest human decision per cluster id from ``<run>/lessons/confirmations.jsonl``."""
    p = lessons_dir(run_dir) / CONFIRMATIONS_FILE
    out: dict[str, dict[str, Any]] = {}
    if p.exists():
        for r in read_jsonl(p):
            if r.get("cluster_id") and r.get("decision") in ("confirmed", "rejected") and r.get("by"):
                out[str(r["cluster_id"])] = r
    return out


def append_confirmation(run_dir: Path, cluster_id: str, *, by: str, decision: str = "confirmed",
                        note: str = "") -> dict[str, Any]:
    if decision not in ("confirmed", "rejected"):
        raise ValueError("decision must be confirmed or rejected")
    if not by.strip():
        raise ValueError("a human reviewer name is required")
    rec = {"cluster_id": cluster_id, "decision": decision, "by": by.strip()[:80], "note": note[:200],
           "ts": datetime.now(UTC).isoformat(timespec="seconds")}
    p = lessons_dir(run_dir) / CONFIRMATIONS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def apply_confirmations(clusters: Sequence[LessonCluster], decisions: Mapping[str, Mapping[str, Any]]
                        ) -> list[LessonCluster]:
    out = []
    for c in clusters:
        d = decisions.get(c.id)
        if d is None:
            out.append(c)
        elif d["decision"] == "confirmed":
            out.append(c.model_copy(update={"status": "confirmed", "human_confirmed": True}))
        else:
            out.append(c.model_copy(update={"status": "rejected", "human_confirmed": False}))
    return out


def mine(trajectories: Iterable[Trajectory], config: MineConfig | None = None,
         decisions: Mapping[str, Mapping[str, Any]] | None = None) -> MineResult:
    """Split by family, then cluster the mined side's failures by fingerprint."""
    cfg = config or MineConfig()
    fails = [t for t in trajectories if t.split in ("evolve", "usage") and is_failure(t)]
    mined, hold = split_by_family(fails, cfg.holdout_fraction)
    early, late = _halves([t.slice for t in mined])
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    fps: dict[str, Fingerprint] = {}
    for t in mined:
        fp = fingerprint(t, window=cfg.window)
        cid = cluster_id(fp)
        fps[cid] = fp
        groups[cid].append(t)
    hold_by: dict[str, list[str]] = defaultdict(list)
    for t in hold:
        hold_by[cluster_id(fingerprint(t, window=cfg.window))].append(t.id)

    res = MineResult(clusters=[], holdout_members={}, n_failures=len(fails), n_mined=len(mined),
                     n_holdout=len(hold))
    for cid in sorted(groups):
        members = sorted(groups[cid], key=lambda t: t.id)
        slices = sorted({t.slice for t in members})
        families = sorted({t.family for t in members})
        reasons = []
        if len(members) < cfg.min_support:
            reasons.append(f"support {len(members)} < {cfg.min_support}")
        if len(families) < cfg.min_families:
            reasons.append(f"families {len(families)} < {cfg.min_families}")
        if len(slices) < cfg.min_slices:
            reasons.append(f"slices {len(slices)} < {cfg.min_slices}")
        share = max(sum(t.slice == s for t in members) for s in slices) / len(members)
        if len(slices) > 1 and share > cfg.max_slice_share:
            reasons.append(f"slice share {share:.2f} > {cfg.max_slice_share}")
        if cfg.require_stability and late and not (set(slices) & early and set(slices) & late):
            reasons.append("unstable: not present in both early and late slices")
        if reasons:
            res.dropped[cid] = reasons
            continue
        trusted = sum(t.trusted for t in members)
        status = "candidate" if trusted >= cfg.min_support else "backlog"
        res.clusters.append(LessonCluster(
            id=cid, fingerprint=fps[cid], members=tuple(t.id for t in members), families=tuple(families),
            slices=tuple(slices), route="R6", status=status, human_confirmed=False))
        res.holdout_members[cid] = sorted(hold_by.get(cid, []))
        res.members[cid] = members
        res.fingerprints[cid] = fps[cid]
    if decisions:
        res.clusters = apply_confirmations(res.clusters, decisions)
    return res
