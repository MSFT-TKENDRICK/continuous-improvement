"""HOOK(M16): mine lesson candidates from one sleep night's rollouts (design §13.3, B3/B8, C10).

The night's judged rollouts (evolve split, reviewed tasks) become :class:`ci_lab.rulespec.Trajectory`
records through the ``ci_lab.lessons`` harvest adapters, sliced by night date. They are kept in a
**local** store, deduplicated by id and accumulated across nights, together with any optional local
``source:path`` inputs such as usage traces. Usage data is always reduced to typed features (B3) and
stays untrusted until a human labels it. Then ``ci_lab.lessons`` mines and routes the store.

Only a sanitized, typed summary leaves the hook. It holds cluster ids, rung routes, counts, rule and
rubric ids, tool-name n-grams and typed feature slots. It never holds excerpts, tool arguments, raw
messages, member ids or hashed values (B8). The night writes that summary into the draft-PR bundle
as ``experiments/sleep/lessons/<night_id>.json``. Candidates are proposals only: nothing is adopted
or enforced automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ci_lab.lessons.cluster import MineConfig
from ci_lab.lessons.common import (
    SealedSplitError,
    clean_ident,
    clean_rule_ids,
    save_trajectories,
)
from ci_lab.lessons.fingerprint import END, START
from ci_lab.lessons.harvest import (
    SOURCES,
    HarvestOptions,
    family_of,
    make_trajectory,
    steps_from_transcript,
)
from ci_lab.lessons.pipeline import run_harvest, run_mine
from ci_lab.rulespec import Trajectory
from ci_lab.sleep.backend import RolloutView

log = logging.getLogger(__name__)

LESSONS_FORMAT = "ci_lab.sleep.lessons.v1"
LESSONS_REL = "experiments/sleep/lessons"
SLEEP_PIN_PREFIX = "sleep-"
MAX_CANDIDATES = 50
MAX_LIST = 16
LOCAL_SOURCES = tuple(s for s in SOURCES if s != "calibrate")


@dataclass(frozen=True)
class LessonsRequest:
    target: str
    night_id: str
    date: str  # yyyymmdd: the convergence slice
    profile: str
    rollouts: tuple[RolloutView, ...]


@dataclass
class LessonsReport:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int | str] = field(default_factory=dict)


LessonsHook = Callable[[LessonsRequest], LessonsReport]


# ----------------------------------------------------------------- rollouts -> trajectories

def _rubric_ids(rule_ids: Iterable[str]) -> list[str]:
    """``check.<op>:<arg>`` judge failures -> ``check.<op>`` (the arg is task text, never kept)."""
    return [r.split(":", 1)[0] for r in rule_ids if r.startswith("check.")]


def trajectories_from_rollouts(req: LessonsRequest) -> list[Trajectory]:
    trial: dict[str, int] = {}
    out: list[Trajectory] = []
    for ro in req.rollouts:
        n = trial[ro.case_id] = trial.get(ro.case_id, -1) + 1
        steps = steps_from_transcript(getattr(ro.transcript, "messages", ()) or (),
                                      getattr(ro.transcript, "tool_calls", ()) or ())
        out.append(make_trajectory(
            source="assert", split="evolve", case_id=ro.case_id, steps=steps, family=family_of(ro.case_id),
            slice=req.date, suite=ro.suite, pin=SLEEP_PIN_PREFIX + req.profile, trial=n, passed=ro.passed,
            oracle_rules=[getattr(v, "rule_id", "") for v in ro.violations],
            rubric_fails=_rubric_ids(ro.rule_ids), labels=(ro.suite,), extra_id=req.night_id))
    return out


# ----------------------------------------------------------------- sanitize (B8)

def _scalar(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    return clean_ident(v)


def _typed(value: Any) -> Any:
    """Typed slots only. Returns ``None`` when the value is not a bool, number or clean identifier."""
    if isinstance(value, (list, tuple)):
        items = [_scalar(v) for v in value[:MAX_LIST]]
        return items if all(i is not None for i in items) else None
    return _scalar(value)


def _features(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = clean_ident(k)
        typed = _typed(v)
        if key and typed is not None:
            out[key] = typed
    return out


def _ident_or(value: Any, default: str) -> str:
    return clean_ident(value) or default


def _ngram_token(value: Any) -> str:
    return value if value in (START, END) else _ident_or(value, "_")


def sanitize_candidate(line: Mapping[str, Any]) -> dict[str, Any]:
    c = line.get("cluster") or {}
    fp = c.get("fingerprint") or {}
    ngrams = [[_ngram_token(tok) for tok in g][:8] for g in (fp.get("tool_ngrams") or [])[:MAX_LIST]
              if isinstance(g, (list, tuple))]
    return {
        "id": _ident_or(c.get("id"), "cluster"),
        "route": _ident_or(c.get("route"), "R6"),
        "status": _ident_or(c.get("status"), "candidate"),
        "human_confirmed": bool(c.get("human_confirmed")),
        "trusted": bool(line.get("trusted")),
        "forced_structural": bool(line.get("forced_structural")),
        "support": len(c.get("members") or ()),
        "families": len(c.get("families") or ()),
        "slices": len(c.get("slices") or ()),
        "holdout": len(line.get("holdout_members") or ()),
        "fingerprint": {
            "pin": _ident_or(fp.get("pin"), "unpinned"),
            "oracle_rules": list(clean_rule_ids(fp.get("oracle_rules") or ()))[:MAX_LIST],
            "rubric_ids": list(clean_rule_ids(fp.get("rubric_ids") or ()))[:MAX_LIST],
            "tool_ngrams": ngrams,
            "error_class": clean_ident(fp.get("error_class")),
        },
        "features": _features(line.get("features")),
    }


# ----------------------------------------------------------------- the hook

def parse_source(spec: str) -> tuple[str, Path]:
    """``source:path`` (e.g. ``usage:artifacts/usage-traces``) for local extra harvest inputs."""
    source, sep, path = spec.partition(":")
    if not sep or source not in LOCAL_SOURCES or not path:
        raise ValueError(f"lessons source must be one of {LOCAL_SOURCES} as 'source:path', got {spec!r}")
    return source, Path(path)


def make_lessons_hook(store_dir: Path, sources: Sequence[tuple[str, Path]] = (), *,
                      config: MineConfig | None = None) -> LessonsHook:
    """Build ``SleepDeps.lessons``. ``store_dir`` is local and should be gitignored (e.g. under
    ``artifacts/``). Each target gets ``<store>/<target>/trajectories.jsonl`` plus a mine run dir."""
    store_root = Path(store_dir)

    def hook(req: LessonsRequest) -> LessonsReport:
        store = store_root / req.target
        counts: dict[str, int | str] = {}
        night_trajs = trajectories_from_rollouts(req)
        counts["trajectories_night"] = len(night_trajs)
        counts["failures_night"] = sum(t.outcome.passed is False for t in night_trajs)
        counts["trajectories_total"] = save_trajectories(store, night_trajs)
        extra = 0
        for source, path in sources:
            if not path.exists():
                counts["sources_missing"] = int(counts.get("sources_missing", 0)) + 1
                continue
            try:
                trajs, stats = run_harvest(source, path, store, HarvestOptions(slice=req.date))
            except (SealedSplitError, ValueError, OSError) as exc:
                log.warning("lessons hook: source %s skipped (%s)", source, type(exc).__name__)
                counts["sources_failed"] = int(counts.get("sources_failed", 0)) + 1
                continue
            extra += len(trajs)
            counts["sealed_refused"] = int(counts.get("sealed_refused", 0)) + int(stats.sealed)
        if sources:
            counts["trajectories_sources"] = extra
        lines = run_mine(store, store / "mine", config=config)
        candidates = [sanitize_candidate(x) for x in lines[:MAX_CANDIDATES]]
        counts["clusters"] = len(lines)
        counts["candidates"] = len(candidates)
        return LessonsReport(candidates=candidates, counts=counts)

    return hook


def proposal_document(night_id: str, reports: Mapping[str, LessonsReport]) -> dict[str, Any] | None:
    """Bundle file content, or ``None`` when no target produced candidates."""
    targets = {name: {"counts": dict(r.counts), "candidates": list(r.candidates)}
               for name, r in sorted(reports.items()) if r.candidates}
    if not targets:
        return None
    return {"format": LESSONS_FORMAT, "night_id": night_id, "status": "proposed",
            "note": ("Lesson candidates mined from this night's evolve-split rollouts. Proposals only: "
                     "they are never adopted or enforced automatically. Confirm with `ci-lab lessons "
                     "confirm`, then synthesize and validate rules in a separate reviewed PR."),
            "targets": targets}


__all__ = [
    "LESSONS_FORMAT", "LESSONS_REL", "LOCAL_SOURCES", "LessonsHook", "LessonsReport", "LessonsRequest",
    "RolloutView", "make_lessons_hook", "parse_source", "proposal_document", "sanitize_candidate",
    "trajectories_from_rollouts",
]
