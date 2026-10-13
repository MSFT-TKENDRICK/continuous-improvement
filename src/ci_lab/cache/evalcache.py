"""Content-addressed eval result cache (design §4, C8).

Key: ``<root>/<pin_hash>/<harness_tree>/<split>/<case>/<trial>.json``. A cached score is
only valid when the evaluator pin, the harness tree and the case/trial all match, so a
changed judge, evaluator or harness can never be served stale results.

Disabled for the Copilot profile (C8): served models behind Copilot can change silently,
so cross-run reuse would mix judge versions. Missing trials (``score is None``) are never
cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ci_lab.contracts import EvaluatorPin, Profile, TaskScore, Violation
from ci_lab.gitops.safe_path import UnsafePathError, safe_join
from ci_lab.ledger.atomic import atomic_write_json, read_json

SCHEMA = 1
_SAFE_SEG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SPLITS = {"evolve", "heldout", "ood", "aa"}


def pin_hash(pin: EvaluatorPin) -> str:
    """Hash of the requested evaluator identity. ``served_judge_models`` is observed after the
    run, so it is stored with each entry and checked on read rather than keyed on."""
    payload = json.dumps({"evaluator_tree": pin.evaluator_tree, "judge_model": pin.judge_model,
                          "judge_provider": pin.judge_provider}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _segment(value: str) -> str:
    """Filesystem-safe, collision-free segment: readable ids pass through, others are hashed."""
    s = str(value)
    if _SAFE_SEG.match(s) and not s.endswith(".") and s.upper().split(".")[0] not in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        return s
    return "h-" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:40]


def default_root() -> Path:
    env = os.environ.get("CI_EVAL_CACHE")
    if env:
        return Path(env)
    from ci_lab.cache.env import cache_root

    return cache_root() / "eval"


def _score_json(score: TaskScore) -> dict[str, Any]:
    return {"case_id": score.case_id, "trial": score.trial, "suite": score.suite, "score": score.score,
            "violations": [{"rule_id": v.rule_id, "severity": v.severity, "detail": v.detail}
                           for v in score.violations],
            "tokens_in": score.tokens_in, "tokens_out": score.tokens_out, "served_model": score.served_model,
            "wall_ms": score.wall_ms, "llm_calls": score.llm_calls, "tool_calls": score.tool_calls,
            "subscores": dict(score.subscores)}


def _score_from(d: dict[str, Any]) -> TaskScore:
    return TaskScore(case_id=d["case_id"], trial=int(d["trial"]), suite=d["suite"], score=float(d["score"]),
                     violations=tuple(Violation(v["rule_id"], v["severity"], v["detail"])
                                      for v in d.get("violations", ())),
                     tokens_in=int(d.get("tokens_in", 0)), tokens_out=int(d.get("tokens_out", 0)),
                     served_model=d.get("served_model"), wall_ms=float(d.get("wall_ms") or 0.0),
                     llm_calls=int(d.get("llm_calls") or 0), tool_calls=int(d.get("tool_calls") or 0),
                     subscores={str(k): float(v) for k, v in (d.get("subscores") or {}).items()})


class EvalCache:
    def __init__(self, root: str | os.PathLike[str] | None = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.enabled = enabled

    @classmethod
    def for_profile(cls, profile: Profile | str, root: str | os.PathLike[str] | None = None) -> "EvalCache":
        return cls(root, enabled=Profile(profile) != Profile.COPILOT)

    def path(self, pin: EvaluatorPin, harness_tree: str, split: str, case_id: str, trial: int) -> Path:
        if split not in _SPLITS:
            raise ValueError(f"bad split {split!r}")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", harness_tree or ""):
            raise ValueError(f"harness_tree must be a git tree oid, got {harness_tree!r}")
        if not isinstance(trial, int) or trial < 0:
            raise ValueError(f"bad trial {trial!r}")
        rel = f"{pin_hash(pin)}/{harness_tree}/{split}/{_segment(case_id)}/{trial}.json"
        return safe_join(self.root, rel)

    def get(self, pin: EvaluatorPin, harness_tree: str, split: str, case_id: str, trial: int,
            *, suite: str | None = None) -> TaskScore | None:
        if not self.enabled:
            return None
        try:
            p = self.path(pin, harness_tree, split, case_id, trial)
        except UnsafePathError:
            return None
        try:
            data = read_json(p, default=None)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None
        try:
            if (data["pin"]["evaluator_tree"] != pin.evaluator_tree or data["pin"]["judge_model"] != pin.judge_model
                    or data["pin"]["judge_provider"] != pin.judge_provider or data["harness_tree"] != harness_tree
                    or data["split"] != split):
                return None
            if pin.served_judge_models and tuple(data["pin"].get("served_judge_models") or ()) != tuple(
                    pin.served_judge_models):
                return None
            score = _score_from(data["score"])
        except (KeyError, TypeError, ValueError):
            return None
        if score.case_id != case_id or score.trial != trial or (suite is not None and score.suite != suite):
            return None
        return score

    def put(self, pin: EvaluatorPin, harness_tree: str, split: str, score: TaskScore) -> Path | None:
        if not self.enabled or score.score is None:
            return None
        p = self.path(pin, harness_tree, split, score.case_id, score.trial)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, {
            "schema": SCHEMA,
            "pin": {"evaluator_tree": pin.evaluator_tree, "judge_model": pin.judge_model,
                    "judge_provider": pin.judge_provider, "served_judge_models": list(pin.served_judge_models)},
            "harness_tree": harness_tree, "split": split, "score": _score_json(score)})
        return p
