"""JSON (de)serialisation of contract records + atomic file helpers."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ci_lab.contracts import (
    ArmResult,
    CriticVerdict,
    Edit,
    EvalResult,
    EvaluatorPin,
    FailureRecord,
    TaskScore,
    Violation,
)

_LOCK = threading.Lock()


def _retry(fn: Any, attempts: int = 20) -> Any:
    """Windows: os.replace/read can transiently fail while another thread holds the file."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01 * (i + 1))
    return None


def write_json(path: Path, obj: Any) -> None:
    """Atomic write (temp + replace) so a crash never leaves a torn marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")
    _retry(lambda: os.replace(tmp, path))


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(_retry(lambda: path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None


def append_jsonl(path: Path, obj: Mapping[str, Any], *, key: str) -> bool:
    """Append unless a line with the same ``obj[key]`` exists (idempotent)."""
    with _LOCK:
        rows = read_jsonl(path)
        if any(r.get(key) == obj[key] for r in rows):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, sort_keys=True, default=_default) + "\n")
        return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ------------------------------------------------------------------ contract records

def eval_to_dict(result: EvalResult) -> dict[str, Any]:
    return asdict(result)


def eval_from_dict(data: Mapping[str, Any]) -> EvalResult:
    pin = data["pin"]
    return EvalResult(
        harness_tree=data["harness_tree"],
        split=data["split"],
        pin=EvaluatorPin(pin["evaluator_tree"], pin["judge_model"], pin["judge_provider"],
                         tuple(pin.get("served_judge_models", ()))),
        scores=[TaskScore(case_id=s["case_id"], trial=s["trial"], suite=s["suite"], score=s["score"],
                          violations=tuple(Violation(**v) for v in s.get("violations", ())),
                          tokens_in=s.get("tokens_in", 0), tokens_out=s.get("tokens_out", 0),
                          served_model=s.get("served_model"), wall_ms=float(s.get("wall_ms") or 0.0),
                          llm_calls=int(s.get("llm_calls") or 0), tool_calls=int(s.get("tool_calls") or 0),
                          subscores={str(k): float(v) for k, v in (s.get("subscores") or {}).items()})
                for s in data.get("scores", ())],
        surface={str(k): float(v) for k, v in (data.get("surface") or {}).items()},
    )


def verdict_to_dict(v: CriticVerdict) -> dict[str, Any]:
    return asdict(v)


def verdict_from_dict(data: Mapping[str, Any]) -> CriticVerdict:
    return CriticVerdict(passed=bool(data["passed"]), reasons=list(data.get("reasons", ())),
                         repairs=int(data.get("repairs", 0)))


def arm_to_dict(arm: ArmResult) -> dict[str, Any]:
    return asdict(arm)


def arm_from_dict(data: Mapping[str, Any]) -> ArmResult:
    return ArmResult(
        arm=data["arm"], base_commit=data["base_commit"], head_commit=data.get("head_commit"),
        harness_tree=data.get("harness_tree"),
        edits=[Edit(e["component"], e["hypothesis"], tuple(e.get("files", ())), e["commit"])
               for e in data.get("edits", ())],
        critic=verdict_from_dict(data["critic"]) if data.get("critic") else None,
        eval=eval_from_dict(data["eval"]) if data.get("eval") else None,
        status=data.get("status", "pending"),
        strategy=data.get("strategy", "agent"),
        cost=data.get("cost"),
    )


def failure_from_dict(data: Mapping[str, Any]) -> FailureRecord:
    return FailureRecord(data["case_id"], data["suite"], data["category"], tuple(data.get("rule_ids", ())),
                         dict(data.get("rubric_scores") or {}), data.get("excerpt", ""))


# ------------------------------------------------------------------ metrics

def mean_score(result: EvalResult) -> float:
    """Mean task score with missing trials counted as 0 (RRSI)."""
    if not result.scores:
        return 0.0
    return sum(s.score or 0.0 for s in result.scores) / len(result.scores)


def critical_count(result: EvalResult) -> int:
    return sum(1 for s in result.scores for v in s.violations if v.severity == "critical")


def tokens(result: EvalResult | None) -> int:
    if result is None:
        return 0
    return sum(s.tokens_in + s.tokens_out for s in result.scores)


def cases(result: EvalResult) -> int:
    return len({s.case_id for s in result.scores})
