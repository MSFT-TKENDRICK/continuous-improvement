"""Run a backend over a dataset and emit per-trace records."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .backends.base import BackendError
from .rubric import Rubric
from .state import project_observable

Progress = Callable[[str], None]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def manifest(backend, rubric: Rubric, dataset_sha: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tool": f"s1eval {__version__}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": backend.provenance(),
        "rubric": {"name": rubric.name, "version": rubric.version, "sha256": rubric.sha256,
                   "min_confidence": rubric.min_confidence, "noul_threshold": rubric.noul_threshold},
        "dataset_sha256": dataset_sha,
        **(extra or {}),
    }


def run_cases(backend, rubric: Rubric, cases: list[dict[str, Any]], *, repeats: int = 1,
              progress: Progress | None = None) -> list[dict[str, Any]]:
    records = []
    for i, case in enumerate(cases):
        state = project_observable(case)
        labels = case.get("labels") or {}
        for r in range(repeats):
            t0 = time.perf_counter()
            rec: dict[str, Any] = {
                "case_id": case["id"],
                "tags": case.get("tags", []),
                "repeat": r,
                "gold": {k: v for k, v in labels.items() if k != "human_pass"},
                "gold_pass": rubric.gold_pass(labels),
                "human_pass": labels.get("human_pass"),
            }
            try:
                d = backend.decide(state, rubric.questions)
            except BackendError as e:
                rec.update({"error": str(e)[:500], "latency_s": time.perf_counter() - t0})
                records.append(rec)
                if progress:
                    progress(f"[{i + 1}/{len(cases)}] {case['id']} ERROR {e}")
                continue
            rec.update({
                "model": d.model,
                "answers": {k: a.to_record() for k, a in d.answers.items()},
                "verdicts": rubric.verdicts(d.answers),
                "composite": rubric.composite(d.answers),
                "latency_s": round(d.latency_s, 3),
                "model_calls": d.model_calls,
                "http_calls": d.http_calls,
                "usage": d.usage,
            })
            records.append(rec)
            if progress:
                c = rec["composite"]
                progress(f"[{i + 1}/{len(cases)}] {case['id']} pass={c['pass']} review={c['needs_review']} "
                         f"gold={rec['gold_pass']} {d.latency_s:.1f}s")
    return records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
