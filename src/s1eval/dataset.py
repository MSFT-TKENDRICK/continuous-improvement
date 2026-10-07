"""Dataset loading + validation (YAML list or JSONL of cases)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .rubric import Rubric
from .state import project_observable

AMBIGUOUS = "ambiguous"


def load_cases(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        cases = yaml.safe_load(text)
        if isinstance(cases, dict):  # mapping form: {cases: [...], _anchors: ...}
            cases = cases.get("cases")
    else:
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(cases, list):
        raise ValueError("dataset must be a list of cases")
    return cases, hashlib.sha256(text.encode()).hexdigest()


def validate_cases(cases: list[dict[str, Any]], rubric: Rubric) -> None:
    seen = set()
    for c in cases:
        cid = c.get("id")
        if not isinstance(cid, str) or cid in seen:
            raise ValueError(f"missing or duplicate case id {cid!r}")
        seen.add(cid)
        project_observable(c)  # raises LeakageError on non-observable fields
        labels = c.get("labels") or {}
        extra = set(labels) - set(rubric.questions) - {"human_pass"}
        if extra:
            raise ValueError(f"{cid}: labels for unknown questions {sorted(extra)}")
        for k, v in labels.items():
            if v == AMBIGUOUS or v is None:
                continue
            if k == "human_pass":
                if not isinstance(v, bool):
                    raise ValueError(f"{cid}: human_pass must be bool or 'ambiguous'")
                continue
            q = rubric.questions[k]
            if q.type == "noul" and not isinstance(v, bool):
                raise ValueError(f"{cid}.{k}: noul label must be bool")
            if q.type == "choice" and v not in q.criteria:
                raise ValueError(f"{cid}.{k}: {v!r} is not an option")
            if q.type == "score" and (isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(q.criteria)):
                raise ValueError(f"{cid}.{k}: score label out of range")
