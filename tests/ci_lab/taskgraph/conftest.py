from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ci_lab.taskgraph.model import (
    Budget,
    ContextRef,
    Criterion,
    Deliverable,
    OutputSpec,
    Rubric,
    TaskGraph,
)

CANARY = "0123456789abcdef"


def make_rubric(deliverable: str = "summary", version: int = 1, **kw: Any) -> Rubric:
    criteria = (
        Criterion("c-format", "Report starts with a level one heading", "deterministic",
                  {"kind": "regex", "pattern": "^# Report", "independent": True}, 1.0, required=True),
        Criterion("c-suite", "Harness agent suite passes on dev", "assert",
                  {"suite": "harness-agent", "split": "dev", "min_score": 0.8}, 0.8, weight=2.0),
        Criterion("c-cites", "Report cites resource identifiers from the log", "s1",
                  {"question": "Does the report cite at least two distinct resource identifiers taken from the log?",
                   "type": "noul", "options": []}, 0.6),
    )
    base: dict[str, Any] = {"id": f"{deliverable}-rubric", "version": version, "deliverable": deliverable,
                            "criteria": criteria, "pass_score": 0.7, "canary": CANARY}
    base.update(kw)
    return Rubric(**base)


def make_graph(commitments: dict[str, str] | None = None) -> TaskGraph:
    c = commitments or {}

    def d(tid: str, deps: tuple[str, ...], text: str, path: str) -> Deliverable:
        return Deliverable(tid, f"Task {tid}", text, OutputSpec("file", path),
                           (ContextRef("file", "data/log.txt"),), deps, c.get(tid, "0" * 64),
                           Budget(max_attempts=2, timeout_s=120, weight={"llm": 1, "s1": 2}))

    return TaskGraph("demo", "Summarise the resource log", (
        d("summary", (), "Summarise the resource log into out/summary.md as Markdown.", "out/summary.md"),
        d("triage", ("summary",), "Triage each escalation into out/triage.json.", "out/triage.json"),
        d("reply", ("triage",), "Draft one reviewer reply into out/reply.md.", "out/reply.md"),
        d("audit", ("summary",), "List open changes into out/audit.md.", "out/audit.md"),
    ))


@pytest.fixture
def rubric_factory() -> Callable[..., Rubric]:
    return make_rubric


@pytest.fixture
def graph_factory() -> Callable[..., TaskGraph]:
    return make_graph
