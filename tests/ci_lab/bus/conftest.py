"""Shared builders for bus voter tests (exposed as the ``kit`` fixture)."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from ci_lab.bus.types import ArtifactRef, ProposalBody
from ci_lab.taskgraph.model import (
    Criterion,
    Deliverable,
    OutputSpec,
    Rubric,
    StudentSpec,
)


class Kit:
    @staticmethod
    def crit(cid: str, measure: str = "deterministic", check: dict[str, Any] | None = None,
             threshold: float = 0.5) -> Criterion:
        return Criterion(cid, f"criterion {cid}", measure, check or {"kind": "regex", "pattern": "x"},  # type: ignore[arg-type]
                         threshold)

    @staticmethod
    def rubric(*criteria: Criterion) -> Rubric:
        return Rubric("r1", 1, "t1", criteria, 0.5, "canary-1")

    @staticmethod
    def spec(path: str | None = "out/answer.txt", kind: str = "file") -> StudentSpec:
        return StudentSpec.of(Deliverable("t1", "Task", "do it", OutputSpec(kind, path)))  # type: ignore[arg-type]

    @staticmethod
    def proposal(artifact: bytes, rubric_version: str = "r1@v1") -> ProposalBody:
        h = hashlib.sha256(artifact).hexdigest()
        return ProposalBody(proposal="t1@1/student:alice", attempt="t1@1", rubric_version=rubric_version,
                            artifact=ArtifactRef(path=f"{h[:2]}/{h}", sha256=h, bytes=len(artifact)),
                            summary="s")


@pytest.fixture
def kit() -> type[Kit]:
    return Kit
