"""Backend protocol shared by all System One-style judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..types import Answer, Question


class BackendError(RuntimeError):
    """Request-level failure (transport, auth, protocol violation)."""


@dataclass
class Decision:
    answers: dict[str, Answer]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    http_calls: int = 0
    model_calls: int = 0
    latency_s: float = 0.0


class Backend(Protocol):
    name: str

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision: ...

    def provenance(self) -> dict[str, Any]: ...
