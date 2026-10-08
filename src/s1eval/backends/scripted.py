"""Deterministic scripted backend for offline tests and dry runs.

``fn(state, name, question) -> Answer`` decides each question. The backend records every
(state, questions) request it sees so tests can assert what reached the "judge".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..types import Answer, Question
from .base import Decision

AnswerFn = Callable[[Any, str, Question], Answer]


def default_answer(state: Any, name: str, q: Question) -> Answer:
    if q.type == "noul":
        return Answer.from_noul_probability(0.5)
    if q.type == "choice":
        n = len(q.criteria)
        return Answer.from_choice_distribution({k: 1.0 / n for k in q.criteria})
    n = len(q.criteria)
    return Answer.from_score_distribution([1.0 / n] * n, list(q.criteria))


class ScriptedBackend:
    def __init__(self, fn: AnswerFn = default_answer, name: str = "scripted") -> None:
        self.fn = fn
        self.name = name
        self.requests: list[tuple[Any, dict[str, Question]]] = []

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision:
        t0 = time.perf_counter()
        self.requests.append((state, dict(questions)))
        answers = {k: self.fn(state, k, q) for k, q in questions.items()}
        return Decision(answers=answers, model=self.name, http_calls=0, model_calls=len(questions),
                        latency_s=time.perf_counter() - t0)

    def provenance(self) -> dict[str, Any]:
        return {"backend": "scripted", "name": self.name}
