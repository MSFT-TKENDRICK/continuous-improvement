"""System One question/answer types and confidence formulas (ported from ``s1eval``).

Wire reference (TypeSafe System One):
  request  {model, state, questions: {name: {type, instructions?, criteria?}}}
  noul     answer {type: "noul",   noul: P(true)}
  choice   answer {type: "choice", choice, probabilities: {option: p}, confidence}
  score    answer {type: "score",  score: E[level], legend, probabilities: {"i": p}, confidence}

Answers also carry a local ``status`` ("ok" | "abstain" | "refusal") and ``diagnostics``.
Non-ok answers are never coerced to false / 0 / the first option.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal["noul", "choice", "score"]
Status = Literal["ok", "abstain", "refusal"]

QUESTION_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
PROB_SUM_TOLERANCE = 0.02


class WireError(ValueError):
    """A request or response does not conform to the System One contract."""


# ------------------------------------------------------------------ confidence

def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else min(x, 1.0)


def choice_confidence(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    """clip((K * p_max - 1) / (K - 1))."""
    ps = list(probabilities.values()) if isinstance(probabilities, Mapping) else list(probabilities)
    k = len(ps)
    if k < 2:
        raise ValueError("choice confidence needs at least 2 options")
    return _clip((k * max(ps) - 1.0) / (k - 1.0))


def score_confidence(probabilities: Sequence[float]) -> float:
    """clip(1 - sum_i p_i |i - peak| / mean_i |i - (K-1)/2|)."""
    ps = list(probabilities)
    k = len(ps)
    if k < 2:
        raise ValueError("score confidence needs at least 2 levels")
    peak = max(range(k), key=lambda i: (ps[i], -i))
    centre = (k - 1) / 2.0
    spread = sum(abs(i - centre) for i in range(k)) / k
    dispersion = sum(p * abs(i - peak) for i, p in enumerate(ps))
    return _clip(1.0 - dispersion / spread)


def noul_confidence(p_true: float) -> float:
    """|2p - 1| (local convention; TypeSafe noul answers carry no confidence)."""
    return _clip(abs(2.0 * p_true - 1.0))


# ------------------------------------------------------------------ questions

def _is_json_content(v: Any) -> bool:
    return isinstance(v, (str, dict, list))


def render_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=False)


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: Any = None
    criteria: Any = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.type not in ("noul", "choice", "score"):
            raise WireError(f"unknown question type {self.type!r}")
        if self.instructions is not None and not _is_json_content(self.instructions):
            raise WireError("instructions must be text, an object, or an array")
        if self.type == "noul":
            if self.criteria is None:
                return
            if not isinstance(self.criteria, dict) or set(self.criteria) - {"true", "false"}:
                raise WireError("noul criteria must be an object with optional 'true'/'false'")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict):
                raise WireError("choice criteria must map option names to descriptions")
            n = len(self.criteria)
            if not 2 <= n <= MAX_CHOICE_OPTIONS:
                raise WireError(f"choice needs 2..{MAX_CHOICE_OPTIONS} options, got {n}")
            for k, v in self.criteria.items():
                if not isinstance(k, str) or not k:
                    raise WireError("choice option names must be non-empty strings")
                if v is not None and not _is_json_content(v):
                    raise WireError(f"choice option {k!r} description must be text/object/array/null")
        else:
            if not isinstance(self.criteria, list):
                raise WireError("score criteria must be an ordered array of level descriptions")
            n = len(self.criteria)
            if not 2 <= n <= MAX_SCORE_LEVELS:
                raise WireError(f"score needs 2..{MAX_SCORE_LEVELS} levels, got {n}")
            for v in self.criteria:
                if not _is_json_content(v):
                    raise WireError("score level descriptions must be text/object/array")

    @property
    def options(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["true", "false"]

    def to_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.instructions is not None:
            d["instructions"] = self.instructions
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


def questions_to_wire(qs: Mapping[str, Question]) -> dict[str, Any]:
    for name in qs:
        if not QUESTION_NAME_RE.match(name):
            raise WireError(f"invalid question name {name!r}")
    return {k: q.to_wire() for k, q in qs.items()}


def _check_prob(p: Any, where: str) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise WireError(f"{where}: probability must be a finite number in [0, 1], got {p!r}")
    return float(p)


def _check_distribution(probs: Mapping[str, float], expected_keys: list[str], where: str) -> None:
    if set(probs) != set(expected_keys):
        raise WireError(f"{where}: probability keys {sorted(probs)} != expected {sorted(expected_keys)}")
    s = sum(probs.values())
    if abs(s - 1.0) > PROB_SUM_TOLERANCE:
        raise WireError(f"{where}: probabilities sum to {s:.4f}, not ~1 (no silent renormalisation)")


# ------------------------------------------------------------------ answers

@dataclass
class Answer:
    type: QuestionType
    status: Status = "ok"
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] | None = None
    legend: dict[str, Any] | None = None
    confidence: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def gate_confidence(self) -> float | None:
        if not self.ok:
            return None
        if self.type == "noul":
            return noul_confidence(self.noul)  # type: ignore[arg-type]
        return self.confidence

    @property
    def level(self) -> int | None:
        """Modal score level (ties -> lower level)."""
        if self.type != "score" or not self.ok or not self.probabilities:
            return None
        probs = self.probabilities
        return int(max(sorted(probs, key=int), key=lambda k: probs[k]))

    def verdict(self, threshold: float = 0.5) -> Any:
        if not self.ok:
            return None
        if self.type == "noul":
            return self.noul >= threshold  # type: ignore[operator]
        if self.type == "choice":
            return self.choice
        return self.level

    @classmethod
    def from_noul_probability(cls, p: float, **diag: Any) -> Answer:
        return cls(type="noul", noul=_check_prob(p, "noul"), diagnostics=dict(diag))

    @classmethod
    def from_choice_distribution(cls, probs: Mapping[str, float], **diag: Any) -> Answer:
        choice = max(probs, key=lambda k: probs[k])  # ties -> first listed
        return cls(type="choice", choice=choice, probabilities=dict(probs),
                   confidence=choice_confidence(probs), diagnostics=dict(diag))

    @classmethod
    def from_score_distribution(cls, probs: Sequence[float], legend: Sequence[Any], **diag: Any) -> Answer:
        if len(probs) != len(legend):
            raise WireError("score distribution and legend length differ")
        return cls(type="score", score=sum(i * p for i, p in enumerate(probs)),
                   probabilities={str(i): p for i, p in enumerate(probs)},
                   legend={str(i): d for i, d in enumerate(legend)},
                   confidence=score_confidence(probs), diagnostics=dict(diag))

    @classmethod
    def non_answer(cls, qtype: QuestionType, status: Status, **diag: Any) -> Answer:
        if status == "ok":
            raise ValueError("non_answer requires a non-ok status")
        return cls(type=qtype, status=status, diagnostics=dict(diag))

    @classmethod
    def from_wire(cls, d: Any, question: Question, where: str = "answer") -> Answer:
        """Parse and strictly validate a System One answer against its question."""
        if not isinstance(d, dict):
            raise WireError(f"{where}: answer must be an object")
        t = d.get("type")
        if t != question.type:
            raise WireError(f"{where}: answer type {t!r} does not match question type {question.type!r}")
        if t == "noul":
            return cls(type="noul", noul=_check_prob(d.get("noul"), f"{where}.noul"))
        probs_raw = d.get("probabilities")
        if not isinstance(probs_raw, dict):
            raise WireError(f"{where}: probabilities must be an object")
        probs = {str(k): _check_prob(v, f"{where}.probabilities[{k}]") for k, v in probs_raw.items()}
        _check_distribution(probs, question.options, where)
        conf = d.get("confidence")
        conf = _check_prob(conf, f"{where}.confidence") if conf is not None else None
        if t == "choice":
            choice = d.get("choice")
            if choice not in question.criteria:
                raise WireError(f"{where}: choice {choice!r} is not one of the options")
            return cls(type="choice", choice=choice, probabilities=probs, confidence=conf)
        score = d.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise WireError(f"{where}: score must be a finite number")
        if not 0 <= score <= len(question.criteria) - 1 + 1e-9:
            raise WireError(f"{where}: score {score} outside level range")
        legend = d.get("legend")
        legend = {str(k): v for k, v in legend.items()} if isinstance(legend, dict) else None
        return cls(type="score", score=float(score), probabilities=probs, legend=legend, confidence=conf)

    def to_record(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "status": self.status,
            "noul": self.noul,
            "choice": self.choice,
            "score": self.score,
            "level": self.level,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "gate_confidence": self.gate_confidence,
            "diagnostics": self.diagnostics,
        }
