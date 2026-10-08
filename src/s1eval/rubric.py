"""Rubric: atomic System One questions + a code-side composite pass rule + a review gate.

Composite logic (deliberately in code, not in a single vague "does_pass" judge question):
  * a component that FAILS with gate confidence >= min_confidence decides ``pass = False``
  * otherwise any abstained/refused or low-confidence component -> ``needs_review``
  * otherwise ``pass = all components satisfied``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import Answer, Question, questions_from_wire, questions_to_wire


@dataclass
class Condition:
    question: str
    expect: Any = None  # bool for noul
    allowed: list[str] | None = None  # choice
    min_level: int | None = None  # score

    def satisfied(self, verdict: Any) -> bool:
        if self.expect is not None:
            return verdict is self.expect
        if self.allowed is not None:
            return verdict in self.allowed
        if self.min_level is not None:
            return verdict is not None and verdict >= self.min_level
        raise ValueError(f"condition for {self.question} has no test")


@dataclass
class Rubric:
    name: str
    version: str
    questions: dict[str, Question]
    pass_rule: list[Condition]
    complements: dict[str, Question] = field(default_factory=dict)
    min_confidence: float = 0.4
    noul_threshold: float = 0.5
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        raw_text = Path(path).read_text(encoding="utf-8")
        d = yaml.safe_load(raw_text)
        qs = questions_from_wire(d["questions"])
        comps = questions_from_wire(d["complements"]) if d.get("complements") else {}
        for k, q in comps.items():
            if k not in qs or q.type != "noul" or qs[k].type != "noul":
                raise ValueError(f"complement {k!r} must pair a noul with a noul of the same name")
        rule = []
        for c in d["pass_rule"]:
            cond = Condition(c["question"], c.get("expect"), c.get("allowed"), c.get("min_level"))
            if cond.question not in qs:
                raise ValueError(f"pass_rule references unknown question {cond.question!r}")
            rule.append(cond)
        gate = d.get("review_gate") or {}
        return cls(
            name=d["name"],
            version=str(d["version"]),
            questions=qs,
            pass_rule=rule,
            complements=comps,
            min_confidence=float(gate.get("min_confidence", 0.4)),
            noul_threshold=float(d.get("noul_threshold", 0.5)),
            sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
        )

    def verdicts(self, answers: dict[str, Answer]) -> dict[str, Any]:
        return {k: a.verdict(self.noul_threshold) for k, a in answers.items()}

    def composite(self, answers: dict[str, Answer]) -> dict[str, Any]:
        confident_fail, uncertain, reasons = [], [], []
        for c in self.pass_rule:
            a = answers.get(c.question)
            if a is None or not a.ok:
                uncertain.append(c.question)
                reasons.append(f"{c.question}: {a.status if a else 'missing'}")
                continue
            ok = c.satisfied(a.verdict(self.noul_threshold))
            conf = a.gate_confidence or 0.0
            if not ok and conf >= self.min_confidence:
                confident_fail.append(c.question)
            elif conf < self.min_confidence:
                uncertain.append(c.question)
                reasons.append(f"{c.question}: low confidence {conf:.2f}")
        if confident_fail:
            return {"pass": False, "needs_review": False, "failed": confident_fail, "uncertain": uncertain, "reasons": reasons}
        if uncertain:
            return {"pass": None, "needs_review": True, "failed": [], "uncertain": uncertain, "reasons": reasons}
        return {"pass": True, "needs_review": False, "failed": [], "uncertain": [], "reasons": []}

    def gold_pass(self, labels: dict[str, Any]) -> bool | None:
        """Pass derived from gold component labels with the same rule.

        False if any unambiguous component fails; None if none fails but one is ambiguous/missing.
        """
        ambiguous = False
        for c in self.pass_rule:
            v = labels.get(c.question)
            if v is None or v == "ambiguous":
                ambiguous = True
                continue
            if not c.satisfied(v):
                return False
        return None if ambiguous else True

    def to_langsmith_questions(self) -> str:
        """TypeSafe questions JSON for the LangSmith decision-model evaluator 'Advanced' editor."""
        return json.dumps(questions_to_wire(self.questions), indent=2, ensure_ascii=False)
