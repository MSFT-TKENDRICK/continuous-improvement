"""Terminal ``submit_*`` tools: validate with pydantic and write JSON into the run dir.

Writes are atomic and idempotent (re-submitting identical content is a no-op; a
different payload replaces the previous one). The orchestrator reads the file with
:func:`read_submission`; a missing file means the agent failed to finish.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ci_lab.contracts import COMPONENTS

__all__ = [
    "SUBMISSION_FILES",
    "AnalysisSubmission",
    "Direction",
    "FailurePattern",
    "ProposalSubmission",
    "ReflectionSubmission",
    "VerdictSubmission",
    "make_submit_tools",
    "read_submission",
    "write_json_atomic",
]

Text = Field(min_length=1, max_length=4000)
Component = Literal[COMPONENTS]  # type: ignore[valid-type] # enum in the tool JSON schema
ShortList = Field(default_factory=list, max_length=50)


def _component(v: str) -> str:
    if v not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    return v


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FailurePattern(_Model):
    name: str = Field(min_length=1, max_length=200)
    description: str = Text
    component: Component
    case_ids: list[str] = ShortList
    rule_ids: list[str] = ShortList

    @field_validator("component")
    @classmethod
    def _check_component(cls, v: str) -> str:
        return _component(v)


class AnalysisSubmission(_Model):
    summary: str = Text
    patterns: list[FailurePattern] = Field(default_factory=list, max_length=20)
    suggested_components: list[Component] = Field(default_factory=list, max_length=len(COMPONENTS))

    @field_validator("suggested_components")
    @classmethod
    def _components(cls, v: list[str]) -> list[str]:
        return [_component(c) for c in v]


class ProposalSubmission(_Model):
    summary: str = Text
    predicted_fixes: list[str] = ShortList
    risks: list[str] = ShortList


class VerdictSubmission(_Model):
    verdict: Literal["accept", "reject"]
    reasons: list[str] = ShortList
    risk_notes: list[str] = ShortList

    @field_validator("reasons")
    @classmethod
    def _reasons(cls, v: list[str]) -> list[str]:
        return [r for r in (s.strip() for s in v) if r]


class Direction(_Model):
    component: Component
    idea: str = Field(min_length=1, max_length=1000)

    @field_validator("component")
    @classmethod
    def _check_component(cls, v: str) -> str:
        return _component(v)


class ReflectionSubmission(_Model):
    summary: str = Text
    lessons: list[str] = ShortList
    next_directions: list[Direction] = Field(default_factory=list, max_length=20)


SUBMISSION_FILES: dict[str, tuple[str, type[_Model]]] = {
    "submit_analysis": ("analysis.json", AnalysisSubmission),
    "submit_proposal_done": ("proposal.json", ProposalSubmission),
    "submit_verdict": ("verdict.json", VerdictSubmission),
    "submit_reflection": ("reflection.json", ReflectionSubmission),
}


def write_json_atomic(path: Path, payload: Any) -> bool:
    """Write JSON atomically; return False when identical content already exists."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


def read_submission(run_dir: Path | str, tool: str) -> _Model | None:
    fname, model = SUBMISSION_FILES[tool]
    path = Path(run_dir) / fname
    if not path.is_file():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:10])


def make_submit_tools(run_dir: Path | str) -> dict[str, Callable[..., str]]:
    """All four terminal tools bound to ``run_dir`` (bind only the one an agent needs)."""
    root = Path(run_dir)

    def _submit(tool: str, data: dict[str, Any]) -> str:
        fname, model = SUBMISSION_FILES[tool]
        try:
            obj = model.model_validate(data)
        except ValidationError as exc:
            return f"ERROR: invalid submission ({_errors(exc)}); fix and call {tool} again"
        changed = write_json_atomic(root / fname, obj.model_dump(mode="json"))
        return "submitted; your task is complete" if changed else "already submitted; your task is complete"

    def submit_analysis(summary: str, patterns: list[FailurePattern] | None = None,
                        suggested_components: list[str] | None = None) -> str:
        """Finish the analysis: overall summary, recurring failure patterns (each with the harness
        component most likely responsible) and the components worth editing next."""
        return _submit("submit_analysis", {
            "summary": summary,
            "patterns": [p.model_dump() if isinstance(p, BaseModel) else p for p in patterns or []],
            "suggested_components": list(suggested_components or []),
        })

    def submit_proposal_done(summary: str, predicted_fixes: list[str] | None = None,
                             risks: list[str] | None = None) -> str:
        """Finish proposing: summarize the committed edits, list case ids you predict they fix,
        and any risks. Call only after commit_edit."""
        return _submit("submit_proposal_done", {"summary": summary, "predicted_fixes": list(predicted_fixes or []),
                                                "risks": list(risks or [])})

    def submit_verdict(verdict: Literal["accept", "reject"], reasons: list[str] | None = None,
                       risk_notes: list[str] | None = None) -> str:
        """Finish the review: verdict accept or reject, with concrete reasons (required to reject)."""
        if verdict == "reject" and not [r for r in reasons or [] if str(r).strip()]:
            return "ERROR: a reject verdict needs at least one reason; call submit_verdict again"
        return _submit("submit_verdict", {"verdict": verdict, "reasons": list(reasons or []),
                                          "risk_notes": list(risk_notes or [])})

    def submit_reflection(summary: str, lessons: list[str] | None = None,
                          next_directions: list[Direction] | None = None) -> str:
        """Finish the reflection: what the round taught us, and concrete next directions per component."""
        return _submit("submit_reflection", {
            "summary": summary,
            "lessons": list(lessons or []),
            "next_directions": [d.model_dump() if isinstance(d, BaseModel) else d for d in next_directions or []],
        })

    return {"submit_analysis": submit_analysis, "submit_proposal_done": submit_proposal_done,
            "submit_verdict": submit_verdict, "submit_reflection": submit_reflection}
