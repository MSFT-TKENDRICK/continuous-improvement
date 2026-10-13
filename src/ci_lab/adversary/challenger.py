"""Challengers (bus contract v2 §11): produce adversary proposals for one attempt.

A challenger sees the :class:`StudentSpec` and an :class:`AdversaryRubricView` (soft-criterion
questions only: no oracle checks, thresholds, weights or canary) and never a student answer.
Proposal ids are ``ids.proposal_id(attempt, "adversary", name)``; the bus never commits them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ci_lab.adversary.gamers import FAMILIES, generate, shape
from ci_lab.bus import ids
from ci_lab.bus.types import ArtifactRef, ProposalBody
from ci_lab.taskgraph.model import Rubric, StudentSpec

__all__ = ["ADVERSARY_SPEC", "AdversaryRubricView", "Challenger", "ChallengerProposal", "DeterministicChallenger",
           "LLMAdversary", "artifact_ref"]

ADVERSARY_SPEC = Path(__file__).resolve().parents[1] / "meta" / "specs" / "adversary.yaml"
_SUMMARY_MAX = 400


@dataclass(frozen=True)
class AdversaryRubricView:
    """What the adversary role may see of a rubric: its version id and soft questions."""

    rubric_version: str
    questions: tuple[str, ...]

    @classmethod
    def of(cls, rubric: Rubric) -> AdversaryRubricView:
        return cls(rubric.version_id, tuple(str(c.check["question"]) for c in rubric.soft()))

    def render(self) -> str:
        return "## Soft criteria (LLM judges)\n" + "".join(f"- {q}\n" for q in self.questions)


@dataclass(frozen=True)
class ChallengerProposal:
    body: ProposalBody
    artifact: bytes
    gamer: str


def artifact_ref(data: bytes) -> ArtifactRef:
    """Content-addressed ref ``<sha[:2]>/<sha>`` for ``data``."""
    sha = hashlib.sha256(data).hexdigest()
    return ArtifactRef(path=f"{sha[:2]}/{sha}", sha256=sha, bytes=len(data))


def _proposal(attempt: str, name: str, view: AdversaryRubricView, data: bytes, summary: str) -> ProposalBody:
    return ProposalBody(proposal=ids.proposal_id(attempt, "adversary", name), attempt=attempt,
                        rubric_version=view.rubric_version, artifact=artifact_ref(data),
                        summary=summary[:_SUMMARY_MAX])


class Challenger(Protocol):
    async def propose(self, spec: StudentSpec, rubric_view: AdversaryRubricView,
                      attempt: str) -> list[ChallengerProposal]: ...


class DeterministicChallenger:
    """One proposal per gamer family; bytes depend only on (family, seed, attempt, spec, questions)."""

    def __init__(self, families: Sequence[str] = FAMILIES, seed: int = 0) -> None:
        unknown = sorted(set(families) - set(FAMILIES))
        if unknown:
            raise ValueError(f"unknown gamer families: {unknown}")
        self.families, self.seed = tuple(families), seed

    async def propose(self, spec: StudentSpec, rubric_view: AdversaryRubricView,
                      attempt: str) -> list[ChallengerProposal]:
        out = []
        for fam in self.families:
            data = generate(fam, spec, rubric_view.questions, seed=self.seed, attempt=attempt)
            out.append(ChallengerProposal(_proposal(attempt, fam, rubric_view, data, f"gamer {fam}"), data, fam))
        return out


def _parse(reply: str) -> tuple[str, str]:
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except ValueError:
        return reply, ""
    if isinstance(obj, dict) and isinstance(obj.get("artifact"), str):
        return obj["artifact"], str(obj.get("summary") or "")
    return reply, ""


class LLMAdversary:
    """White-box LLM attacker driven by ``meta/specs/adversary.yaml`` through an injected ``complete``."""

    def __init__(self, complete: Callable[[str], Awaitable[str]], *, spec_path: Path = ADVERSARY_SPEC,
                 name: str = "llm") -> None:
        ids.proposal_id("t@1", "adversary", name)  # validates the name early
        self.complete, self.spec_path, self.name = complete, spec_path, name

    def prompt(self, spec: StudentSpec, rubric_view: AdversaryRubricView, attempt: str) -> str:
        from ci_lab.meta.spec_loader import load_spec

        instructions = load_spec(self.spec_path).instructions.strip()
        return (f"{instructions}\n\n{spec.render()}\n{rubric_view.render()}\nAttempt: {attempt}\n"
                'Reply with JSON {"artifact": "<deliverable text>", "summary": "<attack idea>"}.\n')

    async def propose(self, spec: StudentSpec, rubric_view: AdversaryRubricView,
                      attempt: str) -> list[ChallengerProposal]:
        """One proposal; an LLM failure yields none (the lane is out-of-band and best effort)."""
        try:
            reply = await self.complete(self.prompt(spec, rubric_view, attempt))
        except Exception:  # noqa: BLE001
            return []
        text, summary = _parse(reply)
        data = shape(text, spec.output)
        body = _proposal(attempt, self.name, rubric_view, data, f"llm: {summary}".strip())
        return [ChallengerProposal(body, data, self.name)]
