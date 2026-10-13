"""Remote voters (bus contract v2 §9, P8b): ASSERT suites, System-1 categorical judges and the
MAF critic agent. Every network/LLM-facing piece is injectable (domain, pin, backend, runner) so
tests run offline. Pools are taken by :func:`~ci_lab.bus.voters.local.run_voters` (``pool``
attribute); the S1 voter additionally holds a host-wide ``judge.admission`` lease before its
timeout starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.bus.types import Measure, ProposalBody, VoteBody
from ci_lab.bus.voters.local import Ballot, BaseVoter, _vote, artifact_path
from ci_lab.judge import admission
from ci_lab.judge.backends import BackendError, make_backend
from ci_lab.judge.s1types import Answer, Question
from ci_lab.taskgraph.model import Criterion, Rubric, StudentSpec

__all__ = ["DEFAULT_S1_MODEL", "AgentVoter", "AssertVoter", "S1RubricVoter", "s1_ballot"]

DEFAULT_S1_MODEL = "s1/llamacpp/qwen3.5-4b"
Materialize = Callable[[bytes, StudentSpec, Path], Path]


def _check(c: Criterion) -> dict[str, Any]:
    return c.to_json()["check"]


def _materialize(artifact: bytes, spec: StudentSpec, root: Path) -> Path:
    path = root / artifact_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(artifact)
    return root


class AssertVoter(BaseVoter):
    """Deliverable-specific ASSERT: ``domain.evaluate(harness_dir, split, k)`` on the materialized
    artifact, mean of the ``suite`` case scores (missing trials count 0), pass iff ``>= min_score``.
    Scores are cached by (artifact sha256, suite, split, evaluator pin) in memory and, when
    ``cache_dir`` is set, as JSON files."""

    per_criterion = True

    def __init__(self, domain: Any, *, name: str = "assert", criteria: Sequence[str] | None = None,
                 pool: str | None = "llm", weight: int = 1, k: int = 1, pin: str | None = None,
                 materialize: Materialize = _materialize, cache_dir: Path | str | None = None,
                 experiment_id: str = "bus-vote") -> None:
        super().__init__(name, "assert", criteria=criteria, pool=pool, weight=weight)
        self.domain, self.k, self.materialize, self.experiment_id = domain, k, materialize, experiment_id
        self._pin, self.cache_dir = pin, None if cache_dir is None else Path(cache_dir)
        self.cache: dict[tuple[str, str, str, str], float] = {}
        self.evaluations = 0

    async def pin(self) -> str:
        if self._pin is None:
            p = await asyncio.to_thread(self.domain.pin) if hasattr(self.domain, "pin") else None
            self._pin = f"{p.evaluator_tree}|{p.judge_model}" if p is not None else "unpinned"
        return self._pin

    def _file(self, key: tuple[str, ...]) -> Path | None:
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        return None if self.cache_dir is None else self.cache_dir / f"{digest}.json"

    async def score(self, artifact: bytes, sha: str, spec: StudentSpec, suite: str, split: str) -> float:
        key = (sha, suite, split, await self.pin())
        if key in self.cache:
            return self.cache[key]
        f = self._file(key)
        if f is not None and f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if tuple(data["key"]) == key:
                self.cache[key] = float(data["score"])
                return self.cache[key]
        with tempfile.TemporaryDirectory(prefix="ci-assert-", ignore_cleanup_errors=True) as tmp:
            harness = self.materialize(artifact, spec, Path(tmp))
            self.evaluations += 1
            result = await self.domain.evaluate(harness, split, self.k, experiment_id=self.experiment_id,
                                                variant=sha[:16])
        by_suite: dict[str, list[float]] = {}
        for s in result.scores:  # one evaluation fills the cache for every suite it covered
            by_suite.setdefault(s.suite, []).append(s.score or 0.0)
        for name, got in by_suite.items():
            k2 = (sha, name, split, key[3])
            self.cache[k2] = value = min(1.0, max(0.0, sum(got) / len(got)))
            if (f2 := self._file(k2)) is not None:
                f2.parent.mkdir(parents=True, exist_ok=True)
                f2.write_text(json.dumps({"key": list(k2), "score": value}), encoding="utf-8")
        if key not in self.cache:
            raise LookupError(f"no {split!r} cases for suite {suite!r}")
        return self.cache[key]

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        out: list[VoteBody] = []
        for c in self.covered(rubric):
            chk = _check(c)
            try:
                s = await self.score(artifact, proposal.artifact.sha256, spec, chk["suite"], chk["split"])
                b = Ballot(s >= chk["min_score"], s, 1.0, (f"{chk['suite']}/{chk['split']}: {s:.3f}",))
            except Exception as exc:  # noqa: BLE001 - evaluation failure abstains this criterion
                b = Ballot(None, reasons=(f"assert error: {type(exc).__name__}: {exc}"[:500],))
            out.append(_vote(self, proposal, c.id, b))
        return out


def s1_ballot(answer: Answer, c: Criterion) -> Ballot:
    """noul: score = P(true); choice: score = P(first option) (the passing option), falling back
    to 1/0 on the modal choice. Pass iff ``score >= threshold``; non-ok answers abstain."""
    if not answer.ok:
        return Ballot(None, reasons=(f"s1 answer status {answer.status}",))
    if answer.type == "noul" and answer.noul is not None:
        score, said = answer.noul, "noul"
    elif answer.type == "choice" and answer.choice is not None:
        good, probs = _check(c)["options"][0], answer.probabilities
        score = probs.get(good, 0.0) if probs else float(answer.choice == good)
        said = f"choice {answer.choice!r}"
    else:
        return Ballot(None, reasons=(f"s1 answer has no {answer.type} value",))
    score = min(1.0, max(0.0, float(score)))
    conf = answer.gate_confidence
    return Ballot(score >= c.threshold, score, None if conf is None else min(1.0, max(0.0, conf)),
                  (f"s1 {said}: pass probability {score:.3f}",))


class S1RubricVoter(BaseVoter):
    """One categorical System-1 question per ``s1`` (or ``llm``) criterion via the ``ci_lab.judge``
    backend for ``model``; local llama.cpp models hold a ``judge.admission`` lease (reentered by
    the backend) before the vote's timeout starts. Backend errors abstain."""

    per_criterion = True

    def __init__(self, model: str = DEFAULT_S1_MODEL, *, backend: Any = None, name: str = "s1",
                 measure: Measure = "s1", criteria: Sequence[str] | None = None, pool: str | None = "s1",
                 weight: int = 1, api_base: str | None = None) -> None:
        super().__init__(name, measure, criteria=criteria, pool=pool, weight=weight)
        self.model, self.api_base, self._backend = model, api_base, backend

    @property
    def backend(self) -> Any:
        if self._backend is None:
            from ci_lab.judge.provider import split_model  # heavy (imports litellm)

            kind, model = split_model(self.model)
            self._backend = make_backend(kind, model, api_base=self.api_base)
        return self._backend

    def admission(self) -> contextlib.AbstractAsyncContextManager[Any]:
        url = admission.s1_local_url(self.model, self.api_base)
        return admission.hold_async(url) if url else contextlib.nullcontext()

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        crits = self.covered(rubric)
        if not crits:
            return []
        questions: dict[str, Question] = {}
        for i, c in enumerate(crits):
            chk = _check(c)
            opts = None if chk["type"] == "noul" else {o: None for o in chk["options"]}
            questions[f"q{i}"] = Question(type=chk["type"], instructions=chk["question"], criteria=opts)
        state = f"# Task\n{spec.instructions}\n\n# Output\n{artifact.decode('utf-8', 'replace')}"
        try:
            decision = await asyncio.to_thread(self.backend.decide, state, questions)
        except BackendError as exc:
            return [_vote(self, proposal, c.id, Ballot(None, reasons=(f"s1 backend error: {exc}"[:500],)))
                    for c in crits]
        return [_vote(self, proposal, c.id, s1_ballot(decision.answers[f"q{i}"], c)
                      if f"q{i}" in decision.answers else Ballot(None, reasons=("s1 answer missing",)))
                for i, c in enumerate(crits)]


AgentRunner = Callable[[str, Path, Mapping[str, Callable[..., Any]]], Awaitable[Any]]


class AgentVoter(BaseVoter):
    """MAF critic agent (``spec``, default ``critic``) once per ``llm`` criterion. It reads a brief
    with the task and the criterion question, the artifact as ``proposal`` and a read-only view of
    the snapshot; ``accept`` passes. ``runner(spec_key, run_dir, bindings) -> VerdictSubmission``
    is injectable; the default calls ``meta.run.run_meta_agent`` with ``client``/``builder``."""

    per_criterion = True

    def __init__(self, client: Any = None, *, spec: str = "critic", runner: AgentRunner | None = None,
                 builder: Any = None, name: str = "critic-agent", criteria: Sequence[str] | None = None,
                 pool: str | None = "llm", weight: int = 1) -> None:
        super().__init__(name, "llm", criteria=criteria, pool=pool, weight=weight)
        if runner is None and client is None:
            raise ValueError("AgentVoter needs a chat client or an injected runner")
        self.client, self.spec_key, self.builder = client, spec, builder
        self.runner: AgentRunner = runner or self._run_meta

    async def _run_meta(self, key: str, run_dir: Path, bindings: Mapping[str, Callable[..., Any]]) -> Any:
        from ci_lab.meta.run import run_meta_agent

        return await run_meta_agent(key, run_dir, self.client, bindings, builder=self.builder, reuse=False)

    async def vote(self, proposal: ProposalBody, artifact: bytes, rubric: Rubric,
                   spec: StudentSpec) -> list[VoteBody]:
        from ci_lab.meta.run import write_brief
        from ci_lab.meta.spec_loader import load_spec
        from ci_lab.tools.arm_fs import make_arm_fs
        from ci_lab.tools.briefs import make_brief_tools

        documents = load_spec(self.spec_key).documents or None
        out: list[VoteBody] = []
        for c in self.covered(rubric):
            with tempfile.TemporaryDirectory(prefix="ci-agent-vote-", ignore_cleanup_errors=True) as tmp:
                run_dir, snap = Path(tmp) / "run", _materialize(artifact, spec, Path(tmp) / "snap")
                write_brief(run_dir, f"# Review one deliverable output\n\n## Task\n{spec.instructions}\n\n"
                                     f"## Question\n{_check(c)['question']}\n\nAccept only if the output "
                                     "(document `proposal`) clearly satisfies the question; otherwise reject "
                                     "with concrete reasons.\n")
                (run_dir / "proposal.json").write_text(json.dumps(
                    {"path": artifact_path(spec), "content": artifact.decode("utf-8", "replace")}), encoding="utf-8")
                fs = make_arm_fs(snap, ["**"], writable_globs=())
                bindings = {**make_brief_tools(run_dir, allowed=documents),
                            "list_files": fs["list_files"], "read_file": fs["read_file"]}
                sub = await self.runner(self.spec_key, run_dir, bindings)
            ok = sub.verdict == "accept"
            out.append(_vote(self, proposal, c.id, Ballot(ok, 1.0 if ok else 0.0, None, tuple(sub.reasons))))
        return out
