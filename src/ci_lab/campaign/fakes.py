"""Offline fakes for the ``fake`` profile and tests (no git, no network, no LLM).

* Slots are directories with a ``HEAD`` file; commits are sha1s of
  ``base|hypothesis`` and their applied edit list lives in ``commits.json``.
* The stub domain scores each case with ``score_fn(edits, case_id, trial)``.
* Proposer/Analyst are real MAF ``Agent``s over :class:`ci_lab.testing.FakeChatClient`
  whose terminal tools (``submit_proposal`` / ``submit_analysis``) write the
  durable outputs the workflow gates on.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Container, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.campaign import records
from ci_lab.campaign.deps import CampaignDeps
from ci_lab.campaign.local import FileLedger, FileOutbox
from ci_lab.contracts import (
    COMPONENTS,
    ArmContext,
    CriticVerdict,
    Edit,
    EvalResult,
    EvaluatorPin,
    FailureRecord,
    Outbox,
    TaskScore,
)

ScoreFn = Callable[[Sequence[str], str, int], float]
HypothesisFn = Callable[[str, str, Mapping[str, Any]], str]


def default_score(edits: Sequence[str], case_id: str, trial: int) -> float:
    """0.1 baseline, +0.3 per ``boost`` edit (capped at 1)."""
    return min(1.0, 0.1 + 0.3 * sum("boost" in e for e in edits))


def default_hypothesis(eid: str, arm: str, directive: Mapping[str, Any]) -> str:
    return f"boost {directive.get('component', 'prompt')} ({eid})" if arm == "v1" else f"tweak {arm} ({eid})"


class FakeRepo:
    """Commit registry + slot provisioning (stand-in for M6 gitops)."""

    ROOT_EDITS: list[str] = []

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        (self.root / "commits").mkdir(parents=True, exist_ok=True)
        self.base_commit = hashlib.sha1(b"root").hexdigest()
        self._register(self.base_commit, [])

    def _register(self, commit: str, edits: list[str]) -> None:
        path = self.root / "commits" / f"{commit}.json"
        with self._lock:  # content-addressed and write-once (Windows: no replace races)
            if not path.exists():
                records.write_json(path, edits)

    def edits(self, commit: str) -> list[str] | None:
        if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
            return None
        return records.read_json(self.root / "commits" / f"{commit}.json")

    def resolve_incumbent(self) -> tuple[str, str]:
        return self.base_commit, self.tree_of(self.base_commit)

    def tree_of(self, commit: str) -> str:
        return hashlib.sha256(json.dumps(self.edits(commit)).encode()).hexdigest()

    def provision_slot(self, eid: str, arm: str, base_commit: str) -> Path:
        if self.edits(base_commit) is None:
            raise ValueError(f"unknown base commit {base_commit}")
        slot = self.root / "slots" / eid / arm
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "HEAD").write_text(base_commit, encoding="utf-8")
        return slot

    def head_commit(self, worktree: Path) -> str:
        return (Path(worktree) / "HEAD").read_text(encoding="utf-8").strip()

    def harness_tree(self, worktree: Path) -> str:
        return self.tree_of(self.head_commit(worktree))

    def apply(self, worktree: Path, base_commit: str, hypothesis: str) -> str:
        commit = hashlib.sha1(f"{base_commit}|{hypothesis}".encode()).hexdigest()
        self._register(commit, [*(self.edits(base_commit) or []), hypothesis])
        (Path(worktree) / "HEAD").write_text(commit, encoding="utf-8")
        return commit


class StubDomain:
    name = "stub"
    surface_globs: Sequence[str] = ("harness/**",)
    frozen_globs: Sequence[str] = ("evals/**",)
    component_globs: Mapping[str, Sequence[str]] = {c: (f"harness/{c}/**",) for c in COMPONENTS}

    def __init__(self, repo: FakeRepo, score_fn: ScoreFn = default_score,
                 evolve: Sequence[str] = ("c1", "c2", "c3", "c4"), heldout: Sequence[str] = ("h1", "h2"),
                 tokens_per_case: int = 10) -> None:
        self.repo = repo
        self.score_fn = score_fn
        self._splits = {"evolve": tuple(evolve), "heldout": tuple(heldout)}
        self.tokens_per_case = tokens_per_case
        self.calls: list[tuple[str, str, str]] = []
        self.fail_on: Callable[[str, str, str], bool] | None = None

    def splits(self) -> Mapping[str, Sequence[str]]:
        return self._splits

    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult:
        if self.fail_on is not None and self.fail_on(experiment_id, variant, split):
            raise RuntimeError(f"injected evaluate failure {experiment_id}/{variant}")
        self.calls.append((experiment_id, variant, split))
        commit = self.repo.head_commit(harness_dir)
        edits = self.repo.edits(commit) or []
        scores = [TaskScore(case, trial, "stub", self.score_fn(edits, case, trial),
                            tokens_in=self.tokens_per_case, tokens_out=0)
                  for case in self._splits[split] for trial in range(k)]
        return EvalResult(self.repo.tree_of(commit), split,  # type: ignore[arg-type]
                          EvaluatorPin("stub-evaluator", "stub-judge", "fake"), scores)

    def failures(self, result: EvalResult) -> list[FailureRecord]:
        return [FailureRecord(s.case_id, s.suite, "low_score", (), {"score": s.score or 0.0})
                for s in result.scores if (s.score or 0.0) < 1.0]


class FakeCritic:
    """Rejects the first critique of arms in ``reject_first``; passes otherwise."""

    def __init__(self, reject_first: Container[str] = ()) -> None:
        self.reject_first = reject_first
        self.calls: list[tuple[str, str, int]] = []

    async def __call__(self, ctx: Any, attempt: int) -> CriticVerdict:
        self.calls.append((ctx.eid, ctx.arm, attempt))
        if attempt == 1 and ctx.arm in self.reject_first:
            return CriticVerdict(False, ["hypothesis lacks a mechanism"])
        return CriticVerdict(True, [])


def _toggle_script(call: Callable[[], Any]) -> Callable[..., Any]:
    """FakeChatClient step: tool call, then final text, alternating per agent turn."""
    state = {"n": 0}

    def step(messages: Any, options: Any) -> Any:
        state["n"] += 1
        return call() if state["n"] % 2 else "done"

    return step


class FakeAgents:
    """``make_agent(role, ctx)`` for the fake profile."""

    def __init__(self, repo: FakeRepo, hypothesis_fn: HypothesisFn = default_hypothesis) -> None:
        self.repo = repo
        self.hypothesis_fn = hypothesis_fn
        self.runs: list[tuple[str, str]] = []

    def __call__(self, role: str, ctx: Any) -> Any:
        if role == "proposer":
            return self._proposer(ctx)
        if role == "analyst":
            return self._analyst(ctx)
        raise ValueError(f"no fake agent for role {role!r}")

    def _proposer(self, ctx: Any) -> Any:
        from agent_framework import Agent

        from ci_lab.testing import Call, FakeChatClient

        hypothesis = self.hypothesis_fn(ctx.eid, ctx.arm, ctx.directive)
        component = ctx.directive.get("component", "prompt")

        def submit_proposal(component: str, hypothesis: str) -> str:
            """Record the proposed edit (terminal tool)."""
            self.runs.append((ctx.eid, ctx.arm))
            commit = self.repo.apply(ctx.worktree, ctx.base_commit, hypothesis)
            records.write_json(ctx.proposal_path, {"edits": [
                {"component": component, "hypothesis": hypothesis, "files": [f"harness/{component}/x"],
                 "commit": commit}]})
            return "submitted"

        client = FakeChatClient(default=_toggle_script(
            lambda: [Call("submit_proposal", {"component": component, "hypothesis": hypothesis})]))
        return Agent(client=client, name="Proposer", instructions="Propose one edit.", tools=[submit_proposal])

    def _analyst(self, ctx: Any) -> Any:
        from agent_framework import Agent

        from ci_lab.testing import Call, FakeChatClient

        def submit_analysis(summary: str) -> str:
            """Record the round analysis (terminal tool)."""
            self.runs.append((ctx.eid, "analyst"))
            records.write_json(ctx.analysis_path, {"summary": summary, "failures": len(ctx.brief()["failures"])})
            return "submitted"

        client = FakeChatClient(default=_toggle_script(
            lambda: [Call("submit_analysis", {"summary": f"analysis for {ctx.eid}"})]))
        return Agent(client=client, name="Analyst", instructions="Analyse.", tools=[submit_analysis])


class FakeStrategy:
    """Stand-in :class:`~ci_lab.contracts.ArmStrategy` (M10): one commit per ``propose``;
    critic feedback (``critic_rejected`` failures) yields a "(repaired)" hypothesis."""

    def __init__(self, name: str, repo: FakeRepo, hypothesis_fn: HypothesisFn = default_hypothesis) -> None:
        self.name = name
        self.repo = repo
        self.hypothesis_fn = hypothesis_fn
        self.calls: list[ArmContext] = []

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        self.calls.append(ctx)
        d = ctx.directive
        component = d.component_focus[0] if d.component_focus else "prompt"
        hypothesis = self.hypothesis_fn(ctx.experiment_id, d.arm, {"component": component,
                                                                   "strategy": d.strategy})
        if any(f.category == "critic_rejected" for f in ctx.failures):
            hypothesis += " (repaired)"
        commit = self.repo.apply(ctx.worktree, ctx.base_commit, hypothesis)
        return [Edit(component, hypothesis, (f"harness/{component}/x",), commit)]


class FakeStrategies:
    """``get_strategy(name, **kw)`` for the fake profile; instances are cached per name."""

    def __init__(self, repo: FakeRepo, hypothesis_fn: HypothesisFn = default_hypothesis) -> None:
        self.repo = repo
        self.hypothesis_fn = hypothesis_fn
        self.instances: dict[str, FakeStrategy] = {}
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, name: str, **kwargs: Any) -> FakeStrategy:
        self.kwargs.append(kwargs)
        if name not in self.instances:
            self.instances[name] = FakeStrategy(name, self.repo, self.hypothesis_fn)
        return self.instances[name]


def fake_deps(root: Path, *, outbox: Outbox | None = None, score_fn: ScoreFn = default_score,
              hypothesis_fn: HypothesisFn = default_hypothesis, reject_first: Container[str] = (),
              publisher: Any = None, repo: str = "example/harness", ledger_root: Path | None = None,
              **overrides: Any) -> CampaignDeps:
    """Fully offline :class:`CampaignDeps`; publishing is a dry-run GitHubPublisher."""
    from ci_lab.publish.github import GitHubPublisher

    root = Path(root)
    fake_repo = FakeRepo(root / "repo")
    outbox = outbox if outbox is not None else FileOutbox(root / "outbox.jsonl")
    if publisher is None:
        publisher = GitHubPublisher(repo, outbox=outbox, dry_run=True, journal=root / "publish-calls.jsonl")
    kwargs: dict[str, Any] = dict(
        domain=StubDomain(fake_repo, score_fn), make_agent=FakeAgents(fake_repo, hypothesis_fn),
        provision_slot=fake_repo.provision_slot, head_commit=fake_repo.head_commit,
        harness_tree=fake_repo.harness_tree, resolve_incumbent=fake_repo.resolve_incumbent,
        get_strategy=FakeStrategies(fake_repo, hypothesis_fn),
        critique=FakeCritic(reject_first), ledger=FileLedger(ledger_root or root / "experiments"),
        publisher=publisher, outbox=outbox)
    kwargs.update(overrides)
    return CampaignDeps(**kwargs)
