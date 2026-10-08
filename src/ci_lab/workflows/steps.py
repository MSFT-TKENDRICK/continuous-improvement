"""Workflow step functions, bound as closures over run contexts (design §5, C5, C6).

Every step is idempotent: it first checks its durable marker under the run dir
(``CI_RUN_DIR/<eid>/...``) and returns the recorded result if present. External
effects go through ``deps.outbox`` with :func:`ci_lab.contracts.op_id` keys.
Exceptions are converted to :class:`~ci_lab.workflows.runtime.StepAborted` so the
declarative runner stops at the failing superstep (it swallows ``Exception``).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import random
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.campaign import bus_adapter, challenger_lane, records
from ci_lab.campaign.deps import ROUND_CONTEXT, CampaignDeps
from ci_lab.contracts import (
    ARM_RE,
    ATTR_CAMPAIGN,
    ATTR_DECISION,
    ATTR_EXPERIMENT,
    ATTR_SCORE,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_ARM,
    STRATEGIES,
    ArmDirective,
    ArmResult,
    CriticVerdict,
    Edit,
    EvalResult,
    FailureRecord,
    Profile,
    arm_branch,
    op_id,
    round_experiment_id,
)
from ci_lab.contracts import ArmContext as StrategyContext
from ci_lab.ledger.looks import LookBudgetExceeded, is_same_look, normalize_hash
from ci_lab.meta.brief import arm_attempt, failure_corpus
from ci_lab.taskgraph.firewall import sanitize_correction
from ci_lab.workflows import ARM_YAMLS
from ci_lab.workflows.progress import HEARTBEAT_S, Progress, Tracker
from ci_lab.workflows.runtime import GatedAgent, StepAborted

INCUMBENT = "inc"
INCUMBENT_STRATEGY = "incumbent"  # status.json label only; not an arm strategy
INCUMBENT_GUARD = "inc-guard"     # run-dir/slot name of the incumbent's paired guard eval (B1)
GUARD_SPLIT_FILE = "guard_split.json"  # split + dataset hash (+ C15 look) of a guard arm's paired eval
OES_SPLITS = ("evolve", "heldout", "ood", "aa")  # splitHashes keys allowed by the OES extension schema
# history.jsonl per-arm fields: the legacy six + what rrsi HistoryRecord needs (L_t, strategy stats)
HISTORY_ARM_FIELDS = ("arm", "component", "hypotheses", "accepted", "score", "status", "strategy", "edits",
                      "evaluated", "cost", "delta_s", "delta_c", "novelty", "admissible", "reasons")
FINAL_CRITIQUE = 3  # critique_1, critique_2, critique_final
_GUARD_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


# ------------------------------------------------------------------ contexts

@dataclass
class CampaignEnv:
    cid: str
    profile: Profile
    hyper: Mapping[str, Any]
    deps: CampaignDeps
    run_root: Path

    @property
    def campaign_dir(self) -> Path:
        return self.run_root / self.cid

    def rel(self, *parts: str) -> str:
        return "/".join(("campaigns", self.cid, *parts))

    def progress(self, eid: str, writer: str, **base: Any) -> Progress:
        return Progress(self.run_root, eid, writer=writer, campaign_id=self.cid,
                        heartbeat_s=float(self.hyper.get("heartbeat_s") or HEARTBEAT_S), **base)

    def span_attrs(self, eid: str) -> dict[str, Any]:
        return {ATTR_CAMPAIGN: self.cid, ATTR_EXPERIMENT: eid}

    def frontier(self) -> dict[str, Any]:
        frontier = self.deps.ledger.read_json(self.rel("frontier.json"))
        if not frontier:
            raise RuntimeError(f"campaign {self.cid} has no frontier (run `campaign new`)")
        return frontier

    def delta(self) -> float:
        cal = self.deps.ledger.read_json(self.rel("calibration.json"))
        if not cal:
            raise RuntimeError(f"campaign {self.cid} is not calibrated")
        return float(cal["delta"])

    def commit_ledger(self, key: str, message: str, paths: list[str]) -> Any:
        return self.deps.outbox.run_once(op_id("ledger-commit", self.cid, key),
                                         lambda: {"sha": self.deps.ledger.commit(message, paths)})

    def remember_incumbent(self, tree: str, result: EvalResult) -> None:
        data = records.eval_to_dict(result)
        records.write_json(self.campaign_dir / "inc-cache" / f"{tree}.json", data)
        records.write_json(self.campaign_dir / "last_incumbent_eval.json", data)

    def round_hyper(self, round_no: int, history: Sequence[Mapping[str, Any]], *,
                    incumbent_commit: str | None = None) -> dict[str, Any]:
        """``hyper`` + :data:`ROUND_CONTEXT` for ``schedule``/``select``: the round, the calibrated
        delta (None before calibration), the incumbent commit and the history rows of earlier
        rounds (the RRSI adapters derive S*, the score trajectory and L_t from them)."""
        cal = self.deps.ledger.read_json(self.rel("calibration.json"))
        prior = [dict(r) for r in history if int(r.get("round") or 0) < round_no]
        return {**self.hyper, ROUND_CONTEXT: {
            "round": round_no, "eid": round_experiment_id(self.cid, round_no),
            "delta": float(cal["delta"]) if cal else None, "incumbent_commit": incumbent_commit,
            "history": prior}}

    def split_hashes(self) -> dict[str, str]:
        """OES ``splitHashes``: ``sha256:`` digest of each split's case ids."""
        return {name: dataset_hash(ids) for name, ids in self.deps.domain.splits().items()
                if name in OES_SPLITS}


@dataclass
class RoundContext:
    env: CampaignEnv
    round_no: int

    @property
    def eid(self) -> str:
        return round_experiment_id(self.env.cid, self.round_no)

    @property
    def dir(self) -> Path:
        return self.env.run_root / self.eid

    @property
    def ckpt(self) -> Path:
        return self.dir / "ckpt"

    @property
    def analysis_path(self) -> Path:
        """Written by the Analyst's terminal ``submit_analysis`` tool (M8a)."""
        return self.dir / "analysis.json"

    def begin(self) -> dict[str, Any]:
        data = records.read_json(self.dir / "begin.json")
        if data is None:
            raise RuntimeError(f"{self.eid}: begin_round has not run")
        return data

    def brief(self) -> dict[str, Any]:
        """Read-only round brief for analyst/proposer tools (typed failures only, C12)."""
        begin = self.begin()
        return {"experiment_id": self.eid, "round": self.round_no, "directives": begin["directives"],
                "failures": begin["failures"], "history": begin["history"],
                "analysis": records.read_json(self.analysis_path)}

    def arm(self, name: str) -> ArmRun:
        directive = next((d for d in self.begin()["directives"] if d["arm"] == name), {"arm": name})
        return ArmRun(self, name, directive)

    def arm_result(self, name: str) -> ArmResult | None:
        data = records.read_json(self.dir / name / "arm.done")
        return records.arm_from_dict(data["result"]) if data else None

    @functools.cached_property
    def progress(self) -> Progress:
        return self.env.progress(self.eid, "round", round=self.round_no)

    @property
    def tracker(self) -> Tracker:
        return Tracker(self.progress, self.env.span_attrs(self.eid))

    def arm_state(self, arm: str, strategy: str, state: str, phase: str | None) -> None:
        """Arm workers write their own marker (``status.d/<arm>.json``)."""
        self.progress.as_writer(arm).write(arms={arm: {"strategy": strategy, "state": state, "phase": phase}})

    async def incumbent_guard_metrics(self, split: str) -> Any:
        """Paired guard-off/on metrics of this round's incumbent (B1 ship baseline for guard arms);
        computed once per round in its own slot (``<run_dir>/inc-guard``), shared by all guard arms."""
        from ci_lab.lessons_arm.paired import guard_paired_eval_step
        from ci_lab.rulespec import GuardMetrics

        run_dir = self.dir / INCUMBENT_GUARD
        lock = _GUARD_LOCKS.setdefault((id(asyncio.get_running_loop()), str(run_dir)), asyncio.Lock())
        async with lock:
            slot = _done(run_dir / "slot.json")
            if slot is None:
                worktree = self.env.deps.provision_slot(self.eid, INCUMBENT_GUARD, self.begin()["base_commit"])
                slot = {"worktree": str(worktree)}
                records.write_json(run_dir / "slot.json", slot)
            data = await guard_paired_eval_step(domain=self.env.deps.domain, worktree=Path(slot["worktree"]),
                                                run_dir=run_dir, split=split, experiment_id=self.eid,
                                                variant=INCUMBENT_GUARD, **guard_eval_options(self.env))
        return GuardMetrics.model_validate(data["metrics"])


def guard_envelope_extension(ctx: RoundContext, arm: str) -> dict[str, Any] | None:
    """``{"com.microsoft.ci.guard": ...}`` (``lessons_arm.envelope.guard_extension``) for a guard arm
    that ran its paired eval, else ``None``; the round envelope carries it for the shipped guard arm."""
    guard = (records.read_json(ctx.dir / arm / "arm.done") or {}).get("guard") or {}
    if guard.get("skipped") or not guard.get("metrics"):
        return None
    from ci_lab.lessons_arm.envelope import guard_extension
    from ci_lab.lessons_arm.paired import GUARD_EVAL_FILE
    from ci_lab.rulespec import GuardMetrics

    ship = guard.get("ship")
    inc = (records.read_json(ctx.dir / INCUMBENT_GUARD / GUARD_EVAL_FILE) or {}).get("metrics") or {}
    return guard_extension(
        GuardMetrics.model_validate(guard["metrics"]), split=guard.get("split", "evolve"), arm=arm,
        ship=(bool(ship.get("ok")), ship.get("reasons") or ()) if ship else None,
        incumbent_digest=inc.get("bundle_digest"), dataset_hash=guard.get("dataset_hash"),
        looks_used=int(guard.get("look_no") or 1), planned_looks=int(guard.get("planned_looks") or 1))


def guard_eval_options(env: CampaignEnv) -> dict[str, Any]:
    """``k``/``trials``/``stochastic``/``margin`` for paired guard evals. B4: a stochastic provider
    (the Copilot profile unless ``guard_stochastic`` says otherwise) gets >= 3 trials per case."""
    from ci_lab.lessons_arm.paired import MIN_STOCHASTIC_TRIALS

    hyper = env.hyper
    k = int(hyper["k"])
    stochastic = hyper.get("guard_stochastic")
    stochastic = env.profile is Profile.COPILOT if stochastic is None else bool(stochastic)
    trials = hyper.get("guard_trials") or (-(-MIN_STOCHASTIC_TRIALS // k) if stochastic else 1)
    return {"k": k, "trials": int(trials), "stochastic": stochastic, "margin": float(hyper.get("guard_margin") or 0.0)}


def normalize_directive(raw: Any) -> dict[str, Any]:
    """Schedule output (``contracts.ArmDirective`` or mapping) -> JSON-able directive dict."""
    d = asdict(raw) if isinstance(raw, ArmDirective) else dict(raw)
    d.setdefault("strategy", "agent")
    if d["strategy"] not in STRATEGIES:
        raise ValueError(f"unknown arm strategy {d['strategy']!r} (expected one of {STRATEGIES})")
    if "component_focus" in d:
        d["component_focus"] = list(d["component_focus"] or ())
        if not d.get("component") and d["component_focus"]:
            d["component"] = d["component_focus"][0]
    return d


def directive_of(d: Mapping[str, Any]) -> ArmDirective:
    focus = d.get("component_focus") or ([d["component"]] if d.get("component") else [])
    return ArmDirective(arm=d["arm"], strategy=d.get("strategy", "agent"), component_focus=tuple(focus),
                        edit_budget=int(d.get("edit_budget", d.get("budget", 1))), explore=bool(d.get("explore")))


@dataclass
class ArmRun:
    """Run-dir view of one arm of a round (not to be confused with ``contracts.ArmContext``,
    the read-only view handed to an :class:`~ci_lab.contracts.ArmStrategy`)."""

    round: RoundContext
    arm: str
    directive: Mapping[str, Any]
    reinvoke_proposer: Callable[[str, list[str]], Awaitable[Any]] | None = field(default=None, repr=False)
    strategy_impl: Any = field(default=None, repr=False)

    @property
    def eid(self) -> str:
        return self.round.eid

    @property
    def strategy(self) -> str:
        return str(self.directive.get("strategy", "agent"))

    @property
    def dir(self) -> Path:
        return self.round.dir / self.arm

    @property
    def ckpt(self) -> Path:
        return self.dir / "ckpt"

    @property
    def proposal_path(self) -> Path:
        """Written by the Proposer's terminal ``submit_proposal`` tool (M8a) or the ``propose`` step."""
        return self.dir / "proposal.json"

    @property
    def base_commit(self) -> str:
        return self.round.begin()["base_commit"]

    @property
    def worktree(self) -> Path:
        slot = records.read_json(self.dir / "slot.json")
        if slot is None:
            raise RuntimeError(f"{self.eid}/{self.arm}: slot not provisioned")
        return Path(slot["worktree"])

    def verdict(self, attempt: int) -> CriticVerdict | None:
        data = records.read_json(self.dir / f"critique_{attempt}.json")
        return records.verdict_from_dict(data) if data else None

    @property
    def span_attrs(self) -> dict[str, Any]:
        return {**self.round.env.span_attrs(self.eid), ATTR_VARIANT: self.arm, ATTR_STRATEGY: self.strategy}

    @property
    def tracker(self) -> Tracker:
        def fields(phase: str) -> Mapping[str, Any]:
            return {"arms": {self.arm: {"strategy": self.strategy, "state": "running", "phase": phase}}}

        return Tracker(self.round.progress.as_writer(self.arm), self.span_attrs, fields)

    def state(self, state: str, phase: str | None) -> None:
        self.round.arm_state(self.arm, self.strategy, state, phase)

    def strategy_context(self, feedback: Sequence[str] = ()) -> StrategyContext:
        env = self.round.env
        failures = [records.failure_from_dict(f) for f in self.round.begin()["failures"]]
        failures += [FailureRecord("critic", "critic", "critic_rejected", (), {}, reason[:500])
                     for reason in feedback]
        budget = env.hyper.get("arm_budget_tokens")
        return StrategyContext(experiment_id=self.eid, directive=directive_of(self.directive),
                               worktree=self.worktree, base_commit=self.base_commit, failures=failures,
                               profile=env.profile, run_dir=self.dir,
                               budget_tokens=int(budget) if budget is not None else None,
                               evolve_case_ids=tuple(env.deps.domain.splits()["evolve"]))

    def optimizer_cost(self) -> Mapping[str, Any] | None:
        """The strategy's ``<arm dir>/optimizer/<arm>-<strategy>.json`` ``cost`` block (C18)."""
        from ci_lab.strategies.base import OPTIMIZER_DIR

        report = _done(self.dir / OPTIMIZER_DIR / f"{self.arm}-{self.strategy}.json")
        cost = report.get("cost") if report else None
        return dict(cost) if isinstance(cost, Mapping) else None

    def edit_scope_violations(self, edits: Sequence[Mapping[str, Any]]) -> list[str]:
        """B2/N5: declared edit files plus, for git worktrees, the actual ``base..HEAD`` diff.
        Guard files are the domain's ``<harness root>/guards`` (and any ``harness/guards``)."""
        from ci_lab.domain.layout import guards_rel
        from ci_lab.strategies.base import edit_scope_violations

        files = [f for e in edits for f in e.get("files", ())]
        files += _changed_files(self.worktree, self.base_commit)
        return edit_scope_violations(self.strategy, files, guards_dir=guards_rel(self.round.env.deps.domain))

    async def run_strategy(self, feedback: Sequence[str] = ()) -> list[Edit]:
        """Run the arm's :class:`~ci_lab.contracts.ArmStrategy` and record its edits
        exactly like the Proposer's ``submit_proposal`` would."""
        if self.strategy_impl is None:
            raise RuntimeError(f"{self.eid}/{self.arm}: no strategy bound for {self.strategy!r}")
        edits = list(await self.strategy_impl.propose(self.strategy_context(list(feedback))))
        for e in edits:
            if not isinstance(e, Edit):
                raise TypeError(f"strategy {self.strategy!r} returned {type(e).__name__}, expected Edit")
        records.write_json(self.proposal_path, {"strategy": self.strategy, "edits": [
            {"component": e.component, "hypothesis": e.hypothesis, "files": list(e.files), "commit": e.commit}
            for e in edits]})
        return edits


# ------------------------------------------------------------------ step wrapper

def step(name: str, fn: Callable[..., Any], tracker: Tracker | None = None) -> Callable[..., Awaitable[Any]]:
    """Async tool wrapper: tolerate literal YAML args, abort the workflow on error.
    With a ``tracker`` the call runs inside a ``ci.step{ci.phase=name}`` span and
    updates/heartbeats the run's ``status.json``."""

    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> Any:
        params = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        try:
            async with (tracker.phase(name) if tracker is not None else contextlib.nullcontext()):
                result = fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
            return result
        except Exception as exc:
            raise StepAborted(name, exc) from exc

    return wrapper


def _done(path: Path) -> dict[str, Any] | None:
    return records.read_json(path)


