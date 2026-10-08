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


def _changed_files(worktree: Path, base: str) -> list[str]:
    """``git diff --name-only base HEAD`` for git worktrees; ``[]`` otherwise (fake repos)."""
    if not (Path(worktree) / ".git").exists() or not base:
        return []
    proc = subprocess.run(["git", "-C", str(worktree), "diff", "--name-only", "--no-renames", base, "HEAD"],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git diff {base[:12]}..HEAD failed in {worktree}: {proc.stderr.strip()[:200]}")
    return [line for line in proc.stdout.splitlines() if line]


# ------------------------------------------------------------------ arm.yaml

def arm_tools(ctx: ArmRun) -> dict[str, Callable[..., Awaitable[Any]]]:
    deps = ctx.round.env.deps
    hyper = ctx.round.env.hyper

    def provision_slot() -> dict[str, Any]:
        marker = ctx.dir / "slot.json"
        if (data := _done(marker)) is not None:
            return data
        base = ctx.base_commit
        worktree = deps.provision_slot(ctx.eid, ctx.arm, base)
        data = {"worktree": str(worktree), "base_commit": base, "branch": arm_branch(ctx.eid, ctx.arm)}
        records.write_json(marker, data)
        return data

    async def critique(attempt: int) -> dict[str, Any]:
        attempt = int(attempt)
        marker = ctx.dir / f"critique_{attempt}.json"
        if (data := _done(marker)) is not None:
            return data
        prev = ctx.verdict(attempt - 1) if attempt > 1 else None
        if prev is not None and prev.passed:
            verdict = prev  # no-op gate: already passed
        elif not ctx.proposal_path.exists():
            verdict = CriticVerdict(False, ["proposer did not call submit_proposal"])
        else:
            verdict = await bus_adapter.critique(ctx, attempt)
        verdict = CriticVerdict(verdict.passed, list(verdict.reasons),
                                verdict.repairs if prev is not None and prev.passed else attempt - 1)
        data = records.verdict_to_dict(verdict)
        records.write_json(marker, data)
        return data

    async def repair(attempt: int) -> dict[str, Any]:
        attempt = int(attempt)
        marker = ctx.dir / f"repair_{attempt}.json"
        if (data := _done(marker)) is not None:
            return data
        verdict = ctx.verdict(attempt)
        if verdict is None:
            raise RuntimeError(f"repair_{attempt} before critique_{attempt}")
        if verdict.passed:
            data = {"repaired": False, "reason": "critique passed"}
        elif (succession := await bus_adapter.repair(ctx, attempt)) is not None:
            data = {**succession, "reasons": verdict.reasons}
        else:
            if ctx.reinvoke_proposer is None:
                raise RuntimeError("no proposer bound for repair")
            corpus = failure_corpus(records.failure_from_dict(f) for f in ctx.round.begin()["failures"])
            text = sanitize_correction(list(verdict.reasons), None, attempt=arm_attempt(ctx.arm, attempt),
                                       extra_corpus=corpus).text
            await ctx.reinvoke_proposer(
                text + "\nRepair the edits with your tools, then call submit_proposal again.",
                list(verdict.reasons))
            data = {"repaired": True, "reasons": verdict.reasons}
        records.write_json(marker, data)
        return data

    async def evaluate(split: str = "evolve") -> dict[str, Any]:
        marker = ctx.dir / "eval.json"
        if (data := _done(marker)) is not None:
            return {k: v for k, v in data.items() if k != "eval"}
        verdict = ctx.verdict(FINAL_CRITIQUE)
        if verdict is None:
            raise RuntimeError("evaluate before critique_final")
        head = deps.head_commit(ctx.worktree)
        if not verdict.passed:
            data = {"skipped": True, "reason": "critic_rejected", "head": head}
        elif head == ctx.base_commit:
            data = {"skipped": True, "reason": "no_edits", "head": head}
        elif bad := ctx.edit_scope_violations((_done(ctx.proposal_path) or {}).get("edits", ())):
            data = {"skipped": True, "reason": f"edit_scope: {ctx.strategy} may not write {', '.join(bad[:5])}",
                    "head": head}
        else:
            tree = deps.harness_tree(ctx.worktree)
            result = await bus_adapter.evaluate(ctx, split, head, functools.partial(
                deps.domain.evaluate, ctx.worktree, split, int(hyper["k"]), experiment_id=ctx.eid, variant=ctx.arm))
            data = {"skipped": False, "head": head, "tree": tree, "eval": records.eval_to_dict(result)}
        records.write_json(marker, data)
        return {k: v for k, v in data.items() if k != "eval"}

    async def guard_paired_eval(split: str = "evolve") -> dict[str, Any]:
        """``arm_guard.yaml`` (v2.4 §13, B1/B4): paired guard-off/on eval of the arm, gated against
        the incumbent's paired metrics (marker ``guard_eval.json``; skipped with ``evaluate``)."""
        from ci_lab.lessons_arm.envelope import holdout_look_required
        from ci_lab.lessons_arm.paired import GUARD_EVAL_FILE, guard_paired_eval_step

        data = _done(ctx.dir / GUARD_EVAL_FILE)
        if data is None:
            ev = _done(ctx.dir / "eval.json")
            if ev is None:
                raise RuntimeError("guard_paired_eval before evaluate")
            if not ev.get("skipped"):
                env = ctx.round.env
                info: dict[str, Any] = {"split": split, "holdout_look": holdout_look_required(split),
                                        "planned_looks": int(env.hyper["holdout_looks"])}
                if info["holdout_look"]:  # C15: one look per round, shared by the round's guard arms
                    info.update(reserve_holdout_look(env, ctx.eid, split))
                else:
                    info["dataset_hash"] = dataset_hash(deps.domain.splits()[split])
                records.write_json(ctx.dir / GUARD_SPLIT_FILE, info)
            incumbent = None if ev.get("skipped") else await ctx.round.incumbent_guard_metrics(split)
            data = await guard_paired_eval_step(domain=deps.domain, worktree=ctx.worktree, run_dir=ctx.dir,
                                                split=split, experiment_id=ctx.eid, variant=ctx.arm,
                                                incumbent=incumbent, **guard_eval_options(ctx.round.env))
        return {"skipped": bool(data.get("skipped")), "ship": (data.get("ship") or {}).get("ok")}

    def finalize_arm() -> dict[str, Any]:
        marker = ctx.dir / "arm.done"
        if (data := _done(marker)) is not None:
            return {"arm": ctx.arm, "status": data["result"]["status"]}
        ev = _done(ctx.dir / "eval.json")
        verdict = ctx.verdict(FINAL_CRITIQUE)
        if ev is None or verdict is None:
            raise RuntimeError("finalize_arm before evaluate")
        proposal = _done(ctx.proposal_path) or {}
        edits = [Edit(e["component"], e.get("hypothesis", ""), tuple(e.get("files", ())), e.get("commit", ""))
                 for e in proposal.get("edits", ())]
        reason = ev.get("reason")
        status = "evaluated" if not ev["skipped"] else "rejected"
        guard = None
        if ctx.strategy == "guard":
            from ci_lab.lessons_arm.paired import GUARD_EVAL_FILE

            guard = _done(ctx.dir / GUARD_EVAL_FILE)
            if guard is None:
                raise RuntimeError("finalize_arm before guard_paired_eval")
            ship = guard.get("ship")
            if status == "evaluated" and ship is not None and not ship.get("ok"):
                status, reason = "rejected", "guard_ship_rule: " + "; ".join(ship.get("reasons") or ())
        result = ArmResult(
            arm=ctx.arm, base_commit=ctx.base_commit, head_commit=ev.get("head"), harness_tree=ev.get("tree"),
            edits=edits, critic=verdict,
            eval=records.eval_from_dict(ev["eval"]) if ev.get("eval") else None,
            status=status, strategy=ctx.strategy, cost=ctx.optimizer_cost())
        done = {"result": records.arm_to_dict(result), "reason": reason, "directive": dict(ctx.directive)}
        if guard is not None:
            done["guard"] = {**guard, **(_done(ctx.dir / GUARD_SPLIT_FILE) or {})}
        records.write_json(marker, done)
        return {"arm": ctx.arm, "status": result.status}

    async def propose(strategy: str = "") -> dict[str, Any]:
        """Non-agent strategies: run ``ArmStrategy.propose`` once (marker: proposal.json)."""
        if (data := _done(ctx.proposal_path)) is not None:
            return {"strategy": data.get("strategy", ctx.strategy), "edits": len(data.get("edits", ()))}
        if strategy and strategy != ctx.strategy:
            raise ValueError(f"workflow strategy {strategy!r} != directive strategy {ctx.strategy!r}")
        edits = await ctx.run_strategy()
        return {"strategy": ctx.strategy, "edits": len(edits)}

    tracker = ctx.tracker
    return {name: step(name, fn, tracker) for name, fn in {
        "provision_slot": provision_slot, "propose": propose, "critique": critique, "repair": repair,
        "evaluate": evaluate, "guard_paired_eval": guard_paired_eval, "finalize_arm": finalize_arm}.items()}


async def run_arm(rctx: RoundContext, name: str) -> None:
    """Launch (or resume) the arm workflow of the directive's strategy
    (``arm_<strategy>.yaml``) with its own checkpoint dir (C5, section 11.2)."""
    env = rctx.env
    deps = env.deps
    ctx = rctx.arm(name)
    strategy = ctx.strategy
    with obs.span(SPAN_ARM, ctx.span_attrs):
        ctx.state("running", "start")
        agents: dict[str, Any] = {}
        if strategy == "agent":
            proposer = deps.make_agent("proposer", ctx)

            async def reinvoke(message: str, reasons: list[str]) -> Any:
                async with ctx.tracker.phase("propose"):
                    return await proposer.run(message)

            agents["Proposer"] = GatedAgent(proposer, ctx.proposal_path, wrap=lambda: ctx.tracker.phase("propose"))
        else:
            ctx.strategy_impl = deps.get_strategy(strategy, **dict(deps.strategy_kwargs))

            async def reinvoke(message: str, reasons: list[str]) -> Any:
                return await ctx.run_strategy(reasons)

        ctx.reinvoke_proposer = reinvoke
        workflow = deps.build_workflow(ARM_YAMLS[strategy], agents, arm_tools(ctx), ctx.ckpt)
        try:
            await deps.run_or_resume(workflow, ctx.ckpt, "start")
        except Exception as exc:
            attempts_path = ctx.dir / "attempts.json"
            attempts = (_done(attempts_path) or {"failures": []})["failures"]
            attempts.append(f"{type(exc).__name__}: {exc}")
            records.write_json(attempts_path, {"failures": attempts})
            if len(attempts) < int(env.hyper["max_arm_attempts"]):
                ctx.state("error", getattr(exc, "step", None))
                raise
            result = ArmResult(arm=name, base_commit=ctx.base_commit, status="failed", strategy=strategy)
            records.write_json(ctx.dir / "arm.done", {"result": records.arm_to_dict(result),
                                                      "reason": attempts[-1], "directive": dict(ctx.directive)})
            ctx.state("failed", "done")
            return
        result = rctx.arm_result(name)
        if result is None:
            raise RuntimeError(f"{rctx.eid}/{name}: workflow completed without arm.done")
        obs.annotate({ATTR_SCORE: records.mean_score(result.eval)} if result.eval else {})
        ctx.state(result.status, "done")


async def run_incumbent(rctx: RoundContext) -> None:
    """Re-evaluate the incumbent this round (C8); cached by tree off the Copilot profile."""
    env = rctx.env
    begin = rctx.begin()
    base, tree = begin["base_commit"], begin["base_tree"]
    cache = env.campaign_dir / "inc-cache" / f"{tree}.json"
    reuse = env.profile is not Profile.COPILOT and env.hyper.get("cache_incumbent", True)
    with obs.span(SPAN_ARM, {**env.span_attrs(rctx.eid), ATTR_VARIANT: INCUMBENT,
                             ATTR_STRATEGY: INCUMBENT_STRATEGY}):
        rctx.arm_state(INCUMBENT, INCUMBENT_STRATEGY, "running", "evaluate")
        if reuse and (cached := _done(cache)) is not None:
            result = records.eval_from_dict(cached)
        else:
            tracker = Tracker(rctx.progress.as_writer(INCUMBENT),
                              {**env.span_attrs(rctx.eid), ATTR_VARIANT: INCUMBENT},
                              lambda p: {"arms": {INCUMBENT: {"strategy": INCUMBENT_STRATEGY, "state": "running",
                                                              "phase": p}}})
            async with tracker.phase("evaluate"):
                worktree = env.deps.provision_slot(rctx.eid, INCUMBENT, base)
                result = await env.deps.domain.evaluate(worktree, "evolve", int(env.hyper["k"]),
                                                        experiment_id=rctx.eid, variant=INCUMBENT)
            env.remember_incumbent(tree, result)
        arm = ArmResult(arm=INCUMBENT, base_commit=base, head_commit=base, harness_tree=tree, eval=result,
                        status="evaluated")
        records.write_json(rctx.dir / INCUMBENT / "arm.done",
                           {"result": records.arm_to_dict(arm), "reason": None, "directive": {}})
        obs.annotate({ATTR_SCORE: records.mean_score(result)})
        rctx.arm_state(INCUMBENT, INCUMBENT_STRATEGY, "evaluated", "done")


# ------------------------------------------------------------------ round.yaml

def round_tools(ctx: RoundContext) -> dict[str, Callable[..., Awaitable[Any]]]:
    env = ctx.env
    deps = env.deps
    hyper = env.hyper

    def begin_round() -> dict[str, Any]:
        marker = ctx.dir / "begin.json"
        if (data := _done(marker)) is not None:
            return {"eid": data["eid"], "arms": [d["arm"] for d in data["directives"]]}
        frontier = env.frontier()
        history = deps.ledger.read_jsonl(env.rel("history.jsonl"))
        directives = [normalize_directive(d) for d in deps.schedule(
            ctx.round_no, env.round_hyper(ctx.round_no, history, incumbent_commit=frontier["incumbent_commit"]),
            history)]
        names = [d.get("arm") for d in directives]
        if not names or len(set(names)) != len(names) or INCUMBENT in names or \
                not all(isinstance(n, str) and ARM_RE.match(n) for n in names):
            raise ValueError(f"bad schedule arm names {names!r}")
        last_inc = _done(env.campaign_dir / "last_incumbent_eval.json")
        failures = [asdict(f) for f in deps.domain.failures(records.eval_from_dict(last_inc))] if last_inc else []
        data = {"eid": ctx.eid, "round": ctx.round_no, "base_commit": frontier["incumbent_commit"],
                "base_tree": frontier["incumbent_tree"], "directives": directives, "failures": failures,
                "history": history}
        records.write_json(marker, data)
        ctx.progress.write(arms={d["arm"]: {"strategy": d["strategy"], "state": "pending", "phase": None}
                                 for d in directives}
                           | {INCUMBENT: {"strategy": INCUMBENT_STRATEGY, "state": "pending", "phase": None}})
        return {"eid": ctx.eid, "arms": names}

    async def run_arms() -> dict[str, Any]:
        marker = ctx.dir / "arms.json"
        if (data := _done(marker)) is not None:
            return data
        arms = [d["arm"] for d in ctx.begin()["directives"]]
        order_path = ctx.dir / "run_order.json"
        order = _done(order_path)
        if order is None:
            order = [*arms, INCUMBENT]
            random.Random(f"{hyper['seed']}|{ctx.eid}").shuffle(order)  # C8 interleaving
            records.write_json(order_path, order)
        sem = asyncio.Semaphore(max(1, int(hyper["max_parallel_arms"])))

        async def one(name: str) -> None:
            async with sem:
                if (ctx.dir / name / "arm.done").exists():
                    return
                await (run_incumbent(ctx) if name == INCUMBENT else run_arm(ctx, name))

        outcomes = await asyncio.gather(*(one(n) for n in order), return_exceptions=True)
        errors = [o for o in outcomes if isinstance(o, BaseException)]
        if errors:
            raise RuntimeError(f"{len(errors)} arm(s) failed; rerun resumes them") from errors[0]
        statuses = {}
        for name in order:
            result = ctx.arm_result(name)
            if result is None:
                raise RuntimeError(f"{ctx.eid}/{name}: missing arm.done")
            statuses[name] = result.status
        await challenger_lane.run_lane(ctx)  # out of band: bus entries + evaluator proposal only
        data = {"order": order, "status": statuses}
        records.write_json(marker, data)
        return data

    def select() -> dict[str, Any]:
        marker = ctx.dir / "selection.json"
        if (data := _done(marker)) is not None:
            return {"decision": data["decision"], "winner": data["winner"]}
        incumbent = ctx.arm_result(INCUMBENT)
        if incumbent is None or incumbent.eval is None:
            raise RuntimeError("incumbent not evaluated")
        begin = ctx.begin()
        arms = {d["arm"]: ctx.arm_result(d["arm"]) for d in begin["directives"]}
        sel_hyper = env.round_hyper(ctx.round_no, begin.get("history") or (),
                                    incumbent_commit=begin["base_commit"])
        verdict = dict(deps.select(incumbent.eval, arms, env.delta(), sel_hyper))
        winner = verdict.get("winner")
        if verdict.get("decision") not in ("ship", "do_not_ship", "rerun"):
            raise ValueError(f"bad decision {verdict.get('decision')!r}")
        if (verdict["decision"] == "ship") != (winner is not None) or \
                (winner is not None and (winner not in arms or arms[winner].status != "evaluated")):
            raise ValueError(f"inconsistent selection {verdict!r}")
        records.write_json(marker, verdict)
        obs.annotate({ATTR_DECISION: verdict["decision"]})
        return {"decision": verdict["decision"], "winner": winner}

    def record() -> dict[str, Any]:
        marker = ctx.dir / "record.done"
        if (data := _done(marker)) is not None:
            return data
        begin = ctx.begin()
        sel = _done(ctx.dir / "selection.json")
        if sel is None:
            raise RuntimeError("record before select")
        incumbent = ctx.arm_result(INCUMBENT)
        arms = {d["arm"]: ctx.arm_result(d["arm"]) for d in begin["directives"]}
        winner = sel.get("winner") if sel["decision"] == "ship" else None
        trace = {t.get("arm"): t for t in sel.get("trace") or ()}
        attrib = {r.get("arm"): r for r in sel.get("attribution") or ()}
        arm_rows = []
        for d in begin["directives"]:
            a = arms[d["arm"]]
            t, at = trace.get(a.arm) or {}, attrib.get(a.arm) or {}
            reasons = t.get("reasons") or ([t["reason"]] if t.get("reason") else [])
            arm_rows.append({"arm": a.arm, "component": d.get("component"), "status": a.status,
                             "head": a.head_commit, "tree": a.harness_tree,
                             "score": records.mean_score(a.eval) if a.eval else None,
                             "hypotheses": [e.hypothesis for e in a.edits],
                             "critic": asdict(a.critic) if a.critic else None, "accepted": a.arm == winner,
                             "strategy": a.strategy or d.get("strategy", "agent"),
                             "edits": [asdict(e) for e in a.edits],
                             "evaluated": a.status == "evaluated" and a.eval is not None,
                             "cost": at.get("cost", t.get("cost")), "delta_s": at.get("delta_s", t.get("delta_s")),
                             "delta_c": at.get("delta_c", t.get("delta_c")),
                             "novelty": int(at.get("novelty", t.get("novelty")) or 0),
                             "admissible": bool(t.get("admissible")), "reasons": list(reasons)})
        tokens = records.tokens(incumbent.eval) + sum(records.tokens(a.eval) for a in arms.values())
        inc_score = sel.get("incumbent_score")
        inc_score = records.mean_score(incumbent.eval) if inc_score is None else inc_score
        score_next = sel.get("score_next")
        if score_next is None:
            won = next((r for r in arm_rows if r["arm"] == winner), None)
            score_next = won["score"] if won and won["score"] is not None else inc_score
        rec = {"eid": ctx.eid, "campaignId": env.cid, "round": ctx.round_no, "base_commit": begin["base_commit"],
               "base_tree": begin["base_tree"], "delta": env.delta(), "decision": sel["decision"],
               "winner": winner, "selection": sel, "arms": arm_rows, "tokens": tokens,
               "incumbent_score": records.mean_score(incumbent.eval)}
        ext = {**((guard_envelope_extension(ctx, winner) or {}) if winner else {}), **bus_adapter.bus_extension(ctx)}
        # lazy: AGT audit chain import; x-ci-governance = audit head + decision counts
        from ci_lab.governance.audit import envelope_extension

        if gov_ext := envelope_extension():
            ext = {**ext, **gov_ext}
        if ext:
            rec["extensions"] = ext
        evals = {"incumbent": records.arm_to_dict(incumbent),
                 "arms": {k: records.arm_to_dict(v) for k, v in arms.items()}}
        envelope_rec = {**rec, "evals": evals, "directives": begin["directives"],
                        "incumbent_commit": begin["base_commit"], "split_hashes": env.split_hashes()}
        rounds = f"rounds/{ctx.eid}"
        deps.ledger.write_json(env.rel(rounds, "envelope.json"), deps.build_envelope("round", envelope_rec))
        if (record_decisions := getattr(deps.ledger, "record_decisions", None)) is not None:
            record_decisions(env.cid, ctx.eid, sel)  # ci_lab.ledger.decisions (verdict check + record span)
        else:
            deps.ledger.write_json(env.rel(rounds, "decisions.json"), sel)
        deps.ledger.write_json(env.rel(rounds, "evals.json"), evals)
        s_star = sel.get("s_star")
        deps.ledger.append_jsonl(env.rel("history.jsonl"), {
            "eid": ctx.eid, "round": ctx.round_no, "decision": sel["decision"], "winner": winner,
            "tokens": tokens, "incumbent_score": inc_score, "score_next": score_next,
            **({"s_star": s_star} if s_star is not None else {}),
            "arms": [{k: r[k] for k in HISTORY_ARM_FIELDS} for r in arm_rows]}, key="eid")
        paths = [env.rel(rounds, n) for n in ("envelope.json", "decisions.json", "evals.json")]
        paths.append(env.rel("history.jsonl"))
        if winner:
            w = arms[winner]
            new = {"incumbent_commit": w.head_commit, "incumbent_tree": w.harness_tree,
                   "score": records.mean_score(w.eval) if w.eval else None, "round": ctx.round_no, "eid": ctx.eid}
            current = env.frontier()
            if current.get("incumbent_commit") != w.head_commit:
                if current.get("incumbent_commit") != begin["base_commit"] or \
                        not deps.ledger.cas_json(env.rel("frontier.json"), current, new):
                    raise RuntimeError(f"frontier CAS conflict for {ctx.eid}")
            paths.append(env.rel("frontier.json"))
        env.commit_ledger(ctx.eid, f"Record {ctx.eid}: {sel['decision']}", paths)
        data = {"decision": sel["decision"], "winner": winner}
        records.write_json(marker, data)
        return data

    def publish() -> dict[str, Any]:
        marker = ctx.dir / "publish.done"
        if (data := _done(marker)) is not None:
            return {k: data[k] for k in ("winner", "layers")}
        begin = ctx.begin()
        rec = _done(ctx.dir / "record.done")
        if rec is None:
            raise RuntimeError("publish before record")
        heads = {}
        for d in begin["directives"]:
            a = ctx.arm_result(d["arm"])
            if a and a.head_commit and a.head_commit != begin["base_commit"]:
                heads[a.arm] = a.head_commit
        winner = rec["winner"]
        stack = deps.ledger.read_json(env.rel("stack.json")) or {"layers": [], "stack_number": None}
        envelope = env.rel("rounds", ctx.eid, "envelope.json")
        title = f"RRSI {ctx.eid}: accept {winner}" if winner else f"RRSI {ctx.eid}"
        edits = ctx.arm_result(winner).edits if winner else []
        body = (f"Accepted arm `{winner}` of experiment `{ctx.eid}` (campaign `{env.cid}`).\n\n"
                f"OES envelope: `experiments/{envelope}`\n\n" +
                "\n".join(f"- {e.component}: {e.hypothesis}" for e in edits))
        result = deps.publisher.publish_round(eid=ctx.eid, winner=winner, heads=heads, stack=stack,
                                              title=title, body=body)
        if result.get("stack") != stack:
            deps.ledger.write_json(env.rel("stack.json"), result["stack"])
            env.commit_ledger(f"{ctx.eid}-stack", f"Record stack after {ctx.eid}", [env.rel("stack.json")])
        data = {"winner": winner, "layers": len(result["stack"]["layers"]), "result": result}
        records.write_json(marker, data)
        return {"winner": winner, "layers": data["layers"]}

    tracker = ctx.tracker
    return {name: step(name, fn, tracker) for name, fn in {
        "begin_round": begin_round, "run_arms": run_arms, "select": select, "record": record,
        "publish": publish}.items()}


# ------------------------------------------------------------------ calibrate.yaml

@dataclass
class CalibrationContext:
    env: CampaignEnv

    @functools.cached_property
    def progress(self) -> Progress:
        return self.env.progress(self.eid, "campaign")

    @property
    def tracker(self) -> Tracker:
        return Tracker(self.progress, self.env.span_attrs(self.eid))

    @property
    def eid(self) -> str:
        return f"{self.env.cid}-cal"

    @property
    def dir(self) -> Path:
        return self.env.run_root / self.eid

    @property
    def ckpt(self) -> Path:
        return self.dir / "ckpt"


def calibrate_tools(ctx: CalibrationContext) -> dict[str, Callable[..., Awaitable[Any]]]:
    env = ctx.env
    deps = env.deps
    hyper = env.hyper
    repeats = int(hyper["aa_repeats"])

    async def aa_runs(split: str = "evolve") -> dict[str, Any]:
        frontier = env.frontier()
        slot = ctx.dir / "slot.json"
        if (data := _done(slot)) is None:
            worktree = deps.provision_slot(ctx.eid, "h0", frontier["incumbent_commit"])
            data = {"worktree": str(worktree), "base_commit": frontier["incumbent_commit"]}
            records.write_json(slot, data)
        worktree = Path(data["worktree"])
        sem = asyncio.Semaphore(max(1, int(hyper["max_parallel_arms"])))

        async def one(i: int) -> None:
            path = ctx.dir / f"aa_{i}.json"
            if path.exists():
                return
            async with sem:
                result = await deps.domain.evaluate(worktree, split, int(hyper["k"]),
                                                    experiment_id=ctx.eid, variant=f"aa{i}")
            records.write_json(path, records.eval_to_dict(result))

        await asyncio.gather(*(one(i) for i in range(repeats)))
        env.remember_incumbent(frontier["incumbent_tree"],
                               records.eval_from_dict(_done(ctx.dir / "aa_0.json")))
        return {"repeats": repeats}

    def delta() -> dict[str, Any]:
        marker = ctx.dir / "delta.json"
        if (data := _done(marker)) is not None:
            return data
        results = [records.eval_from_dict(_done(ctx.dir / f"aa_{i}.json")) for i in range(repeats)]
        value = float(deps.calibrate_delta(results, hyper))
        data = {"delta": value, "means": [records.mean_score(r) for r in results],
                "tokens": sum(records.tokens(r) for r in results)}
        records.write_json(marker, data)
        return data

    def record() -> dict[str, Any]:
        marker = ctx.dir / "record.done"
        if (data := _done(marker)) is not None:
            return data
        cal = _done(ctx.dir / "delta.json")
        existing = deps.ledger.read_json(env.rel("calibration.json"))
        if existing is not None and existing["delta"] != cal["delta"]:
            raise RuntimeError("delta is immutable once calibrated")
        rec = {"eid": ctx.eid, "campaignId": env.cid, "decision": None, **cal,
               "repeats": repeats, "pin": _done(ctx.dir / "aa_0.json")["pin"]}
        envelope_rec = {**rec, "runs": [_done(ctx.dir / f"aa_{i}.json") for i in range(repeats)],
                        "harness_commit": _done(ctx.dir / "slot.json")["base_commit"],
                        "split_hashes": env.split_hashes()}
        deps.ledger.write_json(env.rel("calibration", "envelope.json"),
                               deps.build_envelope("calibration", envelope_rec))
        deps.ledger.write_json(env.rel("calibration.json"), rec)
        env.commit_ledger(ctx.eid, f"Calibrate {env.cid}: delta={cal['delta']:.4f}",
                          [env.rel("calibration.json"), env.rel("calibration", "envelope.json")])
        data = {"delta": cal["delta"]}
        records.write_json(marker, data)
        return data

    tracker = ctx.tracker
    return {name: step(name, fn, tracker) for name, fn in {
        "aa_runs": aa_runs, "delta": delta, "record": record}.items()}


# ------------------------------------------------------------------ confirm.yaml

class HoldoutExhausted(RuntimeError):
    pass


@dataclass
class ConfirmContext:
    env: CampaignEnv

    @functools.cached_property
    def progress(self) -> Progress:
        return self.env.progress(self.eid, "campaign")

    @property
    def tracker(self) -> Tracker:
        return Tracker(self.progress, self.env.span_attrs(self.eid))

    @property
    def eid(self) -> str:
        return f"{self.env.cid}-confirm"

    @property
    def dir(self) -> Path:
        return self.env.run_root / self.eid

    @property
    def ckpt(self) -> Path:
        return self.dir / "ckpt"


LOOKS = "holdout-looks.jsonl"  # global, ledger root (C15)


def dataset_hash(case_ids: Any) -> str:
    """``sha256:<hex>`` of the sorted case ids (OES ``datasetHash`` form; also a valid look-ledger key)."""
    return "sha256:" + hashlib.sha256("\n".join(sorted(str(c) for c in case_ids)).encode()).hexdigest()


def reserve_holdout_look(env: CampaignEnv, eid: str, split: str) -> dict[str, Any]:
    """C15: reserve one look at ``split`` in the global ledger ``holdout-looks.jsonl`` for experiment
    ``eid`` (idempotent per experiment, so a resumed step does not consume a second look); returns
    ``{"dataset_hash", "look_no"}``. Ledgers with ``record_look`` (FileLedger) go through
    :mod:`ci_lab.ledger.looks`."""
    deps = env.deps
    digest = dataset_hash(deps.domain.splits()[split])
    planned = int(env.hyper["holdout_looks"])
    if (record_look := getattr(deps.ledger, "record_look", None)) is not None:
        try:
            look_no = int(record_look(digest, experiment_id=eid, planned=planned, campaign_id=env.cid,
                                      split=split)["look_no"])
        except LookBudgetExceeded as exc:
            raise HoldoutExhausted(str(exc)) from exc
    else:
        key = f"{eid}|{digest}"
        bare = normalize_hash(digest)
        looks = [r for r in deps.ledger.read_jsonl(LOOKS)
                 if isinstance(r.get("dataset_hash"), str) and normalize_hash(r["dataset_hash"]) == bare]
        mine = next((i for i, r in enumerate(looks) if is_same_look(r, digest, eid, env.cid)), None)
        if mine is None:
            if len(looks) >= planned:
                raise HoldoutExhausted(f"held-out {digest[:19]} already looked at {len(looks)} time(s)")
            deps.ledger.append_jsonl(LOOKS, {"key": key, "campaign": env.cid, "dataset_hash": digest,
                                             "split": split, "eid": eid}, key="key")
            mine = len(looks)
        look_no = mine + 1
    env.commit_ledger(f"{eid}-look", f"Reserve held-out look for {eid}", [LOOKS])
    return {"dataset_hash": digest, "look_no": look_no}


def confirm_tools(ctx: ConfirmContext) -> dict[str, Callable[..., Awaitable[Any]]]:
    env = ctx.env
    deps = env.deps
    hyper = env.hyper

    def reserve_look(split: str = "heldout") -> dict[str, Any]:
        marker = ctx.dir / "look.json"
        if (data := _done(marker)) is not None:
            return data
        look = reserve_holdout_look(env, ctx.eid, split)
        data = {"dataset_hash": look["dataset_hash"], "split": split, "look_no": look["look_no"],
                "planned": int(hyper["holdout_looks"])}
        records.write_json(marker, data)
        return data

    async def evaluate_heldout(split: str = "heldout") -> dict[str, Any]:
        if _done(ctx.dir / "look.json") is None:
            raise RuntimeError("evaluate_heldout before reserve_look")
        campaign = deps.ledger.read_json(env.rel("campaign.json"))
        frontier = env.frontier()
        targets = {"h0": campaign["base_commit"], "final": frontier["incumbent_commit"]}
        for name, commit in targets.items():
            path = ctx.dir / f"{name}.json"
            if path.exists():
                continue
            if name == "final" and commit == targets["h0"]:
                records.write_json(path, _done(ctx.dir / "h0.json"))
                continue
            worktree = deps.provision_slot(ctx.eid, name, commit)
            result = await deps.domain.evaluate(worktree, split, int(hyper["k"]),
                                                experiment_id=ctx.eid, variant=name)
            records.write_json(path, records.eval_to_dict(result))
        return {"evaluated": sorted(targets)}

    def decide() -> dict[str, Any]:
        marker = ctx.dir / "decision.json"
        if (data := _done(marker)) is not None:
            return data
        h0 = records.eval_from_dict(_done(ctx.dir / "h0.json"))
        final = records.eval_from_dict(_done(ctx.dir / "final.json"))
        data = dict(deps.confirm_test(h0, final, hyper))
        if data.get("decision") not in ("ship", "do_not_ship"):
            raise ValueError(f"bad confirm decision {data!r}")
        records.write_json(marker, data)
        return data

    def record() -> dict[str, Any]:
        marker = ctx.dir / "record.done"
        if (data := _done(marker)) is not None:
            return data
        decision = _done(ctx.dir / "decision.json")
        rec = {"eid": ctx.eid, "campaignId": env.cid, **decision, "look": _done(ctx.dir / "look.json"),
               "h0": _done(ctx.dir / "h0.json"), "final": _done(ctx.dir / "final.json")}
        campaign = deps.ledger.read_json(env.rel("campaign.json")) or {}
        history = deps.ledger.read_jsonl(env.rel("history.jsonl"))
        envelope_rec = {**rec, "baseline_commit": campaign.get("base_commit"),
                        "final_commit": env.frontier()["incumbent_commit"], "look_ledger_ref": LOOKS,
                        "planned_looks": int(hyper["holdout_looks"]), "split_hashes": env.split_hashes(),
                        "accepted_rounds": [h["eid"] for h in history if h.get("decision") == "ship"]}
        deps.ledger.write_json(env.rel("confirm", "envelope.json"), deps.build_envelope("confirm", envelope_rec))
        deps.ledger.write_json(env.rel("confirm.json"), {"eid": ctx.eid, **decision})
        env.commit_ledger(ctx.eid, f"Confirm {env.cid}: {decision['decision']}",
                          [env.rel("confirm.json"), env.rel("confirm", "envelope.json")])
        data = {"decision": decision["decision"]}
        records.write_json(marker, data)
        return data

    tracker = ctx.tracker
    return {name: step(name, fn, tracker) for name, fn in {
        "reserve_look": reserve_look, "evaluate_heldout": evaluate_heldout, "decide": decide,
        "record": record}.items()}
