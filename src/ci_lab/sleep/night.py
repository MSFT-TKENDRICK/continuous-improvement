"""One SkillOpt-Sleep night: ``sleep.yaml`` = harvest -> consolidate -> assert_gate -> record -> bundle.

Every step iterates the enabled skill targets of the registry (``targets.yaml``). Nothing in
the checkout is modified (C10): candidate skills, the advanced ``experiments/sleep/state.json``
and the night's OES envelope only leave as a patch inside the digest-pinned bundle, which the
privileged ``publish`` job applies to a PR branch.

Observability (design §12.3): one ``ci.sleep.night`` trace per night with ``ci.step{ci.phase}``
children per step and per target, ``ci.optimizer{ci.strategy=skillopt}`` around SkillOpt, and
live ``<run_dir>/<night_id>/status.d/night.json`` markers via :func:`ci_lab.obs.write_status`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.types import TaskRecord

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_COMPONENT,
    ATTR_DECISION,
    ATTR_EXPERIMENT,
    ATTR_NIGHT,
    ATTR_PHASE,
    ATTR_PROFILE,
    ATTR_SPLIT,
    ATTR_STRATEGY,
    ATTR_TARGET,
    SPAN_OPTIMIZER,
    SPAN_SLEEP_NIGHT,
    SPAN_STEP,
    EvalResult,
    EvaluatorPin,
    SafetyOracle,
)
from ci_lab.sleep.backend import OrderSupportSleepBackend, Reflector, RunTarget, Scorer
from ci_lab.sleep.budget import Budget, BudgetExceeded, BudgetLimits
from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle
from ci_lab.sleep.gate import CanaryResult, GateDecision, decide, static_canaries
from ci_lab.sleep.harvest import HarvestResult, from_agl_exports, harvest, reviewed_ids
from ci_lab.sleep.registry import ORDER_SUPPORT, SkillTarget, load_targets, validate_targets
from ci_lab.sleep.runner import RunWorkflow, default_runner

if TYPE_CHECKING:
    from ci_lab.sleep.lessons_hook import LessonsReport, LessonsRequest

WORKFLOW_PATH = Path(__file__).with_name("sleep.yaml")
SKILL_REL = ORDER_SUPPORT.skill_path
MEMORY_REL = ORDER_SUPPORT.memory_path
STATE_REL = "experiments/sleep/state.json"
ENVELOPES_REL = "experiments/sleep/envelopes"
TASKS_REL = ORDER_SUPPORT.tasks_file
STATE_FORMAT = "ci_lab.sleep.state.v1"
STEPS = ("harvest", "consolidate", "assert_gate", "record", "bundle")
HISTORY_KEEP = 60


@dataclass
class SleepConfig:
    repo_root: Path
    out_dir: Path
    profile: str = "fake"
    targets: list[SkillTarget] | None = None  # None = enabled targets of the packaged registry
    tasks_file: Path | None = None  # single-target override of the target's reviewed tasks file
    work_dir: Path | None = None  # checkpoints + temp harness copies (never inside out_dir)
    run_dir: Path | None = None  # live status.json markers + telemetry JSONL (default: work_dir)
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    edit_budget: int = 4
    dream_rollouts: int = 1
    dream_factor: int = 0
    recall_k: int = 0
    gate_metric: str = "mixed"
    gate_mixed_weight: float = 0.5
    evolve_memory: bool = False
    val_fraction: float = 0.34
    delta_default: float = 0.02
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    night_date: str | None = None  # yyyymmdd (UTC today when None)
    run_attempt: int = 1
    base_sha: str | None = None
    lessons_hook: bool = False  # HOOK(M16): mine lesson candidates into the bundle; off by default

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.out_dir = Path(self.out_dir)
        self.targets = validate_targets(load_targets() if self.targets is None else self.targets)
        if self.tasks_file is not None:
            if len(self.targets) != 1:
                raise ValueError("tasks_file override needs exactly one skill target")
            self.tasks_file = Path(self.tasks_file)
        if self.work_dir is None:
            self.work_dir = self.out_dir.parent / f"{self.out_dir.name}-work"
        if self.run_dir is None:
            self.run_dir = self.work_dir
        if self.night_date is not None and not re.fullmatch(r"\d{8}", self.night_date):
            raise ValueError("night_date must be yyyymmdd")
        if not 1 <= int(self.run_attempt) <= 9999:
            raise ValueError("run_attempt out of range")

    def tasks_path(self, target: SkillTarget) -> Path:
        return self.tasks_file if self.tasks_file is not None else self.repo_root / target.tasks_file


@dataclass
class SleepDeps:
    run_target: RunTarget
    oracle: SafetyOracle
    reflector: Reflector
    assert_eval: Callable[[str, str, str], EvalResult]  # (skill, memory, variant) -> ASSERT result
    build_envelope: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    # the ASSERT gate's evaluator pin, recorded in the OES envelope of nights where nothing was
    # gated; None = the ``evaluator_pin`` attribute of ``assert_eval`` (set by the real and fake builders)
    evaluator_pin: Callable[[], EvaluatorPin] | None = None
    scorer: Scorer | None = None
    agl_records: Callable[[], Iterable[Mapping[str, Any]]] | None = None
    run_canaries: Callable[[str, str], list[CanaryResult]] | None = None
    latest_delta: Callable[[], float | None] | None = None
    run_workflow: RunWorkflow | None = None
    git_head: Callable[[Path], str] | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    monotonic: Callable[[], float] = time.monotonic
    # target-specific callables (run_target, oracle, reflector, assert_eval, scorer,
    # run_canaries, latest_delta); None = these deps serve every target
    per_target: Callable[[SkillTarget], SleepDeps] | None = None
    # HOOK(M16): typed rollouts -> sanitized lesson candidates (ci_lab.sleep.lessons_hook);
    # used only when cfg.lessons_hook. Per-target deps fall back to the night-level hook.
    lessons: Callable[[LessonsRequest], LessonsReport] | None = None

@dataclass
class NightResult:
    status: str  # accepted | rejected | budget_exceeded | no_tasks | error
    accepted: bool
    ledger_update: bool
    bundle_dir: Path
    manifest: dict[str, Any]
    night_id: str
    error: str = ""
    targets: dict[str, str] = field(default_factory=dict)  # target -> status
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _TargetRun:
    target: SkillTarget
    deps: SleepDeps
    skill: str = ""
    memory: str = ""
    memory_exists: bool = False
    reviewed_ids: list[str] = field(default_factory=list)
    harvest: HarvestResult | None = None
    backend: OrderSupportSleepBackend | None = None
    consolidation: Any = None
    candidate_skill: str = ""
    candidate_memory: str = ""
    decision: GateDecision | None = None
    lessons: LessonsReport | None = None
    gate_skipped: str = ""
    no_tasks: str = ""
    evals: dict[str, EvalResult] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def status(self) -> str:
        if self.no_tasks:
            return "no_tasks"
        return "accepted" if self.decision is not None and self.decision.accepted else "rejected"

    def reasons(self) -> list[str]:
        if self.no_tasks:
            return [self.no_tasks]
        if self.decision is not None:
            return list(self.decision.reasons)
        return [self.gate_skipped or "no candidate"]


@dataclass
class _Night:
    cfg: SleepConfig
    deps: SleepDeps
    budget: Budget
    night_id: str
    date: str
    base_sha: str
    runs: list[_TargetRun] = field(default_factory=list)
    state_text: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    envelope: dict[str, Any] | None = None
    changes: list[FileChange] = field(default_factory=list)
    status: str = "running"
    error: str = ""
    error_tb: str = ""
    budget_exceeded: BudgetExceeded | None = None
    steps_run: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    otel_ctx: Any = None

    @property
    def night_no(self) -> int:
        return int(self.state.get("night", 0) or 0) + 1

    @property
    def stopped(self) -> bool:
        return self.status in ("error", "no_tasks", "budget_exceeded")


# ----------------------------------------------------------------- helpers

def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def _match_eol(old: str | None, new: str) -> str:
    return new.replace("\r\n", "\n").replace("\n", "\r\n") if old and "\r\n" in old else new


def git_head(repo: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def default_state() -> dict[str, Any]:
    return {"format": STATE_FORMAT, "night": 0, "last_night_id": None, "last_status": None,
            "accepted_total": 0, "history": []}


def _budget_counts(counts: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("tasks", "rollouts", "tokens", "aiu"):
        v = counts.get(prefix + key)
        if v is not None:
            out[key] = v
    minutes = counts.get(prefix + "minutes")
    if minutes is not None:
        out["wallClockSeconds"] = round(float(minutes) * 60.0, 3)
    return out


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evaluator_pin(night: _Night, run: _TargetRun) -> EvaluatorPin:
    for deps in (run.deps, night.deps):
        fn = deps.evaluator_pin or getattr(deps.assert_eval, "evaluator_pin", None)
        if fn is not None:
            return fn()
    raise ValueError(f"no evaluator pin for target {run.name!r}: the OES envelope of a night without an ASSERT "
                     "gate run needs SleepDeps.evaluator_pin (or assert_eval.evaluator_pin)")


def oes_envelope(night: _Night, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Schema-valid OES sleep envelope (``ci_lab.oes.sleep_envelope``) for the night.

    Records the accepted target (else the first one the ASSERT gate evaluated). When no target
    reached the gate (no SkillOpt candidate, no tasks, budget spent first) it records the first
    target with tasks (else the first target) as a "no change, control retained" night: no eval
    results, the gate's evaluator pin, and ``rerun`` when the budget stopped the night.
    """
    from importlib import metadata

    from ci_lab.oes import sleep_envelope

    gated = [r for r in night.runs if "incumbent" in r.evals and r.decision is not None]
    if gated:
        run = next((r for r in gated if r.status == "accepted"), gated[0])
    else:
        run = next((r for r in night.runs if r.harvest is not None and r.harvest.tasks), night.runs[0])
    decision, res = run.decision, run.consolidation
    try:
        skillopt_version = metadata.version("skillopt")
    except metadata.PackageNotFoundError:  # pragma: no cover
        skillopt_version = "unknown"
    tasks = run.harvest.tasks if run.harvest else []
    by_origin = {k: int(v) for k, v in (run.harvest.sources if run.harvest else {}).items() if v}
    if sum(by_origin.values()) != len(tasks):
        by_origin = {"reviewed": len(tasks)}
    snap = payload["budget"]
    date = night.date
    common: dict[str, Any] = {
        "incumbent_commit": night.base_sha, "skillopt_version": skillopt_version, "tasks_by_origin": by_origin,
        "tasks_by_split": {"evolve": len(tasks)}, "budget_used": _budget_counts(snap["used"]),
        "budget_limits": _budget_counts(snap["limits"], "max_"), "incumbent_digest": _digest(run.skill),
        "night_index": int(payload["night"]), "skill_path": run.target.skill_path,
        "exported_at": str(payload["exported_at"])}
    skillopt = {"mode": str(res.gate_action) if res else "none", "score": res.candidate_score if res else None,
                "baselineScore": res.baseline_score if res else None}
    night_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    if decision is None:
        reasons = [str(night.budget_exceeded)] if night.budget_exceeded is not None else run.reasons()
        return sleep_envelope(
            night_date, incumbent=None, candidate=None, evaluator_pin=_evaluator_pin(night, run),
            skillopt_gate={**skillopt, "passed": False, "reasons": reasons},
            assert_gate={"passed": False, "reasons": reasons}, delta=float(night.cfg.delta_default),
            rerun_reason=str(night.budget_exceeded) if night.budget_exceeded is not None else None, **common)
    accepted = decision.accepted
    return sleep_envelope(
        night_date, incumbent=run.evals["incumbent"], candidate=run.evals.get("candidate"),
        skillopt_gate={**skillopt, "passed": bool(res and res.accepted)},
        assert_gate={"passed": accepted, "reasons": list(decision.reasons), "ciLowerBound": decision.delta_lcb},
        delta=float(decision.delta), candidate_digest=_digest(run.candidate_skill) if accepted else None, **common)

def _consolidation_summary(res: Any) -> dict[str, Any] | None:
    if res is None:
        return None
    return {"accepted": res.accepted, "gate_action": res.gate_action,
            "baseline_score": res.baseline_score, "candidate_score": res.candidate_score,
            "holdout_baseline": res.holdout_baseline, "holdout_candidate": res.holdout_candidate,
            "applied_edits": [asdict(e) for e in res.applied_edits],
            "rejected_edits": [asdict(e) for e in res.rejected_edits],
            "call_error": res.call_error}


def _eval_summary(result: EvalResult) -> dict[str, Any]:
    return {"harness_tree": result.harness_tree, "split": result.split, "pin": asdict(result.pin),
            "scores": [{"case_id": s.case_id, "trial": s.trial, "suite": s.suite, "score": s.score,
                        "violations": [v.rule_id for v in s.violations]} for s in result.scores]}


def _status(night: _Night, **fields: Any) -> None:
    """Live dashboard marker; never fails the night."""
    try:
        obs.write_status(Path(night.cfg.run_dir or night.cfg.out_dir), night.night_id, writer="night",
                         kind="sleep", night=night.night_no, **fields)
    except (OSError, TypeError, ValueError):
        pass


@contextmanager
def _target_span(night: _Night, run: _TargetRun, phase: str, **attrs: Any) -> Iterator[Any]:
    base = {ATTR_PHASE: phase, ATTR_COMPONENT: run.name, ATTR_TARGET: run.name,
            ATTR_EXPERIMENT: night.night_id, ATTR_NIGHT: night.night_no}
    with obs.span(SPAN_STEP, {**base, **attrs}) as s:
        yield s


# ----------------------------------------------------------------- steps

def _step(night: _Night, name: str, fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Errors are captured into the night (MAF swallows tool exceptions). Steps may run on a
    worker thread, so the night's trace context is re-attached explicitly."""

    def run(**kwargs: Any) -> dict[str, Any]:
        night.steps_run.append(name)
        token = otel_context.attach(night.otel_ctx) if night.otel_ctx is not None else None
        try:
            with obs.span(SPAN_STEP, {ATTR_PHASE: name, ATTR_EXPERIMENT: night.night_id,
                                      ATTR_NIGHT: night.night_no}) as s:
                _status(night, phase=name, state="running")
                if night.stopped and name not in ("record", "bundle"):
                    s.set_attribute("ci.skipped", night.status)
                    return {"skipped": night.status}
                try:
                    return fn(**kwargs)
                except BudgetExceeded as exc:
                    night.budget_exceeded = exc
                    if night.status != "error":
                        night.status = "budget_exceeded"
                    return {"budget_exceeded": exc.to_dict()}
                except Exception as exc:  # noqa: BLE001 - recorded, surfaced by run_night
                    night.status = "error"
                    night.error = f"{name}: {type(exc).__name__}: {exc}"
                    night.error_tb = traceback.format_exc(limit=8)
                    s.set_attribute("error.type", type(exc).__name__)
                    return {"error": night.error}
        finally:
            if token is not None:
                otel_context.detach(token)

    run.__name__ = f"sleep_{name}"
    return run


