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
from ci_lab.sleep.backend import Reflector, RunTarget, Scorer, SleepBackend
from ci_lab.sleep.budget import Budget, BudgetExceeded, BudgetLimits
from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle
from ci_lab.sleep.gate import CanaryResult, GateDecision, decide, static_canaries
from ci_lab.sleep.harvest import HarvestResult, from_agl_exports, harvest, reviewed_ids
from ci_lab.sleep.registry import (
    HARNESS_EDITING,
    SkillTarget,
    load_targets,
    validate_targets,
)
from ci_lab.sleep.runner import RunWorkflow, default_runner

if TYPE_CHECKING:
    from ci_lab.sleep.lessons_hook import LessonsReport, LessonsRequest

WORKFLOW_PATH = Path(__file__).with_name("sleep.yaml")
SKILL_REL = HARNESS_EDITING.skill_path
MEMORY_REL = HARNESS_EDITING.memory_path
STATE_REL = "experiments/sleep/state.json"
ENVELOPES_REL = "experiments/sleep/envelopes"
TASKS_REL = HARNESS_EDITING.tasks_file
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
    backend: SleepBackend | None = None
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


def _row_target(row: Mapping[str, Any]) -> str | None:
    for key in ("target", "skill_target"):
        if row.get(key):
            return str(row[key])
    body = row.get("task") if isinstance(row.get("task"), Mapping) else row
    return str(body["project"]) if body.get("project") else None


def _route_rows(rows: list[Mapping[str, Any]], runs: list[_TargetRun]) -> dict[str, list[Mapping[str, Any]]]:
    """AGL rows go to the target named by ``target``/``project`` (unlabelled -> first target).
    Rows for unknown targets are still split-checked (C15), then dropped."""
    routed: dict[str, list[Mapping[str, Any]]] = {r.name: [] for r in runs}
    alias = {key: r.name for r in reversed(runs) for key in (r.target.owner_agent, r.name)}
    unknown = []
    for row in rows:
        label = _row_target(row) if isinstance(row, Mapping) else None
        name = runs[0].name if label is None else alias.get(label)
        (routed[name] if name else unknown).append(row)
    if unknown:
        from_agl_exports(unknown)
    return routed


def _harvest(night: _Night, split: str) -> dict[str, Any]:
    if split != "evolve":
        raise ValueError(f"sleep harvest split must be 'evolve' (got {split!r}; C15)")
    cfg = night.cfg
    rows = list(night.deps.agl_records()) if night.deps.agl_records else []
    routed = _route_rows(rows, night.runs)
    remaining = cfg.limits.max_tasks
    counts: dict[str, int] = {}
    for run in night.runs:
        with _target_span(night, run, "harvest", **{ATTR_SPLIT: split}) as s:
            extra = []
            if run.deps is not night.deps and run.deps.agl_records is not None:
                extra = list(run.deps.agl_records())
            path = cfg.tasks_path(run.target)
            run.reviewed_ids = reviewed_ids(path)
            if remaining <= 0:
                run.no_tasks = "task budget exhausted by earlier targets"
                counts[run.name] = 0
                continue
            res = harvest(path, [*routed[run.name], *extra], max_tasks=remaining, val_fraction=cfg.val_fraction)
            run.harvest = res
            remaining -= len(res.tasks)
            counts[run.name] = len(res.tasks)
            s.set_attribute("sleep.n_tasks", len(res.tasks))
            if not res.tasks or not {"train", "val"} <= {t.split for t in res.tasks}:
                run.no_tasks = "no reviewed/evolve tasks with both train and val splits"
    night.budget.admit_tasks(sum(counts.values()))
    if all(r.no_tasks for r in night.runs):
        night.status = "no_tasks"
    _status(night, tasks=counts)
    return {"n_tasks": sum(counts.values()), "targets": counts}


def _consolidate(night: _Night, gate_mode: str) -> dict[str, Any]:
    if gate_mode != "on":
        raise ValueError("SkillOpt gate must stay on for nightly sleep")
    cfg = night.cfg
    out: dict[str, Any] = {}
    for run in night.runs:
        if run.no_tasks:
            continue
        assert run.harvest is not None
        d = run.deps
        with _target_span(night, run, "consolidate"):
            run.backend = SleepBackend(run_target=d.run_target, oracle=d.oracle,
                                       reflector=d.reflector, scorer=d.scorer, budget=night.budget)
            tasks: list[TaskRecord] = run.harvest.tasks
            with obs.span(SPAN_OPTIMIZER, {ATTR_STRATEGY: "skillopt", ATTR_COMPONENT: run.name,
                                           ATTR_TARGET: run.name, ATTR_NIGHT: night.night_no}) as s:
                res = dream_consolidate(
                    run.backend, tasks, run.skill, run.memory,
                    history_tasks=None, recall_k=cfg.recall_k, dream_rollouts=cfg.dream_rollouts,
                    dream_factor=cfg.dream_factor, edit_budget=cfg.edit_budget, gate_metric=cfg.gate_metric,
                    gate_mixed_weight=cfg.gate_mixed_weight, gate_mode=gate_mode, evolve_skill=True,
                    evolve_memory=cfg.evolve_memory, night=night.night_no)
                s.set_attribute("skillopt.accepted", bool(res.accepted))
                s.set_attribute("skillopt.gate_action", res.gate_action)
        run.consolidation = res
        run.candidate_skill, run.candidate_memory = res.new_skill, res.new_memory
        if cfg.lessons_hook:
            _lessons(night, run)
        out[run.name] = {"accepted": bool(res.accepted), "gate_action": str(res.gate_action)}
    _status(night, skillopt={k: v["accepted"] for k, v in out.items()})
    return out


def _lessons(night: _Night, run: _TargetRun) -> None:
    """HOOK(M16): mine lesson candidates from this target's judged rollouts (design §13.3).

    Runs after ``dream_consolidate`` because the trajectories are the rollouts it judged. The hook
    keeps raw trajectories in a local store (B8) and returns only sanitized typed candidates, which
    ``_record`` proposes in the draft-PR bundle. Nothing is adopted or enforced here. Failures are
    soft: the error type is counted and the night continues."""
    from ci_lab.sleep.lessons_hook import LessonsReport, LessonsRequest

    hook = run.deps.lessons or night.deps.lessons
    if hook is None or run.backend is None:
        return
    with _target_span(night, run, "lessons") as s:
        req = LessonsRequest(target=run.name, night_id=night.night_id, date=night.date,
                             profile=night.cfg.profile, rollouts=tuple(run.backend.judged_rollouts()))
        try:
            report = hook(req)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - proposals must never fail the night
            s.set_attribute("error.type", type(exc).__name__)
            report = LessonsReport(counts={"error": type(exc).__name__})
        run.lessons = report
        s.set_attribute("ci.lessons.candidates", len(report.candidates))


def _gate_target(night: _Night, run: _TargetRun, suite_split: str) -> None:
    res = run.consolidation
    changed = res is not None and (run.candidate_skill != run.skill or run.candidate_memory != run.memory)
    if res is None or not res.accepted or not changed:
        run.gate_skipped = "SkillOpt pre-filter produced no candidate"
        return
    night.budget.check_time()
    deps = run.deps
    inc = deps.assert_eval(run.skill, run.memory, "incumbent")
    cand = deps.assert_eval(run.candidate_skill, run.candidate_memory, "candidate")
    for r in (inc, cand):
        if r.split != suite_split:
            raise ValueError(f"ASSERT gate for {run.name} ran on split {r.split!r}, expected {suite_split!r}")
    run.evals = {"incumbent": inc, "candidate": cand}
    canaries = static_canaries(run.skill, run.candidate_skill, run.memory, run.candidate_memory)
    if deps.run_canaries is not None:
        canaries += list(deps.run_canaries(run.candidate_skill, run.candidate_memory))
    delta = deps.latest_delta() if deps.latest_delta else None
    cfg = night.cfg
    run.decision = decide(inc, cand, delta=cfg.delta_default if delta is None else float(delta),
                          canaries=canaries, alpha=cfg.alpha, n_boot=cfg.n_boot, seed=cfg.seed)


def _assert_gate(night: _Night, suite_split: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in night.runs:
        if run.no_tasks:
            continue
        with _target_span(night, run, "assert_gate", **{ATTR_SPLIT: suite_split,
                                                        "sleep.eval_suite": run.target.eval_suite}) as s:
            _gate_target(night, run, suite_split)
            s.set_attribute(ATTR_DECISION, run.status)
        out[run.name] = {"evaluated": run.decision is not None, "accepted": run.status == "accepted"}
    _status(night, gate={k: v["accepted"] for k, v in out.items()})
    return out


def _final_status(night: _Night) -> str:
    if night.status in ("error", "no_tasks", "budget_exceeded"):
        return night.status
    return "accepted" if any(r.status == "accepted" for r in night.runs) else "rejected"


def _reasons(night: _Night) -> list[str]:
    if night.status == "error":
        return [night.error]
    if night.budget_exceeded is not None:
        return [str(night.budget_exceeded)]
    return [f"{r.name}: {reason}" for r in night.runs for reason in r.reasons()]


def _target_summary(run: _TargetRun) -> dict[str, Any]:
    return {**run.target.to_dict(), "status": run.status, "reasons": run.reasons(),
            "harvest": run.harvest.summary() if run.harvest else None,
            "skillopt": _consolidation_summary(run.consolidation),
            "gate": run.decision.to_dict() if run.decision else None}


def _record(night: _Night, envelope_kind: str) -> dict[str, Any]:
    if envelope_kind != "sleep":
        raise ValueError("unknown envelope kind")
    if night.status == "error":
        return {"skipped": "error"}
    cfg, deps = night.cfg, night.deps
    status = _final_status(night)
    accepted = status == "accepted"
    night_no = night.night_no
    exported_at = deps.clock().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, Any] = {
        "night_id": night.night_id, "night": night_no, "date": night.date, "status": status,
        "accepted": accepted, "reasons": _reasons(night), "base_sha": night.base_sha, "profile": cfg.profile,
        "exported_at": exported_at, "budget": night.budget.snapshot(),
        "targets": {r.name: _target_summary(r) for r in night.runs},
    }
    if deps.build_envelope is not None:
        night.envelope = dict(deps.build_envelope(payload))
    else:
        night.envelope = oes_envelope(night, payload)
    from ci_lab.oes import validate_envelope

    if errors := validate_envelope(night.envelope):
        raise ValueError(f"invalid OES envelope ({len(errors)} errors): {'; '.join(errors[:3])}")
    state = dict(night.state)
    history = list(state.get("history") or [])
    history.append({"night_id": night.night_id, "night": night_no, "status": status,
                    "targets": {r.name: {"status": r.status,
                                         "delta_lcb": r.decision.delta_lcb if r.decision else None,
                                         "n_tasks": len(r.harvest.tasks) if r.harvest else 0}
                                for r in night.runs}})
    per_target = dict(state.get("targets") or {})
    for r in night.runs:
        prev = dict(per_target.get(r.name) or {})
        per_target[r.name] = {"last_status": r.status,
                              "accepted_total": int(prev.get("accepted_total", 0)) + (r.status == "accepted")}
    state.update({"format": STATE_FORMAT, "night": night_no, "last_night_id": night.night_id,
                  "last_date": night.date, "last_status": status, "last_base_sha": night.base_sha,
                  "accepted_total": int(state.get("accepted_total", 0)) + (1 if accepted else 0),
                  "targets": per_target,
                  # usage watermark: the nightly gate counts reviewed tasks added after this night
                  "watermark": {"at": exported_at, "task_ids": {r.name: r.reviewed_ids for r in night.runs}},
                  "history": history[-HISTORY_KEEP:]})
    repo = cfg.repo_root
    state_new = json.dumps(state, indent=2, sort_keys=True) + "\n"
    env_new = json.dumps(night.envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    env_rel = f"{ENVELOPES_REL}/{night.night_id}.json"
    env_old = _read(repo / env_rel)
    changes = [FileChange(STATE_REL, night.state_text, _match_eol(night.state_text, state_new)),
               FileChange(env_rel, env_old, _match_eol(env_old, env_new))]
    for r in night.runs:
        if r.status != "accepted":
            continue
        changes.append(FileChange(r.target.skill_path, r.skill, _match_eol(r.skill, r.candidate_skill)))
        if r.candidate_memory != r.memory:
            if not r.target.memory_path:
                raise ValueError(f"target {r.name} evolved memory but has no memory path")
            changes.append(FileChange(r.target.memory_path, r.memory if r.memory_exists else None,
                                      _match_eol(r.memory, r.candidate_memory)))
    lessons = {r.name: r.lessons for r in night.runs if r.lessons is not None}
    if lessons:
        from ci_lab.sleep.lessons_hook import LESSONS_REL, proposal_document

        doc = proposal_document(night.night_id, lessons)
        if doc is not None:
            rel = f"{LESSONS_REL}/{night.night_id}.json"
            old = _read(repo / rel)
            new = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            changes.append(FileChange(rel, old, _match_eol(old, new)))
    night.changes = changes
    return {"status": status, "night": night_no}


def _bundle(night: _Night, layout: str) -> dict[str, Any]:
    if layout != "v1":
        raise ValueError("unknown bundle layout")
    cfg = night.cfg
    status = _final_status(night)
    patch = make_patch(night.changes) if status != "error" else ""
    ledger_update = bool(patch) and status != "error"
    experiment = {
        "night_id": night.night_id, "night": night.night_no, "date": night.date, "profile": cfg.profile,
        "base_sha": night.base_sha,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
                   if k not in ("repo_root", "out_dir", "work_dir", "run_dir", "tasks_file", "targets")},
        "targets": {r.name: {**r.target.to_dict(),
                             "harvest": r.harvest.summary() if r.harvest else None,
                             "skillopt": _consolidation_summary(r.consolidation),
                             "reflect_log": r.backend.reflect_log if r.backend else [],
                             "lessons": dict(r.lessons.counts) if r.lessons else None}
                    for r in night.runs},
        "budget": night.budget.snapshot(),
        "steps": list(night.steps_run),
        "envelope": night.envelope,
    }
    results = {
        "status": status, "accepted": status == "accepted", "reasons": _reasons(night),
        "targets": {r.name: {"status": r.status, "reasons": r.reasons(),
                             "gate": r.decision.to_dict() if r.decision else None,
                             "evals": {k: _eval_summary(v) for k, v in r.evals.items()}}
                    for r in night.runs},
        "error": night.error or None,
        "changed_files": sorted(c.path for c in night.changes if c.old != c.new) if patch else [],
    }
    night.manifest = write_bundle(cfg.out_dir, patch=patch, experiment=experiment, results=results,
                                  base_sha=night.base_sha, night_id=night.night_id, date=night.date,
                                  accepted=status == "accepted", ledger_update=ledger_update, status=status)
    return {"status": status}


# ----------------------------------------------------------------- entry point

def _setup(night: _Night) -> None:
    cfg, deps = night.cfg, night.deps
    try:
        night.state_text = _read(cfg.repo_root / STATE_REL)
        night.state = json.loads(night.state_text) if night.state_text else default_state()
        if not isinstance(night.state.get("night", 0), int):
            raise TypeError("state.json night counter must be an int")
        assert cfg.targets is not None
        for target in cfg.targets:
            tdeps = deps.per_target(target) if deps.per_target is not None else deps
            run = _TargetRun(target=target, deps=tdeps)
            skill = _read(cfg.repo_root / target.skill_path)
            if skill is None:
                raise FileNotFoundError(f"incumbent skill not found: {target.skill_path}")
            run.skill = skill
            mem = _read(cfg.repo_root / target.memory_path) if target.memory_path else None
            run.memory, run.memory_exists = mem or "", mem is not None
            night.runs.append(run)
    except Exception as exc:  # noqa: BLE001
        night.status, night.error = "error", f"setup: {type(exc).__name__}: {exc}"
        night.runs = []


def run_night(cfg: SleepConfig, deps: SleepDeps) -> NightResult:
    date = cfg.night_date or deps.clock().strftime("%Y%m%d")
    base_sha = cfg.base_sha or (deps.git_head or git_head)(cfg.repo_root)
    if not re.fullmatch(r"[0-9a-f]{40}", base_sha or ""):
        raise ValueError(f"base sha must be 40 hex chars, got {base_sha!r}")
    night = _Night(cfg=cfg, deps=deps, budget=Budget(cfg.limits, clock=deps.monotonic),
                   night_id=f"sleep-{date}-{int(cfg.run_attempt)}", date=date, base_sha=base_sha)
    _setup(night)

    harvest_step = _step(night, "harvest", lambda split: _harvest(night, split))
    consolidate_step = _step(night, "consolidate", lambda gate_mode: _consolidate(night, gate_mode))
    gate_step = _step(night, "assert_gate", lambda suite_split: _assert_gate(night, suite_split))
    record_step = _step(night, "record", lambda envelope_kind: _record(night, envelope_kind))
    bundle_step = _step(night, "bundle", lambda layout: _bundle(night, layout))

    # Explicit keyword signatures: MAF binds the YAML's literal arguments by parameter name.
    def sleep_harvest(split: str) -> dict:
        return harvest_step(split=split)

    def sleep_consolidate(gate_mode: str) -> dict:
        return consolidate_step(gate_mode=gate_mode)

    def sleep_assert_gate(suite_split: str) -> dict:
        return gate_step(suite_split=suite_split)

    def sleep_record(envelope_kind: str) -> dict:
        return record_step(envelope_kind=envelope_kind)

    def sleep_bundle(layout: str) -> dict:
        return bundle_step(layout=layout)

    tools = {f.__name__: f for f in (sleep_harvest, sleep_consolidate, sleep_assert_gate,
                                     sleep_record, sleep_bundle)}
    runner = deps.run_workflow or default_runner(cfg.profile)
    assert cfg.work_dir is not None
    ckpt = Path(cfg.work_dir) / "checkpoints" / night.night_id
    attrs = {ATTR_EXPERIMENT: night.night_id, ATTR_NIGHT: night.night_no, ATTR_PROFILE: cfg.profile,
             "sleep.date": date, "sleep.targets": ",".join(r.name for r in night.runs)}
    # one trace per night (C27); re-running a night links to the interrupted trace
    links = obs.previous_link(Path(cfg.run_dir or cfg.out_dir), night.night_id)
    with obs.span(SPAN_SLEEP_NIGHT, attrs, links=links or None, new_trace=True) as root:
        night.otel_ctx = otel_context.get_current()
        _status(night, phase="start", state="running", profile=cfg.profile, base_sha=base_sha,
                targets=[r.name for r in night.runs])
        try:
            runner(WORKFLOW_PATH, tools, ckpt)
        except Exception as exc:  # noqa: BLE001 - runner failure is a night error
            if night.status != "error":
                night.status, night.error = "error", f"workflow: {type(exc).__name__}: {exc}"
        if tuple(night.steps_run) != STEPS and night.status != "error":
            night.status, night.error = "error", f"workflow ran steps {night.steps_run}, expected {list(STEPS)}"
        if night.manifest is None or night.status == "error":
            night.changes = []
            _bundle(night, "v1")
        assert night.manifest is not None
        status = _final_status(night)
        root.set_attribute(ATTR_DECISION, status)
        if night.error:
            root.set_attribute("error.type", night.error.split(":", 1)[0])
        _status(night, phase="done", state=status, accepted=status == "accepted",
                ledger_update=bool(night.manifest["ledger_update"]),
                target_status={r.name: r.status for r in night.runs})
    return NightResult(status=status, accepted=status == "accepted",
                       ledger_update=bool(night.manifest["ledger_update"]), bundle_dir=cfg.out_dir,
                       manifest=night.manifest, night_id=night.night_id, error=night.error,
                       targets={r.name: r.status for r in night.runs},
                       decisions={r.name: r.decision.to_dict() for r in night.runs if r.decision})
