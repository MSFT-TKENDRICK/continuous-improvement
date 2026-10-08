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
from typing import Any

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
    SPAN_OPTIMIZER,
    SPAN_SLEEP_NIGHT,
    SPAN_STEP,
    EvalResult,
    SafetyOracle,
)
from ci_lab.sleep.backend import OrderSupportSleepBackend, Reflector, RunTarget, Scorer
from ci_lab.sleep.budget import Budget, BudgetExceeded, BudgetLimits
from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle
from ci_lab.sleep.gate import CanaryResult, GateDecision, decide, static_canaries
from ci_lab.sleep.harvest import HarvestResult, from_agl_exports, harvest, reviewed_ids
from ci_lab.sleep.registry import ORDER_SUPPORT, SkillTarget, load_targets, validate_targets
from ci_lab.sleep.runner import RunWorkflow, default_runner

WORKFLOW_PATH = Path(__file__).with_name("sleep.yaml")
SKILL_REL = ORDER_SUPPORT.skill_path
MEMORY_REL = ORDER_SUPPORT.memory_path
STATE_REL = "experiments/sleep/state.json"
ENVELOPES_REL = "experiments/sleep/envelopes"
TASKS_REL = ORDER_SUPPORT.tasks_file
STATE_FORMAT = "ci_lab.sleep.state.v1"
STEPS = ("harvest", "consolidate", "assert_gate", "record", "bundle")
HISTORY_KEEP = 60
ATTR_TARGET = "sleep.target"  # not (yet) a contracts constant


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
    lessons_hook: bool = False  # HOOK(M16): lessons mine/replay seam, off by default

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
    # HOOK(M16): (target, typed TaskRecords) -> typed TaskRecords; used only when cfg.lessons_hook
    lessons: Callable[[SkillTarget, list[TaskRecord]], Iterable[TaskRecord]] | None = None

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


def default_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal OES 0.1.0 envelope (used when ``ci_lab.oes`` provides no sleep builder)."""
    outcome = {"accepted": "ship", "budget_exceeded": "rerun"}.get(payload["status"], "do_not_ship")
    return {
        "schemaVersion": "0.1.0", "objectType": "experiment", "exportedAt": payload["exported_at"],
        "sourceSystem": "ci-lab",
        "experiment": {"id": payload["night_id"], "slug": payload["night_id"], "status": "decided",
                       "title": f"SkillOpt-Sleep night {payload['night']} ({payload['date']})"},
        "decision": {"status": "decided", "outcome": outcome, "rationale": "; ".join(payload["reasons"])},
        "provenance": {"baseCommit": payload["base_sha"], "profile": payload["profile"]},
        "extensions": {"com.microsoft.ci.sleep": dict(payload)},
    }


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
            run.backend = OrderSupportSleepBackend(run_target=d.run_target, oracle=d.oracle,
                                                   reflector=d.reflector, scorer=d.scorer, budget=night.budget)
            tasks: list[TaskRecord] = run.harvest.tasks
            # HOOK(M16): lessons mine/replay step (design §13/§13.6, ci_lab.lessons on dev/lessons).
            # Seam between trajectory collection (harvest) and proposal (dream_consolidate/reflect).
            # Gated OFF by default (cfg.lessons_hook); not implemented here. Inputs/outputs must stay
            # local and typed-only (§13.6 B8): typed TaskRecords in, typed TaskRecords out; nothing
            # from reflect/usage data is written to the bundle, envelope, spans or status.
            if cfg.lessons_hook and d.lessons is not None:
                tasks = list(d.lessons(run.target, tasks))
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
        out[run.name] = {"accepted": bool(res.accepted), "gate_action": str(res.gate_action)}
    _status(night, skillopt={k: v["accepted"] for k, v in out.items()})
    return out


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
    builder = deps.build_envelope or default_envelope
    night.envelope = dict(builder(payload))
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
                             "reflect_log": r.backend.reflect_log if r.backend else []}
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
            raise ValueError("state.json night counter must be an int")
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
    runner = deps.run_workflow or default_runner()
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
