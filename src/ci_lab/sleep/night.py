"""One SkillOpt-Sleep night: ``sleep.yaml`` = harvest -> consolidate -> assert_gate -> record -> bundle.

Nothing in the checkout is modified (C10): the candidate skill, the advanced
``experiments/sleep/state.json`` and the night's OES envelope only leave as a patch inside
the digest-pinned bundle, which the privileged ``publish`` job applies to a PR branch.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.types import TaskRecord

from ci_lab.contracts import EvalResult, SafetyOracle
from ci_lab.sleep.backend import OrderSupportSleepBackend, Reflector, RunTarget, Scorer
from ci_lab.sleep.budget import Budget, BudgetExceeded, BudgetLimits
from ci_lab.sleep.bundle import FileChange, make_patch, write_bundle
from ci_lab.sleep.gate import CanaryResult, GateDecision, decide, static_canaries
from ci_lab.sleep.harvest import HarvestResult, harvest
from ci_lab.sleep.runner import RunWorkflow, default_runner

WORKFLOW_PATH = Path(__file__).with_name("sleep.yaml")
SKILL_REL = "src/order_support/harness/skills/order-support/SKILL.md"
MEMORY_REL = "src/order_support/harness/skills/order-support/memory.md"
STATE_REL = "experiments/sleep/state.json"
ENVELOPES_REL = "experiments/sleep/envelopes"
TASKS_REL = "experiments/sleep/tasks.jsonl"
STATE_FORMAT = "ci_lab.sleep.state.v1"
STEPS = ("harvest", "consolidate", "assert_gate", "record", "bundle")
HISTORY_KEEP = 60


@dataclass
class SleepConfig:
    repo_root: Path
    out_dir: Path
    profile: str = "fake"
    tasks_file: Path | None = None
    work_dir: Path | None = None  # checkpoints + temp harness copies (never inside out_dir)
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

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.out_dir = Path(self.out_dir)
        if self.tasks_file is None:
            self.tasks_file = self.repo_root / TASKS_REL
        if self.work_dir is None:
            self.work_dir = self.out_dir.parent / f"{self.out_dir.name}-work"
        if self.night_date is not None and not re.fullmatch(r"\d{8}", self.night_date):
            raise ValueError("night_date must be yyyymmdd")
        if not 1 <= int(self.run_attempt) <= 9999:
            raise ValueError("run_attempt out of range")


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


@dataclass
class NightResult:
    status: str  # accepted | rejected | budget_exceeded | no_tasks | error
    accepted: bool
    ledger_update: bool
    bundle_dir: Path
    manifest: dict[str, Any]
    night_id: str
    error: str = ""
    decision: dict[str, Any] | None = None


@dataclass
class _Night:
    cfg: SleepConfig
    deps: SleepDeps
    budget: Budget
    night_id: str
    date: str
    base_sha: str
    skill: str = ""
    memory: str = ""
    memory_exists: bool = False
    state_text: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    harvest: HarvestResult | None = None
    backend: OrderSupportSleepBackend | None = None
    consolidation: Any = None
    candidate_skill: str = ""
    candidate_memory: str = ""
    decision: GateDecision | None = None
    gate_skipped: str = ""
    evals: dict[str, EvalResult] = field(default_factory=dict)
    envelope: dict[str, Any] | None = None
    changes: list[FileChange] = field(default_factory=list)
    status: str = "running"
    error: str = ""
    error_tb: str = ""
    budget_exceeded: BudgetExceeded | None = None
    steps_run: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

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


# ----------------------------------------------------------------- steps

def _step(night: _Night, name: str, fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Errors are captured into the night (MAF swallows tool exceptions)."""

    def run(**kwargs: Any) -> dict[str, Any]:
        night.steps_run.append(name)
        if night.stopped and name not in ("record", "bundle"):
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
            return {"error": night.error}

    run.__name__ = f"sleep_{name}"
    return run


def _harvest(night: _Night, split: str) -> dict[str, Any]:
    if split != "evolve":
        raise ValueError(f"sleep harvest split must be 'evolve' (got {split!r}; C15)")
    cfg = night.cfg
    rows = list(night.deps.agl_records()) if night.deps.agl_records else []
    res = harvest(cfg.tasks_file, rows, max_tasks=cfg.limits.max_tasks, val_fraction=cfg.val_fraction)
    night.budget.admit_tasks(len(res.tasks))
    night.harvest = res
    splits = {t.split for t in res.tasks}
    if not res.tasks or not {"train", "val"} <= splits:
        night.status = "no_tasks"
    return {"n_tasks": len(res.tasks)}


def _consolidate(night: _Night, gate_mode: str) -> dict[str, Any]:
    if gate_mode != "on":
        raise ValueError("SkillOpt gate must stay on for nightly sleep")
    cfg, deps = night.cfg, night.deps
    assert night.harvest is not None
    night.backend = OrderSupportSleepBackend(run_target=deps.run_target, oracle=deps.oracle,
                                             reflector=deps.reflector, scorer=deps.scorer, budget=night.budget)
    tasks: list[TaskRecord] = night.harvest.tasks
    res = dream_consolidate(
        night.backend, tasks, night.skill, night.memory,
        history_tasks=None, recall_k=cfg.recall_k, dream_rollouts=cfg.dream_rollouts,
        dream_factor=cfg.dream_factor, edit_budget=cfg.edit_budget, gate_metric=cfg.gate_metric,
        gate_mixed_weight=cfg.gate_mixed_weight, gate_mode=gate_mode, evolve_skill=True,
        evolve_memory=cfg.evolve_memory, night=int(night.state.get("night", 0)) + 1)
    night.consolidation = res
    night.candidate_skill, night.candidate_memory = res.new_skill, res.new_memory
    return {"accepted": bool(res.accepted), "gate_action": str(res.gate_action)}


def _assert_gate(night: _Night, suite_split: str) -> dict[str, Any]:
    res = night.consolidation
    changed = res is not None and (night.candidate_skill != night.skill or night.candidate_memory != night.memory)
    if res is None or not res.accepted or not changed:
        night.gate_skipped = "SkillOpt pre-filter produced no candidate"
        return {"evaluated": False}
    night.budget.check_time()
    deps = night.deps
    inc = deps.assert_eval(night.skill, night.memory, "incumbent")
    cand = deps.assert_eval(night.candidate_skill, night.candidate_memory, "candidate")
    for r in (inc, cand):
        if r.split != suite_split:
            raise ValueError(f"ASSERT gate ran on split {r.split!r}, expected {suite_split!r}")
    night.evals = {"incumbent": inc, "candidate": cand}
    canaries = static_canaries(night.skill, night.candidate_skill, night.memory, night.candidate_memory)
    if deps.run_canaries is not None:
        canaries += list(deps.run_canaries(night.candidate_skill, night.candidate_memory))
    delta = deps.latest_delta() if deps.latest_delta else None
    cfg = night.cfg
    night.decision = decide(inc, cand, delta=cfg.delta_default if delta is None else float(delta),
                            canaries=canaries, alpha=cfg.alpha, n_boot=cfg.n_boot, seed=cfg.seed)
    return {"evaluated": True, "accepted": night.decision.accepted}


def _final_status(night: _Night) -> str:
    if night.status in ("error", "no_tasks", "budget_exceeded"):
        return night.status
    return "accepted" if night.decision is not None and night.decision.accepted else "rejected"


def _reasons(night: _Night) -> list[str]:
    if night.status == "error":
        return [night.error]
    if night.budget_exceeded is not None:
        return [str(night.budget_exceeded)]
    if night.status == "no_tasks":
        return ["no reviewed/evolve tasks with both train and val splits"]
    if night.decision is not None:
        return list(night.decision.reasons)
    return [night.gate_skipped or "no candidate"]


def _record(night: _Night, envelope_kind: str) -> dict[str, Any]:
    if envelope_kind != "sleep":
        raise ValueError("unknown envelope kind")
    if night.status == "error":
        return {"skipped": "error"}
    cfg, deps = night.cfg, night.deps
    status = _final_status(night)
    accepted = status == "accepted"
    night_no = int(night.state.get("night", 0)) + 1
    d = night.decision
    payload: dict[str, Any] = {
        "night_id": night.night_id, "night": night_no, "date": night.date, "status": status,
        "accepted": accepted, "reasons": _reasons(night), "base_sha": night.base_sha, "profile": cfg.profile,
        "exported_at": deps.clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harvest": night.harvest.summary() if night.harvest else None,
        "skillopt": _consolidation_summary(night.consolidation),
        "gate": d.to_dict() if d else None,
        "budget": night.budget.snapshot(),
    }
    builder = deps.build_envelope or default_envelope
    night.envelope = dict(builder(payload))
    state = dict(night.state)
    history = list(state.get("history") or [])
    history.append({"night_id": night.night_id, "night": night_no, "status": status,
                    "delta_lcb": d.delta_lcb if d else None, "n_tasks": len(night.harvest.tasks) if night.harvest else 0})
    state.update({"format": STATE_FORMAT, "night": night_no, "last_night_id": night.night_id,
                  "last_date": night.date, "last_status": status, "last_base_sha": night.base_sha,
                  "accepted_total": int(state.get("accepted_total", 0)) + (1 if accepted else 0),
                  "history": history[-HISTORY_KEEP:]})
    repo = cfg.repo_root
    state_new = json.dumps(state, indent=2, sort_keys=True) + "\n"
    env_new = json.dumps(night.envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    env_rel = f"{ENVELOPES_REL}/{night.night_id}.json"
    env_old = _read(repo / env_rel)
    changes = [FileChange(STATE_REL, night.state_text, _match_eol(night.state_text, state_new)),
               FileChange(env_rel, env_old, _match_eol(env_old, env_new))]
    if accepted:
        changes.append(FileChange(SKILL_REL, night.skill, _match_eol(night.skill, night.candidate_skill)))
        if night.candidate_memory != night.memory:
            changes.append(FileChange(MEMORY_REL, night.memory if night.memory_exists else None,
                                      _match_eol(night.memory, night.candidate_memory)))
    night.changes = changes
    return {"status": status, "night": night_no}


def _bundle(night: _Night, layout: str) -> dict[str, Any]:
    if layout != "v1":
        raise ValueError("unknown bundle layout")
    cfg = night.cfg
    status = _final_status(night)
    patch = make_patch(night.changes) if status != "error" else ""
    ledger_update = bool(patch) and status != "error"
    backend = night.backend
    experiment = {
        "night_id": night.night_id, "date": night.date, "profile": cfg.profile, "base_sha": night.base_sha,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
                   if k not in ("repo_root", "out_dir", "work_dir", "tasks_file")},
        "harvest": night.harvest.summary() if night.harvest else None,
        "skillopt": _consolidation_summary(night.consolidation),
        "reflect_log": backend.reflect_log if backend else [],
        "budget": night.budget.snapshot(),
        "steps": list(night.steps_run),
        "envelope": night.envelope,
    }
    results = {
        "status": status, "accepted": status == "accepted", "reasons": _reasons(night),
        "gate": night.decision.to_dict() if night.decision else None,
        "evals": {k: _eval_summary(v) for k, v in night.evals.items()},
        "error": night.error or None,
        "changed_files": sorted(c.path for c in night.changes if c.old != c.new) if patch else [],
    }
    night.manifest = write_bundle(cfg.out_dir, patch=patch, experiment=experiment, results=results,
                                  base_sha=night.base_sha, night_id=night.night_id, date=night.date,
                                  accepted=status == "accepted", ledger_update=ledger_update, status=status)
    return {"status": status}


# ----------------------------------------------------------------- entry point

def run_night(cfg: SleepConfig, deps: SleepDeps) -> NightResult:
    date = cfg.night_date or deps.clock().strftime("%Y%m%d")
    base_sha = cfg.base_sha or (deps.git_head or git_head)(cfg.repo_root)
    if not re.fullmatch(r"[0-9a-f]{40}", base_sha or ""):
        raise ValueError(f"base sha must be 40 hex chars, got {base_sha!r}")
    night = _Night(cfg=cfg, deps=deps, budget=Budget(cfg.limits, clock=deps.monotonic),
                   night_id=f"sleep-{date}-{int(cfg.run_attempt)}", date=date, base_sha=base_sha)
    try:
        skill = _read(cfg.repo_root / SKILL_REL)
        if skill is None:
            raise FileNotFoundError(f"incumbent skill not found: {SKILL_REL}")
        night.skill = skill
        mem = _read(cfg.repo_root / MEMORY_REL)
        night.memory, night.memory_exists = mem or "", mem is not None
        night.state_text = _read(cfg.repo_root / STATE_REL)
        night.state = json.loads(night.state_text) if night.state_text else default_state()
        if not isinstance(night.state.get("night", 0), int):
            raise ValueError("state.json night counter must be an int")
    except Exception as exc:  # noqa: BLE001
        night.status, night.error = "error", f"setup: {type(exc).__name__}: {exc}"

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
    return NightResult(status=status, accepted=status == "accepted",
                       ledger_update=bool(night.manifest["ledger_update"]), bundle_dir=cfg.out_dir,
                       manifest=night.manifest, night_id=night.night_id, error=night.error,
                       decision=night.decision.to_dict() if night.decision else None)
