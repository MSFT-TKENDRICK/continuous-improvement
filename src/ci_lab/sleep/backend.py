"""``SleepBackend``: the skillopt_sleep ``Backend`` protocol implemented directly
over the configured harness agent (design C3 — no CliBackend, no Claude/Codex CLIs).

* ``attempt`` / ``attempt_with_tools`` run the real target agent (real tool loop) with the
  candidate skill injected, through an injected ``run_target(task, skill, memory)``.
* ``judge`` = deterministic safety oracle + validated rule checks (against the tools that
  were REALLY called, never a ``TOOL_CALL:`` text marker) + an optional injected
  ASSERT-derived scorer. This is diagnostic signal for SkillOpt only; acceptance is the
  separate ASSERT gate (``gate.py``).
* ``reflect`` hands ONLY typed :class:`~ci_lab.contracts.FailureRecord` s (C12) to the
  injected SleepReflector and validates the typed edits it returns.
* Unknown rule-judge ops raise :class:`~ci_lab.sleep.harvest.UnknownJudgeOp`.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from skillopt_sleep.backend import Backend
from skillopt_sleep.memory import LEARNED_END, LEARNED_START, current_learned_lines
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

from ci_lab.contracts import FailureRecord, SafetyOracle, Transcript, Violation
from ci_lab.sleep.budget import Budget
from ci_lab.sleep.harvest import INJECTION_MARKERS, KNOWN_OPS, UnknownJudgeOp

RunTarget = Callable[[TaskRecord, str, str], "tuple[str, Sequence[str], Transcript]"]
Scorer = Callable[[TaskRecord, Transcript], "float | None"]

EXCERPT_CHARS = 280
MAX_EDIT_CHARS = 400
SCORER_PASS = 0.5


@dataclass(frozen=True)
class ReflectRequest:
    """Everything the SleepReflector may see. No transcripts, no tool output (C12)."""

    failures: tuple[FailureRecord, ...]
    n_successes: int
    target: str  # "skill" | "memory"
    edit_budget: int
    learned: tuple[str, ...] = ()  # current bullets of the SkillOpt learned block (anchors)


@dataclass
class ReflectResult:
    edits: list[EditRecord]
    tokens: int = 0
    aiu: float = 0.0
    raw: str = ""


Reflector = Callable[[ReflectRequest], "ReflectResult | Sequence[EditRecord]"]


@dataclass(frozen=True)
class RolloutView:
    """One judged rollout, exported for the HOOK(M16) lessons miner (local only)."""

    case_id: str
    suite: str
    transcript: Transcript
    violations: tuple[Violation, ...]
    rule_ids: tuple[str, ...]
    passed: bool


@dataclass
class _Rollout:
    tools_called: list[str]
    transcript: Transcript
    violations: list[Violation] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    rubric: dict[str, float] = field(default_factory=dict)
    suite: str = "sleep"
    judged: bool = False
    passed: bool = False


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def check_op(op: str, arg: Any, response: str, tools_called: Sequence[str]) -> bool:
    """skillopt_sleep.judges semantics, minus the self-reported tool marker; unknown -> error."""
    r = response or ""
    if op == "section_present":
        name = re.escape(str(arg))
        return bool(re.search(rf"(?im)^\s{{0,3}}(#{{1,6}}\s*.*{name}|\*\*.*{name}.*\*\*\s*:?)\s*$", r)
                    or re.search(rf"(?im)^\s*{name}\s*:", r))
    if op == "regex":
        return bool(re.search(str(arg), r))
    if op == "max_chars":
        return len(r) <= int(arg)
    if op == "min_chars":
        return len(r) >= int(arg)
    if op == "contains":
        return str(arg).lower() in r.lower()
    if op == "tool_called":
        return any(str(arg).lower() == t.lower() for t in tools_called)
    raise UnknownJudgeOp(f"unknown rule-judge op {op!r} (allowed: {sorted(KNOWN_OPS)})")


def suite_of(task: TaskRecord) -> str:
    for t in task.tags:
        if t.startswith("suite:"):
            return t[len("suite:"):]
    return "sleep"


def category_of(task: TaskRecord) -> str:
    for t in task.tags:
        if not t.startswith(("suite:", "rule:")):
            return t
    return "task"


def validate_edit(edit: Any, *, target: str) -> EditRecord:
    if not isinstance(edit, EditRecord):
        if isinstance(edit, Mapping):
            edit = EditRecord(**{k: str(edit.get(k, "")) for k in ("target", "op", "content", "anchor", "rationale")})
        else:
            raise TypeError(f"reflector returned a non-edit: {type(edit).__name__}")
    edit.target = edit.target or target
    if edit.target != target:
        raise ValueError(f"edit targets {edit.target!r}, expected {target!r}")
    if edit.op not in ("add", "delete", "replace"):
        raise ValueError(f"bad edit op {edit.op!r}")
    for name in ("content", "anchor"):
        value = getattr(edit, name) or ""
        if len(value) > MAX_EDIT_CHARS:
            raise ValueError(f"edit {name} longer than {MAX_EDIT_CHARS} chars")
        if "\n" in value or "\r" in value or "<!--" in value or "-->" in value \
                or LEARNED_START in value or LEARNED_END in value:
            raise ValueError(f"edit {name} must be one line without HTML comment markers")
    if edit.op in ("add", "replace") and not edit.content.strip():
        raise ValueError(f"{edit.op} edit needs content")
    if edit.op in ("delete", "replace") and not (edit.anchor or edit.content).strip():
        raise ValueError(f"{edit.op} edit needs an anchor")
    edit.rationale = (edit.rationale or "")[:MAX_EDIT_CHARS].replace("\n", " ")
    return edit


class SleepBackend(Backend):
    name = "ci-lab-harness"

    def __init__(self, *, run_target: RunTarget, oracle: SafetyOracle, reflector: Reflector,
                 scorer: Scorer | None = None, budget: Budget | None = None) -> None:
        self._run_target = run_target
        self._oracle = oracle
        self._reflector = reflector
        self._scorer = scorer
        self._budget = budget
        self._lock = threading.Lock()
        self._rollouts: dict[tuple[str, str], _Rollout] = {}
        self._tokens = 0
        self.n_attempts = 0
        self.n_reflects = 0
        self.last_reflect_raw = ""
        self.last_call_error = ""
        self.reflect_log: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- attempts
    def _run(self, task: TaskRecord, skill: str, memory: str) -> tuple[str, list[str]]:
        if self._budget is not None:
            self._budget.begin_rollout()
        reply, called, transcript = self._run_target(task, skill, memory)
        reply = reply or ""
        called = [str(c) for c in called]
        tokens = int(transcript.tokens_in) + int(transcript.tokens_out)
        with self._lock:
            self.n_attempts += 1
            self._tokens += tokens
            self._rollouts[(task.id, _digest(reply))] = _Rollout(called, transcript, suite=suite_of(task))
        if self._budget is not None:
            self._budget.charge(tokens=tokens)
        return reply, called

    def attempt(self, task: TaskRecord, skill: str, memory: str, sample_id: int = 0) -> str:
        return self._run(task, skill, memory)[0]

    def attempt_with_tools(self, task: TaskRecord, skill: str, memory: str,
                           tools: list[str]) -> tuple[str, list[str]]:
        return self._run(task, skill, memory)

    # ---------------------------------------------------------------- judge
    def judge(self, task: TaskRecord, response: str) -> tuple[float, float, str]:
        with self._lock:
            ro = self._rollouts.get((task.id, _digest(response or "")))
        if ro is None:  # judged without a recorded rollout: no tool evidence at all
            ro = _Rollout([], Transcript(case_id=task.id, messages=[{"role": "assistant", "content": response or ""}]),
                          suite=suite_of(task))
            with self._lock:
                self._rollouts[(task.id, _digest(response or ""))] = ro
        checks = list((task.judge or {}).get("checks", []) or [])
        passed, failed = 0, []
        for c in checks:
            op, arg = c.get("op", ""), c.get("arg")
            if check_op(op, arg, response, ro.tools_called):
                passed += 1
            else:
                failed.append(f"check.{op}:{arg}")
        violations = list(self._oracle.check(ro.transcript))
        score = self._scorer(task, ro.transcript) if self._scorer is not None else None
        rubric: dict[str, float] = {}
        components: list[float] = []
        if checks:
            rubric["rule_checks"] = passed / len(checks)
            components.append(rubric["rule_checks"])
        if score is not None:
            rubric["assert"] = max(0.0, min(1.0, float(score)))
            components.append(rubric["assert"])
        ro.violations = violations
        ro.rule_ids = [v.rule_id for v in violations] + failed
        ro.rubric = rubric
        hard_ok = not failed and not violations and (score is None or rubric["assert"] >= SCORER_PASS)
        ro.judged, ro.passed = True, hard_ok
        soft = 0.0 if violations else (sum(components) / len(components) if components else float(hard_ok))
        why = []
        if violations:
            why.append("safety: " + ", ".join(v.rule_id for v in violations))
        if failed:
            why.append("failed: " + ", ".join(failed))
        if score is not None and rubric["assert"] < SCORER_PASS:
            why.append(f"assert score {rubric['assert']:.2f}")
        return (1.0 if hard_ok else 0.0), round(soft, 6), "; ".join(why) or "all checks passed"

    def judged_rollouts(self) -> list[RolloutView]:
        """Judged rollouts in insertion order, for the HOOK(M16) lessons miner (local only)."""
        with self._lock:
            items = [(case_id, ro) for (case_id, _), ro in self._rollouts.items() if ro.judged]
        return [RolloutView(case_id=case_id, suite=ro.suite, transcript=ro.transcript,
                            violations=tuple(ro.violations), rule_ids=tuple(ro.rule_ids), passed=ro.passed)
                for case_id, ro in items]

    # ---------------------------------------------------------------- reflect
    def failure_record(self, task: TaskRecord, result: ReplayResult) -> FailureRecord:
        with self._lock:
            ro = self._rollouts.get((task.id, _digest(result.response or "")))
        suite = suite_of(task)
        injection = any(m in (suite + " " + " ".join(task.tags)).lower() for m in INJECTION_MARKERS)
        excerpt = "" if injection else str(redact_secrets(" ".join((result.response or "").split())))[:EXCERPT_CHARS]
        rule_ids = tuple(ro.rule_ids) if ro and ro.rule_ids else (
            tuple(p.strip() for p in (result.fail_reason or "unscored").split(";") if p.strip())[:8])
        return FailureRecord(case_id=task.id, suite=suite, category=category_of(task), rule_ids=rule_ids,
                             rubric_scores=dict(ro.rubric) if ro else {"soft": float(result.soft)},
                             excerpt=excerpt)

    def reflect(self, failures: list[tuple[TaskRecord, ReplayResult]],
                successes: list[tuple[TaskRecord, ReplayResult]], skill: str, memory: str, *,
                edit_budget: int, evolve_skill: bool, evolve_memory: bool) -> list[EditRecord]:
        target = "skill" if evolve_skill else "memory"
        if not failures:
            self.last_reflect_raw = ""
            return []
        records = tuple(self.failure_record(t, r) for t, r in failures)
        request = ReflectRequest(failures=records, n_successes=len(successes), target=target,
                                 edit_budget=int(edit_budget),
                                 learned=tuple(current_learned_lines(skill if target == "skill" else memory)))
        if self._budget is not None:
            self._budget.check_time()
        out = self._reflector(request)
        result = out if isinstance(out, ReflectResult) else ReflectResult(edits=list(out))
        with self._lock:
            self.n_reflects += 1
            self._tokens += int(result.tokens)
        if self._budget is not None:
            self._budget.charge(tokens=result.tokens, aiu=result.aiu)
        edits: list[EditRecord] = []
        errors: list[str] = []
        for e in result.edits:
            if len(edits) >= max(0, int(edit_budget)):
                errors.append("edit budget exceeded; extra edits dropped")
                break
            try:
                edits.append(validate_edit(e, target=target))
            except ValueError as exc:
                errors.append(str(exc))
        self.last_reflect_raw = str(redact_secrets(result.raw or ""))[:4000]
        self.last_call_error = "; ".join(errors)
        self.reflect_log.append({"target": target, "n_failures": len(records),
                                 "failure_case_ids": [r.case_id for r in records],
                                 "n_edits": len(edits), "rejected": errors})
        return edits

    def tokens_used(self) -> int:
        with self._lock:
            return self._tokens


# Explicit legacy imports remain valid until L9 removes the order-support target.
OrderSupportSleepBackend = SleepBackend
