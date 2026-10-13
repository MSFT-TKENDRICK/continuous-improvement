"""Shared contracts for the self-improving harness (design-v2 §9-10).

Every ``ci_lab`` module codes against these types/protocols so modules can be
built in parallel and wired together later. Keep this file dependency-light
(stdlib + typing only): it is imported by every layer.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------- identifiers

CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
ARM_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}$")
ARM_BRANCH_RE = re.compile(r"^exp/[a-z0-9][a-z0-9-]{2,44}/[a-z][a-z0-9-]{0,15}$")
SLEEP_BRANCH_RE = re.compile(r"^exp/sleep-\d{8}-\d{1,4}/cand$")
NAMESPACE = uuid.UUID("6c1e3a52-55d2-4d0b-9a7a-1b0c0a1c1ab0")


def round_experiment_id(campaign_id: str, round_no: int) -> str:
    if not CAMPAIGN_RE.match(campaign_id):
        raise ValueError(f"bad campaign id {campaign_id!r}")
    if not 0 <= round_no <= 99:
        raise ValueError(f"round out of range: {round_no}")
    return f"{campaign_id}-r{round_no:02d}"


def arm_branch(experiment_id: str, arm: str) -> str:
    """``exp/<cid>-r<tt>/<arm>`` — fixed depth 3, validated."""
    ref = f"exp/{experiment_id}/{arm}"
    if not ARM_RE.match(arm) or not ARM_BRANCH_RE.match(ref):
        raise ValueError(f"bad arm branch {ref!r}")
    return ref


def op_id(*parts: Any) -> str:
    """Deterministic logical operation id (outbox / AGL event dedupe)."""
    return str(uuid.uuid5(NAMESPACE, "|".join(str(p) for p in parts)))


# ---------------------------------------------------------------- inference

class Profile(str, Enum):
    COPILOT = "copilot"  # CopilotChatClient, ambient GitHub auth (default)
    OFFLINE = "offline"  # OpenAI-compatible client -> AGL proxy -> llama-server
    FAKE = "fake"        # ci_lab.testing.FakeChatClient (unit tests)


Purpose = Literal["target", "proposer", "critic", "analyst", "reflector", "judge", "optimizer"]


@dataclass(frozen=True)
class RolloutKey:
    experiment_id: str
    variant: str
    case_id: str
    trial: int = 0
    attempt: int = 0

    @property
    def rollout_id(self) -> str:
        raw = f"{self.experiment_id}|{self.variant}|{self.case_id}|{self.trial}"
        return "ro-" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    @property
    def attempt_id(self) -> str:
        return str(self.attempt)


class ChatClientFactory(Protocol):
    """Builds a MAF chat client (``agent_framework.SupportsChatGetResponse``)."""

    def __call__(self, *, profile: Profile, model: str, purpose: Purpose,
                 rollout: RolloutKey | None = None, **kw: Any) -> Any: ...


class RolloutJournal(Protocol):
    """Journal-first AGL client (C2): append locally, then mirror to agl-server."""

    def start(self, key: RolloutKey, input: Mapping[str, Any]) -> None: ...
    def event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> None: ...
    def finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> None: ...
    def events(self, key: RolloutKey) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------- transcripts / safety

@dataclass(frozen=True)
class ToolCallRecord:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    result: Any
    turn: int


@dataclass(frozen=True)
class Transcript:
    case_id: str
    messages: Sequence[Mapping[str, Any]]  # {"role", "content"} user/assistant turns
    tool_calls: Sequence[ToolCallRecord] = ()
    served_models: Sequence[str] = ()
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class Violation:
    rule_id: str  # e.g. "harness.protected_path"
    severity: Literal["critical", "major"]
    detail: str


class SafetyOracle(Protocol):
    def check(self, transcript: Transcript) -> list[Violation]: ...


# ---------------------------------------------------------------- evaluation

@dataclass(frozen=True)
class TaskScore:
    case_id: str
    trial: int
    suite: str
    score: float | None  # None = missing trial (counted as 0 by RRSI)
    violations: tuple[Violation, ...] = ()
    tokens_in: int = 0
    tokens_out: int = 0
    served_model: str | None = None
    # Evaluator-measured runtime cost of this trial (ci_lab.metrics.RunMeter); never agent-supplied.
    wall_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    # Named diagnostic subscores (e.g. "resource.<criterion>", "resource_score"); ``score`` stays the
    # deliverable-quality score and never folds these in.
    subscores: dict[str, float] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class EvaluatorPin:
    evaluator_tree: str
    judge_model: str
    judge_provider: str
    served_judge_models: tuple[str, ...] = ()


@dataclass
class EvalResult:
    harness_tree: str
    split: Literal["evolve", "heldout", "ood", "aa"]
    pin: EvaluatorPin
    scores: list[TaskScore] = field(default_factory=list)
    # Evaluator-measured surface metrics of the evaluated harness (ci_lab.metrics.surface_metrics).
    surface: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureRecord:
    """Typed failure summary for analyst/reflector — never raw tool output (C12)."""

    case_id: str
    suite: str
    category: str
    rule_ids: tuple[str, ...]
    rubric_scores: Mapping[str, float]
    excerpt: str = ""  # truncated agent-turn excerpt; always empty for injection suites


class Domain(Protocol):
    name: str
    surface_globs: Sequence[str]
    frozen_globs: Sequence[str]
    component_globs: Mapping[str, Sequence[str]]

    def splits(self) -> Mapping[str, Sequence[str]]: ...
    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult: ...
    def failures(self, result: EvalResult) -> list[FailureRecord]: ...


# ---------------------------------------------------------------- arms / RRSI

COMPONENTS = ("prompt", "skill", "client_tool", "config", "memory", "context_mgmt", "guard",
              "agent", "loop", "workflow", "mcp")
# Components text strategies (agent/gepa/skillopt) may target; "guard" (harness/guards/**) is
# written only by the guard strategy (v2.4 §13, frozen for text optimizers).
TEXT_COMPONENTS = tuple(c for c in COMPONENTS if c != "guard")
# Owning optimizer per component. AGL owns only structural components; prompt/skill/guard stay
# exclusively owned by GEPA/SkillOpt/guard.
COMPONENT_OWNERS: Mapping[str, str] = {
    "prompt": "gepa",
    "skill": "skillopt",
    "guard": "guard",
    "agent": "agl",
    "loop": "agl",
    "workflow": "agl",
    "mcp": "agl",
    "client_tool": "agl",
    "config": "agl",
    "context_mgmt": "agl",
    "memory": "agl",
}


def strategy_may_edit(strategy: str, component: str) -> bool:
    """Whether ``strategy`` may write ``component``: the generalist ``agent`` strategy may edit every text
    component, ``guard`` only guards, any other strategy only the components it owns."""
    if strategy == "agent":
        return component in TEXT_COMPONENTS
    if strategy == "guard":
        return component == "guard"
    return COMPONENT_OWNERS.get(component) == strategy


@dataclass(frozen=True)
class Edit:
    component: str
    hypothesis: str
    files: tuple[str, ...]
    commit: str


@dataclass
class CriticVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    repairs: int = 0


# v2.2 (§11): heterogeneous arm strategies. Every strategy only edits the evolvable
# surface of its arm worktree and commits typed Edits; critique, ASSERT evaluation and
# RRSI/OES selection are strategy-agnostic and authoritative (an optimizer's internal
# acceptance gate is diagnostic only).
STRATEGIES = ("agent", "gepa", "skillopt", "guard", "agl")


@dataclass(frozen=True)
class ArmDirective:
    arm: str
    strategy: str = "agent"
    component_focus: tuple[str, ...] = ()
    edit_budget: int = 1
    explore: bool = False


@dataclass
class ArmContext:
    experiment_id: str
    directive: ArmDirective
    worktree: Path
    base_commit: str
    failures: Sequence[FailureRecord]
    profile: Profile
    run_dir: Path
    budget_tokens: int | None = None
    evolve_case_ids: tuple[str, ...] = ()  # optimizers may score only these (never heldout)


class ArmStrategy(Protocol):
    name: str

    async def propose(self, ctx: ArmContext) -> list[Edit]: ...


# ---------------------------------------------------------------- telemetry (v2.3 §12)
# One OTel trace per round / night / calibration / evaluator experiment (never one
# campaign-long span: spans export only on end). Resumed work starts a new trace with a
# span link to the trace recorded in the run dir (``trace.json``). MAF's built-in GenAI
# spans (workflow / executor / invoke_agent / chat / execute_tool) nest under these.

SPAN_CAMPAIGN_ROUND = "ci.round"
SPAN_ARM = "ci.arm"
SPAN_STEP = "ci.step"            # attr ci.phase = propose|critique|repair|evaluate|select|publish|...
SPAN_CASE = "ci.case"            # one ASSERT case x trial (= one AGL rollout)
SPAN_CALIBRATE = "ci.calibrate"
SPAN_CONFIRM = "ci.confirm"
SPAN_SLEEP_NIGHT = "ci.sleep.night"
SPAN_OPTIMIZER = "ci.optimizer"  # attr ci.strategy = gepa|skillopt|agent
SPAN_EVALUATOR_EXPERIMENT = "ci.evaluator"

ATTR_CAMPAIGN = "ci.campaign_id"
ATTR_EXPERIMENT = "oes.experiment_id"
ATTR_VARIANT = "oes.variant"
ATTR_DECISION = "oes.decision"
ATTR_ROUND = "rrsi.round"
ATTR_COMPONENT = "rrsi.component"
ATTR_STRATEGY = "ci.strategy"
ATTR_PHASE = "ci.phase"
ATTR_PROFILE = "ci.profile"
ATTR_PURPOSE = "ci.purpose"
ATTR_SPLIT = "ci.split"
ATTR_CASE = "ci.case_id"
ATTR_TRIAL = "ci.trial"
ATTR_ROLLOUT = "agl.rollout_id"
ATTR_ATTEMPT = "agl.attempt_id"
ATTR_NIGHT = "sleep.night"
ATTR_SCORE = "ci.score"
ATTR_DELTA_S = "rrsi.delta_s"
ATTR_DELTA_C = "rrsi.delta_c"
ATTR_TARGET = "sleep.target"

# v2.4 §13 guards/lessons. One `ci.guard` span event per matching rule (all-match telemetry);
# enforced blocks also set ATTR_GUARD_ACTION on the enclosing execute_tool / invoke_agent span.
SPAN_GUARD = "ci.guard"             # event name on the current span (not a span) for each match
SPAN_LESSONS = "ci.lessons"         # attr ci.phase = harvest|cluster|route|synthesize|replay|propose
SPAN_LINT = "ci.lint"
ATTR_GUARD_RULE = "ci.guard.rule"
ATTR_GUARD_VERSION = "ci.guard.version"
ATTR_GUARD_MODE = "ci.guard.mode"   # shadow|enforce
ATTR_GUARD_ACTION = "ci.guard.action"  # warn|block|redact
ATTR_GUARD_ENFORCED = "ci.guard.enforced"
ATTR_GUARD_BUNDLE = "ci.guard.bundle"
ATTR_GUARD_DEGRADED = "ci.guard.degraded"
ATTR_LESSON = "ci.lesson_id"

# Run-dir progress markers the dashboard canvas reads live (spans arrive only on end).
# One file per writer (no cross-process lost updates):
#   <run_dir>/<experiment_id>/status.d/<writer>.json
#   {"writer", "seq", "updated", "phase", "arms": {arm: {"strategy","state","phase","updated"}},
#    "trace": {"trace_id","span_id"}, ...}
# Readers aggregate with ``ci_lab.obs.read_status`` (latest top-level fields win; arms merged
# per arm by ``updated``). Payloads must not contain PII/secrets.
RUN_STATUS_DIR = "status.d"
RUN_STATUS_FILE = "status.json"  # legacy single-file marker; readers still accept it
DASHBOARD_STATE_ENV = "CI_DASHBOARD_STATE"  # default ~/.ci-lab/dashboard.json
SPAN_SCHEMA_VERSION = 1  # versioned internal span record (see design §12.5)


@dataclass
class ArmResult:
    arm: str
    base_commit: str
    head_commit: str | None = None
    harness_tree: str | None = None
    edits: list[Edit] = field(default_factory=list)
    critic: CriticVerdict | None = None
    eval: EvalResult | None = None
    status: Literal["pending", "proposed", "rejected", "evaluated", "failed"] = "pending"
    strategy: str = "agent"
    cost: Mapping[str, Any] | None = None  # optimizer cost report (tokens, metric calls)


# ---------------------------------------------------------------- durability

class Outbox(Protocol):
    """Durable effect executor keyed by logical op id (C6). ``reconcile`` probes remote
    state (e.g. existing PR for head) and returns a result to record instead of acting."""

    def run_once(self, op: str, fn: Callable[[], Any], *,
                 reconcile: Callable[[], Any | None] | None = None) -> Any: ...
    async def arun_once(self, op: str, fn: Callable[[], Awaitable[Any]], *,
                        reconcile: Callable[[], Awaitable[Any | None]] | None = None) -> Any: ...


# ---------------------------------------------------------------- MAF wiring

@runtime_checkable
class ToolRegistry(Protocol):
    """Name -> callable map for AgentFactory(bindings=) / WorkflowFactory.register_tool."""

    def bindings(self) -> Mapping[str, Callable[..., Any]]: ...


PROVIDER_NAME = "GitHubCopilot"  # declarative YAML `model.provider` value for our client
PROVIDER_MAPPING: dict[str, Any] = {
    "package": "ci_lab.providers.copilot",
    "name": "CopilotChatClient",
    "model_field": "model",
    "endpoint_field": None,
    "api_key_field": None,
}
