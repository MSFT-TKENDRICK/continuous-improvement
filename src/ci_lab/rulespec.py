"""v2.4 contracts — lessons as structure (design §13, §13.6). FROZEN: change only via dev/contracts.

Rules are *data* (YAML → these pydantic models, unknown keys rejected) interpreted by the frozen
engine in ``ci_lab.rules``. A rule *fires* when ``when`` holds (default: always) and ``require``
does NOT hold for the step under evaluation.

Path namespaces (B5): outside ``prior.where`` only ``current.args.<p>`` is addressable; inside
``prior.where`` only ``prior.args.<p>`` / ``prior.result.<p>``. ``prior.same`` is a typed join list
of ``(current_path, prior_path)``. ``state`` flags are subject-scoped and set only by frozen
extractors from successful structured tool results.

Guards see a closed :class:`GuardView` (no suite/split/case/evaluator metadata — B2). Runtime
remediation text comes only from trusted templates (B3); guard tool results are canonical JSON
``str`` (B6), see :func:`guard_result_json`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RULESPEC_SCHEMA_VERSION = 1

# ---------------------------------------------------------------- limits (§13.6)
REGEX_MAX_LEN = 256            # B7: RE2 only, no `re` fallback
TEXT_MAX_BYTES = 16 * 1024     # B7: engine caps evaluated input
MAX_GUARD_BLOCKS_PER_TURN = 2  # B4: then terminal safe response
PROMOTE_MIN_OPPORTUNITIES = 200  # N3 shadow -> enforce
PROMOTE_MIN_FIRES = 20
PROMOTE_MIN_ADJUDICATED = 10
PROMOTE_FP_UCB = 0.02          # Clopper-Pearson 95% upper bound on FP rate
SHADOW_MAX_NIGHTS = 30

# ---------------------------------------------------------------- locations
GUARDS_DIR = "harness/guards"                 # arm-writable only via the `guard` strategy
GUARD_BUNDLE_LOCK = "harness/guards/BUNDLE.lock"  # last-known-good digest (N1); publish-only
TEMPLATES_RESOURCE = ("ci_lab.rules", "templates.yaml")  # trusted catalog (frozen code, B3)
LESSON_REGISTRY = "lessons/registry.yaml"     # lesson -> rules + prose anchors (N4); not arm surface
OES_GUARD_EXT = "com.microsoft.ci.guard"
GUARD_RESULT_KEY = "guard"
TOOL_SIDE_EFFECT_KEY = "side_effect"          # tool_specs.yaml: `side_effect: true` => serialized + fail-closed

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,63}$")
_SEG = r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])"
CURRENT_PATH_RE = re.compile(rf"^current\.args{_SEG}*$")
PRIOR_PATH_RE = re.compile(rf"^prior\.(?:args|result){_SEG}*$")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Scalar = str | int | float | bool | None

# ---------------------------------------------------------------- predicates (closed union)


class AllPred(_M):
    kind: Literal["all"]
    of: list[Pred] = Field(min_length=1)


class AnyPred(_M):
    kind: Literal["any"]
    of: list[Pred] = Field(min_length=1)


class NotPred(_M):
    kind: Literal["not"]
    of: Pred


class ArgPred(_M):
    kind: Literal["arg"]
    path: str
    op: Literal["eq", "ne", "in", "nin", "gt", "ge", "lt", "le", "exists", "matches"]
    value: Scalar | list[Scalar] = None

    @model_validator(mode="after")
    def _check(self) -> ArgPred:
        if self.op in ("in", "nin") and not isinstance(self.value, list):
            raise ValueError(f"op {self.op} needs a list value")
        if self.op == "matches":
            _check_regex(self.value)
        if self.op not in ("in", "nin") and isinstance(self.value, list):
            raise ValueError(f"op {self.op} takes a scalar value")
        return self


class PriorPred(_M):
    """Exists an earlier tool_call step for ``tool`` with ``status`` satisfying ``where``."""

    kind: Literal["prior"]
    tool: str
    status: Literal["ok", "error", "any"] = "ok"
    where: Pred | None = None
    within: int | None = Field(default=None, ge=1)  # counts tool_call steps
    same: list[tuple[str, str]] = Field(default_factory=list)  # (current_path, prior_path)

    @field_validator("same")
    @classmethod
    def _same(cls, v: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for cur, pri in v:
            if not CURRENT_PATH_RE.match(cur) or not PRIOR_PATH_RE.match(pri):
                raise ValueError(f"bad join ({cur}, {pri})")
        return v


class CountPred(_M):
    kind: Literal["count"]
    tool: str
    status: Literal["ok", "error", "blocked", "any"] = "any"
    op: Literal["eq", "ge", "le", "gt", "lt"]
    n: int = Field(ge=0)


class TextPred(_M):
    """Response text only (``on: response``). RE2 semantics."""

    kind: Literal["text"]
    matches: str

    @field_validator("matches")
    @classmethod
    def _re(cls, v: str) -> str:
        _check_regex(v)
        return v


class StatePred(_M):
    kind: Literal["state"]
    flag: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    subject: str | None = None  # current.args.<path> the flag must be bound to
    value: bool = True

    @field_validator("subject")
    @classmethod
    def _subj(cls, v: str | None) -> str | None:
        if v is not None and not CURRENT_PATH_RE.match(v):
            raise ValueError(f"state subject must be current.args.*: {v}")
        return v


Pred = Annotated[
    AllPred | AnyPred | NotPred | ArgPred | PriorPred | CountPred | TextPred | StatePred,
    Field(discriminator="kind"),
]
for _cls in (AllPred, AnyPred, NotPred, ArgPred, PriorPred, CountPred, TextPred, StatePred):
    _cls.model_rebuild()


def _check_regex(v: Any) -> None:
    if not isinstance(v, str) or not v or len(v) > REGEX_MAX_LEN:
        raise ValueError(f"regex must be a non-empty str <= {REGEX_MAX_LEN} chars")


def _walk(p: Any, inside_prior: bool, on: str) -> None:
    if isinstance(p, (AllPred, AnyPred)):
        for c in p.of:
            _walk(c, inside_prior, on)
    elif isinstance(p, NotPred):
        _walk(p.of, inside_prior, on)
    elif isinstance(p, ArgPred):
        rx = PRIOR_PATH_RE if inside_prior else CURRENT_PATH_RE
        if not rx.match(p.path):
            raise ValueError(f"path {p.path!r} not allowed {'inside' if inside_prior else 'outside'} prior.where")
    elif isinstance(p, PriorPred):
        if inside_prior:
            raise ValueError("nested prior is not allowed")
        if p.where is not None:
            _walk(p.where, True, on)
    elif isinstance(p, TextPred):
        if on != "response" or inside_prior:
            raise ValueError("text predicates only apply to on: response")
    elif isinstance(p, (StatePred, CountPred)) and inside_prior:
        raise ValueError(f"{p.kind} not allowed inside prior.where")


# ---------------------------------------------------------------- rules


class Provenance(_M):
    lesson_id: str | None = None
    evidence: list[str] = Field(default_factory=list)  # trajectory/cluster digests only — never text
    envelope: str | None = None  # OES experiment id that accepted it
    source: Literal["seed", "template", "synthesizer", "human"] = "human"


_RUNG_ON = {"R1": "tool_call", "R2": "tool_call", "R3": "response", "R4": "trajectory"}


class RuleSpec(_M):
    id: str
    version: int = Field(ge=1)
    rung: Literal["R1", "R2", "R3", "R4"]
    on: Literal["tool_call", "response", "trajectory"]
    target: str  # tool name, or "*"
    when: Pred | None = None
    require: Pred
    action: Literal["warn", "block", "redact"]
    severity: Literal["critical", "major", "minor"] = "major"
    template: str  # id in the trusted template catalog (B3)
    slots: dict[str, str | int] = Field(default_factory=dict)
    see: str = ""
    mode: Literal["shadow", "enforce"] = "shadow"
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"bad rule id {v!r}")
        return v

    @model_validator(mode="after")
    def _check(self) -> RuleSpec:
        if _RUNG_ON[self.rung] != self.on:
            raise ValueError(f"rung {self.rung} requires on: {_RUNG_ON[self.rung]}")
        if self.target == "*" and self.action == "block":
            raise ValueError("target '*' cannot block (B2)")
        if self.action == "redact" and self.on != "response":
            raise ValueError("redact only applies to on: response")
        if self.rung == "R1":
            for p in _leaves(self.require) + (_leaves(self.when) if self.when else []):
                if not isinstance(p, ArgPred):
                    raise ValueError("R1 rules may only use arg predicates")  # noqa: TRY004 (pydantic needs ValueError)
        for p in (self.when, self.require):
            if p is not None:
                _walk(p, False, self.on)
        for k, v in self.slots.items():
            if not re.match(r"^[a-z_]{1,32}$", k) or (isinstance(v, str) and len(v) > 120):
                raise ValueError(f"bad slot {k}")
        return self


def _leaves(p: Any) -> list[Any]:
    if isinstance(p, (AllPred, AnyPred)):
        return [x for c in p.of for x in _leaves(c)]
    if isinstance(p, NotPred):
        return _leaves(p.of)
    return [p]


class TemplateSpec(_M):
    """Trusted remediation text; ``{slot}`` placeholders must be declared in ``slots``."""

    id: str
    message: str = Field(max_length=300)
    fix: str = Field(max_length=300)
    slots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _slots(self) -> TemplateSpec:
        used = set(re.findall(r"\{([a-z_]+)\}", self.message + self.fix))
        if not used <= set(self.slots):
            raise ValueError(f"undeclared slots {sorted(used - set(self.slots))}")
        return self


class RuleFile(_M):
    """One ``harness/guards/*.yaml`` file."""

    schema_version: Literal[1] = 1
    rules: list[RuleSpec]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def bundle_digest(rules: list[RuleSpec]) -> str:
    """Order-independent digest of a rule set (BUNDLE.lock, envelope ext)."""
    items = sorted(canonical_json(r.model_dump(mode="json")) for r in rules)
    return "sha256:" + hashlib.sha256("\n".join(items).encode()).hexdigest()


# ---------------------------------------------------------------- trajectories / guard view

StepKind = Literal["user", "tool_call", "tool_result", "response"]


class TrajectoryStep(_M):
    i: int = Field(ge=0)
    kind: StepKind
    tool: str | None = None
    call_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)      # tool_call only
    result: dict[str, Any] | None = None                    # tool_result: structured only
    status: Literal["ok", "error", "blocked"] | None = None  # tool_result / blocked call
    text: str | None = None          # response only (user text never enters guards — B3)
    text_digest: str | None = None
    flags: dict[str, list[str]] = Field(default_factory=dict)  # flag -> subjects (frozen extractors)


class GuardView(_M):
    """Everything a guard may see (B2). No suite/split/case/env/evaluator metadata."""

    steps: tuple[TrajectoryStep, ...] = ()
    pending: TrajectoryStep | None = None  # the tool_call / response being evaluated


class Outcome(_M):
    passed: bool | None = None
    oracle_rules: tuple[str, ...] = ()
    rubric_fails: tuple[str, ...] = ()
    error_class: str | None = None
    human_label: Literal["good", "bad"] | None = None
    injection_suspect: bool = False


class Trajectory(_M):
    id: str
    source: Literal["agl", "spans", "assert", "usage", "calibrate", "pr"]
    split: Literal["evolve", "usage"]  # lessons never ingest heldout/ood/confirm (B2)
    family: str          # case/intent family: holdout split unit (N2)
    slice: str           # time or suite slice for convergence
    pin: str             # oracle+evaluator pin (fingerprint version, N2)
    trusted: bool        # False for usage/PR-derived until human-reviewed (B3)
    steps: tuple[TrajectoryStep, ...]
    outcome: Outcome = Field(default_factory=Outcome)
    guard_on: bool = False  # recorded with guards enabled


class GuardDecision(_M):
    """One rule evaluation that matched (all-match telemetry, N6). Recorded before acting (B1)."""

    rule_id: str
    rule_version: int
    mode: Literal["shadow", "enforce"]
    action: Literal["warn", "block", "redact"]
    enforced: bool                 # mode == enforce and action applied
    step_index: int
    target: str
    attempt_digest: str            # sha256 of canonical pending step (pre-enforcement attempt)
    degraded: bool = False         # LKG fallback / warn-only degradation (N1)


def attempt_digest(step: TrajectoryStep) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(step.model_dump(mode="json")).encode()).hexdigest()


def guard_result_json(rule_id: str, violation: str, fix: str, see: str = "", *, terminal: bool = False) -> str:
    """Canonical tool result the model receives when a guard blocks a call (B6)."""
    body = {"rule": rule_id, "violation": violation, "fix": fix, "see": see}
    if terminal:
        body["terminal"] = True
    return canonical_json({GUARD_RESULT_KEY: body})


# ---------------------------------------------------------------- lessons


class Fingerprint(_M):
    pin: str
    oracle_rules: tuple[str, ...] = ()
    rubric_ids: tuple[str, ...] = ()
    tool_ngrams: tuple[tuple[str, ...], ...] = ()  # canonicalized tool-sequence 3-grams
    error_class: str | None = None

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()[:16]


Rung = Literal["R1", "R2", "R3", "R4", "R5", "R6"]


class LessonCluster(_M):
    id: str
    fingerprint: Fingerprint
    members: tuple[str, ...]
    families: tuple[str, ...]
    slices: tuple[str, ...]
    route: Rung
    status: Literal["candidate", "confirmed", "rejected", "backlog"] = "candidate"
    human_confirmed: bool = False


class GuardMetrics(_M):
    """OES envelope ext ``com.microsoft.ci.guard`` payload (B1/B4/N3)."""

    bundle_digest: str
    paired: bool                  # guard-off and guard-on on identical cases/seeds
    trials: int = Field(ge=1)
    attempted_violation_rate: float
    delivered_violation_rate: float
    task_completion: float
    false_denial_rate: float
    block_rate: float
    opportunities: int = 0
    fires: int = 0
    recall: float | None = None
    fp_rate: float | None = None
    fp_ucb: float | None = None
    substitutions: int = 0        # attempted alternative calls after a block


class LessonEntry(_M):
    """``lessons/registry.yaml`` entry (N4)."""

    lesson_id: str
    cluster_id: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    prose_anchors: list[str] = Field(default_factory=list)  # "path#heading"
    status: Literal["proposed", "shadow", "enforced", "retired"] = "proposed"


class CorrectionRecord(_M):
    """Typed dev-transcript correction (B8). No excerpts, args or raw text — ever."""

    kind: Literal["revert", "user_correction", "repeated_fix", "review_comment", "lint_fail"]
    tool: str | None = None
    rule_hint: str | None = None  # lint template id
    file_glob: str | None = None
    count: int = Field(default=1, ge=1)
