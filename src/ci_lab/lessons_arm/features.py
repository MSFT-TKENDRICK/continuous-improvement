"""Typed lesson features — the only thing synthesizers (template or LLM) may see (B3, C12).

Features are identifiers (tool / arg / result field names, flags), enum-ish short tokens and
bounded numbers. Free text cannot pass the validators, so trace-derived prose can never reach
a rule or a prompt. :func:`derive_features` maps a :class:`~ci_lab.rulespec.LessonCluster`
deterministically to features using the frozen oracle-rule table :data:`ORACLE_FEATURES`
(design §13.7); clusters it cannot map are *leftovers* for the ``LessonSynthesizer``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ci_lab.rulespec import LessonCluster

from .templates import SAFE_PATTERNS

Ident = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")]
Token = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,40}$")]
Bounded = Annotated[float, Field(ge=-1e9, le=1e9)]
EnumValue = bool | int | Bounded | Token

FeatureKind = Literal["prior_call", "state_flag", "arg_constraint", "amount_vs_prior", "response_pattern"]
CmpOp = Literal["eq", "ne", "le", "lt", "ge", "gt"]
ArgOp = Literal["eq", "ne", "in", "nin", "gt", "ge", "lt", "le", "exists"]


class LessonFeatures(BaseModel):
    """Typed slots for one lesson. ``kind`` selects the template synthesizer. Unknown keys are
    dropped (never forwarded anywhere)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: FeatureKind
    target_tool: Ident | None = None
    # precondition-from-sequence (R2 prior with `same` join on the subject)
    prior_tool: Ident | None = None
    subject_arg: Ident | None = None            # current.args.<subject_arg>
    prior_subject_field: Ident | None = None    # prior.args.<field> (defaults to subject_arg)
    prior_result_equals: dict[Ident, bool | int | Token] = Field(default_factory=dict)
    within: int | None = Field(default=None, ge=1, le=50)
    # state-flag precondition (frozen extractor flag, subject-scoped)
    flag: Ident | None = None
    via_tool: Ident | None = None
    # arg constraint (R1)
    arg: Ident | None = None
    op: ArgOp | None = None
    values: list[EnumValue] = Field(default_factory=list, max_length=32)
    # amount vs prior (R2 cmp)
    amount_arg: Ident | None = None
    prior_amount_field: Ident | None = None
    cmp_op: CmpOp = "le"
    # response pattern (R3 redact)
    pattern_classes: list[str] = Field(default_factory=list, max_length=8)
    # trust / safety (B3)
    trusted: bool = True
    injection_suspect: bool = False

    @field_validator("pattern_classes")
    @classmethod
    def _classes(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if c not in SAFE_PATTERNS]
        if bad:
            raise ValueError(f"unknown pattern classes {bad}; allowed {sorted(SAFE_PATTERNS)}")
        return sorted(set(v))


# Frozen, reviewed mapping from oracle rule ids to typed features (design §13.7). Injection
# lessons are deliberately absent: not expressible without user text -> R6 prose.
ORACLE_FEATURES: dict[str, LessonFeatures] = {
    "refund.unverified_identity": LessonFeatures(
        kind="state_flag", target_tool="issue_refund", flag="identity_verified", subject_arg="order_id",
        via_tool="verify_identity"),
    "refund.ineligible_order": LessonFeatures(
        kind="prior_call", target_tool="issue_refund", prior_tool="lookup_order", subject_arg="order_id",
        prior_result_equals={"refund_eligible": True, "refund_limit_exceeded": False}),
    "refund.amount_exceeds_eligible": LessonFeatures(
        kind="amount_vs_prior", target_tool="issue_refund", prior_tool="lookup_order", subject_arg="order_id",
        amount_arg="amount", prior_amount_field="total", cmp_op="le"),
    "pii.disclosed_before_verification": LessonFeatures(
        kind="response_pattern", flag="identity_verified", pattern_classes=["email", "phone", "street_address"]),
}


def is_injection(cluster: LessonCluster) -> bool:
    return any(r.startswith("injection.") for r in cluster.fingerprint.oracle_rules)


def derive_features(cluster: LessonCluster) -> LessonFeatures | None:
    """Deterministic features for ``cluster`` or ``None`` (leftover / not structural)."""
    if is_injection(cluster):
        return None
    hits = [ORACLE_FEATURES[r] for r in sorted(set(cluster.fingerprint.oracle_rules)) if r in ORACLE_FEATURES]
    return hits[0] if len(hits) == 1 else None


class Candidate(BaseModel):
    """One ``candidates.jsonl`` line: a cluster plus optional typed features (extra keys such as
    ``route_reasons`` are ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cluster: LessonCluster
    features: LessonFeatures | None = None
    trusted: bool | None = None  # M16 top-level trust bit (False: usage/PR source, needs a human label)

    @field_validator("features", mode="before")
    @classmethod
    def _empty(cls, v: object) -> object:
        return None if v in ({}, None) else v

    def resolved(self) -> LessonFeatures | None:
        """Typed features (given or derived from the frozen oracle map) with the top-level trust bit applied."""
        feats = self.features or derive_features(self.cluster)
        if feats is not None and self.trusted is False and feats.trusted:
            feats = feats.model_copy(update={"trusted": False})
        return feats


def read_candidates(path: Path) -> list[Candidate]:
    """Parse ``<run_dir>/lessons/candidates.jsonl`` (bare ``LessonCluster`` lines or
    ``{"cluster": ..., "features": ...}`` wrappers). Missing file ⇒ ``[]``."""
    if not path.exists():
        return []
    out: list[Candidate] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            out.append(Candidate.model_validate(obj) if "cluster" in obj
                       else Candidate(cluster=LessonCluster.model_validate(obj)))
        except Exception as exc:
            raise ValueError(f"{path.name}:{n}: bad candidate: {exc}") from exc
    return out


def features_for(candidates: Iterable[Candidate]) -> list[tuple[LessonCluster, LessonFeatures | None]]:
    return [(c.cluster, c.resolved()) for c in candidates]
