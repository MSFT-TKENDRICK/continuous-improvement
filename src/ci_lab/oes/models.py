"""Pydantic models for the parts of an OES 0.1.0 envelope that ci_lab produces.

Field names are the OES camelCase names (via aliases); every model keeps unknown
keys (``extra="allow"``) so foreign documents round-trip unchanged. Dates are kept
as ISO-8601 strings (not ``datetime``) so serialisation is byte-stable.

``Envelope.from_dict(d).to_dict() == d`` for any schema-valid document that does
not use explicit JSON ``null`` in typed OES fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

OES_VERSION = "0.1.0"
RRSI_EXT = "com.microsoft.ci.rrsi"
SLEEP_EXT = "com.microsoft.ci.sleep"
GUARD_EXT = "com.microsoft.ci.guard"  # == rulespec.OES_GUARD_EXT (v2.4 §13; built by lessons_arm.envelope)
EXT_VERSION = "0.1.0"
# Metric-level vendor flag (OES items are open objects; namespaced like `growthbook:*`).
NON_COMPENSATORY = "com.microsoft.ci:nonCompensatory"
# int stays int and float stays float, so hashes survive a parse/dump round trip.
Number = int | float

ExperimentStatus = Literal["draft", "planned", "running", "stopped", "analyzed", "decided", "archived"]
DesignType = Literal["ab", "abn", "multivariate", "factorial", "holdout", "switchback", "bandit",
                     "quasi_experiment"]
MultipleTestingPolicy = Literal["none", "bonferroni", "fdr", "hierarchical", "metric_family",
                                "benjamini_hochberg", "custom"]
PeekingPolicy = Literal["fixed_horizon", "sequential", "always_valid", "bayesian_monitoring", "informal"]
VariantRole = Literal["control", "treatment", "holdout", "baseline"]
MetricRole = Literal["primary", "secondary", "guardrail", "diagnostic", "data_quality", "invariant"]
Direction = Literal["increase_is_good", "decrease_is_good", "no_change_expected", "two_sided"]
MetricType = Literal["conversion", "revenue", "count", "duration", "ratio", "retention", "percentile",
                     "custom"]
Aggregation = Literal["mean", "sum", "ratio", "percentile", "capped_mean", "winsorized_mean"]
AnalysisMethod = Literal["frequentist", "bayesian", "sequential", "cuped", "diff_in_diff", "custom"]
VarianceEstimator = Literal["naive", "delta_method", "cluster_robust", "sandwich", "bootstrap"]
ResultStatus = Literal["positive", "negative", "neutral", "inconclusive", "invalid"]
DecisionImpact = Literal["supports_ship", "blocks_ship", "needs_followup", "informational"]
QualityStatus = Literal["valid", "warning", "invalid", "needs_review"]
OverallResult = Literal["win", "loss", "neutral", "mixed", "inconclusive", "invalid"]
RecommendedAction = Literal["ship", "do_not_ship", "iterate", "rerun", "continue_running", "roll_back"]
DecisionStatus = Literal["pending", "decided", "superseded"]
DecisionOutcome = Literal["ship", "do_not_ship", "iterate", "rerun", "rollback", "partial_rollout"]
CheckStatus = Literal["pass", "warn", "fail", "not_run"]
Severity = Literal["low", "medium", "high", "critical"]
ArtifactType = Literal["chart", "screenshot", "sql", "notebook", "csv", "dashboard", "slide", "image",
                       "html_report"]


class OesModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow",
                              validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class Experiment(OesModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    slug: str | None = None
    summary: str | None = None
    description: str | None = None
    hypothesis: str | None = None
    learning_goal: str | None = None
    business_goal: str | None = None
    product_area: str | None = None
    tags: list[str] | None = None
    status: ExperimentStatus | None = None
    owner: dict[str, Any] | None = None
    stakeholders: list[dict[str, Any]] | None = None
    links: list[dict[str, Any]] | None = None


class Design(OesModel):
    type: DesignType | None = None
    randomization_unit: str | None = None
    analysis_unit: str | None = None
    assignment_method: str | None = None
    population: str | None = None
    variant_allocation: dict[str, Number] | None = None
    start_date: str | None = None
    end_date: str | None = None
    exposure_definition: str | None = None
    concurrent_experiments: list[str] | None = None
    power: Number | None = Field(default=None, ge=0, le=1)
    minimum_detectable_effect: Number | None = None
    alpha: Number | None = Field(default=None, ge=0, le=1)
    multiple_testing_policy: MultipleTestingPolicy | None = None
    peeking_policy: PeekingPolicy | None = None
    stopping_rule: str | None = None


class Variant(OesModel):
    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    name: str | None = None
    role: VariantRole | None = None
    description: str | None = None
    allocation: Number | None = Field(default=None, ge=0, le=1)
    config: dict[str, Any] | None = None
    code_references: list[dict[str, Any]] | None = None


class Metric(OesModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    role: MetricRole | None = None
    direction: Direction | None = None
    type: MetricType | None = None
    unit: str | None = None
    aggregation: Aggregation | None = None
    data_source: dict[str, Any] | None = None


class Comparison(OesModel):
    baseline_variant_id: str
    variant_id: str


class Interval(OesModel):
    level: Number | None = Field(default=None, ge=0, le=1)
    lower: Number | None = None
    upper: Number | None = None


class MetricResult(OesModel):
    metric_id: str = Field(min_length=1)
    comparison: Comparison
    role: str | None = None
    baseline_value: Number | None = None
    variant_value: Number | None = None
    absolute_difference: Number | None = None
    relative_difference: Number | None = None
    standard_error: Number | None = None
    confidence_interval: Interval | None = None
    p_value: Number | None = Field(default=None, ge=0, le=1)
    result_status: ResultStatus | None = None
    decision_impact: DecisionImpact | None = None


class Results(OesModel):
    sample_sizes: dict[str, Number] | None = None
    exposures: dict[str, Number] | None = None
    metric_results: list[MetricResult] | None = None
    segment_results: list[dict[str, Any]] | None = None


class Analysis(OesModel):
    method: AnalysisMethod | None = None
    model: str | None = None
    variance_estimator: VarianceEstimator | None = None
    confidence_level: Number | None = Field(default=None, ge=0, le=1)
    alpha: Number | None = Field(default=None, ge=0, le=1)
    multiple_comparison_correction: str | None = None
    missing_data_handling: str | None = None
    generated_at: str | None = None


class Scorecard(OesModel):
    summary: str | None = None
    primary_outcome: dict[str, Any] | None = None
    guardrail_outcomes: list[dict[str, Any]] | None = None
    quality_status: QualityStatus | None = None
    overall_result: OverallResult | None = None
    recommended_action: RecommendedAction | None = None
    key_findings: list[str] | None = None
    risks: list[str] | None = None


class Decision(OesModel):
    status: DecisionStatus | None = None
    outcome: DecisionOutcome | None = None
    rationale: str | None = None
    decided_by: dict[str, Any] | None = None
    decided_at: str | None = None
    follow_ups: list[dict[str, Any]] | None = None
    product_changes: str | None = None


class QualityCheck(OesModel):
    check_type: str = Field(min_length=1)
    status: CheckStatus | None = None
    severity: Severity | None = None
    observed: Any = None
    expected: Any = None
    p_value: Number | None = Field(default=None, ge=0, le=1)
    message: str | None = None


class Artifact(OesModel):
    type: ArtifactType
    uri: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    generated_at: str | None = None
    source: str | None = None
    hash: str | None = None


class Provenance(OesModel):
    created_by: dict[str, Any] | None = None
    exported_by: dict[str, Any] | None = None
    analysis_generated_by: str | None = None
    data_sources: list[dict[str, Any]] | None = None
    query_ids: list[str] | None = None
    code_version: str | None = None
    metric_definition_version: str | None = None
    result_hash: str | None = None
    attachments_hash: str | None = None


class Envelope(OesModel):
    schema_version: str = OES_VERSION
    object_type: Literal["experiment"] = "experiment"
    exported_at: str | None = None
    source_system: str | None = None
    source_system_version: str | None = None
    canonical_url: str | None = None
    external_ids: dict[str, str] | None = None
    experiment: Experiment
    design: Design | None = None
    variants: list[Variant] | None = None
    metrics: list[Metric] | None = None
    analysis: Analysis | None = None
    results: Results | None = None
    scorecard: Scorecard | None = None
    decision: Decision | None = None
    quality_checks: list[QualityCheck] | None = None
    artifacts: list[Artifact] | None = None
    provenance: Provenance | None = None
    extensions: dict[str, Any] | None = None
    content_hash: str | None = None  # ci_lab hash-lock (see ci_lab.oes.canonical)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Envelope:
        return cls.model_validate(dict(data))


# ---------------------------------------------------------------- extensions (typed views)

class ExtModel(BaseModel):
    """Our closed extension objects: snake_case in Python, camelCase on the wire."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        # Nullable extension fields are meaningful (e.g. winner=None) -> keep them.
        return self.model_dump(mode="json", by_alias=True, exclude_unset=False)


class EvaluatorPinExt(ExtModel):
    evaluator_tree: str
    judge_model: str
    judge_provider: str
    served_judge_models: list[str] = Field(default_factory=list)


class EditExt(ExtModel):
    component: str
    hypothesis: str
    commit: str
    files: list[str] = Field(default_factory=list)


class CriticExt(ExtModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    repairs: int = 0


class VariantExt(ExtModel):
    status: Literal["incumbent", "pending", "proposed", "rejected", "evaluated", "failed"]
    base_commit: str
    head_commit: str | None = None
    harness_tree: str | None = None
    edits: list[EditExt] = Field(default_factory=list)
    critic: CriticExt | None = None
    archive_ref: str | None = None


class CandidateExt(ExtModel):
    variant_id: str
    admissible: bool
    delta_s: Number | None = None
    delta_c: Number | None = None
    novelty: Number | None = None
    ci_lower_bound: Number | None = None
    ci_level: Number = 0.95
    rule: Literal["cost", "weighted", "none"] = "none"
    objective: Number | None = None
    reasons: list[str] = Field(default_factory=list)


class SelectionExt(ExtModel):
    winner: str | None
    candidates: list[CandidateExt] = Field(default_factory=list)
    incumbent_score: Number
    new_incumbent_score: Number

