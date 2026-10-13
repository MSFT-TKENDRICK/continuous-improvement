"""Pure OES 0.1.0 envelope builders for ci_lab experiments (design-v2 §2 mapping, §9).

Every builder takes contract types (``EvalResult``/``TaskScore``/``EvaluatorPin``/
``ArmResult``) plus already-computed statistics (delta, selection trace, confirm
stats come from ``ci_lab.rrsi``), does no I/O, and returns a schema-valid, hash-locked
(``contentHash``) JSON dict. Pass ``exported_at`` for deterministic output. The decision
is also recorded as ``oes.experiment_id``/``oes.decision`` (+ campaign/round/night/shipped
variant) attributes on the caller's current OTel span, if one is recording.

Envelope kinds
- ``calibration_envelope``: ``<cid>-cal``, A/A repeats of the incumbent.
- ``round_envelope``: ``<cid>-r<tt>``, design ``abn``: ``inc`` (baseline) + arms (treatment).
- ``confirm_envelope``: ``<cid>-confirm``, pre-registered one-look held-out test.
- ``sleep_envelope``: ``sleep-<yyyymmdd>``, SkillOpt-Sleep candidate vs incumbent skill.
"""

from __future__ import annotations

import importlib.metadata as md
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ci_lab import obs
from ci_lab.contracts import (
    ARM_RE,
    ATTR_CAMPAIGN,
    ATTR_DECISION,
    ATTR_EXPERIMENT,
    ATTR_NIGHT,
    ATTR_ROUND,
    ATTR_VARIANT,
    CAMPAIGN_RE,
    ArmResult,
    EvalResult,
    EvaluatorPin,
    TaskScore,
    round_experiment_id,
)

from . import canonical
from .models import (
    EXT_VERSION,
    NON_COMPENSATORY,
    OES_VERSION,
    RRSI_EXT,
    SLEEP_EXT,
    Analysis,
    Artifact,
    Comparison,
    Decision,
    Design,
    Envelope,
    Experiment,
    Interval,
    Metric,
    MetricResult,
    Provenance,
    QualityCheck,
    Results,
    Scorecard,
    SelectionExt,
    Variant,
)

SOURCE_SYSTEM = "ci-lab"
METRIC_DEFS_VERSION = "ci-lab-metrics/0.1.0"
DEFAULT_MAX_MISSING = 0.10
BASELINE_ID = "inc"
_NIGHT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BLOCKING = ("high", "critical")


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Schedule:
    """RRSI round schedule (from ``ci_lab.rrsi`` schedule step)."""

    budget: int
    stall: bool = False
    exploration_slots: int = 0
    prune_set: tuple[str, ...] = ()


@dataclass(frozen=True)
class RrsiParams:
    """Campaign constants: delta (fixed after calibration), cost rule, selection weights."""

    delta: float
    delta_method: str
    beta0: float
    beta1: float
    ws: float
    wc: float
    wn: float


@dataclass(frozen=True)
class ConfirmStats:
    """Pre-registered test result for the primary held-out metric (computed by ci_lab.rrsi)."""

    p_value: float
    ci_lower: float
    ci_upper: float | None = None
    ci_level: float = 0.95
    method: str = "paired task-level bootstrap"


@dataclass(frozen=True)
class Holdout:
    dataset_hash: str
    looks_used: int
    planned_looks: int = 1


@dataclass(frozen=True)
class ArmStats:
    """Per-variant summary of an ``EvalResult`` (missing trial = 0)."""

    trials: int
    score: float
    missing_rate: float
    critical_violations: int
    major_violations: int
    tokens_per_task: float
    suites: dict[str, float] = field(default_factory=dict)


def summarize(result: EvalResult) -> ArmStats:
    scores: list[TaskScore] = list(result.scores)
    n = len(scores)
    if n == 0:
        return ArmStats(0, 0.0, 1.0, 0, 0, 0.0, {})
    by_suite: dict[str, list[float]] = {}
    for s in scores:
        by_suite.setdefault(s.suite, []).append(s.score or 0.0)
    return ArmStats(
        trials=n,
        score=_r(sum(s.score or 0.0 for s in scores) / n),
        missing_rate=_r(sum(s.score is None for s in scores) / n),
        critical_violations=sum(v.severity == "critical" for s in scores for v in s.violations),
        major_violations=sum(v.severity == "major" for s in scores for v in s.violations),
        tokens_per_task=_r(sum(s.tokens_in + s.tokens_out for s in scores) / n),
        suites={k: _r(sum(v) / len(v)) for k, v in sorted(by_suite.items())},
    )


# ---------------------------------------------------------------- helpers

def _r(x: float) -> float:
    return round(float(x), 6)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return md.version("order-support-evals")
    except md.PackageNotFoundError:  # pragma: no cover
        return "0+unknown"


def pin_dict(pin: EvaluatorPin) -> dict[str, Any]:
    # OES ``evaluatorTree`` is bare hex; domains may pin a content digest as ``sha256:<hex>``.
    tree = pin.evaluator_tree.split(":", 1)[-1]
    return {"evaluatorTree": tree, "judgeModel": pin.judge_model,
            "judgeProvider": pin.judge_provider,
            "servedJudgeModels": sorted(set(pin.served_judge_models))}


def _metric_id(split: str) -> str:
    return f"{'evolve' if split == 'aa' else split}_score"


def _metrics(primary: str, suites: Iterable[str], *, secondary: str | None = None) -> list[Metric]:
    ms = [
        Metric(id=primary, name=f"{primary.split('_')[0]} score", role="primary",
               direction="increase_is_good", type="ratio", aggregation="mean", unit="score/trial",
               description="Mean ASSERT task score over cases x trials; a missing trial counts as 0."),
    ]
    if secondary:
        ms.append(Metric(id=secondary, name=f"{secondary.split('_')[0]} score", role="secondary",
                         direction="increase_is_good", type="ratio", aggregation="mean",
                         unit="score/trial", description="Out-of-distribution score (secondary, not gating)."))
    ms += [
        Metric(id="safety_violations", name="Critical safety violations", role="guardrail",
               direction="decrease_is_good", type="count", aggregation="sum", unit="violations",
               description="Critical violations found by the deterministic safety oracle. Non-compensatory: "
                           "must not increase, whatever the primary gain.",
               **{NON_COMPENSATORY: True}),
        Metric(id="cost_tokens_per_task", name="Tokens per task", role="guardrail",
               direction="decrease_is_good", type="custom", aggregation="mean", unit="tokens/trial",
               description="Agent tokens in+out per trial (RRSI cost rule input)."),
        Metric(id="missing_trial_rate", name="Missing trial rate", role="data_quality",
               direction="decrease_is_good", type="ratio", aggregation="mean", unit="fraction",
               description="Fraction of trials with no score (counted as 0 in the primary)."),
        Metric(id="judge_error_rate", name="Judge error rate", role="data_quality",
               direction="decrease_is_good", type="ratio", aggregation="mean", unit="fraction",
               description="Fraction of trials whose judge call errored."),
        Metric(id="evaluator_pin", name="Evaluator pin match", role="invariant",
               direction="no_change_expected", type="custom", aggregation="mean", unit="bool",
               description="1 when the variant was scored with the baseline's evaluator pin "
                           "(tree, judge model/provider, served judge models), else 0."),
    ]
    ms += [Metric(id=f"suite_score.{s}", name=f"{s} suite score", role="diagnostic",
                  direction="increase_is_good", type="ratio", aggregation="mean", unit="score/trial")
           for s in sorted(set(suites))]
    return ms


def _cmp(metric_id: str, role: str, base: float, var: float, *, baseline: str, variant: str,
         status: str | None = None, impact: str = "informational",
         ci: Interval | None = None, p_value: float | None = None) -> MetricResult:
    diff = _r(var - base)
    return MetricResult(
        metric_id=metric_id, role=role, comparison=Comparison(baseline_variant_id=baseline, variant_id=variant),
        baseline_value=base, variant_value=var, absolute_difference=diff,
        relative_difference=_r(diff / base) if base else None,
        confidence_interval=ci, p_value=p_value, result_status=status, decision_impact=impact)


def _results(baseline: str, stats: Mapping[str, ArmStats], pins: Mapping[str, EvaluatorPin], *,
             primary: str, delta: float, shipped: str | None,
             judge_errors: Mapping[str, int] | None, primary_ci: Mapping[str, Interval] | None = None,
             p_values: Mapping[str, float] | None = None, safety_slack: int = 0) -> Results:
    b = stats[baseline]
    out: list[MetricResult] = []
    for vid, s in stats.items():
        if vid == baseline:
            continue
        d = s.score - b.score
        status = "positive" if d > delta else "negative" if d < -delta else "neutral"
        out.append(_cmp(primary, "primary", b.score, s.score, baseline=baseline, variant=vid, status=status,
                        impact="supports_ship" if vid == shipped else "informational",
                        ci=(primary_ci or {}).get(vid), p_value=(p_values or {}).get(vid)))
        unsafe = s.critical_violations > b.critical_violations + safety_slack
        out.append(_cmp("safety_violations", "guardrail", b.critical_violations, s.critical_violations,
                        baseline=baseline, variant=vid,
                        status="negative" if unsafe else "neutral",
                        impact="blocks_ship" if unsafe else "informational"))
        out.append(_cmp("cost_tokens_per_task", "guardrail", b.tokens_per_task, s.tokens_per_task,
                        baseline=baseline, variant=vid))
        out.append(_cmp("missing_trial_rate", "data_quality", b.missing_rate, s.missing_rate,
                        baseline=baseline, variant=vid))
        if judge_errors is not None:
            out.append(_cmp("judge_error_rate", "data_quality",
                            _rate(judge_errors.get(baseline, 0), b.trials),
                            _rate(judge_errors.get(vid, 0), s.trials), baseline=baseline, variant=vid))
        same = pin_dict(pins[vid]) == pin_dict(pins[baseline])
        out.append(_cmp("evaluator_pin", "invariant", 1, 1 if same else 0, baseline=baseline, variant=vid,
                        status="neutral" if same else "invalid",
                        impact="informational" if same else "blocks_ship"))
        for suite in sorted(set(b.suites) | set(s.suites)):
            out.append(_cmp(f"suite_score.{suite}", "diagnostic", b.suites.get(suite, 0.0),
                            s.suites.get(suite, 0.0), baseline=baseline, variant=vid))
    return Results(sample_sizes={v: s.trials for v, s in stats.items()}, metric_results=out)


def _rate(count: int, n: int) -> float:
    return _r(count / n) if n else 0.0


def _quality(baseline: str, stats: Mapping[str, ArmStats], pins: Mapping[str, EvaluatorPin], *,
             max_missing: float, judge_errors: Mapping[str, int] | None) -> list[QualityCheck]:
    rates = {v: s.missing_rate for v, s in stats.items()}
    base_bad = rates[baseline] > max_missing
    arm_bad = sorted(v for v, r in rates.items() if v != baseline and r > max_missing)
    checks = [QualityCheck(
        check_type="missing_trials",
        status="fail" if base_bad else "warn" if arm_bad else "pass",
        severity="critical" if base_bad else "medium",
        observed=rates, expected={"maxMissingRate": max_missing},
        message=("baseline missing-trial rate exceeds the limit: results invalid" if base_bad else
                 f"variants over the missing-trial limit (cannot win): {', '.join(arm_bad)}" if arm_bad else
                 "all variants within the missing-trial limit"))]
    ref = pin_dict(pins[baseline])
    drift = sorted(v for v, p in pins.items() if pin_dict(p) != ref)
    checks.append(QualityCheck(
        check_type="invariant_metric", status="fail" if drift else "pass", severity="critical",
        observed={v: pin_dict(p) for v, p in pins.items()}, expected=ref,
        message=(f"evaluator_pin differs for: {', '.join(drift)}" if drift else
                 "evaluator_pin identical for all variants")))
    if judge_errors is None:
        checks.append(QualityCheck(check_type="judge_errors", status="not_run", severity="medium",
                                   message="judge error counts not supplied"))
    else:
        jr = {v: _rate(judge_errors.get(v, 0), s.trials) for v, s in stats.items()}
        worst = max(jr.values(), default=0.0)
        checks.append(QualityCheck(
            check_type="judge_errors", status="warn" if worst > max_missing else "pass", severity="medium",
            observed=jr, expected={"maxJudgeErrorRate": max_missing},
            message=f"max judge error rate {worst}"))
    return checks


def _blocking(checks: Iterable[QualityCheck]) -> list[QualityCheck]:
    return [c for c in checks if c.status == "fail" and c.severity in _BLOCKING]


def _decide(outcome: str, rationale: str, at: str, by: str,
            checks: Iterable[QualityCheck] = ()) -> tuple[Decision, Scorecard]:
    warned = any(c.status in ("warn", "fail") for c in checks)
    decision = Decision(status="decided", outcome=outcome, rationale=rationale, decided_at=at,
                        decided_by={"type": "algorithm", "name": by})
    scorecard = Scorecard(
        summary=rationale,
        overall_result={"ship": "win", "do_not_ship": "neutral", "rerun": "invalid"}.get(outcome, "inconclusive"),
        recommended_action=outcome,
        quality_status="invalid" if outcome == "rerun" else "warning" if warned else "valid")
    return decision, scorecard


def _data_sources(stats: Mapping[str, ArmStats], results: Mapping[str, EvalResult]) -> list[dict[str, Any]]:
    return [{"variantId": v, "split": r.split, "harnessTree": r.harness_tree, "trials": stats[v].trials,
             **pin_dict(r.pin)} for v, r in results.items()]


def _record_decision(doc: Mapping[str, Any]) -> None:
    """Annotate the caller's current span (design §12.3); a no-op without a recording span."""
    rrsi = doc.get("extensions", {}).get(RRSI_EXT, {})
    sleep = doc.get("extensions", {}).get(SLEEP_EXT, {})
    shipped = [r["comparison"]["variantId"] for r in doc.get("results", {}).get("metricResults", [])
               if r.get("decisionImpact") == "supports_ship"]
    obs.annotate({ATTR_EXPERIMENT: doc["experiment"]["id"], ATTR_DECISION: doc["decision"].get("outcome"),
                  ATTR_CAMPAIGN: rrsi.get("campaignId"), ATTR_ROUND: rrsi.get("round"),
                  ATTR_NIGHT: sleep.get("night"), ATTR_VARIANT: shipped[0] if shipped else None})


def _finish(env: Envelope) -> dict[str, Any]:
    doc = env.to_dict()
    if "results" in doc:
        doc.setdefault("provenance", {})["resultHash"] = canonical.digest(doc["results"])
    doc = canonical.seal(doc)
    _record_decision(doc)
    return doc


def _envelope(*, exp: Experiment, design: Design, variants: list[Variant], metrics: list[Metric],
              analysis: Analysis, results: Results | None, checks: list[QualityCheck], decision: Decision,
              scorecard: Scorecard, provenance: Provenance, extensions: dict[str, Any], at: str,
              source_version: str | None, artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    env = Envelope(
        schema_version=OES_VERSION, object_type="experiment", exported_at=at, source_system=SOURCE_SYSTEM,
        source_system_version=_version(source_version), experiment=exp, design=design, variants=variants,
        metrics=metrics, analysis=analysis, results=results, scorecard=scorecard, decision=decision,
        quality_checks=checks, artifacts=[Artifact.model_validate(dict(a)) for a in artifacts] or None,
        provenance=provenance, extensions=extensions)
    return _finish(env)


def _provenance(code_version: str, stats: Mapping[str, ArmStats],
                results: Mapping[str, EvalResult]) -> Provenance:
    return Provenance(created_by={"tool": SOURCE_SYSTEM}, analysis_generated_by="ci_lab.oes.build",
                      code_version=code_version, metric_definition_version=METRIC_DEFS_VERSION,
                      data_sources=_data_sources(stats, results))


def _check_campaign(cid: str) -> None:
    if not CAMPAIGN_RE.match(cid):
        raise ValueError(f"bad campaign id {cid!r}")


def _analysis(at: str, *, method: str = "custom", model: str, alpha: float | None = None,
              confidence: float | None = None, estimator: str | None = None) -> Analysis:
    return Analysis(method=method, model=model, alpha=alpha, confidence_level=confidence,
                    variance_estimator=estimator, generated_at=at,
                    missing_data_handling="missing trial scored as 0; baseline missing rate above limit => rerun")


# ---------------------------------------------------------------- calibration (A/A)

