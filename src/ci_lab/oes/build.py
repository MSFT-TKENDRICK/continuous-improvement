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
import math
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
from ci_lab.metrics.simplicity import simplicity_score

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


SURFACE_METRICS = (
    Metric(id="surface_complexity", name="Surface complexity", role="diagnostic", direction="decrease_is_good",
           type="custom", aggregation="mean", unit="complexity",
           description="Weighted surface-complexity composite of the evaluated harness "
                       "(ci_lab.metrics.simplicity; RRSI dX input, never part of the score)."),
    Metric(id="simplicity_score", name="Simplicity score", role="diagnostic", direction="increase_is_good",
           type="ratio", aggregation="mean", unit="score",
           description="ci_lab.metrics.simplicity.simplicity_score against the baseline surface "
                       "(1 at dX <= -10%, 0.5 unchanged, 0 at dX >= +25%)."),
)


def _complexity(r: EvalResult) -> float | None:
    v = (r.surface or {}).get("complexity")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _surface(baseline: str, by_id: Mapping[str, EvalResult]) -> tuple[list[Metric], list[MetricResult]]:
    """Diagnostic surface metrics, only when the baseline and at least one variant carry surface complexity."""
    base = by_id[baseline]
    bx = _complexity(base)
    if bx is None:
        return [], []
    out: list[MetricResult] = []
    for vid, r in by_id.items():
        vx = _complexity(r)
        if vid == baseline or vx is None:
            continue
        out.append(_cmp("surface_complexity", "diagnostic", bx, vx, baseline=baseline, variant=vid))
        out.append(_cmp("simplicity_score", "diagnostic", simplicity_score(base.surface, base.surface),
                        simplicity_score(r.surface, base.surface), baseline=baseline, variant=vid))
    return ([m.model_copy() for m in SURFACE_METRICS] if out else []), out


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

def calibration_envelope(campaign_id: str, runs: Sequence[EvalResult], *, delta: float, delta_method: str,
                         harness_commit: str, split_hashes: Mapping[str, str],
                         judge_errors: Mapping[str, int] | None = None,
                         max_missing_rate: float = DEFAULT_MAX_MISSING, exported_at: str | None = None,
                         source_version: str | None = None,
                         artifacts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """A/A calibration: ``runs`` are R>=2 repeated evaluations of the incumbent (``rep0`` = baseline)."""
    _check_campaign(campaign_id)
    if len(runs) < 2:
        raise ValueError("A/A calibration needs at least 2 repeats")
    at = exported_at or _now()
    ids = [f"rep{i}" for i in range(len(runs))]
    by_id = dict(zip(ids, runs, strict=True))
    stats = {v: summarize(r) for v, r in by_id.items()}
    pins = {v: r.pin for v, r in by_id.items()}
    primary = _metric_id(runs[0].split)
    results = _results(ids[0], stats, pins, primary=primary, delta=delta, shipped=None, judge_errors=judge_errors)
    checks = _quality(ids[0], stats, pins, max_missing=max_missing_rate, judge_errors=judge_errors)
    pairs = {f"{a}-{b}": _r(abs(stats[a].score - stats[b].score))
             for i, a in enumerate(ids) for b in ids[i + 1:]}
    widest = max(pairs.values())
    checks.append(QualityCheck(
        check_type="aa_noise_band", status="pass" if widest <= delta else "warn", severity="low",
        observed={"maxAbsDeltaS": widest, "pairs": pairs}, expected={"delta": delta},
        message=f"A/A max |dS| {widest} vs delta {delta} ({delta_method})"))
    trees = {r.harness_tree for r in runs}
    checks.append(QualityCheck(
        check_type="harness_identity", status="pass" if len(trees) == 1 else "fail", severity="critical",
        observed=sorted(trees), expected=runs[0].harness_tree,
        message="all repeats evaluate the same harness tree" if len(trees) == 1 else
                "A/A repeats evaluated different harness trees"))
    blocking = _blocking(checks)
    if blocking:
        outcome, why = "rerun", "calibration invalid: " + "; ".join(c.message or c.check_type for c in blocking)
    else:
        outcome, why = "do_not_ship", (f"A/A calibration of the incumbent ({len(runs)} repeats); no change to ship. "
                                       f"delta fixed at {delta} for campaign {campaign_id}.")
    decision, scorecard = _decide(outcome, why, at, "rrsi.calibration", checks)
    variants = [Variant(id=v, key=v, name=f"incumbent repeat {i}", role="baseline" if i == 0 else "treatment",
                        description="A/A copy of the incumbent harness", config={"harnessTree": by_id[v].harness_tree},
                        code_references=[{"type": "commit", "sha": harness_commit}])
                for i, v in enumerate(ids)]
    ext = {
        "version": EXT_VERSION, "kind": "calibration", "campaignId": campaign_id, "round": 0,
        "split": runs[0].split, "multipleTesting": "none", "repeats": len(runs), "delta": delta,
        "deltaMethod": delta_method, "harnessTree": runs[0].harness_tree, "evaluatorPin": pin_dict(runs[0].pin),
        "splitHashes": dict(split_hashes), "supersedes": None,
    }
    return _envelope(
        exp=Experiment(id=f"{campaign_id}-cal", slug=f"{campaign_id}-cal", status="decided",
                       title=f"{campaign_id}: A/A noise calibration",
                       hypothesis="Repeated evaluations of an unchanged harness differ only by evaluation noise.",
                       learning_goal="Fix the campaign noise band delta used by every RRSI round.",
                       tags=["rrsi", "calibration", campaign_id]),
        design=Design(type="ab" if len(runs) == 2 else "abn", randomization_unit="task",
                      analysis_unit="task_trial", assignment_method="every task evaluated in every repeat",
                      population=f"{runs[0].split} split", multiple_testing_policy="none",
                      peeking_policy="fixed_horizon"),
        variants=variants, metrics=_metrics(primary, stats[ids[0]].suites),
        analysis=_analysis(at, model=f"A/A repeat comparison; delta via {delta_method}"),
        results=results, checks=checks, decision=decision, scorecard=scorecard,
        provenance=_provenance(harness_commit, stats, by_id),
        extensions={RRSI_EXT: ext}, at=at, source_version=source_version, artifacts=artifacts)


# ---------------------------------------------------------------- RRSI round

def _variant_ext(arm: ArmResult, archive_ref: str | None) -> dict[str, Any]:
    return {
        "status": arm.status, "baseCommit": arm.base_commit, "headCommit": arm.head_commit,
        "harnessTree": arm.harness_tree,
        "edits": [{"component": e.component, "hypothesis": e.hypothesis, "commit": e.commit,
                   "files": list(e.files)} for e in arm.edits],
        "critic": None if arm.critic is None else {"passed": arm.critic.passed, "reasons": list(arm.critic.reasons),
                                                   "repairs": arm.critic.repairs},
        "archiveRef": archive_ref,
    }


def round_envelope(campaign_id: str, round_no: int, *, incumbent: EvalResult, incumbent_commit: str,
                   arms: Sequence[ArmResult], selection: SelectionExt | Mapping[str, Any], schedule: Schedule,
                   params: RrsiParams, split_hashes: Mapping[str, str],
                   lineage: Mapping[str, Any] | None = None, supersedes: str | None = None,
                   archive_refs: Mapping[str, str] | None = None,
                   judge_errors: Mapping[str, int] | None = None,
                   max_missing_rate: float = DEFAULT_MAX_MISSING, exported_at: str | None = None,
                   source_version: str | None = None,
                   artifacts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """One exploratory RRSI round: baseline ``inc`` (incumbent) vs each arm (treatment).

    ``selection`` is the Alg.2 trace from ``ci_lab.rrsi`` (winner + per-candidate rule trace).
    Arms without an ``eval`` (rejected/failed) are listed as variants with no results.
    """
    _check_campaign(campaign_id)
    exp_id = round_experiment_id(campaign_id, round_no)
    if round_no < 1:
        raise ValueError("rounds start at 1 (round 0 is calibration)")
    if incumbent.split != "evolve":
        raise ValueError("rounds evaluate the evolve split only")
    names = [a.arm for a in arms]
    if len(set(names)) != len(names) or BASELINE_ID in names or not all(ARM_RE.match(n) for n in names):
        raise ValueError(f"arm names must be unique, valid and not {BASELINE_ID!r}: {names}")
    sel = selection if isinstance(selection, SelectionExt) else SelectionExt.model_validate(dict(selection))
    evaluated = {a.arm: a.eval for a in arms if a.eval is not None}
    if sel.winner is not None and sel.winner not in evaluated:
        raise ValueError(f"winner {sel.winner!r} is not an evaluated arm")
    at = exported_at or _now()

    by_id: dict[str, EvalResult] = {BASELINE_ID: incumbent, **evaluated}
    stats = {v: summarize(r) for v, r in by_id.items()}
    pins = {v: r.pin for v, r in by_id.items()}
    cands = {c.variant_id: c for c in sel.candidates}
    ci = {v: Interval(level=c.ci_level, lower=c.ci_lower_bound)
          for v, c in cands.items() if c.ci_lower_bound is not None and v in evaluated}
    checks = _quality(BASELINE_ID, stats, pins, max_missing=max_missing_rate, judge_errors=judge_errors)
    for a in arms:
        if a.critic is not None:
            checks.append(QualityCheck(
                check_type=f"critic_{a.arm}", status="pass" if a.critic.passed else "fail", severity="medium",
                observed={"reasons": list(a.critic.reasons), "repairs": a.critic.repairs},
                message=f"critic {'passed' if a.critic.passed else 'rejected'} arm {a.arm}"))
    blocking = _blocking(checks)
    winner = None if blocking else sel.winner
    results = _results(BASELINE_ID, stats, pins, primary="evolve_score", delta=params.delta, shipped=winner,
                       judge_errors=judge_errors, primary_ci=ci)
    surface_defs, surface_results = _surface(BASELINE_ID, by_id)
    results.metric_results.extend(surface_results)
    if blocking:
        outcome = "rerun"
        why = f"round {round_no} invalid: " + "; ".join(c.message or c.check_type for c in blocking)
    elif winner:
        c = cands.get(winner)
        outcome = "ship"
        why = (f"arm {winner} admissible under the {c.rule if c else 'selection'} rule"
               + (f" (dS={c.delta_s}, dC={c.delta_c})" if c else "") + "; becomes the new incumbent.")
    else:
        outcome, why = "do_not_ship", f"no admissible arm in round {round_no}; incumbent kept."
    decision, scorecard = _decide(outcome, why, at, "rrsi.alg2", checks)
    if winner:
        scorecard.primary_outcome = {"metricId": "evolve_score", "variantId": winner,
                                     "absoluteDifference": _r(stats[winner].score - stats[BASELINE_ID].score)}
        scorecard.guardrail_outcomes = [
            {"metricId": "safety_violations", "variantId": winner,
             "status": "pass" if stats[winner].critical_violations <= stats[BASELINE_ID].critical_violations
             else "fail"}]

    variants = [Variant(id=BASELINE_ID, key=BASELINE_ID, name="incumbent", role="baseline",
                        description="Current campaign incumbent harness",
                        config={"harnessTree": incumbent.harness_tree},
                        code_references=[{"type": "commit", "sha": incumbent_commit}])]
    refs = dict(archive_refs or {})
    for a in arms:
        code_refs = [{"type": "commit", "sha": e.commit, "component": e.component} for e in a.edits]
        if a.head_commit:
            code_refs.append({"type": "branch", "ref": f"exp/{exp_id}/{a.arm}", "sha": a.head_commit})
        if a.arm in refs:
            code_refs.append({"type": "tag", "ref": refs[a.arm]})
        variants.append(Variant(
            id=a.arm, key=a.arm, name=f"arm {a.arm}", role="treatment",
            description="; ".join(f"[{e.component}] {e.hypothesis}" for e in a.edits) or f"arm {a.arm} ({a.status})",
            config={"harnessTree": a.harness_tree, "status": a.status}, code_references=code_refs or None))

    parent = f"{campaign_id}-cal" if round_no == 1 else round_experiment_id(campaign_id, round_no - 1)
    lin = {"parentExperimentId": parent, "incumbentCommit": incumbent_commit, "incumbentTree": incumbent.harness_tree}
    lin.update(lineage or {})
    ext = {
        "version": EXT_VERSION, "kind": "round", "campaignId": campaign_id, "round": round_no, "split": "evolve",
        "multipleTesting": "exploratory", "budget": schedule.budget, "stall": schedule.stall,
        "explorationSlots": schedule.exploration_slots, "pruneSet": list(schedule.prune_set),
        "delta": params.delta, "deltaMethod": params.delta_method,
        "costRule": {"beta0": params.beta0, "beta1": params.beta1},
        "weights": {"ws": params.ws, "wc": params.wc, "wn": params.wn},
        "selection": sel.to_dict(),
        "variants": {BASELINE_ID: {"status": "incumbent", "baseCommit": incumbent_commit,
                                   "headCommit": incumbent_commit, "harnessTree": incumbent.harness_tree,
                                   "edits": [], "critic": None, "archiveRef": None},
                     **{a.arm: _variant_ext(a, refs.get(a.arm)) for a in arms}},
        "harnessTree": incumbent.harness_tree, "evaluatorPin": pin_dict(incumbent.pin),
        "splitHashes": dict(split_hashes), "supersedes": supersedes, "lineage": lin,
        "ciLowerBound": cands[winner].ci_lower_bound if winner and winner in cands else None,
    }
    return _envelope(
        exp=Experiment(id=exp_id, slug=exp_id, status="decided", title=f"{campaign_id}: RRSI round {round_no}",
                       hypothesis="At least one proposed harness edit improves the evolve score beyond the noise "
                                  "band without worsening safety (exploratory; not an inferential claim).",
                       learning_goal="Select the next incumbent via RRSI Alg. 2.",
                       tags=["rrsi", "round", campaign_id]),
        design=Design(type="abn", randomization_unit="task", analysis_unit="task_trial",
                      assignment_method="every task evaluated on every variant (paired)",
                      population="evolve split", multiple_testing_policy="custom", peeking_policy="informal",
                      stopping_rule=f"fixed budget b_t={schedule.budget}; one evaluation per variant",
                      minimum_detectable_effect=params.delta),
        variants=variants, metrics=_metrics("evolve_score", stats[BASELINE_ID].suites) + surface_defs,
        analysis=_analysis(at, model="RRSI Alg.2 regularized selection (cost rule / weighted rule, "
                                     "non-compensatory safety guard, floor S* - delta)", estimator="bootstrap"),
        results=results, checks=checks, decision=decision, scorecard=scorecard,
        provenance=_provenance(incumbent_commit, stats, by_id),
        extensions={RRSI_EXT: ext}, at=at, source_version=source_version, artifacts=artifacts)


# ---------------------------------------------------------------- confirmation

def confirm_envelope(campaign_id: str, *, baseline: EvalResult, final: EvalResult, baseline_commit: str,
                     final_commit: str, stats: ConfirmStats, holdout: Holdout, look_ledger_ref: str,
                     split_hashes: Mapping[str, str], alpha: float = 0.05, sided: str = "one",
                     non_inferiority_margin: int = 0, registered_at: str | None = None,
                     ood: tuple[EvalResult, EvalResult] | None = None, accepted_rounds: Sequence[str] = (),
                     supersedes: str | None = None, judge_errors: Mapping[str, int] | None = None,
                     max_missing_rate: float = DEFAULT_MAX_MISSING, exported_at: str | None = None,
                     source_version: str | None = None,
                     artifacts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Pre-registered confirmation H0 (``h0``, baseline) vs H_final (``final``) on the sealed held-out split.

    Ship iff quality is valid, ``p_value < alpha``, the CI lower bound is > 0 and safety is
    non-inferior (final critical violations <= h0 + margin). OOD (optional) is secondary.
    """
    _check_campaign(campaign_id)
    if baseline.split != "heldout" or final.split != "heldout":
        raise ValueError("confirmation must evaluate the held-out split")
    if not 1 <= holdout.looks_used <= holdout.planned_looks:
        raise ValueError(f"held-out look budget exceeded: {holdout.looks_used}/{holdout.planned_looks}")
    at = exported_at or _now()
    by_id = {"h0": baseline, "final": final}
    st = {v: summarize(r) for v, r in by_id.items()}
    pins = {v: r.pin for v, r in by_id.items()}
    checks = _quality("h0", st, pins, max_missing=max_missing_rate, judge_errors=judge_errors)
    blocking = _blocking(checks)
    safe = st["final"].critical_violations <= st["h0"].critical_violations + non_inferiority_margin
    sig = stats.p_value < alpha and stats.ci_lower > 0
    shipped = "final" if (not blocking and sig and safe) else None
    results = _results("h0", st, pins, primary="heldout_score", delta=0.0, shipped=shipped,
                       judge_errors=judge_errors, safety_slack=non_inferiority_margin,
                       primary_ci={"final": Interval(level=stats.ci_level, lower=stats.ci_lower,
                                                     upper=stats.ci_upper)},
                       p_values={"final": stats.p_value})
    metrics = _metrics("heldout_score", st["h0"].suites, secondary="ood_score" if ood else None)
    if ood:
        ob, of = summarize(ood[0]), summarize(ood[1])
        results.metric_results.append(_cmp("ood_score", "secondary", ob.score, of.score,
                                           baseline="h0", variant="final"))
    if blocking:
        outcome = "rerun"
        why = "confirmation invalid: " + "; ".join(c.message or c.check_type for c in blocking)
    elif shipped:
        outcome = "ship"
        why = (f"H_final beats H0 on held-out (p={stats.p_value} < alpha={alpha}, CI lower {stats.ci_lower} > 0) "
               f"and safety is non-inferior (margin {non_inferiority_margin}).")
    else:
        outcome = "do_not_ship"
        why = ("pre-registered test not passed: " + ", ".join(
            x for x, bad in (("not significant", not sig), ("safety inferior", not safe)) if bad))
    decision, scorecard = _decide(outcome, why, at, "rrsi.confirm", checks)
    ext = {
        "version": EXT_VERSION, "kind": "confirm", "campaignId": campaign_id,
        "round": len(accepted_rounds), "split": "heldout", "multipleTesting": "confirmatory",
        "harnessTree": baseline.harness_tree, "evaluatorPin": pin_dict(baseline.pin),
        "splitHashes": dict(split_hashes),
        "holdout": {"datasetHash": holdout.dataset_hash, "plannedLooks": holdout.planned_looks,
                    "looksUsed": holdout.looks_used},
        "lookLedgerRef": look_ledger_ref,
        "preRegistration": {"alpha": alpha, "sided": sided, "primaryMetric": "heldout_score",
                            "nonInferiorityMargin": non_inferiority_margin,
                            **({"registeredAt": registered_at} if registered_at else {}),
                            "stats": {"method": stats.method, "pValue": stats.p_value, "ciLevel": stats.ci_level,
                                      "ciLower": stats.ci_lower, "ciUpper": stats.ci_upper}},
        "supersedes": supersedes,
        "lineage": {"parentExperimentId": accepted_rounds[-1] if accepted_rounds else f"{campaign_id}-cal",
                    "incumbentCommit": baseline_commit, "incumbentTree": baseline.harness_tree,
                    "finalCommit": final_commit, "finalTree": final.harness_tree,
                    "acceptedRounds": list(accepted_rounds)},
        "ciLowerBound": stats.ci_lower,
    }
    return _envelope(
        exp=Experiment(id=f"{campaign_id}-confirm", slug=f"{campaign_id}-confirm", status="decided",
                       title=f"{campaign_id}: held-out confirmation",
                       hypothesis="The final harness scores higher than the campaign starting harness on the sealed "
                                  "held-out split, without more critical safety violations.",
                       tags=["rrsi", "confirm", campaign_id]),
        design=Design(type="ab", randomization_unit="task", analysis_unit="task",
                      assignment_method="every held-out task evaluated on both variants (paired)",
                      population="sealed held-out split", alpha=alpha, multiple_testing_policy="none",
                      peeking_policy="fixed_horizon",
                      stopping_rule=f"single pre-registered look ({holdout.looks_used}/{holdout.planned_looks})"),
        variants=[Variant(id="h0", key="h0", name="campaign start (H0)", role="baseline",
                          config={"harnessTree": baseline.harness_tree},
                          code_references=[{"type": "commit", "sha": baseline_commit}]),
                  Variant(id="final", key="final", name="final incumbent (H_final)", role="treatment",
                          config={"harnessTree": final.harness_tree},
                          code_references=[{"type": "commit", "sha": final_commit}])],
        metrics=metrics,
        analysis=_analysis(at, method="frequentist", model=stats.method, alpha=alpha,
                           confidence=stats.ci_level, estimator="bootstrap"),
        results=results, checks=checks, decision=decision, scorecard=scorecard,
        provenance=_provenance(final_commit, st, by_id),
        extensions={RRSI_EXT: ext}, at=at, source_version=source_version, artifacts=artifacts)


# ---------------------------------------------------------------- SkillOpt-Sleep

def sleep_envelope(night: str, *, incumbent: EvalResult | None, candidate: EvalResult | None,
                   incumbent_commit: str,
                   skillopt_version: str, tasks_by_origin: Mapping[str, int], tasks_by_split: Mapping[str, int],
                   skillopt_gate: Mapping[str, Any], assert_gate: Mapping[str, Any], delta: float,
                   budget_used: Mapping[str, Any], budget_limits: Mapping[str, Any] | None = None,
                   candidate_digest: str | None = None, incumbent_digest: str | None = None,
                   adoption_pr: Mapping[str, Any] | None = None, night_index: int | None = None,
                   skill_path: str | None = None, judge_errors: Mapping[str, int] | None = None,
                   evaluator_pin: EvaluatorPin | None = None, rerun_reason: str | None = None,
                   max_missing_rate: float = DEFAULT_MAX_MISSING, exported_at: str | None = None,
                   source_version: str | None = None,
                   artifacts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Nightly SkillOpt-Sleep record: ``incumbent`` (baseline) vs ``candidate`` skill (treatment).

    ``skillopt_gate``/``assert_gate`` need at least ``passed``; ``deltaS``/``delta``/
    ``safetyViolations`` are filled from the eval results. Ship (= open the adoption PR,
    never auto-merge) iff a candidate exists and both gates passed.

    ``incumbent=None`` records a night where nothing reached the ASSERT gate (no candidate,
    no tasks, budget spent first): a single ``incumbent`` baseline variant, no results, an
    ``assert_gate`` quality check ``not_run`` and a ``do_not_ship`` "no change, control
    retained" decision. It needs ``evaluator_pin`` (the gate's pin, from the ASSERT domain).
    ``rerun_reason`` adds a failing ``sleep_budget`` check, which turns the decision into ``rerun``.
    """
    if not _NIGHT_RE.match(night):
        raise ValueError(f"night must be YYYY-MM-DD: {night!r}")
    if incumbent is None and candidate is not None:
        raise ValueError("a candidate eval needs the incumbent eval it is paired with")
    if incumbent is None and evaluator_pin is None:
        raise ValueError("a night without an incumbent eval needs evaluator_pin")
    at = exported_at or _now()
    by_id = {**({"incumbent": incumbent} if incumbent is not None else {}),
             **({"candidate": candidate} if candidate is not None else {})}
    st = {v: summarize(r) for v, r in by_id.items()}
    pins = {v: r.pin for v, r in by_id.items()}
    if incumbent is not None:
        checks = _quality("incumbent", st, pins, max_missing=max_missing_rate, judge_errors=judge_errors)
    else:
        checks = [QualityCheck(check_type="assert_gate", status="not_run", severity="low",
                               message="no candidate reached the ASSERT gate; the incumbent was not re-evaluated")]
    if rerun_reason:
        checks.append(QualityCheck(check_type="sleep_budget", status="fail", severity="high", message=rerun_reason))
    blocking = _blocking(checks)
    sk = {"reasons": [], **dict(skillopt_gate)}
    ag = {"reasons": [], **dict(assert_gate)}
    ag.setdefault("delta", delta)
    ag.setdefault("deltaS", _r(st["candidate"].score - st["incumbent"].score) if candidate else None)
    if incumbent is not None:
        ag.setdefault("safetyViolations", {"baseline": st["incumbent"].critical_violations,
                                           "candidate": st["candidate"].critical_violations if candidate else None})
    ag.setdefault("ciLowerBound", None)
    passed = candidate is not None and bool(sk.get("passed")) and bool(ag.get("passed")) and candidate_digest
    shipped = "candidate" if passed and not blocking else None
    split = incumbent.split if incumbent is not None else "evolve"
    results = _results("incumbent", st, pins, primary=_metric_id(split), delta=delta, shipped=shipped,
                       judge_errors=judge_errors) if incumbent is not None else None
    if blocking:
        outcome, why = "rerun", "sleep gate invalid: " + "; ".join(c.message or c.check_type for c in blocking)
    elif shipped:
        outcome, why = "ship", "candidate skill passed the SkillOpt and ASSERT gates; open adoption PR (human merge)."
    elif incumbent is None:
        outcome = "do_not_ship"
        why = "; ".join(["no candidate reached the ASSERT gate", *ag["reasons"]]) + \
            "; no change, control retained (incumbent skill kept)."
    else:
        reasons = ["no candidate"] if candidate is None else [
            n for n, g in (("SkillOpt gate failed", sk), ("ASSERT gate failed", ag)) if not g.get("passed")]
        outcome, why = "do_not_ship", "; ".join(reasons or ["no candidate digest"]) + "; incumbent skill kept."
    decision, scorecard = _decide(outcome, why, at, "sleep.gate", checks)
    exp_id = f"sleep-{night.replace('-', '')}"
    pin = incumbent.pin if incumbent is not None else evaluator_pin
    assert pin is not None
    variants = [Variant(id="incumbent", key="incumbent", name="incumbent skill", role="baseline",
                        config={"harnessTree": incumbent.harness_tree if incumbent is not None else None,
                                "skillDigest": incumbent_digest},
                        code_references=[{"type": "commit", "sha": incumbent_commit}])]
    if candidate is not None:
        variants.append(Variant(id="candidate", key="candidate", name="SkillOpt-Sleep candidate skill",
                                role="treatment",
                                config={"harnessTree": candidate.harness_tree, "skillDigest": candidate_digest}))
    for v in variants:
        v.config = {k: x for k, x in (v.config or {}).items() if x is not None}
    ext: dict[str, Any] = {
        "version": EXT_VERSION, "night": night, "skilloptVersion": skillopt_version,
        "tasks": {"total": sum(tasks_by_origin.values()), "byOrigin": dict(tasks_by_origin),
                  "bySplit": dict(tasks_by_split)},
        "gate": {"skillopt": sk, "assert": ag},
        "budget": {"used": dict(budget_used), **({"limits": dict(budget_limits)} if budget_limits else {})},
        "candidateDigest": candidate_digest, "adoptionPr": dict(adoption_pr) if adoption_pr else None,
        "evaluatorPin": pin_dict(pin),
    }
    if night_index is not None:
        ext["nightIndex"] = night_index
    if skill_path:
        ext["skillPath"] = skill_path
    if incumbent_digest:
        ext["incumbentDigest"] = incumbent_digest
    return _envelope(
        exp=Experiment(id=exp_id, slug=exp_id, status="decided", title=f"SkillOpt-Sleep night {night}",
                       hypothesis="The consolidated skill improves the evolve score without worsening safety.",
                       tags=["sleep", "skillopt", *(["no-change"] if incumbent is None else [])]),
        design=Design(type="ab", randomization_unit="task", analysis_unit="task_trial",
                      assignment_method="every gate task evaluated on both variants (paired)",
                      population=f"{split} split", multiple_testing_policy="none",
                      peeking_policy="fixed_horizon", minimum_detectable_effect=delta),
        variants=variants,
        metrics=_metrics(_metric_id(split), st["incumbent"].suites if incumbent is not None else ()),
        analysis=_analysis(at, model=f"SkillOpt {skillopt_version} gate + ASSERT gate (non-compensatory safety)"),
        results=results, checks=checks, decision=decision, scorecard=scorecard,
        provenance=_provenance(incumbent_commit, st, by_id),
        extensions={SLEEP_EXT: ext}, at=at, source_version=source_version, artifacts=artifacts)


__all__ = ["ArmStats", "ConfirmStats", "Holdout", "RrsiParams", "Schedule", "calibration_envelope",
           "confirm_envelope", "pin_dict", "round_envelope", "sleep_envelope", "summarize"]
