"""Metric (resource) criteria: fail-closed MetricVoter, quality-only score, resource subscores, legacy bodies."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from ci_lab.bus.judge import aggregate
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.types import CriterionResult, ProposalBody, VerdictBody, VoteBody
from ci_lab.bus.voters.local import DeterministicCheckVoter, MetricVoter, run_voters
from ci_lab.taskgraph.model import Criterion, Rubric, SpecError
from ci_lab.taskgraph.validate import validate_rubric

SIZE = {"kind": "metric", "metric": "output_chars", "op": "le", "target": 10, "worst": 110}


def votes(kit: Any, rubric: Rubric, artifact: bytes, measurements: dict[str, float]) -> list[VoteBody]:
    body = replace(kit.proposal(artifact), measurements=measurements)
    return asyncio.run(run_voters([DeterministicCheckVoter(), MetricVoter()], body, artifact, rubric, kit.spec(),
                                  pools=ResourcePools({}), timeout_s=10))


def test_metric_criteria_are_resource_and_role_is_validated(kit: Any) -> None:
    m = kit.crit("m", "metric", SIZE)
    q = kit.crit("q")
    assert m.resource and m.role == "resource" and m.oracle and q.role == "quality" and not q.resource
    assert "role" not in q.to_json() and m.to_json()["role"] == "resource"
    assert Criterion.from_json(m.to_json()) == m and Criterion.from_json(q.to_json()) == q
    with pytest.raises(SpecError, match="always resource"):
        Criterion("m", "d", "metric", SIZE, 0.5, role="quality")
    with pytest.raises(SpecError, match="role"):
        Criterion("q", "d", "deterministic", {"kind": "regex", "pattern": "x"}, 0.5, role="bogus")
    soft_resource = Criterion("s", "d", "s1", {"question": "Is it lean?"}, 0.5, role="resource")
    assert soft_resource.resource and not soft_resource.oracle
    def rub(*cs: Criterion) -> Rubric:
        return replace(kit.rubric(*cs), canary="0123456789abcdef")

    assert not validate_rubric(rub(q, m))
    probs = {p.code for p in validate_rubric(rub(m))}
    assert "rubric.oracle" in probs  # a metric criterion alone is not a quality oracle
    bad = {p.code for p in validate_rubric(rub(q, kit.crit("m", "metric", {**SIZE, "worst": 1})))}
    assert "check.metric" in bad


def test_metric_voter_scores_measurements(kit: Any) -> None:
    m = kit.crit("m", "metric", SIZE)
    assert MetricVoter.ballot(m, {"output_chars": 10.0}).score == 1.0
    half = MetricVoter.ballot(m, {"output_chars": 60.0})
    assert half.passed is True and half.score == pytest.approx(0.5) and half.confidence == 1.0
    worst = MetricVoter.ballot(m, {"output_chars": 500.0})
    assert worst.passed is False and worst.score == 0.0


def test_missing_or_invalid_measurement_fails_closed_never_abstains(kit: Any) -> None:
    m = kit.crit("m", "metric", SIZE)
    miss = MetricVoter.ballot(m, {"tokens": 1.0})
    assert (miss.passed, miss.score, miss.confidence) == (False, 0.0, 1.0)
    assert miss.reasons == ("metric.missing: output_chars",)
    bad = MetricVoter.ballot(m, {"output_chars": float("nan")})
    assert bad.passed is False and bad.score == 0.0 and bad.reasons[0].startswith("metric.invalid")
    broken = Criterion("b", "d", "metric", {**SIZE, "metric": "nope"}, 0.5)
    assert MetricVoter.ballot(broken, {"output_chars": 1.0}).reasons[0].startswith("metric.invalid")
    rubric = kit.rubric(kit.crit("q"), m)
    vs = votes(kit, rubric, b"x", {})
    metric_vote = next(v for v in vs if v.voter == "metric")
    assert metric_vote.answered and metric_vote.passed is False and metric_vote.score == 0.0
    assert [v.criterion for v in vs] == ["q", "m"]


def test_quality_score_excludes_resource_and_records_subscores(kit: Any) -> None:
    rubric = kit.rubric(kit.crit("q"), kit.crit("m", "metric", SIZE))
    vs = votes(kit, rubric, b"x", {"output_chars": 85.0})
    draft = aggregate(rubric, vs, 1, attempts_left=1)
    assert draft.criteria["m"].resource and draft.criteria["m"].passed is False and not draft.criteria["q"].resource
    assert draft.score == 1.0 and draft.vetoed == () and draft.decision == "commit"  # resource never vetoes
    assert dict(draft.subscores) == pytest.approx({"resource.m": 0.25, "resource_score": 0.25})
    body = draft.to_body(dict(enumerate(vs)))
    assert body.subscores == draft.subscores and VerdictBody.from_json(body.to_json()) == body
    missing = aggregate(rubric, votes(kit, rubric, b"x", {}), 1, attempts_left=1)
    assert missing.score == 1.0 and dict(missing.subscores) == {"resource.m": 0.0, "resource_score": 0.0}
    req = kit.rubric(kit.crit("q"), replace(kit.crit("m", "metric", SIZE), required=True))
    blocked = aggregate(req, votes(kit, req, b"x", {}), 1, attempts_left=0)
    assert blocked.failed_required == ("m",) and blocked.vetoed == () and blocked.decision != "commit"
    plain = kit.rubric(kit.crit("q"))
    assert dict(aggregate(plain, votes(kit, plain, b"x", {}), 1, attempts_left=1).subscores) == {}


def test_legacy_bodies_round_trip_byte_for_byte(kit: Any) -> None:
    legacy = kit.proposal(b"x").to_json()
    assert "measurements" not in legacy and ProposalBody.from_json(legacy).to_json() == legacy
    rich = replace(kit.proposal(b"x"), measurements={"wall_ms": 12.0, "tokens": 3.0})
    assert rich.to_json()["measurements"] == {"wall_ms": 12.0, "tokens": 3.0}
    assert ProposalBody.from_json(rich.to_json()) == rich
    with pytest.raises(ValueError, match="measurements"):
        replace(kit.proposal(b"x"), measurements={"": 1.0})
    cr = CriterionResult(True, 1.0, False, True, 1)
    assert "resource" not in cr.to_json() and CriterionResult.from_json(cr.to_json()) == cr
    assert CriterionResult.from_json({**cr.to_json(), "resource": True}).resource
    rubric = kit.rubric(kit.crit("q"))
    qv = votes(kit, rubric, b"x", {})
    vb = aggregate(rubric, qv, 1, attempts_left=1).to_body(dict(enumerate(qv)))
    assert "subscores" not in vb.to_json() and VerdictBody.from_json(vb.to_json()) == vb
    with pytest.raises(ValueError, match="subscores"):
        replace(vb, subscores={"resource_score": 2.0})
