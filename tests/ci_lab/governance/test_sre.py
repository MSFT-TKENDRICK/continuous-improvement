from __future__ import annotations

import pytest

from ci_lab.governance.sre import ArmReliability, JudgePoolReliability


def test_unknown_arm_is_not_vetoed():
    book = ArmReliability()
    assert not book.vetoed("lessons") and book.vetoes() == [] and book.snapshot() == {}


def test_budget_exhaustion_circuit_breaks_only_that_arm():
    book = ArmReliability(max_failures=2)
    assert book.record("lessons", True) is False
    assert book.record("lessons", False) is False
    assert book.status("lessons").status == "degraded"
    assert book.record("lessons", False) is True
    book.record("baseline", False)
    assert book.vetoed("lessons") and not book.vetoed("baseline")
    assert book.vetoes() == ["lessons"]
    s = book.status("lessons")
    assert (s.good, s.bad, s.status, s.budget_remaining, s.kind) == (1, 2, "exhausted", 0.0, "arm")
    assert s.success_rate == pytest.approx(1 / 3)


def test_successes_never_refill_budget_and_reset_clears():
    book = ArmReliability(max_failures=1)
    book.record("a", False)
    for _ in range(50):
        book.record("a", True)
    assert book.vetoed("a")
    book.reset("a")
    assert not book.vetoed("a")


def test_judge_pool_snapshot_and_slo_wiring():
    pools = JudgePoolReliability(max_failures=3, target=0.95)
    pools.record("s1-gpt", True)
    pools.record("s1-gpt", False)
    snap = pools.snapshot()
    assert snap["s1-gpt"]["kind"] == "judge_pool" and snap["s1-gpt"]["vetoed"] is False
    assert snap["s1-gpt"]["budget_remaining"] == pytest.approx(2 / 3)
    slo = pools._slos["s1-gpt"]
    assert slo.name == "ci.judge_pool.s1-gpt" and slo.error_budget.exhaustion_action.value == "circuit_break"


@pytest.mark.parametrize("kwargs", [{"max_failures": 0}, {"target": 1.0}, {"target": 0.0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ArmReliability(**kwargs)


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        ArmReliability().record("", True)
