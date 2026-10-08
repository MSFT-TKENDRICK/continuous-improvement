import math

import pytest

from ci_lab.lessons import stats
from ci_lab.rulespec import PROMOTE_FP_UCB


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.3084971),
        (1, 10, 0.0025286, 0.4450161),
        (5, 10, 0.1870860, 0.8129140),
        (10, 10, 0.6915029, 1.0),
        (0, 200, 0.0, 0.0182753),
    ],
)
def test_clopper_pearson_known_values(k, n, lo, hi):
    got_lo, got_hi = stats.clopper_pearson(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-6)
    assert got_hi == pytest.approx(hi, abs=1e-6)


@pytest.mark.parametrize(("k", "n"), [(3, 17), (12, 400), (1, 5000)])
def test_bounds_invert_exact_cdf(k, n):
    lo, hi = stats.clopper_pearson(k, n)
    assert stats.binom_cdf(k, n, hi) == pytest.approx(0.025, abs=1e-9)
    assert stats.binom_sf(k, n, lo) == pytest.approx(0.025, abs=1e-9)
    assert lo < k / n < hi


def test_one_sided_is_tighter_and_closed_form():
    assert stats.cp_upper(0, 100, one_sided=True) == pytest.approx(1 - 0.05 ** (1 / 100))
    assert stats.cp_upper(2, 100, one_sided=True) < stats.cp_upper(2, 100)


def test_cdf_matches_direct_sum():
    n, p = 30, 0.17
    direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(8))
    assert stats.binom_cdf(7, n, p) == pytest.approx(direct, rel=1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        stats.cp_upper(5, 3)
    with pytest.raises(ValueError):
        stats.cp_upper(1, 3, confidence=1.0)


def test_promotion_gate():
    ok, reasons = stats.promotion_ok(200, 20, 10, 0)
    assert ok, reasons
    ok, reasons = stats.promotion_ok(199, 20, 10, 0)
    assert not ok and any("opportunities" in r for r in reasons)
    ok, reasons = stats.promotion_ok(1000, 30, 9, 0)
    assert not ok and any("adjudicated" in r for r in reasons)
    ok, reasons = stats.promotion_ok(200, 25, 10, 1)  # 1/190 -> UCB ~0.029 > 0.02
    assert not ok and any("fp_ucb" in r for r in reasons)
    ok, _ = stats.promotion_ok(2000, 40, 20, 1)
    assert ok and stats.cp_upper(1, 1980) <= PROMOTE_FP_UCB
    ok, reasons = stats.promotion_ok(10, 20, 5, 0)
    assert not ok and reasons[0].startswith("inconsistent")


def test_promotion_stratified_requires_each_intent():
    ok, reasons = stats.promotion_ok_stratified({"refund": (1500, 30, 15, 0), "lookup": (40, 0, 0, 0)})
    assert not ok and any("stratum lookup" in r for r in reasons)
    ok, reasons = stats.promotion_ok_stratified({"refund": (1500, 30, 15, 0), "lookup": (500, 0, 0, 0)})
    assert ok, reasons
