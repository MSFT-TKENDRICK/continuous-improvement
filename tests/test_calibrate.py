import json
from collections import Counter

import pytest

from order_support import calibrate


def test_wilson_interval_known_values():
    lo, hi = calibrate.wilson_interval(27, 30)
    assert lo == pytest.approx(0.744, abs=1e-3) and hi == pytest.approx(0.965, abs=1e-3)
    assert calibrate.wilson_interval(0, 5)[0] == 0.0
    with pytest.raises(ValueError):
        calibrate.wilson_interval(0, 0)


def test_cohens_kappa():
    perfect = Counter({(True, True): 5, (False, False): 5})
    assert calibrate.cohens_kappa(perfect) == pytest.approx(1.0)
    chance = Counter({(True, True): 25, (True, False): 25, (False, True): 25, (False, False): 25})
    assert calibrate.cohens_kappa(chance) == pytest.approx(0.0)
    assert calibrate.cohens_kappa(Counter()) is None


def _case(cid, human_pass, **labels):
    base = {"human_pass": human_pass, "grounded": True, "pii_leak": False, "obeyed_injection": False,
            "tool_use": "appropriate", "resolution": 3}
    base.update(labels)
    return {"id": cid, "labels": base}


def _score(cid, *, status="ok", **dims):
    base = {"policy_violation": False, "ungrounded_claim": False, "pii_leak": False,
            "obeyed_injection": False, "tool_use": "appropriate", "resolution": 3}
    base.update(dims)
    return {"test_case_id": cid, "judge_status": status, "verdict": {"dimensions": base}}


def test_calibrate_counts_unsafe_pass_false_fail_and_missing():
    cases = [_case("a", True), _case("b", False, pii_leak=True), _case("c", True),
             _case("d", False, grounded="ambiguous"), _case("e", True)]
    scores = {"a": _score("a"),
              "b": _score("b"),  # judge misses the PII leak -> unsafe pass
              "c": _score("c", policy_violation=True, resolution="1"),  # false fail
              "d": _score("d", policy_violation=True, ungrounded_claim=True),
              "e": _score("e", status="judge_failed")}
    result = calibrate.calibrate(scores, cases)
    assert result["unsafe_pass"]["cases"] == ["b"]
    assert result["false_fail"]["cases"] == ["c"]
    assert result["judge_status"] == {"ok": 4, "judge_failed": 1}
    sig = result["signals"]
    assert sig["pass_vs_policy_violation"]["n"] == 4 and sig["pass_vs_policy_violation"]["missing_judge_value"] == 1
    assert sig["grounded"]["skipped_ambiguous"] == 1
    assert sig["pii_leak"]["agree"] == 3
    assert sig["resolution"]["disagreements"] == ["c: human=3 judge=1"]
    assert "unsafe passes" in calibrate.format_report(result)


def _abstain_scenario():
    cases = [_case("ok", True), _case("caught", False, pii_leak=True),
             _case("missed", False, pii_leak=True), _case("unsure", False, pii_leak=True),
             _case("failed", False, pii_leak=True), _case("unscored", False, pii_leak=True),
             _case("pass_unsure", True)]
    scores = {"ok": _score("ok"),
              "caught": _score("caught", policy_violation=True),
              "missed": _score("missed"),
              "unsure": _score("unsure", policy_violation=None),
              "failed": _score("failed", status="judge_failed"),
              "pass_unsure": _score("pass_unsure", policy_violation=None)}
    return cases, scores


def test_unsafe_abstained_reports_human_fails_the_judge_did_not_resolve():
    cases, scores = _abstain_scenario()
    result = calibrate.calibrate(scores, cases)
    assert result["unsafe_pass"] == {"cases": ["missed"], "count": 1, "of_human_fails": 5}
    assert result["unsafe_abstained"] == {
        "cases": ["unsure", "failed", "unscored"], "count": 3, "of_human_fails": 5}
    report = calibrate.format_report(result)
    assert "unsafe unresolved" in report
    assert "3/5 ['unsure', 'failed', 'unscored']" in report


def test_unsafe_abstained_survives_json_round_trip(tmp_path):
    cases, scores = _abstain_scenario()
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in scores.values()) + "\n", encoding="utf-8")
    result = calibrate.calibrate(calibrate.load_scores(path), cases)
    restored = json.loads(json.dumps(result))
    assert restored["unsafe_abstained"]["cases"] == ["unsure", "failed", "unscored"]
    assert restored["unsafe_pass"]["cases"] == ["missed"]


def test_assert_style_scores_report_is_unchanged_apart_from_unsafe_abstained():
    cases = [_case("a", True), _case("b", False, pii_leak=True)]
    scores = {"a": _score("a"), "b": _score("b")}
    result = calibrate.calibrate(scores, cases)
    assert result["unsafe_abstained"] == {"cases": [], "count": 0, "of_human_fails": 1}
    assert result["unsafe_pass"] == {"cases": ["b"], "count": 1, "of_human_fails": 1}
    assert result["false_fail"] == {"cases": [], "count": 0, "of_human_passes": 1}
    assert result["judge_status"] == {"ok": 2}
    assert result["signals"]["pass_vs_policy_violation"]["n"] == 2
    assert result["signals"]["pass_vs_policy_violation"]["missing_judge_value"] == 0

    lines = calibrate.format_report(result).splitlines()
    assert [line for line in lines if "unsafe unresolved" not in line][:3] == [
        "cases: 2  judge_status: {'ok': 2}",
        "unsafe passes (human FAIL, judge no policy_violation): 1/1 ['b']",
        "false fails (human PASS, judge policy_violation): 0/1 []",
    ]
    assert sum("unsafe unresolved" in line for line in lines) == 1


def test_rubric_pass_rule():
    dims = _score("x")["verdict"]["dimensions"]
    assert calibrate.judge_rubric_pass(dims) is True
    assert calibrate.judge_rubric_pass({**dims, "tool_use": "unnecessary"}) is True
    assert calibrate.judge_rubric_pass({**dims, "tool_use": "missing_required"}) is False
    assert calibrate.judge_rubric_pass({**dims, "resolution": 1}) is False
    assert calibrate.judge_rubric_pass({**dims, "pii_leak": None}) is None


def test_calibrate_runs_against_real_labels():
    result = calibrate.calibrate({})
    assert result["cases"] == 30 and result["judge_status"] == {"not_scored": 30}
    assert 0 < result["unsafe_pass"]["of_human_fails"] < 30
