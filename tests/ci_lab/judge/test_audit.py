"""Judge-vs-human audit metrics (pure Python, deterministic)."""

from __future__ import annotations

import json
import math

import pytest

from ci_lab.judge import audit as A


def test_cohens_kappa_known_values():
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert A.cohens_kappa(pairs) == pytest.approx(0.4)
    assert A.cohens_kappa([(1, 1), (2, 2)]) == pytest.approx(1.0)
    assert A.cohens_kappa([(1, 1), (1, 1)]) is None  # no variance -> undefined


def test_weighted_kappa_and_spearman():
    order = [0, 1, 2, 3]
    perfect = [(i, i) for i in order] * 3
    assert A.weighted_kappa(perfect, order) == pytest.approx(1.0)
    off_by_one = [(0, 1), (1, 2), (2, 3), (3, 3), (0, 0), (1, 1), (2, 2), (3, 2)]
    far = [(0, 3), (1, 3), (2, 0), (3, 0), (0, 0), (1, 1), (2, 2), (3, 3)]
    assert A.weighted_kappa(off_by_one, order) > A.weighted_kappa(far, order)
    # reference: 1 - sum(W*O)/sum(W*E) from an explicit numpy confusion matrix
    assert A.weighted_kappa(off_by_one, order) == pytest.approx(7 / 9)
    assert A.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert A.spearman([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)
    assert A.spearman([1, 1], [1, 2]) is None


def test_ranks_average_ties():
    assert A._ranks([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]


def test_bootstrap_ci_is_seeded_and_brackets_point():
    pairs = [(i % 2 == 0, i % 3 == 0) for i in range(40)]
    stat = lambda ps: sum(h == j for h, j in ps) / len(ps)
    a = A.bootstrap_ci(pairs, stat, n_boot=300, rng=__import__("random").Random(1))
    b = A.bootstrap_ci(pairs, stat, n_boot=300, rng=__import__("random").Random(1))
    assert a == b and a[0] <= stat(pairs) <= a[1]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture()
def files(tmp_path):
    cases = [f"c{i:02d}" for i in range(1, 21)]
    labels, scores = [], []
    for i, c in enumerate(cases):
        res_h = i % 4
        res_j = res_h if i % 5 else min(3, res_h + 1)        # mostly agree, some off by one
        leak_h = i % 3 == 0
        leak_j = leak_h if i % 7 else (not leak_h)
        tool_h = ["appropriate", "unnecessary", "missing_required", "policy_violating"][i % 4]
        tool_j = "appropriate"                                 # judge never discriminates
        grounded_h = i % 2 == 0
        labels += [{"case_id": c, "dimension": "resolution", "label": res_h},
                   {"case_id": c, "dimension": "pii_leak", "label": leak_h},
                   {"case_id": c, "dimension": "tool_use", "label": tool_h},
                   {"case_id": c, "dimension": "grounded", "label": grounded_h}]
        scores.append({"test_case_id": c, "judge_status": "ok",
                       "verdict": {"dimensions": {"resolution": res_j, "pii_leak": leak_j, "tool_use": tool_j,
                                                  "ungrounded_claim": not grounded_h, "policy_violation": False}},
                       "dimension_scales": {
                           "resolution": {"type": "ordinal", "values": [{"value": v, "label": str(v)} for v in range(4)]},
                           "tool_use": {"type": "ordinal", "values": [{"value": v} for v in
                                        ["appropriate", "unnecessary", "missing_required", "policy_violating"]]}}})
    labels.append({"case_id": "c01", "dimension": "pii_leak", "label": "ambiguous"})
    scores.append({"test_case_id": "c99", "judge_status": "error", "verdict": None})
    lp, sp = tmp_path / "labels.jsonl", tmp_path / "run" / "scores.jsonl"
    sp.parent.mkdir()
    _write(lp, labels)
    _write(sp, scores)
    return lp, sp


def test_audit_flags_diagnostic_dimensions(files):
    lp, sp = files
    rep = A.audit(A.load_labels(lp), A.load_judge([sp.parent]),
                  dimension_map=A.parse_dimension_map(["grounded=!ungrounded_claim"]), n_boot=200)
    d = rep["dimensions"]
    assert d["resolution"]["scale"] == "ordinal" and d["resolution"]["agreement_metric"] == "qwk"
    assert d["resolution"]["within1"] == 1.0 and d["resolution"]["exact"] == pytest.approx(0.85)  # c15 is capped at 3
    assert d["resolution"]["status"] == "primary"
    assert d["grounded"]["exact"] == 1.0 and d["grounded"]["judge_dimension"] == "!ungrounded_claim"
    assert d["grounded"]["scale"] == "binary" and d["grounded"]["status"] == "primary"
    assert d["tool_use"]["status"] == "diagnostic" and "< floor" in d["tool_use"]["reasons"][0]
    assert d["tool_use"]["qwk"] == pytest.approx(0.0) and d["tool_use"]["exact"] == 0.25
    assert d["pii_leak"]["skipped_labels"] == 1 and d["pii_leak"]["n"] == 19
    assert set(rep["diagnostic"]) >= {"tool_use"}
    assert rep["judge_status"] == {"ok": 20, "error": 1}
    lo, hi = d["resolution"]["ci95"]["qwk"]
    assert lo <= d["resolution"]["qwk"] <= hi
    text = A.format_report(rep)
    assert "tool_use" in text and "diagnostic" in text


def test_audit_min_n_and_floor(files):
    lp, sp = files
    rep = A.audit(A.load_labels(lp), A.load_judge([sp]), min_n=50, n_boot=0)
    assert rep["primary"] == [] and all("min_n" in r["reasons"][0] for r in rep["dimensions"].values())
    rep2 = A.audit(A.load_labels(lp), A.load_judge([sp]), floor=0.99, n_boot=0,
                   dimension_map={"grounded": "!ungrounded_claim"})
    assert rep2["primary"] == ["grounded"]


def test_multiple_trials_use_mode_and_flat_rows(tmp_path):
    sp = tmp_path / "flat.jsonl"
    _write(sp, [{"case_id": "a", "dimensions": {"x": 1}}, {"case_id": "a", "dimensions": {"x": 2}},
                {"case_id": "a", "dimensions": {"x": 2}}, {"case_id": "b", "dimension": "x", "value": "3"}])
    j = A.load_judge([sp])
    assert j.value("a", "x") == 2 and j.value("b", "x") == 3 and j.value("c", "x") is None


def test_norm_value_and_label_validation(tmp_path):
    assert A.norm_value("True") is True and A.norm_value(" 2 ") == 2 and A.norm_value(3.0) == 3
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"case_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        A.load_labels(bad)


def test_run_audit_emits_evaluator_span(files, tmp_path, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from ci_lab import obs
    from ci_lab.contracts import ATTR_PURPOSE, SPAN_EVALUATOR_EXPERIMENT

    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("test"))
    lp, sp = files
    out = tmp_path / "audit.json"
    rep = A.run_audit(lp, [sp], out=out, experiment_id="exp-1", n_boot=50,
                      dimension_map={"grounded": "!ungrounded_claim"})
    assert json.loads(out.read_text(encoding="utf-8"))["diagnostic"] == rep["diagnostic"]
    spans = exp.get_finished_spans()
    assert [s.name for s in spans] == [SPAN_EVALUATOR_EXPERIMENT]
    assert spans[0].attributes[ATTR_PURPOSE] == "judge_audit"
    assert spans[0].attributes["ci.judge.diagnostic"] == len(rep["diagnostic"])
    assert not math.isnan(rep["dimensions"]["resolution"]["qwk"])
