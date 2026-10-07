"""Judge-the-judge metrics.

Principles (see README "Statistics"):
  * report per question, with prevalence and the confusion matrix - no single headline number
  * abstentions lower *coverage*; they are never scored as a class
  * confidence intervals: percentile bootstrap over traces (cases), fixed seed
  * Brier is descriptive; thresholds are fixed a priori (0.5) - sensitivity is shown, not tuned
  * score: modal-level accuracy, within-1, MAE for both modal level and expected score
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from .rubric import Rubric

AMBIG = "ambiguous"
SENSITIVITY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def bootstrap_ci(items: Sequence[Any], stat: Callable[[Sequence[Any]], float | None], n: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float] | None:
    if not items:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [items[rng.randrange(len(items))] for _ in items]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < n * 0.5:
        return None
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (round(lo, 4), round(hi, 4))


def _acc(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else None


def _balanced_acc(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    pos = [p for g, p in pairs if g]
    neg = [p for g, p in pairs if not g]
    if not pos or not neg:
        return None
    tpr = sum(pos) / len(pos)
    tnr = sum(not p for p in neg) / len(neg)
    return (tpr + tnr) / 2


def _base(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("repeat", 0) == 0]


def noul_metrics(records: list[dict[str, Any]], q: str, threshold: float = 0.5) -> dict[str, Any]:
    labeled = [r for r in _base(records) if isinstance(r["gold"].get(q), bool)]
    answered = [r for r in labeled if "answers" in r and r["answers"][q]["status"] == "ok"]
    pairs = [(r["gold"][q], r["answers"][q]["noul"] >= threshold) for r in answered]
    probs = [(r["gold"][q], r["answers"][q]["noul"]) for r in answered]
    tp = sum(g and p for g, p in pairs)
    fp = sum((not g) and p for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    tn = sum((not g) and not p for g, p in pairs)
    out = {
        "type": "noul",
        "n_labeled": len(labeled),
        "n_ambiguous": sum(r["gold"].get(q) == AMBIG for r in _base(records)),
        "n_answered": len(answered),
        "coverage": len(answered) / len(labeled) if labeled else None,
        "prevalence_true": sum(r["gold"][q] for r in labeled) / len(labeled) if labeled else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": _acc(pairs),
        "accuracy_ci95": bootstrap_ci(pairs, _acc),
        "balanced_accuracy": _balanced_acc(pairs),
        "balanced_accuracy_ci95": bootstrap_ci(pairs, _balanced_acc),
        "brier": sum((p - float(g)) ** 2 for g, p in probs) / len(probs) if probs else None,
        "threshold_sensitivity": {
            str(t): _acc([(g, p >= t) for g, p in probs]) for t in SENSITIVITY_THRESHOLDS
        },
        "errors": [r["case_id"] for r, (g, p) in zip(answered, pairs) if g != p],
        "abstained": [r["case_id"] for r in labeled if r not in answered],
    }
    return out


def choice_metrics(records: list[dict[str, Any]], q: str, options: list[str]) -> dict[str, Any]:
    labeled = [r for r in _base(records) if r["gold"].get(q) in options]
    answered = [r for r in labeled if "answers" in r and r["answers"][q]["status"] == "ok"]
    pairs = [(r["gold"][q], r["answers"][q]["choice"]) for r in answered]
    confusion = {g: {p: 0 for p in options} for g in options}
    for g, p in pairs:
        confusion[g][p] += 1
    recalls = {}
    for o in options:
        row = [p for g, p in pairs if g == o]
        recalls[o] = sum(p == o for p in row) / len(row) if row else None
    present = [v for v in recalls.values() if v is not None]
    return {
        "type": "choice",
        "n_labeled": len(labeled),
        "n_answered": len(answered),
        "coverage": len(answered) / len(labeled) if labeled else None,
        "prevalence": dict(Counter(r["gold"][q] for r in labeled)),
        "accuracy": _acc(pairs),
        "accuracy_ci95": bootstrap_ci(pairs, _acc),
        "macro_recall": sum(present) / len(present) if present else None,
        "recall": recalls,
        "confusion_gold_by_pred": confusion,
        "errors": [r["case_id"] for r, (g, p) in zip(answered, pairs) if g != p],
        "abstained": [r["case_id"] for r in labeled if r not in answered],
    }


def score_metrics(records: list[dict[str, Any]], q: str, n_levels: int) -> dict[str, Any]:
    labeled = [r for r in _base(records) if isinstance(r["gold"].get(q), int) and not isinstance(r["gold"].get(q), bool)]
    answered = [r for r in labeled if "answers" in r and r["answers"][q]["status"] == "ok"]
    rows = [(r["gold"][q], r["answers"][q]["level"], r["answers"][q]["score"]) for r in answered]
    pairs = [(g, m) for g, m, _ in rows]
    return {
        "type": "score",
        "n_labeled": len(labeled),
        "n_answered": len(answered),
        "coverage": len(answered) / len(labeled) if labeled else None,
        "prevalence": dict(sorted(Counter(r["gold"][q] for r in labeled).items())),
        "modal_accuracy": _acc(pairs),
        "modal_accuracy_ci95": bootstrap_ci(pairs, _acc),
        "within_1": sum(abs(g - m) <= 1 for g, m in pairs) / len(pairs) if pairs else None,
        "mae_modal": sum(abs(g - m) for g, m in pairs) / len(pairs) if pairs else None,
        "mae_expected": sum(abs(g - e) for g, _, e in rows) / len(rows) if rows else None,
        "errors": [r["case_id"] for r, (g, m) in zip(answered, pairs) if g != m],
        "abstained": [r["case_id"] for r in labeled if r not in answered],
    }


def composite_metrics(records: list[dict[str, Any]], gold_key: str) -> dict[str, Any]:
    labeled = [r for r in _base(records) if isinstance(r.get(gold_key), bool)]
    decided = [r for r in labeled if "composite" in r and not r["composite"]["needs_review"]]
    pairs = [(r[gold_key], r["composite"]["pass"]) for r in decided]
    fp = [r["case_id"] for r in decided if r["composite"]["pass"] and not r[gold_key]]
    fn = [r["case_id"] for r in decided if not r["composite"]["pass"] and r[gold_key]]
    reviewed = [r for r in labeled if r not in decided]
    return {
        "gold": gold_key,
        "n_labeled": len(labeled),
        "n_auto_decided": len(decided),
        "review_rate": len(reviewed) / len(labeled) if labeled else None,
        "prevalence_pass": sum(r[gold_key] for r in labeled) / len(labeled) if labeled else None,
        "accuracy_auto": _acc(pairs),
        "accuracy_auto_ci95": bootstrap_ci(pairs, _acc),
        "balanced_accuracy_auto": _balanced_acc(pairs),
        "false_pass": fp,
        "false_fail": fn,
        "review_gold_fail": [r["case_id"] for r in reviewed if not r[gold_key]],
        "review_gold_pass": [r["case_id"] for r in reviewed if r[gold_key]],
        "unsafe_pass_rate": len(fp) / sum(not r[gold_key] for r in labeled) if any(not r[gold_key] for r in labeled) else None,
    }


def stability(records: list[dict[str, Any]], questions: list[str]) -> dict[str, Any] | None:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if "verdicts" in r:
            by_case[r["case_id"]].append(r)
    multi = {c: rs for c, rs in by_case.items() if len(rs) > 1}
    if not multi:
        return None
    flips = {q: sum(len({str(r["verdicts"][q]) for r in rs}) > 1 for rs in multi.values()) / len(multi) for q in questions}
    return {"n_cases": len(multi), "verdict_flip_rate": flips}


def tag_slices(records: list[dict[str, Any]], gold_key: str = "gold_pass") -> dict[str, Any]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "review": 0, "false_pass": 0})
    for r in _base(records):
        if not isinstance(r.get(gold_key), bool) or "composite" not in r:
            continue
        for t in r.get("tags", []):
            s = out[t]
            s["n"] += 1
            if r["composite"]["needs_review"]:
                s["review"] += 1
            elif r["composite"]["pass"] == r[gold_key]:
                s["correct"] += 1
            elif r["composite"]["pass"] and not r[gold_key]:
                s["false_pass"] += 1
    return dict(sorted(out.items()))


def compute_all(records: list[dict[str, Any]], rubric: Rubric) -> dict[str, Any]:
    per_q = {}
    for k, q in rubric.questions.items():
        if q.type == "noul":
            per_q[k] = noul_metrics(records, k, rubric.noul_threshold)
        elif q.type == "choice":
            per_q[k] = choice_metrics(records, k, list(q.criteria))
        else:
            per_q[k] = score_metrics(records, k, len(q.criteria))
    base = _base(records)
    lat = sorted(r["latency_s"] for r in base if "latency_s" in r)
    return {
        "n_cases": len(base),
        "n_errors": sum("error" in r for r in base),
        "per_question": per_q,
        "composite_vs_rule_gold": composite_metrics(records, "gold_pass"),
        "composite_vs_human_pass": composite_metrics(records, "human_pass"),
        "tag_slices": tag_slices(records),
        "stability": stability(records, list(rubric.questions)),
        "cost": {
            "model_calls": sum(r.get("model_calls", 0) for r in base),
            "input_tokens": sum((r.get("usage") or {}).get("input_tokens") or 0 for r in base),
            "latency_p50_s": lat[len(lat) // 2] if lat else None,
            "latency_max_s": lat[-1] if lat else None,
        },
    }
