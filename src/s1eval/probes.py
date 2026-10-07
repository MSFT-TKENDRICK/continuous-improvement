"""Metamorphic / adversarial probes of the judge itself.

Every probe states an invariant, a tolerance and why it matters. Probes reuse baseline answers
from the main run (``base_records``) so only the perturbed requests cost model calls.

  complement       hand-authored complementary noul (e.g. grounded vs has_unsupported_claim):
                   verdicts must disagree and |p + p' - 1| <= 0.3. Catches yes-bias and
                   literal-reading failures. (Not mechanical "not X" negation.)
  choice_order     rotate the option order of each choice question through all K rotations.
                   An order-invariant judge picks the option shown first exactly 1/K of the time;
                   the excess is raw position bias (Jev docs warn of first-option bias).
  code_permutation local backend only: same option order, shuffled letter codes. Separates
                   letter bias from position bias.
  distractor       add a fixed, reviewed, task-irrelevant field to the state, first or last.
                   Verdicts must not change. Jev docs: accuracy degrades with distractor-heavy state.
  batch_vs_single  ask each question in its own request vs. all together. Verdicts must match
                   (cross-question interference / KV-cache effects).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .backends.base import BackendError
from .rubric import Rubric
from .state import project_observable
from .types import Question

COMPLEMENT_TOLERANCE = 0.3


def _base_answers(base_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["case_id"]: r["answers"] for r in base_records if r.get("repeat", 0) == 0 and "answers" in r}


def _verdict(rec: dict[str, Any], q: Question, threshold: float = 0.5) -> Any:
    if rec["status"] != "ok":
        return None
    if q.type == "noul":
        return rec["noul"] >= threshold
    if q.type == "choice":
        return rec["choice"]
    return rec["level"]


class _Comparison:
    """Per-question tally of baseline-vs-perturbed answers.

    ``flip_rate`` only counts pairs where both answers are decided (ok), with that count as the
    denominator; ok<->non-answer transitions are reported separately as ``status_change_rate``
    over all attempted pairs, so a question that often abstains cannot look artificially stable.
    """

    def __init__(self, questions: dict[str, Question], threshold: float) -> None:
        self.qs, self.t = questions, threshold
        self.attempted: dict[str, int] = defaultdict(int)
        self.comparable: dict[str, int] = defaultdict(int)
        self.flips: dict[str, list[str]] = defaultdict(list)
        self.status_changes: dict[str, list[str]] = defaultdict(list)
        self.deltas: dict[str, list[float]] = defaultdict(list)

    def add(self, k: str, base_rec: dict[str, Any], ans, tag: str) -> None:
        q = self.qs[k]
        self.attempted[k] += 1
        b_ok = base_rec["status"] == "ok"
        if b_ok != ans.ok:
            self.status_changes[k].append(f"{tag}: {base_rec['status']} -> {ans.status}")
            return
        if not b_ok:
            return
        self.comparable[k] += 1
        v0, v1 = _verdict(base_rec, q, self.t), ans.verdict(self.t)
        if v0 != v1:
            self.flips[k].append(f"{tag}: {v0} -> {v1}")
        if q.type == "noul":
            self.deltas[k].append(abs(ans.noul - base_rec["noul"]))

    def summary(self) -> dict[str, Any]:
        ks = list(self.qs)
        return {
            "n_comparable": {k: self.comparable[k] for k in ks},
            "flip_rate": {k: len(self.flips[k]) / self.comparable[k] if self.comparable[k] else None for k in ks},
            "status_change_rate": {k: len(self.status_changes[k]) / self.attempted[k] if self.attempted[k] else None
                                   for k in ks},
            "mean_abs_dp_noul": {k: sum(v) / len(v) for k, v in self.deltas.items() if v},
            "flipped": {k: v for k, v in self.flips.items() if v},
            "status_changes": {k: v for k, v in self.status_changes.items() if v},
        }


def probe_complement(backend, rubric: Rubric, cases, base_records, progress=None) -> dict[str, Any]:
    base = _base_answers(base_records)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    # Cases outer, questions inner: consecutive calls share the state prefix (KV-cache reuse).
    for c in cases:
        state = project_observable(c)
        for qname, comp in rubric.complements.items():
            b = base.get(c["id"], {}).get(qname)
            if not b or b["status"] != "ok":
                skipped[qname] += 1
                continue
            try:
                a = backend.decide(state, {f"{qname}__complement": comp}).answers[f"{qname}__complement"]
            except BackendError:
                skipped[qname] += 1
                continue
            if not a.ok:
                skipped[qname] += 1
                continue
            p, pc = b["noul"], a.noul
            rows[qname].append({"case_id": c["id"], "p": round(p, 4), "p_complement": round(pc, 4),
                                "sum_dev": round(abs(p + pc - 1), 4), "verdicts_agree": (p >= 0.5) == (pc >= 0.5)})
            if progress:
                progress(f"complement {qname} {c['id']} p={p:.2f} p'={pc:.2f}")
    out = {}
    for qname in rubric.complements:
        rs = rows[qname]
        n = len(rs)
        out[qname] = {
            "invariant": "verdict(q) != verdict(q') and |p + p' - 1| <= %.1f" % COMPLEMENT_TOLERANCE,
            "n": n,
            "skipped": skipped[qname],
            "contradiction_rate": sum(r["verdicts_agree"] for r in rs) / n if n else None,
            "tolerance_violation_rate": sum(r["sum_dev"] > COMPLEMENT_TOLERANCE for r in rs) / n if n else None,
            "mean_sum_dev": sum(r["sum_dev"] for r in rs) / n if n else None,
            "violations": [r for r in rs if r["verdicts_agree"] or r["sum_dev"] > COMPLEMENT_TOLERANCE],
        }
    return out


def _rotate(q: Question, r: int) -> Question:
    items = list(q.criteria.items())
    return q.with_criteria(dict(items[r:] + items[:r]))


def probe_choice_order(backend, rubric: Rubric, cases, base_records, progress=None) -> dict[str, Any]:
    base = _base_answers(base_records)
    out = {}
    for qname, q in rubric.questions.items():
        if q.type != "choice":
            continue
        k = len(q.criteria)
        per_case, first_pos_hits, total = {}, 0, 0
        for c in cases:
            b = base.get(c["id"], {}).get(qname)
            picks = [b["choice"] if b and b["status"] == "ok" else None]
            state = project_observable(c)
            for r in range(1, k):
                try:
                    a = backend.decide(state, {qname: _rotate(q, r)}).answers[qname]
                    picks.append(a.choice if a.ok else None)
                except BackendError:
                    picks.append(None)
            for r, pick in enumerate(picks):
                if pick is None:
                    continue
                total += 1
                shown_first = list(_rotate(q, r).criteria)[0]
                first_pos_hits += pick == shown_first
            per_case[c["id"]] = picks
            if progress:
                progress(f"choice_order {qname} {c['id']} {picks}")
        valid = {cid: p for cid, p in per_case.items() if None not in p}
        flips = [cid for cid, p in valid.items() if len(set(p)) > 1]
        out[qname] = {
            "invariant": "same option chosen under all K rotations; first-shown rate == 1/K",
            "k": k,
            "n_cases": len(valid),
            "flip_rate": len(flips) / len(valid) if valid else None,
            "first_shown_rate": first_pos_hits / total if total else None,
            "first_shown_expected": 1 / k,
            "flipped_cases": {cid: per_case[cid] for cid in flips},
        }
    return out


def probe_variant_backend(variant, rubric: Rubric, cases, base_records, *, label: str, only_types=("choice",),
                          progress=None) -> dict[str, Any]:
    """Compare verdicts of a backend variant (e.g. shuffled codes) against the baseline."""
    base = _base_answers(base_records)
    qs = {k: q for k, q in rubric.questions.items() if q.type in only_types}
    cmp = _Comparison(qs, rubric.noul_threshold)
    errors = 0
    for c in cases:
        if c["id"] not in base:
            continue
        try:
            d = variant.decide(project_observable(c), qs)
        except BackendError:
            errors += 1
            continue
        for k in qs:
            cmp.add(k, base[c["id"]][k], d.answers[k], c["id"])
        if progress:
            progress(f"{label} {c['id']}")
    return {"invariant": "verdicts unchanged", "backend_errors": errors, **cmp.summary()}


def probe_distractor(backend, rubric: Rubric, cases, base_records, distractors: list[dict[str, str]],
                     progress=None) -> dict[str, Any]:
    base = _base_answers(base_records)
    cmp = _Comparison(rubric.questions, rubric.noul_threshold)
    n, errors = 0, 0
    for d in distractors:
        for pos in ("first", "last"):
            for c in cases:
                if c["id"] not in base:
                    continue
                s = project_observable(c)
                state = {d["key"]: d["text"], **s} if pos == "first" else {**s, d["key"]: d["text"]}
                try:
                    dec = backend.decide(state, rubric.questions)
                except BackendError:
                    errors += 1
                    continue
                n += 1
                for k in rubric.questions:
                    cmp.add(k, base[c["id"]][k], dec.answers[k], f"{c['id']}@{d['key']}:{pos}")
                if progress:
                    progress(f"distractor {d['key']}:{pos} {c['id']}")
    return {"invariant": "verdicts unchanged by a task-irrelevant field", "n_variants": n, "backend_errors": errors,
            **cmp.summary()}


def probe_batch_vs_single(backend, rubric: Rubric, cases, base_records, progress=None) -> dict[str, Any]:
    base = _base_answers(base_records)
    cmp = _Comparison(rubric.questions, rubric.noul_threshold)
    errors = 0
    for c in cases:
        if c["id"] not in base:
            continue
        s = project_observable(c)
        for k, q in rubric.questions.items():
            try:
                a = backend.decide(s, {k: q}).answers[k]
            except BackendError:
                errors += 1
                continue
            cmp.add(k, base[c["id"]][k], a, c["id"])
        if progress:
            progress(f"batch_vs_single {c['id']}")
    return {"invariant": "per-question verdict identical when asked alone", "backend_errors": errors, **cmp.summary()}
