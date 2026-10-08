"""Markdown report rendering."""

from __future__ import annotations

import json
from typing import Any


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return f"[{x[0]:.2f}, {x[1]:.2f}]"
    return str(x)


def render(manifest: dict[str, Any], metrics: dict[str, Any], probes: dict[str, Any] | None = None,
           conformance: dict[str, Any] | None = None, title: str = "s1eval report") -> str:
    L: list[str] = [f"# {title}", ""]
    L += [
        "> Smoke evaluation on an author-labelled conformance suite (n is small; traces and labels were",
        "> written by the same author). Treat numbers as a *specification test of the judge*, not as a",
        "> benchmark. Local results describe a System One-style approximation, not Jev.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(manifest, indent=2, default=str),
        "```",
        "",
    ]
    if conformance:
        L += ["## Local backend conformance", "", f"passed: **{conformance['passed']}**", ""]
        if conformance["failures"]:
            L += [f"- FAIL: {f}" for f in conformance["failures"]] + [""]
        L += ["| check | value |", "|---|---|"]
        L += [f"| {k} | {_f(v, 6)} |" for k, v in conformance["checks"].items() if k != "template_tail"]
        L += [""]

    m = metrics
    L += ["## Summary", "", f"cases: {m['n_cases']}, request errors: {m['n_errors']}", ""]
    for key, label in (("composite_vs_rule_gold", "rule-derived gold"), ("composite_vs_human_pass", "independent human_pass")):
        c = m[key]
        L += [
            f"### Composite pass vs {label}",
            "",
            "| n labelled | auto-decided | review rate | gold pass prevalence | accuracy (auto) | 95% CI | balanced acc | unsafe-pass rate |",
            "|---|---|---|---|---|---|---|---|",
            f"| {c['n_labeled']} | {c['n_auto_decided']} | {_f(c['review_rate'])} | {_f(c['prevalence_pass'])} | "
            f"{_f(c['accuracy_auto'])} | {_f(c['accuracy_auto_ci95'])} | {_f(c['balanced_accuracy_auto'])} | {_f(c['unsafe_pass_rate'])} |",
            "",
            f"- false pass (judge passed, gold fail): {c['false_pass'] or 'none'}",
            f"- false fail: {c['false_fail'] or 'none'}",
            f"- sent to review (gold fail / gold pass): {c['review_gold_fail'] or 'none'} / {c['review_gold_pass'] or 'none'}",
            "",
        ]

    L += ["## Per question", ""]
    for q, s in m["per_question"].items():
        L += [f"### `{q}` ({s['type']})", ""]
        if s["type"] == "noul":
            cm = s["confusion"]
            L += [
                "| n labelled | ambiguous | coverage | prevalence(true) | accuracy | 95% CI | balanced acc | 95% CI | Brier |",
                "|---|---|---|---|---|---|---|---|---|",
                f"| {s['n_labeled']} | {s['n_ambiguous']} | {_f(s['coverage'])} | {_f(s['prevalence_true'])} | {_f(s['accuracy'])} | "
                f"{_f(s['accuracy_ci95'])} | {_f(s['balanced_accuracy'])} | {_f(s['balanced_accuracy_ci95'])} | {_f(s['brier'])} |",
                "",
                f"confusion (positive = true): TP {cm['tp']}  FP {cm['fp']}  FN {cm['fn']}  TN {cm['tn']}",
                "",
                "threshold sensitivity (descriptive only, not tuned): "
                + ", ".join(f"{t}: {_f(v)}" for t, v in s["threshold_sensitivity"].items()),
            ]
        elif s["type"] == "choice":
            opts = list(s["confusion_gold_by_pred"])
            L += [
                f"n labelled {s['n_labeled']}, coverage {_f(s['coverage'])}, accuracy {_f(s['accuracy'])} "
                f"(95% CI {_f(s['accuracy_ci95'])}), macro recall {_f(s['macro_recall'])}, prevalence {s['prevalence']}",
                "",
                "| gold \\ pred | " + " | ".join(opts) + " |",
                "|---|" + "---|" * len(opts),
            ]
            L += [f"| {g} | " + " | ".join(str(s["confusion_gold_by_pred"][g][p]) for p in opts) + " |" for g in opts]
        else:
            L += [
                f"n labelled {s['n_labeled']}, coverage {_f(s['coverage'])}, modal accuracy {_f(s['modal_accuracy'])} "
                f"(95% CI {_f(s['modal_accuracy_ci95'])}), within-1 {_f(s['within_1'])}, MAE(modal) {_f(s['mae_modal'])}, "
                f"MAE(expected) {_f(s['mae_expected'])}, prevalence {s['prevalence']}",
            ]
        L += ["", f"errors: {s['errors'] or 'none'}; abstained: {s['abstained'] or 'none'}", ""]

    L += ["## Tag slices (composite vs rule gold)", "", "| tag | n | correct | review | false pass |", "|---|---|---|---|---|"]
    L += [f"| {t} | {v['n']} | {v['correct']} | {v['review']} | {v['false_pass']} |" for t, v in m["tag_slices"].items()]
    L += [""]
    if m.get("stability"):
        L += ["## Repeat stability", "", "```json", json.dumps(m["stability"], indent=2), "```", ""]
    L += ["## Cost", "", "```json", json.dumps(m["cost"], indent=2), "```", ""]

    if probes:
        L += ["## Probes (judge-the-judge)", ""]
        for name, res in probes.items():
            L += [f"### {name}", "", "```json", json.dumps(res, indent=2, default=str), "```", ""]
    return "\n".join(L)
