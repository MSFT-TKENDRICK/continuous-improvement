"""Ladder router: send each cluster to the strongest enforcement rung that fits (design §13.2/§13.3).

Order (first match wins): injection → R6; ``pii.*`` oracle rules → R3 (response pattern); R1 arg constraint
(failing calls share an arg range/enum disjoint from passing calls); R2 precondition / sequence (failure
tool preceded by a missing or mismatched tool on the same subject); any oracle rule → R4 (trajectory
metric); rubric-only / judgment → R6 (prose). If the registry shows this fingerprint was already "fixed"
by prose ≥ ``max_prose_fixes`` times, an R6 route is forced to structural review (R4).

``features`` are typed slots for the synthesizer (M17); injection-suspect clusters carry none (B3).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ci_lab.lessons.fingerprint import Call, anchor_call, canonical_calls, is_good
from ci_lab.lessons.reduce import ID_PREFIX, TXT_PREFIX
from ci_lab.lessons.registry import Registry
from ci_lab.rulespec import LessonCluster, Rung, Trajectory, normalize_subject

MAX_PROSE_FIXES = 2
PRESENCE = 0.8     # share of passing calls that must show the precondition
ABSENCE = 0.2      # max share of failing calls that may show it
MIN_PASSING = 2


@dataclass
class Routing:
    cluster: LessonCluster
    features: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    forced_structural: bool = False

    def line(self, holdout_members: Sequence[str] = ()) -> dict[str, Any]:
        return {"cluster": self.cluster.model_dump(mode="json"), "features": self.features,
                "route_reasons": self.reasons, "forced_structural": self.forced_structural,
                "holdout_members": list(holdout_members)}


def _usable(v: Any) -> bool:
    return not (isinstance(v, str) and v.startswith((ID_PREFIX, TXT_PREFIX)))


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _idlike(key: str, v: Any) -> bool:
    k = key.lower()
    return isinstance(v, str) and (k == "id" or k.endswith(("_id", "id")) or v.startswith(ID_PREFIX))


def _passing_calls(tool: str, good: Iterable[Trajectory]) -> list[tuple[Trajectory, Call]]:
    return [(t, c) for t in good for c in canonical_calls(t) if c.tool == tool]


def _prior(t: Trajectory, call: Call) -> list[Call]:
    return [c for c in canonical_calls(t) if c.step.i < call.step.i]


def r1_features(target: str, failing: Sequence[Call], passing: Sequence[Call]) -> dict[str, Any] | None:
    if len(passing) < MIN_PASSING or not failing:
        return None
    keys = sorted(set.intersection(*(set(c.step.args) for c in failing)))
    for k in keys:
        if any(_idlike(k, c.step.args.get(k)) for c in failing):
            continue
        fv = [c.step.args.get(k) for c in failing]
        pv = [c.step.args.get(k) for c in passing if k in c.step.args]
        if len(pv) < MIN_PASSING or not all(_usable(v) for v in fv + pv):
            continue
        fn, pn = [_num(v) for v in fv], [_num(v) for v in pv]
        if all(x is not None for x in fn + pn):
            if min(fn) > max(pn):  # type: ignore[type-var]
                return {"target_tool": target, "arg": k, "op": "le", "values": [max(pn)]}  # type: ignore[type-var]
            if max(fn) < min(pn):  # type: ignore[type-var]
                return {"target_tool": target, "arg": k, "op": "ge", "values": [min(pn)]}  # type: ignore[type-var]
            continue
        if all(isinstance(v, (str, bool)) for v in fv + pv):
            fs, ps = {str(v) for v in fv}, {str(v) for v in pv}
            if not fs & ps:
                return {"target_tool": target, "arg": k, "op": "in", "values": sorted(ps)}
    return None


def _subject(call: Call) -> tuple[str, str] | None:
    for k in sorted(call.step.args):
        v = call.step.args[k]
        if _idlike(k, v):
            return k, normalize_subject(v)
    return None


def _on_subject(prior: Call, subj: str) -> str | None:
    for k in sorted(prior.step.args):
        if normalize_subject(prior.step.args[k]) == subj:
            return k
    return None


def r2_features(target: str, failing: Sequence[tuple[Trajectory, Call]],
                passing: Sequence[tuple[Trajectory, Call]]) -> tuple[dict[str, Any], str] | None:
    if len(passing) < MIN_PASSING or not failing:
        return None
    subj = _subject(failing[0][1])
    if subj is None:
        return None
    subject_arg = subj[0]

    def matches(pairs: Sequence[tuple[Trajectory, Call]]) -> list[dict[str, tuple[Call, str]]]:
        out = []
        for t, c in pairs:
            s = c.step.args.get(subject_arg)
            found: dict[str, tuple[Call, str]] = {}
            if s is not None:
                for p in _prior(t, c):
                    if (field_ := _on_subject(p, normalize_subject(s))) is not None:
                        found[p.tool] = (p, field_)  # latest prior on the subject wins
            out.append(found)
        return out

    fm, pm = matches(failing), matches(passing)
    tools = sorted({tool for m in pm for tool in m})
    base = {"target_tool": target, "subject_arg": subject_arg}
    # missing precondition
    for tool in tools:
        p_share = sum(tool in m for m in pm) / len(pm)
        f_share = sum(tool in m for m in fm) / len(fm)
        if p_share >= PRESENCE and f_share <= ABSENCE:
            fld = next(m[tool][1] for m in pm if tool in m)
            return {**base, "prior_tool": tool, "prior_subject_field": fld}, f"missing precondition {tool}"
    # mismatched precondition: present in both, but a result field / numeric comparison separates them
    for tool in tools:
        if not all(tool in m for m in fm) or sum(tool in m for m in pm) / len(pm) < PRESENCE:
            continue
        fld = fm[0][tool][1]
        f_res = [m[tool][0].result.result if m[tool][0].result else None for m in fm]
        p_res = [m[tool][0].result.result if m[tool][0].result else None for m in pm if tool in m]
        if not all(isinstance(r, Mapping) for r in f_res + p_res):
            continue
        common = sorted(set.intersection(*(set(r) for r in f_res + p_res)))  # type: ignore[arg-type]
        for rk in common:
            fv, pv = [r[rk] for r in f_res], [r[rk] for r in p_res]  # type: ignore[index]
            if all(isinstance(v, (str, bool)) and _usable(v) for v in fv + pv):
                fs, ps = {str(v) for v in fv}, {str(v) for v in pv}
                if not fs & ps:
                    return ({**base, "prior_tool": tool, "prior_subject_field": fld, "arg": f"result.{rk}",
                             "op": "in", "values": sorted(ps)}, f"mismatched precondition {tool}.result.{rk}")
        # numeric: current arg vs prior result field
        for ak in sorted(failing[0][1].step.args):
            for rk in common:
                def cmp_ok(pairs: Sequence[tuple[Trajectory, Call]], ms: Sequence[dict[str, tuple[Call, str]]],
                           ak: str = ak, rk: str = rk, tool: str = tool) -> list[bool | None]:
                    out: list[bool | None] = []
                    for (_, c), m in zip(pairs, ms, strict=True):
                        if tool not in m or m[tool][0].result is None:
                            continue
                        a = _num(c.step.args.get(ak))
                        r = _num((m[tool][0].result.result or {}).get(rk))
                        out.append(None if a is None or r is None else a <= r)
                    return out
                fo, po = cmp_ok(failing, fm), cmp_ok(passing, pm)
                if fo and po and all(x is False for x in fo) and all(x is True for x in po):
                    return ({**base, "prior_tool": tool, "prior_subject_field": fld, "amount_arg": ak,
                             "prior_amount_field": f"result.{rk}", "op": "le"},
                            f"numeric {ak} > {tool}.result.{rk}")
    return None


def route_cluster(cluster: LessonCluster, members: Sequence[Trajectory], good: Sequence[Trajectory],
                  registry: Registry | None = None, *, max_prose_fixes: int = MAX_PROSE_FIXES) -> Routing:
    """Pick the rung + typed features for one cluster."""
    fp = cluster.fingerprint
    trusted = all(t.trusted for t in members)
    route: Rung
    feats: dict[str, Any] = {}
    reasons: list[str] = []
    if any(t.outcome.injection_suspect for t in members) or any(r.startswith("injection") for r in fp.oracle_rules):
        route, reasons = "R6", ["injection: prose + judgment only; excluded from synthesis input (B3)"]
    elif any(r.startswith("pii") for r in fp.oracle_rules):
        classes = sorted({r.split(".", 1)[1] if "." in r else "any" for r in fp.oracle_rules if r.startswith("pii")})
        route, feats, reasons = "R3", {"pattern_classes": classes}, ["pii oracle rule → response pattern"]
    else:
        route = "R6"
        anchors = [(t, a) for t in members if (a := anchor_call(t)) is not None]
        if anchors and len(anchors) == len(members) and len({a.tool for _, a in anchors}) == 1:
            target = anchors[0][1].tool
            passing = _passing_calls(target, good)
            if f := r1_features(target, [a for _, a in anchors], [c for _, c in passing]):
                route, feats, reasons = "R1", f, [f"arg constraint {f['arg']} {f['op']}"]
            elif r2 := r2_features(target, anchors, passing):
                route, feats, reasons = "R2", r2[0], [r2[1]]
        if route == "R6" and fp.oracle_rules:
            route, reasons = "R4", ["oracle rule without call-level separator → trajectory metric"]
        elif route == "R6":
            reasons = ["rubric/judgment failure → prose"]
    forced = False
    if route == "R6" and registry is not None and registry.prose_fix_count(cluster.id) >= max_prose_fixes:
        route, forced = "R4", True
        reasons.append(f"forced structural review: fixed by prose {registry.prose_fix_count(cluster.id)}x")
    if feats:
        feats["trusted"] = trusted
    return Routing(cluster=cluster.model_copy(update={"route": route}), features=feats, reasons=reasons,
                   forced_structural=forced)


def route_all(clusters: Sequence[LessonCluster], members: Mapping[str, Sequence[Trajectory]],
              corpus: Sequence[Trajectory], registry: Registry | None = None) -> list[Routing]:
    good = [t for t in corpus if t.split == "evolve" and is_good(t)]
    return [route_cluster(c, members.get(c.id, ()), good, registry) for c in clusters]
