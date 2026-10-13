"""Counterfactual replay validator — a cheap **rejection filter only** (B4, design §13.3 step 4).

Given candidate rules and harvested trajectories it can only ``reject`` or ``pass_to_closed_loop``;
acceptance is decided solely by the guard-on/guard-off closed loop + OES gates. Checks:

* safety: every RuleSpec validates, every regex RE2-compiles, the bundle loads via ``ci_lab.rules``;
* recall on the cluster's **held-out families** (same fingerprint, family hash, never mined);
* false positives on passing / human-good evolve trajectories: Clopper-Pearson UCB ≤ ε;
* block-rate ceiling per rule and in aggregate on evolve trajectories;
* leak screen: rule literals vs eval dataset texts (word 3-gram overlap, memorized ids, long verbatim
  substrings) + deterministic metamorphic paraphrase probes of text patterns.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ci_lab.lessons.cluster import HOLDOUT_FRACTION, is_holdout
from ci_lab.lessons.common import digest, log
from ci_lab.lessons.fingerprint import (
    DEFAULT_WINDOW,
    cluster_id,
    fingerprint,
    is_failure,
    is_good,
)
from ci_lab.lessons.reduce import _ID_RES
from ci_lab.lessons.stats import cp_upper
from ci_lab.rulespec import (
    PROMOTE_FP_UCB,
    AllPred,
    AnyPred,
    ArgPred,
    LessonCluster,
    NotPred,
    PriorPred,
    RuleFile,
    RuleSpec,
    TextPred,
    Trajectory,
    bundle_digest,
)


class _R(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LeakFinding(_R):
    rule_id: str
    kind: Literal["ngram", "id", "verbatim", "paraphrase_fragile"]
    literal_digest: str   # never the literal itself
    detail: str = ""


class RuleReplay(_R):
    rule_id: str
    fires_evolve: int = 0
    block_rate: float = 0.0
    fp: int = 0
    recall_hits: int = 0


class ReplayReport(_R):
    verdict: Literal["reject", "pass_to_closed_loop"]
    ok: bool
    reasons: list[str] = Field(default_factory=list)
    bundle_digest: str = ""
    cluster_id: str | None = None
    per_rule: list[RuleReplay] = Field(default_factory=list)
    recall: float | None = None
    holdout_n: int = 0
    fp: int = 0
    negatives: int = 0
    fp_rate: float | None = None
    fp_ucb: float | None = None
    block_rate: float = 0.0
    n_evolve: int = 0
    leaks: list[LeakFinding] = Field(default_factory=list)


@dataclass(frozen=True)
class ReplayConfig:
    epsilon: float = PROMOTE_FP_UCB
    confidence: float = 0.95
    min_recall: float = 0.6
    min_negatives: int = 30
    max_block_rate_rule: float = 0.10
    max_block_rate_total: float = 0.20
    require_holdout: bool = True
    holdout_fraction: float = HOLDOUT_FRACTION
    window: int = DEFAULT_WINDOW
    paraphrase_fragile_share: float = 0.5


class RuleLoadFailure(Exception):
    pass


# ---------------------------------------------------------------- engine

def _engine(engine: Any = None) -> Any:
    if engine is not None:
        return engine
    import importlib

    return importlib.import_module("ci_lab.rules")  # M14 engine, imported lazily


def _rule_id(match: Any) -> str:
    r = getattr(match, "rule", match)
    return str(getattr(r, "id", r))


def _load(rules: Sequence[RuleSpec], eng: Any, work_dir: Path | None, extractors: Sequence[Path] = ()) -> Any:
    def go(d: Path) -> Any:
        p = d / "candidate_rules.yaml"
        body = RuleFile(rules=list(rules)).model_dump(mode="json", exclude_none=True)
        p.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
        # state flags come only from extractors; a flag rule without its extractor fails to load
        return eng.load_bundle([p], list(extractors)) if extractors else eng.load_bundle([p])

    try:
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
            return go(work_dir)
        with tempfile.TemporaryDirectory(prefix="ci-lessons-", dir=None) as d:
            return go(Path(d))
    except Exception as exc:  # RuleLoadError(list[str]) or validation failures
        errs = getattr(exc, "problems", None) or getattr(exc, "errors", None)
        msg = "; ".join(map(str, errs)) if isinstance(errs, (list, tuple)) else str(exc)
        raise RuleLoadFailure(msg[:500]) from exc


# ---------------------------------------------------------------- safety / literals

def _preds(p: Any) -> Iterable[Any]:
    if p is None:
        return
    yield p
    if isinstance(p, (AllPred, AnyPred)):
        for c in p.of:
            yield from _preds(c)
    elif isinstance(p, NotPred):
        yield from _preds(p.of)
    elif isinstance(p, PriorPred):
        yield from _preds(p.where)


def rule_regexes(rule: RuleSpec) -> list[str]:
    out = []
    for p in (*_preds(rule.when), *_preds(rule.require)):
        if isinstance(p, TextPred):
            out.append(p.matches)
        elif isinstance(p, ArgPred) and p.op == "matches" and isinstance(p.value, str):
            out.append(p.value)
    return out


def regex_literals(pattern: str, min_len: int = 4) -> list[str]:
    """Literal runs of a regex (metacharacters split runs; escaped chars kept literally)."""
    out, cur = [], []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt in "dDwWsSbBnrtfvAzZ0123456789pPx":
                out.append("".join(cur))
                cur = []
            else:
                cur.append(nxt)
            i += 2
            continue
        if ch in "[](){}|*+?^$.":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s for s in (x.strip() for x in out) if len(s) >= min_len]


def rule_literals(rule: RuleSpec) -> list[str]:
    lits: list[str] = []
    for p in (*_preds(rule.when), *_preds(rule.require)):
        if isinstance(p, ArgPred) and p.op != "matches":
            vals = p.value if isinstance(p.value, list) else [p.value]
            lits += [v for v in vals if isinstance(v, str)]
    for rx in rule_regexes(rule):
        lits += regex_literals(rx)
    lits += [v for v in rule.slots.values() if isinstance(v, str)]
    return sorted({x for x in lits if x})


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.casefold())


def _grams(ws: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


_SYNONYMS = {
    "change": "edit", "file": "document", "validate": "check", "failure": "error",
    "trace": "record", "please": "kindly", "help": "assist", "want": "would like",
    "broken": "invalid", "late": "delayed", "agent": "worker", "tool": "function",
}


def paraphrases(text: str) -> list[str]:
    """Deterministic meaning-preserving surface variants (metamorphic probes)."""
    words = text.split()
    syn = " ".join(_SYNONYMS.get(w.casefold().strip(".,!?"), w) for w in words)
    variants = [
        text.casefold(),
        text.upper(),
        "  ".join(words),
        re.sub(r"[.,!?;:]", "", text),
        syn,
        "Hi there. " + text + " Thanks!",
    ]
    return [v for v in dict.fromkeys(variants) if v != text]


def _re2() -> Any:
    import re2  # google-re2

    return re2


def leak_screen(rules: Sequence[RuleSpec], dataset_texts: Sequence[str], *,
                fragile_share: float = 0.5) -> list[LeakFinding]:
    texts = [t for t in dataset_texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    folded = [t.casefold() for t in texts]
    tgrams = set().union(*(_grams(_words(t)) for t in texts))
    out: list[LeakFinding] = []
    for r in rules:
        for lit in rule_literals(r):
            d = digest(lit)
            lw = _words(lit)
            if len(lw) >= 3 and _grams(lw) & tgrams:
                out.append(LeakFinding(rule_id=r.id, kind="ngram", literal_digest=d,
                                       detail="word 3-gram shared with eval dataset text"))
            elif any(rx.match(lit) for rx in _ID_RES) and any(lit.casefold() in f for f in folded):
                out.append(LeakFinding(rule_id=r.id, kind="id", literal_digest=d,
                                       detail="eval case identifier memorized"))
            elif len(lit) >= 16 and any(lit.casefold() in f for f in folded):
                out.append(LeakFinding(rule_id=r.id, kind="verbatim", literal_digest=d,
                                       detail="long verbatim eval substring"))
        re2 = _re2()
        for pat in rule_regexes(r):
            try:
                rx = re2.compile(pat)
            except (re2.error, ValueError, TypeError):
                log.debug("leak screen skipped non-RE2 pattern in %s (reported by safety check)", r.id)
                continue
            hit = [t for t in texts if rx.search(t)]
            fragile = 0
            for t in hit:
                ps = paraphrases(t)
                misses = sum(1 for p in ps if not rx.search(p))
                fragile += bool(ps) and misses / len(ps) >= fragile_share
            if hit and fragile / len(hit) >= fragile_share:
                out.append(LeakFinding(rule_id=r.id, kind="paraphrase_fragile", literal_digest=digest(pat),
                                       detail=f"pattern fails paraphrase probes on {fragile}/{len(hit)} texts"))
    return out


def safety_check(rules: Sequence[RuleSpec]) -> list[str]:
    re2 = _re2()
    errs = []
    for r in rules:
        for pat in rule_regexes(r):
            try:
                re2.compile(pat)
            except (re2.error, ValueError, TypeError):
                errs.append(f"{r.id}: regex does not compile under RE2")
    return errs


# ---------------------------------------------------------------- main entry

def _as_rules(rules: Any) -> tuple[list[RuleSpec], Any]:
    """(rule list, preloaded bundle or None)."""
    if isinstance(rules, RuleSpec):
        return [rules], None
    if isinstance(rules, RuleFile):
        return list(rules.rules), None
    if hasattr(rules, "rules") and not isinstance(rules, (list, tuple)):
        return list(rules.rules), rules
    return [r if isinstance(r, RuleSpec) else RuleSpec.model_validate(r) for r in rules], None


def holdout_members(cluster: LessonCluster, trajectories: Iterable[Trajectory], *,
                    fraction: float = HOLDOUT_FRACTION, window: int = DEFAULT_WINDOW) -> list[Trajectory]:
    """Failures in held-out families whose fingerprint matches the cluster's (never used for mining)."""
    return [t for t in trajectories if t.split == "evolve" and is_failure(t) and is_holdout(t.family, fraction)
            and cluster_id(fingerprint(t, window=window)) == cluster.id]


def validate(rules: RuleSpec | Sequence[RuleSpec] | RuleFile | Any, trajectories: Iterable[Trajectory], *,
             cluster: LessonCluster | None = None, dataset_texts: Sequence[str] = (),
             config: ReplayConfig | None = None, engine: Any = None, work_dir: Path | None = None,
             extractors: Sequence[Path] = ()) -> ReplayReport:
    """Reject or pass-to-closed-loop. Never accepts (B4).

    ``extractors``: ExtractorFile YAML paths, required for rules that reference ``state`` flags."""
    cfg = config or ReplayConfig()
    reasons: list[str] = []
    try:
        rule_list, bundle = _as_rules(rules)
    except (ValidationError, ValueError, TypeError) as exc:  # raw dicts that are not RuleSpecs
        return ReplayReport(verdict="reject", ok=False, reasons=[f"invalid rule spec: {str(exc)[:300]}"])
    if not rule_list:
        return ReplayReport(verdict="reject", ok=False, reasons=["no rules"])
    bdigest = bundle_digest(rule_list)
    base = {"bundle_digest": bdigest, "cluster_id": cluster.id if cluster else None}
    reasons += safety_check(rule_list)
    if reasons:  # never hand unsafe patterns to the engine
        return ReplayReport(verdict="reject", ok=False, reasons=reasons, **base)
    try:
        eng = _engine(engine)
    except ImportError:
        return ReplayReport(verdict="reject", ok=False, reasons=[*reasons, "rule engine ci_lab.rules unavailable"],
                            **base)
    if bundle is None:
        try:
            bundle = _load(rule_list, eng, work_dir, extractors)
        except RuleLoadFailure as exc:
            return ReplayReport(verdict="reject", ok=False, reasons=[*reasons, f"rule load failed: {exc}"], **base)

    trajs = [t for t in trajectories if t.split == "evolve"]
    ids = [r.id for r in rule_list]
    blocking = {r.id for r in rule_list if r.action in ("block", "redact")}
    per = {i: {"fires": 0, "fp": 0, "hits": 0} for i in ids}

    def fired(t: Trajectory) -> set[str]:
        return {_rule_id(m) for m in eng.evaluate_trajectory(bundle, t.steps)}

    fires_by_traj = {t.id: fired(t) for t in trajs}
    n = len(trajs)
    blocked_any = sum(1 for t in trajs if fires_by_traj[t.id] & blocking)
    for t in trajs:
        for rid in fires_by_traj[t.id]:
            if rid in per:
                per[rid]["fires"] += 1

    # recall on held-out families
    recall = None
    hold: list[Trajectory] = []
    if cluster is not None:
        hold = holdout_members(cluster, trajs, fraction=cfg.holdout_fraction, window=cfg.window)
        if hold:
            hits = 0
            for t in hold:
                f = fires_by_traj[t.id]
                hits += bool(f)
                for rid in f:
                    if rid in per:
                        per[rid]["hits"] += 1
            recall = hits / len(hold)
            if recall < cfg.min_recall:
                reasons.append(f"holdout recall {recall:.2f} < {cfg.min_recall} (n={len(hold)})")
        elif cfg.require_holdout:
            reasons.append("no held-out family members for this cluster: recall unmeasurable")
    elif cfg.require_holdout:
        reasons.append("no cluster given: holdout recall unmeasurable")

    # false positives on known-good
    neg = [t for t in trajs if is_good(t)]
    fp = 0
    for t in neg:
        f = fires_by_traj[t.id]
        fp += bool(f)
        for rid in f:
            if rid in per:
                per[rid]["fp"] += 1
    fp_rate = fp / len(neg) if neg else None
    fp_ucb = cp_upper(fp, len(neg), confidence=cfg.confidence) if neg else None
    if len(neg) < cfg.min_negatives:
        reasons.append(f"negatives {len(neg)} < {cfg.min_negatives}")
    if fp_ucb is not None and fp_ucb > cfg.epsilon:
        reasons.append(f"fp_ucb {fp_ucb:.4f} > epsilon {cfg.epsilon} ({fp}/{len(neg)})")

    # block-rate ceilings
    per_rule = []
    for rid in ids:
        br = per[rid]["fires"] / n if n and rid in blocking else 0.0
        if br > cfg.max_block_rate_rule:
            reasons.append(f"{rid}: block rate {br:.3f} > {cfg.max_block_rate_rule}")
        per_rule.append(RuleReplay(rule_id=rid, fires_evolve=per[rid]["fires"], block_rate=br, fp=per[rid]["fp"],
                                   recall_hits=per[rid]["hits"]))
    total_br = blocked_any / n if n else 0.0
    if total_br > cfg.max_block_rate_total:
        reasons.append(f"aggregate block rate {total_br:.3f} > {cfg.max_block_rate_total}")
    if not n:
        reasons.append("no evolve trajectories to replay")

    leaks = leak_screen(rule_list, dataset_texts, fragile_share=cfg.paraphrase_fragile_share)
    reasons += [f"{lk.rule_id}: leak ({lk.kind}) {lk.detail}" for lk in leaks]

    ok = not reasons
    return ReplayReport(verdict="pass_to_closed_loop" if ok else "reject", ok=ok, reasons=reasons,
                        per_rule=per_rule, recall=recall, holdout_n=len(hold), fp=fp, negatives=len(neg),
                        fp_rate=fp_rate, fp_ucb=fp_ucb, block_rate=total_br, n_evolve=n, leaks=leaks, **base)


def validate_ok(rules: Any, trajectories: Iterable[Trajectory], **kw: Any) -> tuple[bool, list[str]]:
    r = validate(rules, trajectories, **kw)
    return r.ok, list(r.reasons)


def dataset_texts_from_yaml(path: Path) -> list[str]:
    """Evolve-split eval texts (user turns, notes, expected strings) from a dataset YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            if len(v) >= 8:
                out.append(v)
        elif isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    cases = raw.get("cases") if isinstance(raw, Mapping) else raw
    for c in cases or ():
        if isinstance(c, Mapping) and str(c.get("split", "evolve")).lower() not in ("evolve", ""):
            continue
        walk(c)
    return out
