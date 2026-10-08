"""Compiled, immutable rule bundle + transactional validation (design §13.2, §13.6 B3/B7/N6)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import re2

from ci_lab.rulespec import (
    REGEX_MAX_LEN,
    AllPred,
    AnyPred,
    ArgPred,
    CountPred,
    ExtractorSpec,
    NotPred,
    PriorPred,
    RuleSpec,
    StatePred,
    TemplateSpec,
    TextPred,
    bundle_digest,
    canonical_json,
)

RUNG_ORDER = {"R1": 1, "R2": 2, "R3": 3, "R4": 4}


@dataclass(frozen=True)
class Problem:
    """One load/validation problem, rendered lint-arch style by the CLI."""

    file: str
    violation: str
    fix: str

    def __str__(self) -> str:
        return f"{self.file}: {self.violation}"


class RuleLoadError(Exception):
    """The bundle failed validation. ``problems`` lists *every* problem found (transactional load)."""

    def __init__(self, problems: list[str], details: Sequence[Problem] | None = None) -> None:
        self.problems = list(problems)
        self.details = list(details) if details is not None else [Problem("<bundle>", p, "") for p in problems]
        super().__init__("\n".join(self.problems) or "rule bundle failed to load")


def rule_order(r: RuleSpec) -> tuple[int, str]:
    """Deterministic composition order (N6): rung, then id."""
    return (RUNG_ORDER[r.rung], r.id)


@dataclass(frozen=True)
class Bundle:
    """Immutable, validated and RE2-compiled rule bundle. Build via :func:`load_bundle` /
    :func:`build_bundle`; never construct directly.

    ``rules`` are sorted by (rung, id) — the deterministic composition order (N6).
    ``digest`` is :func:`ci_lab.rulespec.bundle_digest` over the rules (BUNDLE.lock, OES ext);
    ``config_digest`` additionally covers extractors and the templates the rules use.
    """

    rules: tuple[RuleSpec, ...]
    extractors: tuple[ExtractorSpec, ...]
    templates: Mapping[str, TemplateSpec]
    digest: str
    config_digest: str = ""
    sources: tuple[Path, ...] = ()         # rule files (LKG copies these)
    source_digests: tuple[str, ...] = ()   # sha256 of each source at load time
    _patterns: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _rendered: Mapping[str, tuple[str, str]] = field(default_factory=dict, repr=False, compare=False)
    _redact_patterns: Mapping[str, tuple[str, ...]] = field(default_factory=dict, repr=False, compare=False)
    _by_on: Mapping[str, tuple[RuleSpec, ...]] = field(default_factory=dict, repr=False, compare=False)

    def __hash__(self) -> int:
        return hash((self.digest, self.config_digest))

    def rule(self, rule_id: str) -> RuleSpec:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)

    @property
    def flags(self) -> frozenset[str]:
        return frozenset(e.flag for e in self.extractors)

    def pattern(self, pattern: str) -> Any:
        """Precompiled RE2 regex for a pattern that appears in this bundle."""
        return self._patterns[pattern]

    def render(self, rule: RuleSpec) -> tuple[str, str]:
        """(message, fix) rendered from the trusted template catalog (B3)."""
        got = self._rendered.get(rule.id)
        if got is None:
            got = _render(self.templates[rule.template], rule.slots)
        return got

    def rules_for(self, on: str) -> tuple[RuleSpec, ...]:
        return self._by_on.get(on, ())

    def redact_patterns(self, rule: RuleSpec) -> tuple[str, ...]:
        return self._redact_patterns.get(rule.id, ())


def compile_pattern(pattern: Any) -> Any:
    """Compile with RE2 only (B7). Raises ``ValueError`` for unsafe/unsupported patterns."""
    if not isinstance(pattern, str) or not pattern or len(pattern) > REGEX_MAX_LEN:
        raise ValueError(f"regex must be a non-empty str <= {REGEX_MAX_LEN} chars")
    opts = re2.Options()
    opts.log_errors = False
    try:
        return re2.compile(pattern, opts)
    except (re2.error, TypeError, ValueError) as exc:  # never fall back to `re`
        raise ValueError(f"RE2 rejected pattern {pattern!r}: {exc}") from None


def _render(t: TemplateSpec, slots: Mapping[str, Any]) -> tuple[str, str]:
    msg, fix = t.message, t.fix
    for k in t.slots:
        v = str(slots.get(k, ""))
        msg = msg.replace("{" + k + "}", v)
        fix = fix.replace("{" + k + "}", v)
    return msg, fix


def leaves(p: Any) -> list[Any]:
    """All leaf predicates (prior counts as a leaf and its ``where`` leaves are included)."""
    if p is None:
        return []
    if isinstance(p, (AllPred, AnyPred)):
        return [x for c in p.of for x in leaves(c)]
    if isinstance(p, NotPred):
        return leaves(p.of)
    if isinstance(p, PriorPred):
        return [p, *leaves(p.where)]
    return [p]


def _conjuncts(p: Any) -> list[Any]:
    """Top-level facts a predicate *requires* (flattening nested ``all``)."""
    if isinstance(p, AllPred):
        return [x for c in p.of for x in _conjuncts(c)]
    return [p]


def _contradicts(a: Any, b: Any) -> str | None:
    """Return a reason if requiring both leaf facts at once is unsatisfiable."""
    from ci_lab.rules.engine import strict_eq

    if isinstance(a, ArgPred) and isinstance(b, ArgPred) and a.path == b.path:
        ops = {a.op, b.op}
        if a.op == b.op == "eq" and not strict_eq(a.value, b.value):
            return f"{a.path} eq {a.value!r} vs eq {b.value!r}"
        if ops == {"eq", "ne"} and strict_eq(a.value, b.value):
            return f"{a.path} eq/ne {a.value!r}"
        if ops == {"eq", "in"}:
            eq, lst = (a, b) if a.op == "eq" else (b, a)
            if not any(strict_eq(eq.value, x) for x in lst.value):
                return f"{a.path} eq {eq.value!r} not in {lst.value!r}"
        if ops == {"eq", "nin"}:
            eq, lst = (a, b) if a.op == "eq" else (b, a)
            if any(strict_eq(eq.value, x) for x in lst.value):
                return f"{a.path} eq {eq.value!r} but nin {lst.value!r}"
        if a.op == b.op == "in" and not any(strict_eq(x, y) for x in a.value for y in b.value):
            return f"{a.path} in disjoint sets {a.value!r} / {b.value!r}"
        if a.op == b.op == "exists" and (a.value is not False) != (b.value is not False):
            return f"{a.path} exists vs not exists"
    if (isinstance(a, StatePred) and isinstance(b, StatePred) and a.flag == b.flag
            and a.subject == b.subject and a.value != b.value):
        return f"state {a.flag} required both {a.value} and {b.value}"
    return None


def _pred_json(p: Any) -> str:
    return "null" if p is None else canonical_json(p.model_dump(mode="json"))


def build_bundle(
    rules: Iterable[RuleSpec],
    extractors: Iterable[ExtractorSpec] = (),
    *,
    templates: Mapping[str, TemplateSpec] | None = None,
    origins: Mapping[str, str] | None = None,
    sources: Sequence[Path] = (),
    source_digests: Sequence[str] = (),
) -> Bundle:
    """Validate + compile in-memory rules (used by the loader, synthesizers and replay).

    ``templates`` defaults to the trusted catalog. ``origins`` maps rule id -> file for messages.
    Raises :class:`RuleLoadError` listing every problem; nothing is partially built.
    """
    from ci_lab.rules.loader import default_templates

    rules = list(rules)
    extractors = list(extractors)
    tmpl: Mapping[str, TemplateSpec] = default_templates() if templates is None else templates
    origins = origins or {}
    probs: list[Problem] = []

    def where(r: RuleSpec) -> str:
        return f"{origins.get(r.id, '<rules>')}#{r.id}"

    for k, t in tmpl.items():
        if not isinstance(t, TemplateSpec) or t.id != k:
            probs.append(Problem("<templates>", f"template catalog entry {k!r} is invalid",
                                 "Key each TemplateSpec by its own id."))

    seen: dict[str, str] = {}
    for r in rules:
        if r.id in seen:
            probs.append(Problem(where(r), f"duplicate rule id {r.id!r} (also in {seen[r.id]})",
                                 "Give every rule a unique id; bump `version` instead of duplicating."))
        else:
            seen[r.id] = origins.get(r.id, "<rules>")

    flags = {e.flag for e in extractors}
    ex_seen: set[tuple[str, str, str, str]] = set()
    for e in extractors:
        key = (e.flag, e.tool, e.result_path, e.subject)
        if key in ex_seen:
            probs.append(Problem("<extractors>", f"duplicate extractor {key}", "Remove the duplicate extractor."))
        ex_seen.add(key)

    patterns: dict[str, Any] = {}
    rendered: dict[str, tuple[str, str]] = {}
    redact_pats: dict[str, tuple[str, ...]] = {}
    for r in rules:
        t = tmpl.get(r.template) if isinstance(tmpl.get(r.template), TemplateSpec) else None
        if t is None:
            probs.append(Problem(where(r), f"unknown template id {r.template!r}",
                                 "Use a template id from the trusted catalog (ci_lab/rules/templates.yaml)."))
        else:
            want, got = set(t.slots), set(r.slots)
            if want != got:
                probs.append(Problem(
                    where(r), f"template {t.id!r} slots mismatch: missing {sorted(want - got)}, "
                    f"unexpected {sorted(got - want)}",
                    f"Provide exactly the slots {sorted(want)}."))
            else:
                rendered[r.id] = _render(t, r.slots)
        for leaf in leaves(r.when) + leaves(r.require):
            pat = None
            if isinstance(leaf, ArgPred):
                if leaf.op == "matches":
                    pat = leaf.value
                if leaf.op == "exists" and leaf.value is not None and not isinstance(leaf.value, bool):
                    probs.append(Problem(where(r), f"exists on {leaf.path} takes true/false/null",
                                         "Use `value: true` (exists) or `value: false` (absent)."))
            elif isinstance(leaf, TextPred):
                pat = leaf.matches
            elif isinstance(leaf, StatePred) and leaf.flag not in flags:
                probs.append(Problem(where(r), f"state flag {leaf.flag!r} has no extractor",
                                     "Add an ExtractorSpec for the flag or pass its extractor file."))
            elif isinstance(leaf, (PriorPred, CountPred)) and not leaf.tool:
                probs.append(Problem(where(r), f"{leaf.kind} needs a tool", "Name the tool (or '*')."))
            if pat is not None and pat not in patterns:
                try:
                    patterns[pat] = compile_pattern(pat)
                except ValueError as exc:
                    probs.append(Problem(where(r), str(exc),
                                         "Use a short RE2 pattern without backreferences or lookaround."))
        if r.action == "redact":
            redact_pats[r.id] = tuple(x.matches for x in leaves(r.require) if isinstance(x, TextPred))
            if not redact_pats[r.id]:
                probs.append(Problem(where(r), "redact rule has no text predicate in `require`",
                                     "Express what to mask as `text` predicates under `require`."))

    probs.extend(_conflicts(rules, where))
    if probs:
        raise RuleLoadError([str(p) for p in probs], probs)

    ordered = tuple(sorted(rules, key=rule_order))
    used = {r.template for r in ordered}
    digest = bundle_digest(list(ordered))
    cfg = canonical_json({
        "rules": digest,
        "extractors": sorted(canonical_json(e.model_dump(mode="json")) for e in extractors),
        "templates": {k: tmpl[k].model_dump(mode="json") for k in sorted(used)},
    })
    by_on = {on: tuple(r for r in ordered if r.on == on) for on in ("tool_call", "response", "trajectory")}
    return Bundle(
        rules=ordered,
        extractors=tuple(extractors),
        templates=MappingProxyType(dict(tmpl)),
        digest=digest,
        config_digest="sha256:" + hashlib.sha256(cfg.encode()).hexdigest(),
        sources=tuple(sources),
        source_digests=tuple(source_digests),
        _patterns=MappingProxyType(patterns),
        _rendered=MappingProxyType(rendered),
        _redact_patterns=MappingProxyType(redact_pats),
        _by_on=MappingProxyType(by_on),
    )


def _conflicts(rules: list[RuleSpec], where: Any) -> list[Problem]:
    out: list[Problem] = []
    groups: dict[tuple[str, str], list[RuleSpec]] = {}
    for r in rules:
        groups.setdefault((r.target, r.on), []).append(r)
    for (target, on), rs in sorted(groups.items()):
        rs = sorted(rs, key=rule_order)
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                if a.id == b.id:
                    continue
                wa, wb = _pred_json(a.when), _pred_json(b.when)
                if wa == wb and _pred_json(a.require) == _pred_json(b.require):
                    if a.action != b.action or a.mode != b.mode:
                        out.append(Problem(
                            where(b), f"conflict: {a.id!r} and {b.id!r} ({on} {target}) have identical "
                            f"predicates but different action/mode ({a.action}/{a.mode} vs {b.action}/{b.mode})",
                            "Keep one rule (strongest action wins) and delete the other."))
                    continue
                if not (wa == wb or a.when is None or b.when is None):
                    continue
                reasons = {x for la in _conjuncts(a.require) for lb in _conjuncts(b.require)
                           if (x := _contradicts(la, lb))}
                for reason in sorted(reasons):
                    out.append(Problem(
                        where(b), f"conflict: {a.id!r} and {b.id!r} ({on} {target}) require contradictory "
                        f"facts: {reason}",
                        "Merge the rules or scope them with disjoint `when` predicates."))
    return out
