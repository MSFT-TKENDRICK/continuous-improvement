"""Bundle validation seam: :func:`ci_lab.rules.load_bundle` when present, else a strict local
fallback (schema + unique ids + known templates/slots + RE2 compile).

# HOOK(M14): once ``ci_lab.rules`` lands, :func:`load_rules` always uses ``load_bundle``
# (conflict detection, extractor checks, regex safety) and the fallback is dead code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ci_lab.rulespec import (
    AllPred,
    AnyPred,
    ArgPred,
    NotPred,
    PriorPred,
    RuleFile,
    RuleSpec,
    TemplateSpec,
    TextPred,
    bundle_digest,
)

from .templates import trusted_templates


class BundleError(ValueError):
    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@dataclass(frozen=True)
class LoadedRules:
    rules: tuple[RuleSpec, ...]
    digest: str
    engine: str  # "ci_lab.rules" | "fallback"
    bundle: Any = None


def rule_files(guards_dir: Path) -> list[Path]:
    """Every rule file of a guards dir (``*.yaml``; BUNDLE.lock / extractor files excluded)."""
    if not guards_dir.is_dir():
        return []
    return sorted(p for p in guards_dir.glob("*.yaml") if p.is_file() and "extractor" not in p.name.casefold())


def read_rule_file(path: Path) -> RuleFile:
    return RuleFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def dump_rule_file(rules: Sequence[RuleSpec]) -> str:
    doc = RuleFile(rules=list(rules)).model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=False, width=120)


def load_rules(rule_paths: Sequence[Path], extractor_paths: Sequence[Path] = (), *,
               templates: Mapping[str, TemplateSpec] | None = None) -> LoadedRules:
    """Load + validate the full bundle. Raises :class:`BundleError`."""
    try:
        from ci_lab import rules as engine  # type: ignore[attr-defined]
    except ImportError:
        engine = None
    if engine is not None and hasattr(engine, "load_bundle"):
        try:
            bundle = engine.load_bundle(list(rule_paths), list(extractor_paths), templates=templates)
        except Exception as exc:  # RuleLoadError(list[str]) or pydantic errors
            errs = exc.args[0] if exc.args and isinstance(exc.args[0], list) else [str(exc)]
            raise BundleError([str(e) for e in errs]) from exc
        return LoadedRules(tuple(bundle.rules), bundle.digest, "ci_lab.rules", bundle)
    return _fallback(rule_paths, templates if templates is not None else trusted_templates()[0])


def subset_bundle(loaded: LoadedRules, rules: Sequence[RuleSpec], *,
                  templates: Mapping[str, TemplateSpec] | None = None) -> Any:
    """``rules`` compiled with the full bundle's extractors (for per-lesson replay), or the bare rule
    list when the engine is unavailable."""
    if loaded.bundle is None:
        return list(rules)
    from ci_lab import rules as engine  # type: ignore[attr-defined]

    return engine.build_bundle(list(rules), loaded.bundle.extractors, templates=templates)


def _patterns(p: Any) -> list[str]:
    if isinstance(p, (AllPred, AnyPred)):
        return [x for c in p.of for x in _patterns(c)]
    if isinstance(p, NotPred):
        return _patterns(p.of)
    if isinstance(p, PriorPred):
        return _patterns(p.where) if p.where is not None else []
    if isinstance(p, TextPred):
        return [p.matches]
    if isinstance(p, ArgPred) and p.op == "matches":
        return [str(p.value)]
    return []


def _fallback(rule_paths: Sequence[Path], templates: Mapping[str, TemplateSpec]) -> LoadedRules:
    import re2

    errors: list[str] = []
    rules: list[RuleSpec] = []
    for path in rule_paths:
        try:
            rules += read_rule_file(path).rules
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
    seen: set[str] = set()
    for r in rules:
        if r.id in seen:
            errors.append(f"duplicate rule id {r.id}")
        seen.add(r.id)
        tpl = templates.get(r.template)
        if tpl is None:
            errors.append(f"{r.id}: unknown template {r.template!r}")
        elif extra := sorted(set(r.slots) - set(tpl.slots)):
            errors.append(f"{r.id}: undeclared slots {extra}")
        for pat in _patterns(r.when) + _patterns(r.require):
            try:
                re2.compile(pat)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{r.id}: RE2 rejects pattern: {exc}")
    if errors:
        raise BundleError(errors)
    return LoadedRules(tuple(rules), bundle_digest(rules), "fallback")
