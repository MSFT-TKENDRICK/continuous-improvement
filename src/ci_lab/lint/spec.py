"""Dev-lint rule schema (design §13.1 rung R5, §13.4). Rules are data in ``lint/rules/*.yaml``;
the checks that interpret them are frozen code in :mod:`ci_lab.lint.engine`. Unknown keys are
rejected; text patterns are RE2 (B7) and compiled at load so a bad rule fails fast."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import re2
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ci_lab.rulespec import REGEX_MAX_LEN

LINT_SCHEMA_VERSION = 1
RULES_DIR = "lint/rules"
_ID_PATTERN = r"^[a-z][a-z0-9_.-]{2,63}$"
DECLARATIVE_BANNED_KINDS = ("If", "ConditionGroup", "Foreach", "GotoAction")
DECLARATIVE_ROOT_KINDS = ("Workflow", "Prompt", "Agent", "AdaptiveDialog")


class RuleLoadError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Rule(_M):
    id: str = Field(pattern=_ID_PATTERN)
    include: list[str] = Field(min_length=1)  # repo-relative posix globs (`**` spans dirs)
    exclude: list[str] = Field(default_factory=list)  # allowlist: paths where the rule does not apply
    message: str = Field(min_length=1, max_length=300)
    fix: str = Field(min_length=1, max_length=300)
    see: str = ""
    severity: Literal["error", "warn"] = "error"

    @field_validator("include", "exclude")
    @classmethod
    def _globs(cls, v: list[str]) -> list[str]:
        for g in v:
            if not g or g.startswith("/") or "\\" in g or ".." in g.split("/"):
                raise ValueError(f"glob must be repo-relative posix: {g!r}")
        return v


def _check_re2(pattern: str) -> str:
    if not pattern or len(pattern) > REGEX_MAX_LEN:
        raise ValueError(f"pattern must be 1..{REGEX_MAX_LEN} chars")
    try:
        re2.compile(pattern)
    except re2.error as e:
        raise ValueError(f"pattern rejected by RE2: {type(e).__name__}") from None
    return pattern


class BannedCall(_Rule):
    """Calls by dotted name, resolved through import aliases. ``*.name`` matches any receiver."""

    kind: Literal["banned_call"]
    names: list[str] = Field(min_length=1)


class BannedImport(_Rule):
    """Imports (incl. relative, ``importlib.import_module("…")`` and ``__import__``) of module prefixes."""

    kind: Literal["banned_import"]
    modules: list[str] = Field(min_length=1)


class BannedAttrArg(_Rule):
    """Arguments to attribute-setting calls built by stringifying objects (PII via ``str()`` — D6).
    Dict-literal values are checked at any position; positional arg 0 (span name / key) is not."""

    kind: Literal["banned_attr_arg"]
    calls: list[str] = Field(min_length=1)  # callee final name, e.g. set_attribute, span
    banned: list[Literal["str", "repr", "fstring", "format", "percent"]] = Field(
        default_factory=lambda: ["str", "repr", "fstring", "format", "percent"])


class BannedText(_Rule):
    kind: Literal["banned_text"]
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _p(cls, v: str) -> str:
        return _check_re2(v)


class GhaBannedTrigger(_Rule):
    kind: Literal["gha_banned_trigger"]
    triggers: list[str] = Field(min_length=1)


class GhaPinnedSha(_Rule):
    """Every ``uses:`` is ``owner/repo[/path]@<40-hex>``; local ``./`` actions and
    ``docker://…@sha256:<64-hex>`` are allowed."""

    kind: Literal["gha_pinned_sha"]


class DeclarativeYamlExpressionFree(_Rule):
    """MAF declarative YAML (I3, no .NET): no ``=``-prefixed PowerFx strings, no control-flow kinds.
    Applies to YAML documents whose top-level ``kind`` is one of ``root_kinds``."""

    kind: Literal["declarative_yaml_expression_free"]
    banned_kinds: list[str] = Field(default_factory=lambda: list(DECLARATIVE_BANNED_KINDS))
    root_kinds: list[str] = Field(default_factory=lambda: list(DECLARATIVE_ROOT_KINDS))


class MaxLines(_Rule):
    kind: Literal["max_lines"]
    max: int = Field(ge=1)


LintRule = Annotated[
    BannedCall | BannedImport | BannedAttrArg | BannedText | GhaBannedTrigger | GhaPinnedSha
    | DeclarativeYamlExpressionFree | MaxLines,
    Field(discriminator="kind"),
]


class LintRuleFile(_M):
    schema_version: Literal[1] = LINT_SCHEMA_VERSION
    rules: list[LintRule]

    @model_validator(mode="after")
    def _unique(self) -> LintRuleFile:
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate rule ids {dup}")
        return self


def rule_files(root: Path) -> list[Path]:
    d = Path(root) / RULES_DIR
    return sorted([*d.glob("*.yaml"), *d.glob("*.yml")]) if d.is_dir() else []


def parse_rules(text: str, source: str = "<string>") -> list[LintRule]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RuleLoadError([f"{source}: invalid YAML ({type(e).__name__})"]) from None
    try:
        return list(LintRuleFile.model_validate(data).rules)
    except ValidationError as e:
        raise RuleLoadError([f"{source}: {'.'.join(map(str, err['loc']))}: {err['msg']}"
                             for err in e.errors()]) from None


def load_rules(paths: Iterable[Path]) -> list[LintRule]:
    """Load and validate rule files; ids must be unique across files. Raises RuleLoadError."""
    rules: list[LintRule] = []
    errors: list[str] = []
    for p in paths:
        try:
            rules.extend(parse_rules(Path(p).read_text(encoding="utf-8"), Path(p).as_posix()))
        except RuleLoadError as e:
            errors.extend(e.errors)
        except OSError as e:
            errors.append(f"{Path(p).as_posix()}: unreadable ({type(e).__name__})")
    seen: set[str] = set()
    for r in rules:
        if r.id in seen:
            errors.append(f"duplicate rule id across files: {r.id}")
        seen.add(r.id)
    if errors:
        raise RuleLoadError(errors)
    return rules
