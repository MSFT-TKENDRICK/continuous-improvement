"""Trusted template ids + slots and the fixed safe RE2 pattern library (B3, B7).

Runtime ``message``/``fix`` text lives in the frozen catalog ``ci_lab/rules/templates.yaml``
(M14, :func:`ci_lab.rules.default_templates`). The synthesizers here only pick a template id
from :data:`REQUIRED_TEMPLATES` and fill typed slots (tool names, arg names, flags, enum
classes) — never trace-derived prose. :data:`REQUIRED_TEMPLATES` documents the ids/slots this
module needs from the catalog; :func:`catalog_gaps` reports any the catalog lacks.

Response patterns come only from :data:`SAFE_PATTERNS` (keyed by a pattern *class*); a rule
never embeds a literal taken from a trace.
"""

from __future__ import annotations

from collections.abc import Mapping

from ci_lab.rulespec import REGEX_MAX_LEN, TemplateSpec

TPL_PRIOR_CALL = "precondition.prior_call"
TPL_STATE_FLAG = "precondition.state_flag"
TPL_ARG_CONSTRAINT = "arg.constraint"
TPL_AMOUNT_PRIOR = "amount.not_exceed_prior"
TPL_REDACT_PATTERN = "response.redact_pattern"

REQUIRED_TEMPLATES: Mapping[str, TemplateSpec] = {
    t.id: t
    for t in (
        TemplateSpec(id=TPL_PRIOR_CALL, slots=["tool", "prior_tool", "subject"],
                     message="{tool} requires a prior successful {prior_tool} for the same {subject}.",
                     fix="Call {prior_tool} for this {subject} first, then retry {tool}."),
        TemplateSpec(id=TPL_STATE_FLAG, slots=["tool", "flag", "subject", "via_tool"],
                     message="{tool} requires {flag} for this {subject}.",
                     fix="Establish {flag} with {via_tool} for this {subject} before calling {tool}."),
        TemplateSpec(id=TPL_ARG_CONSTRAINT, slots=["tool", "arg", "constraint"],
                     message="{tool} argument {arg} is not allowed ({constraint}).",
                     fix="Correct {arg} so it satisfies {constraint}, then retry {tool}."),
        TemplateSpec(id=TPL_AMOUNT_PRIOR, slots=["tool", "arg", "prior_tool", "prior_field"],
                     message="{tool} {arg} exceeds {prior_field} returned by {prior_tool}.",
                     fix="Use a {arg} no greater than {prior_field} from {prior_tool}."),
        TemplateSpec(id=TPL_REDACT_PATTERN, slots=["pattern_class", "flag"],
                     message="Response contained {pattern_class} before {flag}.",
                     fix="Do not include {pattern_class} until {flag} is established."),
    )
}

# Fixed, reviewed RE2 patterns (linear time; no backrefs/lookaround; single-level quantifiers).
SAFE_PATTERNS: Mapping[str, str] = {
    "email": r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}",
    "phone": r"\+?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}",
    "street_address": (r"(?i)\b\d{1,6} [a-z][a-z .'-]{1,40} "
                       r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b"),
}
for _name, _pat in SAFE_PATTERNS.items():
    assert len(_pat) <= REGEX_MAX_LEN, _name


def trusted_templates() -> tuple[Mapping[str, TemplateSpec], bool]:
    """``(catalog, frozen)``: the frozen M14 catalog when ``ci_lab.rules`` is importable, else
    :data:`REQUIRED_TEMPLATES` with ``frozen=False`` (development fallback only).

    # HOOK(M14): ``ci_lab.rules.default_templates()`` must contain every id in REQUIRED_TEMPLATES.
    """
    try:
        from ci_lab.rules import default_templates  # type: ignore[import-not-found]
    except ImportError:
        return REQUIRED_TEMPLATES, False
    return default_templates(), True


def catalog_gaps(catalog: Mapping[str, TemplateSpec]) -> list[str]:
    """Template ids (or ``id:slot``) this module needs that ``catalog`` lacks."""
    gaps: list[str] = []
    for tid, need in REQUIRED_TEMPLATES.items():
        have = catalog.get(tid)
        if have is None:
            gaps.append(tid)
            continue
        gaps += [f"{tid}:{s}" for s in need.slots if s not in have.slots]
    return gaps


def render(template: TemplateSpec, slots: Mapping[str, str | int]) -> tuple[str, str]:
    """``(message, fix)`` with slots filled (missing slots stay as ``{slot}``)."""

    class _Keep(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    values = _Keep({k: str(v) for k, v in slots.items()})
    return template.message.format_map(values), template.fix.format_map(values)
