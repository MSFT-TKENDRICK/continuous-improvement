"""Deterministic custom ACS policy adapters and annotators for the harness policies.

Each adapter is a pure function of the ACS invocation ``{policy_id, policy, binding, input}``;
host state (kill switch, budget, history) arrives in the snapshot, never via side channels.
Snapshot shape written by the hosts (``ci_lab.governance.maf`` and the campaign gate):

* ``governance: {kill_switch: bool}``; ``agent: {name, role}``
* ``input: {text}`` / ``output: {text}`` / ``model: {id, allowed?: [host allowlist]}``
* ``call: {name, arguments}`` + ``history: [TrajectoryStep dicts]`` at tool points
* ``campaign: {dry_run, publish, budget_exhausted, arms: [{id, edit_scope: [glob]}]}``
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import Any

import re2

from ci_lab.tools.paths import glob_match, glob_prefix

__all__ = ["ANNOTATORS", "DISPATCHER", "Annotator", "globs_overlap", "redact",
           "register_leak_screen"]

_INJECTION = re2.compile(
    r"(?i)(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions|rules|prompts?)|\b(system|developer)\s+prompt\b|\byou\s+are\s+now\b"
    r"|\bdeveloper\s+mode\b|</?\s*(system|assistant)\s*>|\bjailbreak\b")
_SECRETS = re2.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|xox[abpr]-[A-Za-z0-9-]{10,})\b"
    r"|(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*\S{6,}")
_PII = {
    "email": re2.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re2.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re2.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
}
_LEAK_SCREENS: list[Callable[[str], list[str]]] = []


def register_leak_screen(screen: Callable[[str], list[str]] | None) -> None:
    """Install (or clear with ``None``) the student rubric-leak screen, e.g. a bus firewall."""
    _LEAK_SCREENS.clear()
    if screen is not None:
        _LEAK_SCREENS.append(screen)


def _luhn(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n > 4 else n * 2)
                for i, n in enumerate(reversed(nums)))
    return total % 10 == 0


def redact(text: str, *, pii: bool = True) -> tuple[str, list[str]]:
    """``(redacted text, kinds found)``: secrets always; e-mail/SSN/Luhn-valid cards if ``pii``."""
    kinds: list[str] = []

    def sub(kind: str) -> Callable[[Any], str]:
        def repl(m: Any) -> str:
            if kind == "card" and not _luhn(m.group(0)):
                return m.group(0)
            kinds.append(kind)
            return f"[redacted:{kind}]"
        return repl

    out = _SECRETS.sub(sub("secret"), text)
    if pii:
        for kind, rx in _PII.items():
            out = rx.sub(sub(kind), out)
    return out, sorted(set(kinds))


def globs_overlap(scope: str, protected: str) -> bool:
    """Whether edit-scope glob ``scope`` can reach protected glob ``protected``. Anchored globs
    overlap when one literal prefix contains the other; unanchored (``**/x/**``) ones when a
    concrete instance of ``scope`` matches."""
    sp, pp = glob_prefix(scope + "/x"), glob_prefix(protected + "/x")
    if pp:
        a, b = sp.split("/") if sp else [], pp.split("/")
        return a == b[: len(a)] or b == a[: len(b)]
    probe = scope.replace("**", "x").replace("*", "x").replace("?", "x")
    return glob_match(probe.lower(), protected) or glob_match(probe.lower() + "/x", protected)


def _matches(path: str, globs: Iterable[str]) -> bool:
    norm = path.replace("\\", "/").lower()
    while norm.startswith("./"):
        norm = norm[2:]
    return any(glob_match(norm, g.lower()) for g in globs)


def _parts(inv: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    pi = inv["input"]
    return inv["policy"], pi.get("snapshot") or {}, pi["policy_target"]["value"]


def _killed(snap: Mapping[str, Any]) -> dict[str, Any] | None:
    if (snap.get("governance") or {}).get("kill_switch"):
        return {"decision": "deny", "reason": "kill_switch_engaged",
                "message": "Kill switch engaged; all agent actions are halted."}
    return None


@lru_cache(maxsize=1)
def _extractors() -> tuple[Any, ...]:
    from ci_lab.guards.domains.order_support import EXTRACTORS
    from ci_lab.rules.loader import load_yaml
    from ci_lab.rulespec import ExtractorFile

    return tuple(ExtractorFile.model_validate(load_yaml(EXTRACTORS.read_text("utf-8"))).extractors)


def _verified(history: list[Mapping[str, Any]], order_id: Any) -> bool:
    """The frozen ``identity_verified`` flag for ``order_id`` (``ci_lab.rules`` extractors)."""
    from ci_lab.rules.engine import TrajectoryIndex
    from ci_lab.rulespec import TrajectoryStep, normalize_subject

    steps = [TrajectoryStep(i=k, **{f: s[f] for f in ("kind", "tool", "call_id", "args", "result",
                                                     "status") if s.get(f) is not None})
             for k, s in enumerate(history)]
    flags = TrajectoryIndex(steps, _extractors()).flags_at(len(steps))
    return normalize_subject(order_id) in flags.get("identity_verified", set())


def order_support_tools(inv: Mapping[str, Any]) -> dict[str, Any]:
    policy, snap, args = _parts(inv)
    if killed := _killed(snap):
        return killed
    call = snap.get("call") or {}
    if call.get("name") != "issue_refund":
        return {"decision": "allow"}
    if not _verified(list(snap.get("history") or []), args.get("order_id")):
        return {"decision": "deny", "reason": "identity_not_verified",
                "message": "Verify the customer's identity for this order before refunding."}
    from order_support.data import REFUND_LIMIT

    limit = float(policy.get("refund_limit", REFUND_LIMIT))
    amount = args.get("amount")
    if not isinstance(amount, int | float) or isinstance(amount, bool) or amount < 0:
        return {"decision": "deny", "reason": "refund_amount_invalid"}
    if amount > limit:
        return {"decision": "escalate", "reason": "refund_over_limit",
                "message": f"Refunds above {limit:.2f} need human approval.",
                "approval": {"kind": "refund_over_limit"}}
    return {"decision": "allow"}


def meta_tools(inv: Mapping[str, Any]) -> dict[str, Any]:
    policy, snap, args = _parts(inv)
    if killed := _killed(snap):
        return killed
    if (snap.get("call") or {}).get("name") not in set(policy.get("write_tools") or ()):
        return {"decision": "allow"}
    for key in policy.get("path_args") or ("path",):
        value = (args or {}).get(key)
        if isinstance(value, str) and _matches(value, policy.get("protected_globs") or ()):
            return {"decision": "deny", "reason": "protected_path",
                    "message": "Meta-agents may not write governance, guard, rule, workflow, "
                               "eval or sealed paths."}
    return {"decision": "allow"}


def model_allowlist(inv: Mapping[str, Any]) -> dict[str, Any]:
    from ci_lab.maf.models import meta_model_override, with_extra_allowed

    policy, _, model = _parts(inv)
    model_id = (model or {}).get("id")
    if not model_id:
        return {"decision": "allow", "reason": "model_unspecified"}
    allowed = set(with_extra_allowed(policy.get("allowed_models") or ()))
    allowed.update(model.get("allowed") or ())  # host allowlist already enforced at spec load
    allowed.update(m for m in [meta_model_override()] if m)
    if model_id in allowed:
        return {"decision": "allow"}
    return {"decision": "deny", "reason": "model_not_allowed",
            "message": "Model is not in the allowlist (extend via CI_ALLOWED_MODELS)."}


def screen_input(inv: Mapping[str, Any]) -> dict[str, Any]:
    markers = (inv["input"]["annotations"].get("injection") or {}).get("markers") or []
    if markers:
        return {"decision": "warn", "reason": "injection_marker",
                "message": f"{len(markers)} prompt-injection marker(s) in untrusted input."}
    return {"decision": "allow"}


def redact_output(inv: Mapping[str, Any]) -> dict[str, Any]:
    policy, snap, text = _parts(inv)
    leaks = (inv["input"]["annotations"].get("rubric_leak") or {}).get("hits") or []
    if leaks and (snap.get("agent") or {}).get("role") == "student":
        return {"decision": "deny", "reason": "rubric_leak",
                "message": "Student output contains sealed rubric material."}
    if not isinstance(text, str):
        return {"decision": "allow"}
    out, kinds = redact(text, pii=bool(policy.get("pii", True)))
    if not kinds:
        return {"decision": "allow"}
    return {"decision": "transform", "reason": "redacted:" + "/".join(kinds),
            "transform": {"path": "$target", "value": out}}


def campaign_startup(inv: Mapping[str, Any]) -> dict[str, Any]:
    policy, snap, campaign = _parts(inv)
    if killed := _killed(snap):
        return killed
    if campaign.get("budget_exhausted"):
        return {"decision": "deny", "reason": "budget_exhausted",
                "message": "SRE error budget exhausted; campaign launch refused."}
    protected = policy.get("protected_globs") or ()
    for arm in campaign.get("arms") or ():
        if any(globs_overlap(s, p) for s in arm.get("edit_scope") or () for p in protected):
            return {"decision": "deny", "reason": "protected_scope",
                    "message": f"Arm {arm.get('id')!r} edit scope touches protected paths."}
    if campaign.get("publish") and not campaign.get("dry_run", True):
        return {"decision": "escalate", "reason": "publish_requires_approval",
                "message": "Non-dry-run publish needs an approval (ci-lab governance approve).",
                "approval": {"kind": "campaign_publish"}}
    return {"decision": "allow"}


DISPATCHER: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "ci.order_support.tools": order_support_tools,
    "ci.meta.tools": meta_tools,
    "ci.model_allowlist": model_allowlist,
    "ci.screen_input": screen_input,
    "ci.redact_output": redact_output,
    "ci.campaign.startup": campaign_startup,
}


def _injection(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else ""
    return {"markers": sorted({m.group(0).lower() for m in _INJECTION.finditer(text)})[:16]}


def _rubric_leak(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else ""
    return {"hits": [h for screen in _LEAK_SCREENS for h in screen(text)][:16]}


ANNOTATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "injection_markers": _injection,
    "rubric_leak": _rubric_leak,
}


class Annotator:
    """ACS annotator dispatcher: ``classifier`` in the declaration names an :data:`ANNOTATORS` entry."""

    def dispatch(self, name: str, config: Mapping[str, Any], policy_input: Any) -> dict[str, Any]:
        return ANNOTATORS[config["classifier"]](config.get("value"))
