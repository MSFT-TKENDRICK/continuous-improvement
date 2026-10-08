"""redact(): idempotent, original-text spans, only redact rules; RE2 fuzz under a time budget (B7)."""

import time

import pytest

from ci_lab.rules import REDACTED, evaluate, redact
from ci_lab.rules._bundle import compile_pattern, leaves
from ci_lab.rulespec import TEXT_MAX_BYTES, ArgPred, GuardView, TextPred, TrajectoryStep


def _resp(text):
    return TrajectoryStep(i=0, kind="response", text=text)


def _matches(bundle, text):
    return evaluate(bundle, GuardView(pending=_resp(text)), on="response")


TEXTS = [
    "Email jo@example.com or call +1 (555) 010-0000; ship to 12 Elm Street.",
    "a@b.co a@b.co",
    "nothing to see",
    "jo@example.com+1 555 010 0000",        # adjacent spans merge
    "[redacted] then jo@example.com [redacted]",
    "Ünïcødé ✓ jo.doe@exämple.com 12 Main St and 🚀 +44 20 7946 0958",
]


@pytest.mark.parametrize("text", TEXTS)
def test_redact_is_idempotent_and_removes_pii(seeds, text):
    ms = _matches(seeds, text)
    once = redact(text, ms, seeds)
    assert redact(once, ms, seeds) == once
    assert redact(once, _matches(seeds, once), seeds) == once
    assert "jo@example.com" not in once and "010-0000" not in once
    if not ms:
        assert once == text


def test_redact_masks_with_fixed_token_and_keeps_rest(seeds):
    t = "Hi Jo, email jo@example.com today."
    assert redact(t, _matches(seeds, t), seeds) == f"Hi Jo, email {REDACTED} today."


def test_later_rules_see_original_text(mk):
    b = mk({"id": "r.word", "rung": "R3", "target": "*", "action": "redact", "template": "pii.redacted", "slots": {},
            "require": {"kind": "not", "of": {"kind": "text", "matches": "secret code"}}},
           {"id": "r.code", "rung": "R3", "target": "*", "action": "redact", "template": "pii.redacted", "slots": {},
            "require": {"kind": "not", "of": {"kind": "text", "matches": r"code \d+"}}})
    t = "the secret code 1234 is here"
    ms = evaluate(b, GuardView(pending=_resp(t)), on="response")
    assert len(ms) == 2
    assert redact(t, ms, b) == f"the {REDACTED} is here"  # overlapping spans merged on the original


def test_redact_ignores_non_redact_matches(mk):
    b = mk({"id": "r.warn", "rung": "R3", "target": "*", "action": "warn", "template": "response.blocked",
            "slots": {}, "require": {"kind": "not", "of": {"kind": "text", "matches": "x+"}}})
    ms = evaluate(b, GuardView(pending=_resp("xxx")), on="response")
    assert ms and redact("xxx", ms, b) == "xxx"


def test_text_predicates_see_capped_input_only(mk):
    b = mk({"rung": "R3", "target": "*", "action": "warn", "template": "response.blocked", "slots": {},
            "require": {"kind": "not", "of": {"kind": "text", "matches": "NEEDLE"}}})
    assert evaluate(b, GuardView(pending=_resp("a" * 100 + "NEEDLE")), on="response")
    assert not evaluate(b, GuardView(pending=_resp("a" * TEXT_MAX_BYTES + "NEEDLE")), on="response")


ADVERSARIAL = [
    "a" * TEXT_MAX_BYTES,
    "a" * (TEXT_MAX_BYTES - 1) + "!",
    "1" * TEXT_MAX_BYTES,
    "1 " * (TEXT_MAX_BYTES // 2),
    ("x" * 63 + "@") * (TEXT_MAX_BYTES // 64),
    "é" * TEXT_MAX_BYTES,
    "🚀" * (TEXT_MAX_BYTES // 2),
    "\u202e\u0000\ufffd" * 2000,
    "(" * TEXT_MAX_BYTES,
    ("12 " + "a" * 30 + " ") * 400,
]
EVIL = [r"(a+)+$", r"(a|aa)+$", r"(.*a){12}", r"(\d+\s?)+x", r"([a-z]+)*@", r"(x+x+)+y", r"^(\w+\s?)*$"]


def test_fuzz_patterns_under_time_budget(seeds):
    pats = [x.matches if isinstance(x, TextPred) else x.value
            for r in seeds.rules for x in leaves(r.when) + leaves(r.require)
            if isinstance(x, TextPred) or (isinstance(x, ArgPred) and x.op == "matches")]
    assert len(pats) >= 3
    compiled = [seeds.pattern(p) for p in pats] + [compile_pattern(p) for p in EVIL]
    start = time.perf_counter()
    worst = 0.0
    for rx in compiled:
        for s in ADVERSARIAL:
            t0 = time.perf_counter()
            rx.search(s)
            worst = max(worst, time.perf_counter() - t0)
    total = time.perf_counter() - start
    assert worst < 0.25 and total < 5.0, (worst, total)  # catastrophic backtracking would take minutes


def test_redact_on_adversarial_inputs_is_bounded(seeds):
    start = time.perf_counter()
    for s in ADVERSARIAL:
        redact(s, _matches(seeds, s), seeds)
    assert time.perf_counter() - start < 5.0
