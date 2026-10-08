"""Student context firewall (bus contract v2 §6).

``LeakScreen`` finds rubric material (canaries, verbatim criterion descriptions/questions, shared
8-grams, criterion ids, assert suite names, plus any ``extra_corpus``) in text; ``sanitize_correction``
turns verdict reasons into a leak-free ``StudentCorrection``; ``StudentFirewallMiddleware`` is the
MAF middleware pair that aborts a student run (``ContextLeak``) when a model message or tool result
would carry such material. ``ContextLeak`` is a ``MiddlewareFailure`` so it is fail-closed: it is
never absorbed into a tool result and always propagates to the ``Agent.run`` caller, which records
``abort{reason:"context_leak"}``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from agent_framework import (
    AgentContext,
    AgentMiddleware,
    Content,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    MiddlewareFailure,
)

from ci_lab.bus.types import StudentCorrection
from ci_lab.taskgraph.model import Rubric
from ci_lab.taskgraph.validate import LEAK_N, rubric_material
from ci_lab.tools.critic_checks import WORD_RE, LeakCorpus

__all__ = [
    "CATEGORIES",
    "GENERIC_CORRECTION",
    "ContextLeak",
    "LeakScreen",
    "StudentFirewallAgentMiddleware",
    "StudentFirewallFunctionMiddleware",
    "StudentFirewallMiddleware",
    "sanitize_correction",
]

GENERIC_CORRECTION = ("The previous attempt did not meet the deliverable's requirements; "
                      "re-read the instructions and verify the output artifact.")
MAX_CORRECTION = 800
REDACTED = "[redacted]"
VERBATIM_MIN_WORDS = 3
CATEGORIES = ("canary", "criterion_id", "suite", "rubric_text", "ngram", "corpus")

_WORD = re.compile(WORD_RE.pattern, re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\.?\d+(?:[.,/:]\d+)*(?:\s*(?:%|percent\b|pct\b))?", re.IGNORECASE)
_HEADER = "The previous attempt was rejected:"


def _phrase(text: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(w) for w in text.split()), re.IGNORECASE)


def _literal(lit: str) -> re.Pattern[str]:
    # Stricter than LeakCorpus on the left: a dotted prefix ("judge.<criterion>") still leaks the identifier.
    return re.compile(r"(?<![\w@-])" + re.escape(lit) + r"(?![\w@-])", re.IGNORECASE)


class LeakScreen:
    """Screens text for the secret material of ``rubrics`` and ``extra_corpus`` (8-gram logic of ``LeakCorpus``)."""

    def __init__(self, rubrics: Iterable[Rubric] = (), extra_corpus: Iterable[str] = (), *, n: int = LEAK_N) -> None:
        self.n = n
        texts: list[str] = []
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for rubric in rubrics:
            m = rubric_material(rubric)
            texts += m["text"]
            self._patterns += [("canary", re.compile(re.escape(c), re.IGNORECASE)) for c in m["canary"] if c]
            self._patterns += [(k, _literal(s.strip())) for k in ("criterion_id", "suite") for s in m[k] if s.strip()]
            self._patterns += [("rubric_text", _phrase(t)) for t in m["text"] if len(t.split()) >= VERBATIM_MIN_WORDS]
        extra = [s for s in extra_corpus if isinstance(s, str) and s.strip()]
        texts += extra
        self._patterns += [("corpus", _phrase(s) if len(s.split()) >= VERBATIM_MIN_WORDS else _literal(s.strip()))
                           for s in extra]
        self._grams = LeakCorpus.build(texts, (), n=n).ngrams

    def _spans(self, text: str) -> list[tuple[int, int, str]]:
        spans = [(m.start(), m.end(), cat) for cat, rx in self._patterns for m in rx.finditer(text)]
        words = list(_WORD.finditer(text))
        for i in range(len(words) - self.n + 1):
            window = words[i:i + self.n]
            if tuple(w.group().lower() for w in window) in self._grams:
                spans.append((window[0].start(), window[-1].end(), "ngram"))
        return spans

    def hits(self, text: str) -> list[str]:
        """Categories of leaked material in ``text`` (subset of ``CATEGORIES``); never quotes the material."""
        found = {cat for _, _, cat in self._spans(text)}
        return [c for c in CATEGORIES if c in found]

    def redact(self, text: str) -> str:
        """``text`` with every leaked span replaced by ``[redacted]``."""
        merged: list[list[int]] = []
        for start, end, _ in sorted(self._spans(text)):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out, pos = [], 0
        for start, end in merged:
            out += [text[pos:start], REDACTED]
            pos = end
        return "".join(out) + text[pos:]


def _clean(reason: str, screen: LeakScreen) -> str:
    text = _NUMBER.sub("", screen.redact(reason))
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    text = re.sub(r"\s+([)\],;:.])", r"\1", re.sub(r"\s+", " ", text)).strip(" -:;,.")
    residue = re.sub(r"[\W\d_]+", " ", text.replace(REDACTED, " ")).split()
    return text if len(residue) >= 2 else ""


def sanitize_correction(verdict_reasons: Iterable[str] | str, rubric: Rubric | None, *, attempt: str,
                        extra_corpus: Iterable[str] = ()) -> StudentCorrection:
    """Actionable, leak-free correction for ``attempt`` (``task@n``) built from failing verdict reasons."""
    reasons = [verdict_reasons] if isinstance(verdict_reasons, str) else list(verdict_reasons)
    screen = LeakScreen([rubric] if rubric is not None else (), extra_corpus)
    lines: list[str] = []
    for reason in reasons:
        line = _clean(str(reason), screen) if reason else ""
        if not line or f"- {line}" in lines:
            continue
        room = MAX_CORRECTION - len(_HEADER) - sum(len(x) + 1 for x in lines) - 3
        if room < 40:
            break
        if len(line) > room:
            line = line[:room - 3].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {line}")
    text = "\n".join([_HEADER, *lines]) if lines else GENERIC_CORRECTION
    if screen.hits(text) or _NUMBER.search(text):
        text = GENERIC_CORRECTION
    return StudentCorrection(text=text, attempt=attempt)


class ContextLeak(MiddlewareFailure):
    """Rubric material reached a student's context; the caller records ``abort{reason:"context_leak"}``."""

    reason = "context_leak"

    def __init__(self, where: str, hits: Sequence[str]) -> None:
        self.where = where
        self.hits = tuple(hits)
        super().__init__(f"context_leak in {where}: {', '.join(self.hits)}")


def _text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Message):
        return _text(obj.contents)
    if isinstance(obj, Content):
        return "\n".join(_text(getattr(obj, a, None)) for a in ("text", "result", "arguments", "items"))
    if isinstance(obj, (list, tuple)):
        return "\n".join(_text(o) for o in obj)
    return str(obj)


class _Screening:
    def __init__(self, screen: LeakScreen, leaks: list[ContextLeak]) -> None:
        self.screen = screen
        self.leaks = leaks

    def check(self, where: str, payload: Any) -> None:
        hits = self.screen.hits(_text(payload))
        if hits:
            leak = ContextLeak(where, hits)
            self.leaks.append(leak)
            raise leak


class StudentFirewallAgentMiddleware(_Screening, AgentMiddleware):
    """Screens every message handed to the student model before the run starts."""

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        for i, message in enumerate(context.messages):
            self.check(f"message[{i}]", message)
        await call_next()


class StudentFirewallFunctionMiddleware(_Screening, FunctionMiddleware):
    """Screens every tool result before it is returned to the student model."""

    async def process(self, context: FunctionInvocationContext,
                      call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        self.check(f"tool:{context.function.name}", context.result)


class StudentFirewallMiddleware(list[Any]):
    """``[agent, function]`` firewall middleware; pass as ``Agent(middleware=...)``. ``leaks`` records aborts."""

    def __init__(self, screen: LeakScreen) -> None:
        self.screen = screen
        self.leaks: list[ContextLeak] = []
        super().__init__([StudentFirewallAgentMiddleware(screen, self.leaks),
                          StudentFirewallFunctionMiddleware(screen, self.leaks)])
