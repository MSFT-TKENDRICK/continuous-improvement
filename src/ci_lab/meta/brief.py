"""Student-facing views for the harness Proposer (bus contract v2 §6).

The Proposer is a *student* of the campaign's evals: it must never see suite names, case ids,
criterion names or scores. Non-student roles (analyst, failure_analyst, critic) keep the full
records; whatever they produce reaches the Proposer only through
:func:`~ci_lab.taskgraph.firewall.sanitize_correction`, as ``StudentCorrection`` JSON.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_framework import FunctionInvocationContext, FunctionMiddleware

from ci_lab.contracts import FailureRecord
from ci_lab.taskgraph.firewall import LeakScreen, sanitize_correction
from ci_lab.tools.briefs import make_brief_tools

__all__ = [
    "SANITIZED_DOCUMENTS",
    "SANITIZED_TOOLS",
    "SanitizeResultsMiddleware",
    "arm_attempt",
    "failure_corpus",
    "failure_section",
    "student_brief_tools",
    "student_text",
]

SANITIZED_DOCUMENTS = ("analysis", "critique", "history")
SANITIZED_TOOLS = ("background_agents_get_task_results",)
CRITIC_CATEGORY = "critic_rejected"


def arm_attempt(arm: str, n: int = 1) -> str:
    """A ``task@n`` attempt id for an arm name."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(arm).lower()).strip("-_")[:64] or "arm"
    return f"{slug}@{max(1, int(n))}"


def failure_corpus(failures: Iterable[FailureRecord]) -> list[str]:
    """Identifiers of ``failures`` the Proposer must not see: case ids, suites, scored criteria."""
    out: list[str] = []
    for f in failures:
        if f.category != CRITIC_CATEGORY:
            out += [f.case_id, f.suite, *f.rubric_scores]
    return [s for s in out if s]


def _leaves(data: Any, key: str = "") -> list[str]:
    if isinstance(data, str):
        return [f"{key}: {data}" if key else data]
    if isinstance(data, bool):
        return [f"{key}: {'yes' if data else 'no'}"] if key else []
    if isinstance(data, Mapping):
        return [x for k, v in data.items() for x in _leaves(v, str(k))]
    if isinstance(data, (list, tuple)):
        return [x for v in data for x in _leaves(v, key)]
    return []  # numbers (scores) and nulls never reach the student


def _parse(name: str, text: str) -> Any:
    try:
        data = json.loads(text)
    except ValueError:
        data = []
        for line in text.splitlines():
            try:
                data.append(json.loads(line))
            except ValueError:
                data.append(line)
    if name == "critique" and isinstance(data, Mapping):
        return data.get("reasons") or []
    return data


def student_text(name: str, text: str, *, attempt: str, corpus: Sequence[str] = ()) -> str:
    """A non-student document rendered as ``StudentCorrection`` JSON."""
    reasons = _leaves(_parse(name, text))
    return json.dumps(sanitize_correction(reasons, None, attempt=attempt, extra_corpus=corpus).to_json())


def failure_section(failures: Sequence[FailureRecord], *, attempt: str) -> str:
    """Brief section describing ``failures`` without suites, case ids, criteria or scores."""
    if not failures:
        return ""
    reasons = [f.category + (f" (rules: {', '.join(f.rule_ids)})" if f.rule_ids else "")
               + (f": {f.excerpt}" if f.excerpt else "") for f in failures]
    text = sanitize_correction(reasons, None, attempt=attempt, extra_corpus=failure_corpus(failures)).text
    return f"\n## Failures to address ({len(failures)} record(s))\n\n{text}\n"


def student_brief_tools(run_dir: Path | str, failures: Sequence[FailureRecord], *, attempt: str,
                        allowed: Sequence[str] | None = None) -> dict[str, Callable[..., str]]:
    """``make_brief_tools`` for the Proposer: analysis/critique/history come back as
    ``StudentCorrection`` JSON, everything else is redacted against the failure corpus."""
    base = make_brief_tools(run_dir, allowed=allowed)
    corpus = failure_corpus(failures)
    screen = LeakScreen((), corpus)

    def notice(text: str) -> bool:
        return text.startswith(("ERROR", "(document", "(no history"))

    @functools.wraps(base["read_brief"])
    def read_brief(name: str = "brief") -> str:
        text = base["read_brief"](name)
        if notice(text) or name not in SANITIZED_DOCUMENTS:
            return screen.redact(text)
        return student_text(name, text, attempt=attempt, corpus=corpus)

    @functools.wraps(base["read_history"])
    def read_history(limit: int = 20) -> str:
        text = base["read_history"](limit)
        return text if notice(text) else student_text("history", text, attempt=attempt, corpus=corpus)

    return {**base, "read_brief": read_brief, "read_history": read_history}


class SanitizeResultsMiddleware(FunctionMiddleware):
    """Rewrites results of ``tools`` (the failure_analyst's answers) into ``StudentCorrection`` JSON."""

    def __init__(self, failures: Sequence[FailureRecord], *, attempt: str,
                 tools: Sequence[str] = SANITIZED_TOOLS) -> None:
        self.corpus, self.attempt, self.tools = failure_corpus(failures), attempt, tuple(tools)

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        if context.function.name in self.tools:
            raw = context.result
            text = raw if isinstance(raw, str) else "\n".join(
                str(getattr(c, "text", None) or getattr(c, "result", None) or "") for c in raw or ())
            context.result = student_text("result", text, attempt=self.attempt, corpus=self.corpus)
