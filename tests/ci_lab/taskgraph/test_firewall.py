"""Student context firewall (bus contract v2 §6): LeakScreen, sanitize_correction, MAF middleware."""

from __future__ import annotations

import asyncio
import random
import re

import pytest
from agent_framework import Agent, MiddlewareFailure, tool

from ci_lab.bus.types import StudentCorrection
from ci_lab.taskgraph.firewall import (
    GENERIC_CORRECTION,
    REDACTED,
    ContextLeak,
    LeakScreen,
    StudentFirewallMiddleware,
    sanitize_correction,
)
from ci_lab.taskgraph.model import Criterion, Rubric
from ci_lab.testing import Call, FakeChatClient

CANARY = "0123456789abcdef"  # conftest.make_rubric canary
QUESTION = "Does the report cite at least two distinct resource identifiers taken from the log?"
DESCRIPTION = "Report cites resource identifiers from the log"
CLEAN = "Summarise the resource log into out/summary.md as Markdown."


@pytest.fixture
def screen(rubric_factory) -> LeakScreen:
    return LeakScreen([rubric_factory()], extra_corpus=["change-abc"])


@pytest.mark.parametrize(("text", "category"), [
    (f"marker {CANARY} here", "canary"),
    (f"embedded x{CANARY.upper()}y", "canary"),
    ("see criterion c-cites for details", "criterion_id"),
    ("rule judge.c-cites fired", "criterion_id"),
    ("the harness-agent suite failed", "suite"),
    (f"Judge asks: {QUESTION.lower()}", "rubric_text"),
    (f"Note: {DESCRIPTION}.", "rubric_text"),
    ("so does the report cite at least two distinct resource things", "ngram"),
    ("case change-abc regressed", "corpus"),
])
def test_hits_each_category(screen: LeakScreen, text: str, category: str) -> None:
    assert category in screen.hits(text)
    assert category in screen.hits(f"prefix\n{text}\nsuffix")


def test_clean_text_has_no_hits(screen: LeakScreen) -> None:
    assert screen.hits(CLEAN) == []
    assert screen.hits("c-citesx and harness-agentive and xc-cites are different words") == []
    assert screen.redact(CLEAN) == CLEAN


def test_hits_never_quote_material(screen: LeakScreen) -> None:
    hits = screen.hits(f"{QUESTION} {CANARY} c-suite harness-agent")
    assert set(hits) == {"canary", "criterion_id", "suite", "rubric_text", "ngram"}
    assert not any(CANARY in h or "harness-agent" in h for h in hits)


def test_redact_removes_every_span(screen: LeakScreen) -> None:
    text = f"Fix c-format. {QUESTION} Suite harness-agent; canary {CANARY}; case change-abc. Keep going."
    out = screen.redact(text)
    assert screen.hits(out) == []
    assert out.startswith("Fix [redacted].") and out.endswith("Keep going.")
    assert out.count(REDACTED) == 5


def test_sanitize_strips_all_forbidden_items(rubric_factory) -> None:
    rubric = rubric_factory()
    reasons = ["criterion c-suite: harness-agent scored 0.42 < threshold 0.8 (42%)",
               f"c-cites failed: {QUESTION}",
               f"canary {CANARY} leaked; also the heading is missing at the top of the file",
               "the output file out/summary.md is missing a level one heading"]
    corr = sanitize_correction(reasons, rubric, attempt="summary@2")
    assert isinstance(corr, StudentCorrection) and corr.attempt == "summary@2"
    for bad in ("c-suite", "c-cites", "harness-agent", "0.42", "0.8", "42", "%", QUESTION, CANARY):
        assert bad not in corr.text
    assert "heading is missing" in corr.text and LeakScreen([rubric]).hits(corr.text) == []


def test_sanitize_fallback_and_limits(rubric_factory) -> None:
    rubric = rubric_factory()
    for reasons in ([], [QUESTION], ["c-suite 0.4", f"{CANARY}"], "   "):
        assert sanitize_correction(reasons, rubric, attempt="summary@1").text == GENERIC_CORRECTION
    long = [f"paragraph {chr(97 + i % 26)} needs a clearer topic sentence and supporting evidence" * 3
            for i in range(40)]
    corr = sanitize_correction(long, None, attempt="summary@3")
    assert len(corr.text) <= 800 and corr.text != GENERIC_CORRECTION
    assert sanitize_correction("x", None, attempt="summary@1", extra_corpus=["secret-suite"]).text == GENERIC_CORRECTION
    with pytest.raises(Exception, match="attempt"):
        sanitize_correction(["fine reason here"], None, attempt="bad attempt")


_WORDS = ("ledger", "change", "escalate", "repository", "config", "resource", "workflow", "review", "report",
          "heading", "section", "table")


@pytest.mark.parametrize("seed", range(25))
def test_sanitize_property_random_rubrics(seed: int) -> None:
    rng = random.Random(seed)

    def phrase(n: int) -> str:
        return " ".join(rng.choice(_WORDS) for _ in range(n))

    criteria = tuple(Criterion(f"k{seed}-{i}", phrase(rng.randint(3, 10)), m,
                               {"question": phrase(rng.randint(8, 14)) + "?", "type": "noul"} if m == "s1"
                               else {"suite": f"suite_{phrase(1)}_{i}", "split": "dev", "min_score": 0.5},
                               rng.random() or 0.5) for i, m in enumerate(("assert", "s1", "s1", "assert")))
    rubric = Rubric(f"r{seed}", 1, "summary", criteria, 0.5, f"{rng.getrandbits(64):016x}")
    secrets = [c.id for c in criteria] + [rubric.canary] + [c.check["suite"] for c in criteria if "suite" in c.check]
    reasons = []
    for _ in range(rng.randint(1, 6)):
        c = rng.choice(criteria)
        bits = [c.id, c.description, str(c.check.get("question", c.check.get("suite"))), rubric.canary,
                f"{rng.random():.3f}", f"{rng.randint(1, 99)}%", phrase(rng.randint(1, 6))]
        rng.shuffle(bits)
        reasons.append(" ".join(bits[:rng.randint(2, len(bits))]))
    text = sanitize_correction(reasons, rubric, attempt=f"t{seed}@1").text
    assert len(text) <= 800
    assert LeakScreen([rubric]).hits(text) == []
    assert not re.search(r"\d", text)
    assert not any(s.lower() in text.lower() for s in secrets)


def _agent(script: list, tool_result: str, screen: LeakScreen) -> tuple[Agent, StudentFirewallMiddleware]:
    def read_input(path: str) -> str:
        """Read an input file."""
        return tool_result

    mw = StudentFirewallMiddleware(screen)
    return Agent(client=FakeChatClient(script), tools=[tool(read_input)], middleware=mw), mw


def test_middleware_passes_clean_run(screen: LeakScreen) -> None:
    agent, mw = _agent([[Call("read_input", {"path": "data/log.txt"})], "done"], "resource 1 updated", screen)
    assert asyncio.run(agent.run(CLEAN)).text == "done"
    assert mw.leaks == []


def test_middleware_blocks_message_leak(screen: LeakScreen) -> None:
    agent, mw = _agent(["done"], "ok", screen)
    with pytest.raises(ContextLeak) as err:
        asyncio.run(agent.run(f"{CLEAN}\nHint: {QUESTION}"))
    assert err.value.reason == "context_leak" and "rubric_text" in err.value.hits
    assert QUESTION not in str(err.value) and mw.leaks == [err.value]
    assert isinstance(err.value, MiddlewareFailure)


def test_middleware_blocks_tool_result_leak(screen: LeakScreen) -> None:
    client_script = [[Call("read_input", {"path": "notes.txt"})], "done"]
    agent, mw = _agent(client_script, f"grader notes: suite harness-agent, canary {CANARY}", screen)
    with pytest.raises(ContextLeak) as err:
        asyncio.run(agent.run(CLEAN))
    assert err.value.where == "tool:read_input"
    assert {"canary", "suite"} <= set(err.value.hits) and len(mw.leaks) == 1
    client = agent.client
    assert not any(CANARY in str(m) for req, _ in client.requests for m in req)  # never reached the model
