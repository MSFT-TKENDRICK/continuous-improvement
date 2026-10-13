"""GEPA and SkillOpt-Sleep arms end to end over a real HTTP reflection model.

``tests/ci_lab/strategies/test_text_strategies.py`` drives the strategies with an injected
``DummyLM``. Here nothing is injected: ``GepaStrategy`` / ``SkillOptStrategy`` build their
reflection LM with ``make_lm(Profile.OFFLINE, "optimizer")`` exactly as a campaign arm does.
That LM is a real ``dspy.LM`` -> LiteLLM ``openai/<model>`` -> HTTP to a deterministic
:class:`~ci_lab.testing.LoopbackLLM` on 127.0.0.1. The real ``gepa.optimize`` engine and the real
``skillopt_sleep.dream.dream_consolidate`` gate run unpatched over the ``KeywordDomain`` scorer.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid

import pytest

from ci_lab.contracts import ArmContext, ArmDirective, Profile
from ci_lab.optim.gepa import GepaConfig
from ci_lab.optim.skillopt import SkillOptConfig
from ci_lab.strategies import GepaStrategy, SkillOptStrategy
from ci_lab.testing import LoopbackLLM

PROMPT = "harness/prompts/system.md"
SKILL = "harness/skills/harness-editing/SKILL.md"
ALL_KW = "change tracking escalate polite"


def git(cwd, *args):
    import subprocess

    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def last_user_text(body) -> str:
    content = body["messages"][-1]["content"]
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)


@pytest.fixture
def llm(monkeypatch):
    """Loopback model behind the offline profile's env; a fresh model name per test keeps the
    (enabled, C19) DSPy optimizer cache from answering instead of the server."""
    srv = LoopbackLLM().start()
    monkeypatch.setenv("OPENAI_API_BASE", f"{srv.url}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback")
    monkeypatch.setenv("CI_LAB_OPTIMIZER_MODEL", f"reallib-{uuid.uuid4().hex[:12]}")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AGL_OPENAI_BASE_URL", raising=False)
    yield srv
    srv.stop()


def mkctx(wt, tmp_path, strategy, focus=(), budget=1):
    return ArmContext("exp-rl", ArmDirective("arm-1", strategy, tuple(focus), budget), wt,
                      git(wt, "rev-parse", "HEAD"), [], Profile.OFFLINE, tmp_path / "run")


def test_gepa_arm_runs_real_gepa_engine_over_http_reflection_lm(worktree, domain, tmp_path, llm):
    def respond(body):
        prompt = last_user_text(body)
        current = re.search(r"```\n?(.*?)\n?```", prompt, re.DOTALL)
        assert current and "excerpt: agent forgot" in prompt  # GEPA's template + our typed feedback
        return f"Improved instruction:\n```\n{current.group(1).strip()} Always mention: {ALL_KW}.\n```"

    llm.respond = respond
    s = GepaStrategy(domain=domain, config=GepaConfig(max_metric_calls=24))
    ctx = mkctx(worktree, tmp_path, "gepa")
    (edit,) = asyncio.run(s.propose(ctx))

    assert edit.files == (PROMPT,) and edit.commit == git(worktree, "rev-parse", "HEAD") != ctx.base_commit
    text = (worktree / PROMPT).read_text(encoding="utf-8")
    assert text.startswith("Improve this repository's self-hosted harness.") and ALL_KW in text
    (opt,) = s.last
    assert opt.seed_score == 0.0 and opt.best_score == 1.0 and opt.diagnostics["num_candidates"] >= 2
    cost = opt.cost
    chats = llm.of_kind("chat")
    assert cost.reflection_calls == len(chats) >= 1
    assert {b["model"] for b in chats} == {s._lm(ctx).model.split("/", 1)[1]}
    assert cost.reflection_tokens == 8 * len(chats)  # LiteLLM usage from the server, not an estimate
    assert 0 < cost.metric_calls <= 24 and cost.scorer_tokens > 0 and cost.candidates >= 2
    rep = json.loads((tmp_path / "run" / "optimizer" / "arm-1-gepa.json").read_text(encoding="utf-8"))
    assert rep["cost"]["reflection_tokens"] == cost.reflection_tokens and rep["edits"][0]["commit"] == edit.commit
    assert set(domain.splits_called) == {"evolve"}


def skill_reply(content):
    def respond(body):
        prompt = last_user_text(body)
        assert "agent forgot" in prompt and "evolve case" in prompt  # typed failures reach the reflector
        return json.dumps([{"op": "add", "content": content, "rationale": "evolve failures"}])
    return respond


def test_skillopt_arm_accepts_improving_rule_through_real_dream_gate(worktree, domain, tmp_path, llm):
    llm.respond = skill_reply(f"Always {ALL_KW}.")
    s = SkillOptStrategy(domain=domain, config=SkillOptConfig(max_metric_calls=60))
    ctx = mkctx(worktree, tmp_path, "skillopt")
    (edit,) = asyncio.run(s.propose(ctx))

    assert edit.files == (SKILL,) and f"Always {ALL_KW}." in (worktree / SKILL).read_text(encoding="utf-8")
    (opt,) = s.last
    d = opt.diagnostics
    assert d["accepted"] is True and d["applied_edits"] == 1 and d["reflect_raw_chars"] > 0
    assert opt.best_score > opt.seed_score
    assert opt.cost.reflection_calls == len(llm.of_kind("chat")) >= 1 and opt.cost.reflection_tokens > 0


def test_skillopt_arm_rejects_useless_rule_through_real_dream_gate(worktree, domain, tmp_path, llm):
    llm.respond = skill_reply("Be nice.")
    s = SkillOptStrategy(domain=domain, config=SkillOptConfig(max_metric_calls=60))
    ctx = mkctx(worktree, tmp_path, "skillopt")

    assert asyncio.run(s.propose(ctx)) == []
    assert git(worktree, "rev-parse", "HEAD") == ctx.base_commit
    (opt,) = s.last
    assert opt.diagnostics["accepted"] is False and opt.changed == {}
    assert opt.note.startswith("gate ") and llm.of_kind("chat")
