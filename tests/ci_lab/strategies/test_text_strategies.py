import asyncio
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_STRATEGY,
    SPAN_OPTIMIZER,
    ArmContext,
    ArmDirective,
    Edit,
    Profile,
)
from ci_lab.optim.gepa import GepaConfig
from ci_lab.optim.lm import make_lm
from ci_lab.optim.skillopt import SkillOptConfig
from ci_lab.optim.targets import TargetError
from ci_lab.strategies import GepaStrategy, SkillOptStrategy, get_strategy
from ci_lab.strategies.base import COMMIT_TRAILER

PROMPT = "harness/prompts/system.md"
SKILL = "harness/skills/harness-editing/SKILL.md"
MEMORY = "harness/skills/harness-editing/memory.md"
ALL_KW = "change tracking escalate polite"


def git(cwd, *args):
    import subprocess

    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def spans(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


def mkctx(wt, tmp_path, strategy, focus=(), budget=1, budget_tokens=None):
    base = git(wt, "rev-parse", "HEAD")
    return ArmContext("exp-9", ArmDirective("arm-2", strategy, tuple(focus), budget), wt, base, [],
                      Profile.FAKE, tmp_path / "run", budget_tokens=budget_tokens)


def gepa_lm(text=ALL_KW, n=20):
    return make_lm("fake", "optimizer", fake_answers=[{"new_instruction": f"```\n{text}\n```"}] * n)


def skill_lm(content=f"Always {ALL_KW}.", n=6):
    arr = json.dumps([{"op": "add", "content": content, "rationale": "evolve failures"}])
    return make_lm("fake", "optimizer", fake_answers=[{"text": arr}] * n)


def test_gepa_strategy_commits_edit_and_reports_cost(worktree, domain, tmp_path, spans):
    s = get_strategy("gepa", domain=domain, lm=gepa_lm(), config=GepaConfig(max_metric_calls=24))
    c = mkctx(worktree, tmp_path, "gepa")
    (e,) = asyncio.run(s.propose(c))
    assert isinstance(e, Edit) and e.component == "prompt" and e.files == (PROMPT,)
    assert e.commit == git(worktree, "rev-parse", "HEAD") != c.base_commit
    assert git(worktree, "show", "--name-only", "--format=", e.commit).splitlines() == [PROMPT]
    assert COMMIT_TRAILER in git(worktree, "log", "-1", "--format=%B")
    assert git(worktree, "log", "-1", "--format=%(trailers:key=RRSI-Component,valueonly)") == "prompt"
    assert git(worktree, "log", "-1", "--format=%(trailers:key=RRSI-Hypothesis,valueonly)")
    assert (worktree / PROMPT).read_text() == ALL_KW
    assert git(worktree, "status", "--porcelain") == ""
    assert set(domain.splits_called) == {"evolve"}
    rep = json.loads((tmp_path / "run" / "optimizer" / "arm-2-gepa.json").read_text())
    assert rep["strategy"] == "gepa" and rep["acceptance"] == "diagnostic-only"
    assert 0 < rep["cost"]["metric_calls"] <= 24 and rep["cost"]["reflection_calls"] >= 1
    assert rep["edits"] == [{"component": "prompt", "files": [PROMPT], "commit": e.commit}]
    (sp,) = spans.get_finished_spans()
    assert sp.name == SPAN_OPTIMIZER and sp.attributes[ATTR_STRATEGY] == "gepa"
    assert sp.attributes["ci.optimizer.metric_calls"] == rep["cost"]["metric_calls"]
    assert sp.attributes["ci.edits"] == 1


def test_gepa_budget_tokens_too_small_yields_no_edits(worktree, domain, tmp_path):
    s = GepaStrategy(domain=domain, lm=gepa_lm())
    c = mkctx(worktree, tmp_path, "gepa", budget_tokens=2 * 4000)
    assert asyncio.run(s.propose(c)) == []
    assert domain.splits_called == [] and git(worktree, "rev-parse", "HEAD") == c.base_commit


def test_gepa_edit_budget_caps_targets(worktree, domain, tmp_path):
    focus = (PROMPT, SKILL)
    s = GepaStrategy(domain=domain, lm=gepa_lm(n=40), config=GepaConfig(max_metric_calls=40))
    c = mkctx(worktree, tmp_path, "gepa", focus=focus, budget=1)
    edits = asyncio.run(s.propose(c))
    assert len(edits) == 1 and edits[0].files == (PROMPT,)
    assert sorted(s.last[0].seed) == [PROMPT]  # only budgeted components entered the optimizer
    assert (worktree / SKILL).read_text() == "# Harness editing\nMake bounded harness changes.\n"


def test_gepa_rejects_off_surface_focus(worktree, domain, tmp_path):
    (worktree / "README.md").write_text("x")
    guards = worktree / "harness/guards/rules.yaml"
    guards.parent.mkdir(parents=True)
    guards.write_text("rules: []\n")
    s = GepaStrategy(domain=domain, lm=gepa_lm())
    for focus in ("README.md", "harness/guards/rules.yaml"):
        with pytest.raises(TargetError):
            asyncio.run(s.propose(mkctx(worktree, tmp_path, "gepa", focus=(focus,))))


def test_gepa_zero_budget_and_non_text_focus(worktree, domain, tmp_path):
    s = GepaStrategy(domain=domain, lm=gepa_lm())
    assert asyncio.run(s.propose(mkctx(worktree, tmp_path, "gepa", budget=0))) == []
    assert asyncio.run(s.propose(mkctx(worktree, tmp_path, "gepa", focus=("client_tool",)))) == []
    assert domain.splits_called == []


def test_gepa_uses_make_lm_for_profile_when_no_lm_injected(worktree, domain, tmp_path, monkeypatch):
    seen = []
    import ci_lab.optim.lm as lm_mod

    def fake_make_lm(profile, purpose="optimizer", **kw):
        seen.append((profile, purpose))
        return gepa_lm()

    monkeypatch.setattr(lm_mod, "make_lm", fake_make_lm)
    s = GepaStrategy(domain=domain, config=GepaConfig(max_metric_calls=24))
    assert len(asyncio.run(s.propose(mkctx(worktree, tmp_path, "gepa")))) == 1
    assert seen == [(Profile.FAKE, "optimizer")]


def test_skillopt_strategy_ignores_agl_owned_memory(worktree, domain, tmp_path, spans):
    s = SkillOptStrategy(domain=domain, lm=skill_lm(), config=SkillOptConfig(max_metric_calls=60))
    c = mkctx(worktree, tmp_path, "skillopt", focus=("skill", "memory"), budget=2)
    edits = asyncio.run(s.propose(c))
    assert [e.files for e in edits] == [(SKILL,)]
    assert (worktree / MEMORY).read_text() == "- remember harness lessons\n"
    assert f"Always {ALL_KW}." in (worktree / SKILL).read_text()
    assert set(domain.splits_called) == {"evolve"}
    art = tmp_path / "run" / "optimizer" / "arm-2-skillopt" / "harness-editing"
    assert (art / "best_skill.md").read_text() == (worktree / SKILL).read_text()
    assert not (art / "best_memory.md").exists()
    rep = json.loads((tmp_path / "run" / "optimizer" / "arm-2-skillopt.json").read_text())
    assert rep["runs"][0]["diagnostics"]["accepted"] is True
    assert rep["runs"][0]["targets"] == [SKILL]
    (sp,) = spans.get_finished_spans()
    assert sp.attributes[ATTR_STRATEGY] == "skillopt" and sp.attributes["ci.edits"] == len(edits)


def test_skillopt_budget_one_skips_memory(worktree, domain, tmp_path):
    s = SkillOptStrategy(domain=domain, lm=skill_lm(), config=SkillOptConfig(max_metric_calls=60))
    c = mkctx(worktree, tmp_path, "skillopt", focus=("skill", "memory"), budget=1)
    edits = asyncio.run(s.propose(c))
    assert [e.files for e in edits] == [(SKILL,)]
    assert s.last[0].seed.keys() == {SKILL}
    assert (worktree / MEMORY).read_text() == "- remember harness lessons\n"


def test_skillopt_gate_rejection_gives_no_edit(worktree, domain, tmp_path):
    s = SkillOptStrategy(domain=domain, lm=skill_lm("Be nice."), config=SkillOptConfig(max_metric_calls=60))
    c = mkctx(worktree, tmp_path, "skillopt")
    assert asyncio.run(s.propose(c)) == []
    assert git(worktree, "rev-parse", "HEAD") == c.base_commit
    assert not (tmp_path / "run" / "optimizer" / "arm-2-skillopt").exists()
