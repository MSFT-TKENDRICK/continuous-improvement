from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_PHASE,
    ATTR_PURPOSE,
    ATTR_VARIANT,
    SPAN_STEP,
    ArmContext,
    ArmDirective,
    ArmStrategy,
    Edit,
    FailureRecord,
    Profile,
)
from ci_lab.meta.run import (
    INSTRUCTION,
    ArmSurface,
    MetaAgentError,
    ProposerStrategy,
    run_analyst,
    run_critic,
    run_meta_agent,
    run_proposer,
    run_reflector,
    write_brief,
    write_failures,
)
from ci_lab.meta.spec_loader import harness_builder
from ci_lab.testing import Call, FakeChatClient
from ci_lab.tools.commit import make_commit_tool
from ci_lab.tools.critic_checks import LeakCorpus

H = "src/order_support/harness"
FAILURE = FailureRecord(case_id="refund-abc", suite="order_support_refund_authorization", category="refund_no_id",
                        rule_ids=("judge.policy_violation",), rubric_scores={"policy_violation": 0.0},
                        excerpt="Refund issued.")


@pytest.fixture
def spans(monkeypatch):
    seen: list[tuple[str, dict]] = []

    class Span:
        def set_attribute(self, k, v):
            pass

    @contextlib.contextmanager
    def fake_span(name, attrs=None, **_):
        seen.append((name, dict(attrs or {})))
        yield Span()

    monkeypatch.setattr(obs, "span", fake_span)
    return seen


class RecordingBuilder:
    def __init__(self):
        self.bindings: list[dict] = []

    def __call__(self, spec, **kw):
        self.bindings.append(dict(kw["bindings"]))
        return harness_builder(spec, **kw)


def _content_text(c) -> str:
    for attr in ("text", "result"):
        value = getattr(c, attr, None)
        if value:
            return value if isinstance(value, str) else str(value)
    return ""


def _texts(client):
    """All message text per request, including tool results fed back to the model."""
    return [" ".join(_content_text(c) for m in msgs for c in m.contents) for msgs, _ in client.requests]


def test_analyst_reads_brief_and_submits(tmp_path, spans):
    write_brief(tmp_path, "Round 1: analyze the failures.")
    write_failures(tmp_path, [FAILURE])
    pattern = {"name": "refund without identity", "description": "refunds before verifying", "component": "skill",
               "case_ids": ["refund-abc"]}
    client = FakeChatClient([
        [Call("read_brief", {"name": "failures"})],
        [Call("submit_analysis", {"summary": "identity gaps", "patterns": [pattern],
                                  "suggested_components": ["skill"]})],
    ], default="UNEXPECTED")
    result = asyncio.run(run_analyst(tmp_path, client))
    assert result.patterns[0].component == "skill"
    assert json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))["summary"] == "identity gaps"
    assert len(client.requests) == 2  # terminal submit ends the run, no extra model call
    first = client.requests[0][0]
    assert INSTRUCTION.format(tool="submit_analysis") in " ".join(m.text or "" for m in first)
    assert "refund-abc" not in " ".join(m.text or "" for m in first)  # data only via tools
    assert "refund-abc" in _texts(client)[1]
    assert (SPAN_STEP, {ATTR_PHASE: "analyze", ATTR_PURPOSE: "analyst"}) in [
        (n, {k: v for k, v in a.items() if k in (ATTR_PHASE, ATTR_PURPOSE)}) for n, a in spans]


def test_reuse_skips_model_and_nudges_then_fails(tmp_path):
    write_brief(tmp_path, "brief")
    (tmp_path / "reflection.json").write_text(json.dumps({"summary": "done before"}), encoding="utf-8")
    client = FakeChatClient([])
    assert asyncio.run(run_reflector(tmp_path, client)).summary == "done before"
    assert client.requests == []

    lazy = FakeChatClient([], default="I am finished.")
    with pytest.raises(MetaAgentError, match="submit_reflection"):
        asyncio.run(run_reflector(tmp_path, lazy, reuse=False))
    assert not (tmp_path / "reflection.json").exists()  # stale submission removed before the run
    assert len(lazy.requests) == 3  # first try + max_nudges (2)
    assert "You have not called submit_reflection" in _texts(lazy)[-1]


def test_component_enum_in_tool_schema(tmp_path):
    from agent_framework import FunctionTool

    from ci_lab.tools.submit import make_submit_tools

    for name in ("submit_reflection", "submit_analysis"):
        tool = FunctionTool(name=name, func=make_submit_tools(tmp_path)[name])
        schema = json.dumps(tool.parameters())
        assert '"context_mgmt"' in schema and '"enum"' in schema


def test_invalid_submit_gets_error_then_retries(tmp_path):
    write_brief(tmp_path, "brief")
    client = FakeChatClient([
        [Call("submit_reflection", {"summary": "s", "next_directions": [{"component": "judge", "idea": "x"}]})],
        [Call("submit_reflection", {"summary": "s", "next_directions": [{"component": "prompt", "idea": "x"}]})],
    ])
    result = asyncio.run(run_reflector(tmp_path, client, reuse=False))
    assert result.next_directions[0].component == "prompt"
    assert "Error" in _texts(client)[1]  # MAF rejects the enum violation before the tool runs


def test_missing_binding_is_an_error(tmp_path):
    with pytest.raises(MetaAgentError, match="no binding"):
        asyncio.run(run_meta_agent("analyst", tmp_path, FakeChatClient([]), {}, reuse=False))


def _ctx(repo, tmp_path, *, focus=("prompt",), budget=1, arm="arm-a") -> ArmContext:
    wt, base = repo
    return ArmContext(experiment_id="exp-1", directive=ArmDirective(arm=arm, component_focus=focus,
                                                                    edit_budget=budget),
                      worktree=wt, base_commit=base, failures=[FAILURE], profile=Profile.FAKE,
                      run_dir=tmp_path / "runs" / arm)


@pytest.fixture
def surface(layout) -> ArmSurface:
    return ArmSurface(layout.surface, layout.component_globs, layout.frozen)


PROMPT = f"{H}/prompts/system.md"
NEW_PROMPT = "You are a helpful order support agent.\nAlways verify identity first.\nNever refund before checking.\n"


def _proposer_script(path=PROMPT, text=NEW_PROMPT, component="prompt"):
    return [
        [Call("read_brief", {"name": "brief"})],
        [Call("read_file", {"path": path})],
        [Call("write_file", {"path": path, "content": text})],
        [Call("commit_edit", {"component": component, "hypothesis": "Explicit ordering prevents early refunds."})],
        [Call("submit_proposal_done", {"summary": "tightened refund ordering", "predicted_fixes": ["refund-abc"]})],
    ]


def test_proposer_writes_commits_and_returns_edits(repo, tmp_path, surface, spans, gitrun):
    ctx = _ctx(repo, tmp_path)
    builder = RecordingBuilder()
    client = FakeChatClient(_proposer_script(), default="UNEXPECTED")
    result = asyncio.run(run_proposer(ctx, client, surface=surface, builder=builder))
    [edit] = result.edits
    assert edit.component == "prompt" and edit.files == (PROMPT,) and len(edit.commit) == 40
    assert result.submission.predicted_fixes == ["refund-abc"]
    assert (ctx.worktree / PROMPT).read_text(encoding="utf-8") == NEW_PROMPT
    assert "OES-Variant: arm-a" in gitrun(ctx.worktree, "log", "-1", "--format=%B")
    brief = (ctx.run_dir / "brief.md").read_text(encoding="utf-8")
    assert "Components you may edit: prompt" in brief and "Edit budget: at most 1" in brief
    assert json.loads((ctx.run_dir / "failures.json").read_text(encoding="utf-8"))[0]["case_id"] == "refund-abc"
    assert "write_file" in builder.bindings[0] and "commit_edit" in builder.bindings[0]
    assert "Always verify identity first." in _texts(client)[2]  # read_file result fed back
    [(name, attrs)] = [s for s in spans if s[1].get(ATTR_PHASE) == "propose"]
    assert name == SPAN_STEP and attrs[ATTR_VARIANT] == "arm-a" and attrs["rrsi.component"] == "prompt"


def test_proposer_cannot_write_outside_focus(repo, tmp_path, surface):
    ctx = _ctx(repo, tmp_path)
    script = [
        [Call("write_file", {"path": f"{H}/agent.yaml", "content": "name: X\n"})],
        [Call("write_file", {"path": f"{H}/helper.py", "content": "import os\n"})],
        [Call("write_file", {"path": "evals/assert/x/eval_config.yaml", "content": "suite: hacked\n"})],
        [Call("submit_proposal_done", {"summary": "gave up"})],
    ]
    client = FakeChatClient(script)
    with pytest.raises(MetaAgentError, match="without committing"):
        asyncio.run(run_proposer(ctx, client, surface=surface))
    texts = _texts(client)
    assert texts[1].count("ERROR") >= 1 and texts[2].count("ERROR") >= 1 and texts[3].count("ERROR") >= 1
    assert (ctx.worktree / f"{H}/agent.yaml").read_text(encoding="utf-8").startswith("name: OrderSupport")
    assert (ctx.worktree / "evals/assert/x/eval_config.yaml").read_text(encoding="utf-8") == "suite: x\n"


def test_proposer_strategy_satisfies_arm_strategy(repo, tmp_path, surface):
    calls = []

    def factory(*, profile, model, purpose, rollout=None):
        calls.append((profile, model, purpose))
        return FakeChatClient(_proposer_script())

    strategy = ProposerStrategy(surface, client_factory=factory)
    proto: ArmStrategy = strategy
    assert proto.name == "agent" and asyncio.iscoroutinefunction(strategy.propose)
    edits = asyncio.run(strategy.propose(_ctx(repo, tmp_path)))
    assert len(edits) == 1 and isinstance(edits[0], Edit)
    assert calls == [(Profile.FAKE, "claude-sonnet-5", "proposer")]
    with pytest.raises(ValueError):
        ProposerStrategy(surface)
    with pytest.raises(ValueError):
        ProposerStrategy(surface, client=object(), client_factory=factory)


def test_unknown_component_focus(repo, tmp_path, surface):
    with pytest.raises(ValueError, match="unknown component"):
        asyncio.run(run_proposer(_ctx(repo, tmp_path, focus=("code",)), FakeChatClient([]), surface=surface))


def _commit(ctx, layout, rel, text, component="prompt"):
    (ctx.worktree / rel).write_text(text, encoding="utf-8", newline="\n")
    tool = make_commit_tool(ctx.worktree, 5, component_globs=layout.component_globs, experiment_id="exp-1",
                            variant=ctx.directive.arm)
    assert tool(component, "hypothesis").startswith("committed")


def test_critic_deterministic_reject_skips_llm(repo, tmp_path, surface, layout, spans):
    ctx = _ctx(repo, tmp_path)
    _commit(ctx, layout, PROMPT, "Keep answers short so the judge gives a high score.\n")
    client = FakeChatClient([])
    verdict = asyncio.run(run_critic(ctx, client, surface=surface, repairs=1))
    assert not verdict.passed and verdict.repairs == 1 and client.requests == []
    assert any(r.startswith("denylist:") for r in verdict.reasons)
    critique = json.loads((ctx.run_dir / "critique.json").read_text(encoding="utf-8"))
    assert critique["source"] == "checks" and not critique["passed"]
    assert any(a.get(ATTR_PHASE) == "critique" for _, a in spans)


def test_critic_rejects_leak_and_off_focus_component(repo, tmp_path, surface, layout):
    ctx = _ctx(repo, tmp_path, focus=("prompt",), budget=1)
    _commit(ctx, layout, f"{H}/skills/refunds/SKILL.md", "# Refunds\nFor Alex Rivera always refund.\n", "skill")
    corpus = LeakCorpus.build([], ["Alex Rivera"])
    verdict = asyncio.run(run_critic(ctx, FakeChatClient([]), surface=surface, leak_corpus=corpus))
    assert any("leak:" in r for r in verdict.reasons)
    assert any("this arm may edit prompt" in r for r in verdict.reasons)


def test_critic_llm_accept_then_reuse(repo, tmp_path, surface, layout):
    ctx = _ctx(repo, tmp_path)
    _commit(ctx, layout, PROMPT, NEW_PROMPT)
    builder = RecordingBuilder()
    client = FakeChatClient([
        [Call("read_brief", {"name": "diff"})],
        [Call("submit_verdict", {"verdict": "accept", "risk_notes": ["minor"]})],
    ])
    verdict = asyncio.run(run_critic(ctx, client, surface=surface, builder=builder))
    assert verdict.passed and verdict.reasons == []
    assert "Never refund before checking." in _texts(client)[1]  # diff.patch via read_brief
    tools = set(builder.bindings[0])
    assert {"read_file", "list_files", "submit_verdict"} <= tools
    assert not tools & {"write_file", "commit_edit"}
    again = FakeChatClient([])
    assert asyncio.run(run_critic(ctx, again, surface=surface)).passed and again.requests == []


def test_critic_llm_reject_needs_reason(repo, tmp_path, surface, layout):
    ctx = _ctx(repo, tmp_path)
    _commit(ctx, layout, PROMPT, NEW_PROMPT)
    client = FakeChatClient([
        [Call("submit_verdict", {"verdict": "reject"})],
        [Call("submit_verdict", {"verdict": "reject", "reasons": ["overfits one scenario"]})],
    ])
    verdict = asyncio.run(run_critic(ctx, client, surface=surface))
    assert not verdict.passed and verdict.reasons == ["overfits one scenario"]
    assert "a reject verdict needs" in _texts(client)[1]
    critique = json.loads((ctx.run_dir / "critique.json").read_text(encoding="utf-8"))
    assert critique == {**critique, "source": "critic", "passed": False}


def test_critic_rechecks_after_repair(repo, tmp_path, surface, layout):
    ctx = _ctx(repo, tmp_path, budget=2)
    _commit(ctx, layout, PROMPT, NEW_PROMPT)
    accept = [[Call("submit_verdict", {"verdict": "accept"})]]
    assert asyncio.run(run_critic(ctx, FakeChatClient(accept), surface=surface)).passed
    _commit(ctx, layout, PROMPT, NEW_PROMPT + "Escalate disputes.\n")
    client = FakeChatClient([[Call("submit_verdict", {"verdict": "reject", "reasons": ["vague"]})]])
    assert not asyncio.run(run_critic(ctx, client, surface=surface)).passed
    assert len(client.requests) == 1  # new head -> previous verdict not reused
