from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from agent_framework import Agent

import ci_lab.maf.workflows as wf_mod
from ci_lab.maf.workflows import (
    CheckpointNotWrittenError,
    WorkflowSpecError,
    assert_expression_free,
    build_workflow,
    declarative_allowlist,
    latest_checkpoint,
    run_or_resume,
    secure_dir,
)
from ci_lab.testing import FakeChatClient

WORKFLOW = """\
kind: Workflow
trigger:
  kind: OnConversationStart
  id: demo_wf
  actions:
    - kind: InvokeFunctionTool
      id: prep
      functionName: prep
      arguments: {x: "1"}
    - kind: InvokeAzureAgent
      id: think
      agent: {name: Thinker}
      input: {messages: "do the thing"}
    - kind: InvokeFunctionTool
      id: finish
      functionName: finish
      arguments: {}
"""


class Harness:
    """Builds identical workflows over one checkpoint dir and records what ran."""

    def __init__(self, tmp_path: Path) -> None:
        self.yaml = tmp_path / "wf.yaml"
        self.yaml.write_text(WORKFLOW, encoding="utf-8")
        self.checkpoint_dir = tmp_path / "ckpt"
        self.calls: list[str] = []
        self.clients: list[FakeChatClient] = []

    def prep(self, x: str) -> str:
        self.calls.append("prep")
        return f"prepped{x}"

    def finish(self) -> str:
        self.calls.append("finish")
        return "done"

    def kwargs(self) -> dict[str, Any]:
        client = FakeChatClient(["agent says hi"])
        self.clients.append(client)
        agent = Agent(client=client, name="Thinker", instructions="Think.")
        return {"agents": {"Thinker": agent}, "tools": {"prep": self.prep, "finish": self.finish},
                "checkpoint_dir": self.checkpoint_dir}

    def agent_calls(self) -> int:
        return sum(len(c.requests) for c in self.clients)

    def reset(self) -> None:
        self.calls.clear()
        self.clients.clear()


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_workflow_runs_and_checkpoints(h: Harness) -> None:
    async def go() -> None:
        wf, storage = build_workflow(h.yaml, **h.kwargs())
        assert wf.name == "demo_wf"
        result = await wf.run("start")
        assert list(result.get_outputs()) == ["prepped1", "agent says hi", "done"]
        assert h.calls == ["prep", "finish"] and h.agent_calls() == 1
        checkpoints = await storage.list_checkpoints(workflow_name=wf.name)
        assert len(checkpoints) > 0
        assert sorted(c.iteration_count for c in checkpoints) == list(range(len(checkpoints)))

    asyncio.run(go())


def test_resume_from_mid_checkpoint_runs_only_remaining_steps(h: Harness) -> None:
    async def go() -> None:
        wf, storage = build_workflow(h.yaml, **h.kwargs())
        await wf.run("start")
        checkpoints = {c.iteration_count: c for c in await storage.list_checkpoints(workflow_name=wf.name)}

        # After superstep 2, prep has run; the agent and finish have not.
        h.reset()
        wf2, storage2 = build_workflow(h.yaml, **h.kwargs())
        result = await wf2.run(checkpoint_id=checkpoints[2].checkpoint_id, checkpoint_storage=storage2)
        assert list(result.get_outputs()) == ["agent says hi", "done"]
        assert h.calls == ["finish"] and h.agent_calls() == 1

        # After superstep 3, only finish remains.
        h.reset()
        wf3, storage3 = build_workflow(h.yaml, **h.kwargs())
        result = await wf3.run(checkpoint_id=checkpoints[3].checkpoint_id, checkpoint_storage=storage3)
        assert list(result.get_outputs()) == ["done"]
        assert h.calls == ["finish"] and h.agent_calls() == 0

    asyncio.run(go())


def test_run_or_resume_fresh_then_resume(h: Harness) -> None:
    async def go() -> None:
        assert await run_or_resume(h.yaml, "start", **h.kwargs()) == ["prepped1", "agent says hi", "done"]

        # Simulate a crash after superstep 2: drop the later checkpoints.
        storage = build_workflow(h.yaml, **h.kwargs())[1]
        for cp in await storage.list_checkpoints(workflow_name="demo_wf"):
            if cp.iteration_count > 2:
                await storage.delete(cp.checkpoint_id)
        assert (await latest_checkpoint(storage, "demo_wf")).iteration_count == 2

        h.reset()
        assert await run_or_resume(h.yaml, "ignored on resume", **h.kwargs()) == ["agent says hi", "done"]
        assert h.calls == ["finish"] and h.agent_calls() == 1

        # A completed run resumes from its final checkpoint and does nothing more.
        h.reset()
        assert await run_or_resume(h.yaml, "again", **h.kwargs()) == []
        assert h.calls == [] and h.agent_calls() == 0

    asyncio.run(go())


def test_run_or_resume_detects_dropped_checkpoints(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # Without the declarative allowlist MAF logs and skips the intermediate checkpoints.
    monkeypatch.setattr(wf_mod, "declarative_allowlist", lambda extra=(): [])
    with pytest.raises(CheckpointNotWrittenError, match="skipped checkpoints"):
        asyncio.run(run_or_resume(h.yaml, "start", **h.kwargs()))


def test_assert_checkpoints_written() -> None:
    wf_mod._assert_checkpoints_written("w", {3, 4, 5}, 3, "d")
    with pytest.raises(CheckpointNotWrittenError, match="no checkpoint"):
        wf_mod._assert_checkpoints_written("w", set(), 0, "d")
    with pytest.raises(CheckpointNotWrittenError, match=r"\[1, 2\]"):
        wf_mod._assert_checkpoints_written("w", {0, 3}, 0, "d")


def test_declarative_allowlist() -> None:
    base = "agent_framework_declarative._workflows._declarative_base"
    allow = declarative_allowlist(["my.mod:Thing"])
    for cls in ("DeclarativeWorkflowState", "ActionTrigger", "ActionComplete", "ConditionResult",
                "LoopIterationResult", "LoopControl", "DeclarativeStateData"):
        assert f"{base}:{cls}" in allow
    assert "my.mod:Thing" in allow
    assert all(":" in entry for entry in allow)
    assert not any(entry.endswith(":Any") or "typing" in entry for entry in allow)


@pytest.mark.parametrize("kind", sorted(wf_mod.FORBIDDEN_ACTIONS))
def test_assert_expression_free_rejects_forbidden_actions(kind: str) -> None:
    text = WORKFLOW + f"    - kind: {kind}\n      id: bad\n"
    with pytest.raises(WorkflowSpecError, match=kind):
        assert_expression_free(text)


def test_assert_expression_free_rejects_nested_forbidden_action() -> None:
    text = WORKFLOW + "    - kind: SendActivity\n      id: s\n      nested:\n        - {kind: If, id: x}\n"
    with pytest.raises(WorkflowSpecError, match="If"):
        assert_expression_free(text)


@pytest.mark.parametrize("snippet", [
    'arguments: {x: "=Env.SECRET"}',
    'arguments: {x: "=Local.y"}',
    'arguments: {"=key": "v"}',
])
def test_assert_expression_free_rejects_expressions(snippet: str) -> None:
    with pytest.raises(WorkflowSpecError, match="'=' expressions"):
        assert_expression_free(WORKFLOW.replace('arguments: {x: "1"}', snippet))


@pytest.mark.parametrize("text", ["- a\n", "kind: [unclosed\n"])
def test_assert_expression_free_rejects_bad_yaml(text: str) -> None:
    with pytest.raises(WorkflowSpecError):
        assert_expression_free(text)


def test_assert_expression_free_accepts_literal_equals_inside() -> None:
    data = assert_expression_free(WORKFLOW.replace('x: "1"', 'x: "a=b"'))
    assert data["kind"] == "Workflow"


@pytest.mark.parametrize("old, new, match", [
    ("kind: Workflow", "kind: Prompt", "kind: Workflow"),
    ("name: Thinker", "name: Stranger", "unknown agent"),
    ("functionName: finish", "functionName: rm_rf", "unknown tool"),
    ("kind: Workflow\n", "kind: Workflow\nagents:\n  Inline: {kind: Prompt}\n", "inline agent"),
])
def test_build_workflow_rejects_bad_references(h: Harness, old: str, new: str, match: str) -> None:
    h.yaml.write_text(WORKFLOW.replace(old, new), encoding="utf-8")
    with pytest.raises(WorkflowSpecError, match=match):
        build_workflow(h.yaml, **h.kwargs())


def test_secure_dir(tmp_path: Path) -> None:
    d = secure_dir(tmp_path / "a" / "ckpt")
    assert d.is_dir()
    if os.name != "nt":
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    link = tmp_path / "link"
    try:
        os.symlink(d, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(WorkflowSpecError, match="symlink"):
        secure_dir(link)
