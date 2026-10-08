"""Target-agent workflows (``harness/workflows``) and skills (``harness/skills``) of the harness tree."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from ci_lab.harness_tree import HarnessTree, repo_harness_dir, skill_frontmatter
from ci_lab.meta.spec_loader import load_spec
from ci_lab.testing import Call, FakeChatClient
from ci_lab.workflows import agent_names, assert_expression_free, function_names
from ci_lab.workflows.runtime import build_workflow, run_or_resume

REPO = HarnessTree(repo_harness_dir())
EXPECTED = {"triage": (["failure_analyst", "analyst"], {"CiFailureAnalyst", "CiAnalyst"}, set()),
            "propose": (["proposer", "self_check"], {"CiProposer"}, {"self_check"})}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    dest = tmp_path / "harness"
    shutil.copytree(repo_harness_dir(), dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_target_workflows_validate(name: str) -> None:
    path = REPO.workflow_path(name)
    ids, agents, functions = EXPECTED[name]
    assert [a["id"] for a in assert_expression_free(path)["trigger"]["actions"]] == ids
    assert agent_names(path) == agents and function_names(path) == functions
    spec_names = {load_spec(k).name for k in REPO.agent_names()}
    assert agents <= spec_names and functions <= set(REPO.manifest["workflow_functions"])


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_target_workflows_run_with_fake_agents(name: str, tmp_path: Path) -> None:
    from agent_framework import Agent

    path = REPO.workflow_path(name)
    log: list[str] = []

    def agent(agent_name: str) -> Any:
        def submit(note: str = "") -> str:
            log.append(agent_name)
            return "ok"

        submit.__name__ = submit.__doc__ = "submit_stub"
        return Agent(client=FakeChatClient([[Call("submit_stub", {"note": "x"})], "done"]), name=agent_name,
                     instructions="stub", tools=[submit])

    def step(fn: str):
        async def call(**kwargs: Any) -> dict[str, Any]:
            log.append(f"{fn}:{kwargs['step']}")
            return {"ok": True}
        return call

    workflow = build_workflow(path, {a: agent(a) for a in agent_names(path)},
                              {fn: step(fn) for fn in function_names(path)}, tmp_path / "ckpt")
    assert asyncio.run(run_or_resume(workflow, tmp_path / "ckpt"))["status"] == "completed"
    want = {"triage": ["CiFailureAnalyst", "CiAnalyst"], "propose": ["CiProposer", "self_check:self_check"]}
    assert log == want[name]


@pytest.mark.parametrize(("old", "new", "match"), [
    ("name: CiAnalyst", "name: Analyst", "'Analyst' is not an agent of this tree"),
    ("messages: \"Read your brief", "messages: \"=Read your brief", "expression"),
])
def test_bad_triage_workflow_fails_validation(tree: Path, old: str, new: str, match: str) -> None:
    p = tree / "workflows" / "triage.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    assert any("workflows/triage.yaml" in e and match in e for e in HarnessTree(tree).validate())


def test_workflow_function_outside_the_manifest_fails(tree: Path) -> None:
    p = tree / "workflows" / "propose.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("functionName: self_check", "functionName: publish"),
                 encoding="utf-8")
    assert any("function 'publish' is not one of ['self_check']" in e for e in HarnessTree(tree).validate())


def test_skills_have_matching_frontmatter(tree: Path) -> None:
    for d in ("harness-editing", "trace-triage"):
        meta = skill_frontmatter((repo_harness_dir() / "skills" / d / "SKILL.md").read_text(encoding="utf-8"))
        assert meta is not None and meta["name"] == d and meta["description"].strip()
    p = tree / "skills" / "trace-triage" / "SKILL.md"
    p.write_text(p.read_text(encoding="utf-8").replace("name: trace-triage", "name: other"), encoding="utf-8")
    assert any("skills/trace-triage/SKILL.md" in e for e in HarnessTree(tree).validate())


def test_repo_tree_is_complete_and_valid() -> None:
    assert REPO.validate() == []
    present = {REPO.component_of(rel) for rel in REPO.files()} - {None}
    assert {"prompt", "agent", "skill", "workflow", "loop", "client_tool", "mcp", "guard"} <= present
