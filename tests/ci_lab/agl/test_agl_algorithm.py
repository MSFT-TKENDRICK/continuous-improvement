from __future__ import annotations

import ast
import asyncio
import json
import shutil
from pathlib import Path

import pytest

from ci_lab.agl.algorithm import (
    ComponentOwnershipError,
    LlmResourceAlgorithm,
    StructuralProposalError,
)
from ci_lab.agl.journal import FileRolloutJournal
from ci_lab.contracts import ArmContext, ArmDirective, Edit, Profile, RolloutKey
from ci_lab.harness_tree import HarnessTree
from ci_lab.testing import FakeChatClient

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    root = tmp_path / "arm"
    shutil.copytree(REPO / "harness", root / "harness")
    assert HarnessTree(root / "harness").validate() == []
    return root


def context(worktree: Path, tmp_path: Path, focus: tuple[str, ...]) -> ArmContext:
    return ArmContext("camp-r01", ArmDirective("v1", "agl", focus, 1), worktree, "base", [],
                      Profile.FAKE, tmp_path / "run")


def journal(tmp_path: Path) -> FileRolloutJournal:
    out = FileRolloutJournal(tmp_path / "journal", fsync=False)
    key = RolloutKey("camp-r00", "base", "case-cost")
    out.start(key, {"suite": "harness_proposal", "component_touches": {"loop": 2}})
    out.event(key, "ci.score", {
        "name": "assert", "value": 0.2, "suite": "harness_proposal",
        "rule_ids": ["budget.exceeded"], "llm_calls": 20, "tool_calls": 10,
    }, event_id="score")
    out.finish(key, "succeeded")
    return out


def credit_reply(component: str, reason: str = "cost.calls") -> str:
    return json.dumps({"credits": [{
        "component": component, "weight": 0.9, "reason_code": reason,
        "evidence_ids": ["case-cost:calls"],
    }]})


def edit_reply(path: str, content: str, component: str = "loop") -> str:
    return json.dumps({"component": component, "path": path, "operation": "replace",
                       "content": content, "hypothesis": "tighten proposer nudges"})


@pytest.mark.parametrize(("component", "owner"), [
    ("prompt", "gepa"),
    ("skill", "skillopt"),
    ("guard", "guard"),
])
def test_agl_refuses_components_owned_by_other_strategies(
        worktree: Path, tmp_path: Path, component: str, owner: str) -> None:
    client = FakeChatClient()
    algorithm = LlmResourceAlgorithm(client=client)
    with pytest.raises(ComponentOwnershipError, match=rf"owned by {owner}"):
        asyncio.run(algorithm.propose(context(worktree, tmp_path, (component,))))
    assert client.requests == []


def test_agl_commits_one_contained_structural_edit(
        worktree: Path, tmp_path: Path) -> None:
    path = "harness/loops/loops.yaml"
    original = (worktree / path).read_text(encoding="utf-8")
    changed = original.replace("proposer: {max_nudges: 2", "proposer: {max_nudges: 1")
    client = FakeChatClient([credit_reply("loop"), edit_reply(path, changed)])
    committed: list[tuple[Path, tuple[str, ...], str]] = []

    def commit(root: Path, files: tuple[str, ...] | list[str], message: str) -> str:
        committed.append((root, tuple(files), message))
        return "abc123"

    algorithm = LlmResourceAlgorithm(journal=journal(tmp_path), client=client, committer=commit)
    edits = asyncio.run(algorithm.propose(context(worktree, tmp_path, ("loop",))))
    assert edits == [Edit("loop", "tighten proposer nudges", (path,), "abc123")]
    assert "proposer: {max_nudges: 1" in (worktree / path).read_text(encoding="utf-8")
    expected_message = "\n".join((
        "v1: agl tighten proposer nudges",
        "",
        "RRSI-Component: loop",
        "RRSI-Hypothesis: tighten proposer nudges",
    ))
    assert committed == [(worktree, (path,), expected_message)]
    assert HarnessTree(worktree / "harness").validate() == []
    assert "Prefer deletion or tightening" in "\n".join(m.text for m in client.requests[1][0])
    report = json.loads((tmp_path / "run" / "optimizer" / "v1-agl.json").read_text(encoding="utf-8"))
    assert report["component"] == "loop" and report["edits"][0]["files"] == [path]
    credit_events = [e for e in algorithm.source.events(RolloutKey("camp-r00", "base", "case-cost"))
                     if e["event_type"] == "ci.credit"]
    assert len(credit_events) == 1


def test_agl_uses_copilot_optimizer_factory(worktree: Path, tmp_path: Path) -> None:
    path = "harness/loops/loops.yaml"
    content = (worktree / path).read_text(encoding="utf-8").replace(
        "proposer: {max_nudges: 2", "proposer: {max_nudges: 1")
    client = FakeChatClient([credit_reply("loop"), edit_reply(path, content)])
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeChatClient:
        calls.append(kwargs)
        return client

    algorithm = LlmResourceAlgorithm(journal=journal(tmp_path), client_factory=factory,
                                     committer=lambda *_: "sha")
    asyncio.run(algorithm.propose(context(worktree, tmp_path, ("loop",))))
    assert calls == [{"profile": Profile.COPILOT, "model": "gpt-5-mini", "purpose": "optimizer"}]


def test_agl_rejects_path_outside_component(worktree: Path, tmp_path: Path) -> None:
    client = FakeChatClient([credit_reply("loop"), edit_reply("src/ci_lab/contracts.py", "bad")])
    algorithm = LlmResourceAlgorithm(client=client, committer=lambda *_: "sha")
    with pytest.raises(StructuralProposalError, match="escaped"):
        asyncio.run(algorithm.propose(context(worktree, tmp_path, ("loop",))))
    assert not (worktree / "src").exists()


def test_agl_fails_closed_on_changed_frozen_manifest(worktree: Path, tmp_path: Path) -> None:
    manifest = worktree / "harness" / "harness.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n# widened\n", encoding="utf-8")
    client = FakeChatClient()
    with pytest.raises(StructuralProposalError, match="invalid before AGL"):
        asyncio.run(LlmResourceAlgorithm(client=client).propose(context(worktree, tmp_path, ("loop",))))
    assert client.requests == []


def test_importing_algorithm_does_not_import_verl() -> None:
    import ci_lab.agl.algorithm as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name == "verl" or name.startswith("verl.") for name in imported)
