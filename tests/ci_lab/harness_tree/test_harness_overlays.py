"""``loops/loops.yaml`` and ``tools/tools.yaml`` overlays: validation, clamping and enforcement."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from ci_lab.harness_tree import HarnessTree, repo_harness_dir
from ci_lab.meta.run import MetaAgentError, run_analyst, write_brief, write_failures
from ci_lab.meta.spec_loader import SpecError, load_spec
from ci_lab.testing import Call, FakeChatClient

SUBMIT = Call("submit_analysis", {"summary": "done", "patterns": [], "suggested_components": []})


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    dest = tmp_path / "harness"
    shutil.copytree(repo_harness_dir(), dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _edit(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    write_brief(run_dir, "Round 1: analyze the failures.")
    write_failures(run_dir, [])
    return run_dir


def _tool_results(client: FakeChatClient) -> list[str]:
    return [str(getattr(c, "result", "")) for msgs, _ in client.requests for m in msgs for c in m.contents
            if getattr(c, "type", "") == "function_result"]


def test_repo_overlays_load_within_caps() -> None:
    t = HarnessTree(repo_harness_dir())
    caps = t.manifest["caps"]
    for name, knobs in t.loops()["agents"].items():
        assert all(v <= caps["agents"][k] for k, v in knobs.items()), name
    spec = load_spec("proposer")
    assert (spec.max_nudges, spec.max_tool_calls, spec.max_turns) == (2, 80, 40)
    assert spec.terminal_tool in spec.tools and t.agent_tools("proposer") is not None


def test_loop_knobs_above_the_caps_fail_validation_and_are_clamped(tree: Path) -> None:
    _edit(tree / "loops" / "loops.yaml", lambda d: d["agents"]["analyst"].update(max_nudges=99, max_turns=1000))
    errs = HarnessTree(tree).validate()
    assert any("max_nudges=99 exceeds the frozen cap 4" in e for e in errs)
    spec = load_spec("analyst", harness_dir=tree)
    assert (spec.max_nudges, spec.max_turns) == (4, 40)


def test_missing_overlays_keep_the_defaults(tree: Path) -> None:
    (tree / "loops" / "loops.yaml").unlink()
    (tree / "tools" / "tools.yaml").unlink()
    assert HarnessTree(tree).validate() == []
    spec = load_spec("analyst", harness_dir=tree)
    assert (spec.max_nudges, spec.max_tool_calls, spec.max_turns, dict(spec.tool_descriptions)) == (2, None, None, {})
    assert "read_history" in spec.tools


def test_malformed_loops_fail_closed(tree: Path) -> None:
    _edit(tree / "loops" / "loops.yaml", lambda d: d["agents"]["analyst"].update(max_retries=3))
    assert any("max_retries: unknown knob" in e for e in HarnessTree(tree).validate())
    with pytest.raises(SpecError, match="unknown knob"):
        load_spec("analyst", harness_dir=tree)


def test_tools_overlay_restricts_and_describes(tmp_path: Path, tree: Path) -> None:
    _edit(tree / "tools" / "tools.yaml", lambda d: d["agents"].update(
        analyst={"read_brief": "Read a run document; start with failures.", "submit_analysis": None}))
    spec = load_spec("analyst", harness_dir=tree)
    assert spec.tools == ("read_brief", "submit_analysis")
    assert dict(spec.tool_descriptions) == {"read_brief": "Read a run document; start with failures."}
    client = FakeChatClient([[SUBMIT]], default="UNEXPECTED")
    asyncio.run(run_analyst(_run_dir(tmp_path), client, harness_dir=tree))
    exposed = {t.name: t.description for t in client.requests[0][1]["tools"]}
    assert exposed["read_brief"] == "Read a run document; start with failures."
    assert "read_history" not in exposed and "list_documents" not in exposed


@pytest.mark.parametrize(("overlay", "match"), [
    ({"read_brief": None, "write_file": None, "submit_analysis": None}, "does not bind"),
    ({"read_brief": None}, "must keep the terminal tool"),
])
def test_tools_overlay_cannot_add_capabilities(tree: Path, overlay: dict, match: str) -> None:
    _edit(tree / "tools" / "tools.yaml", lambda d: d["agents"].update(analyst=overlay))
    with pytest.raises(SpecError, match=match):
        load_spec("analyst", harness_dir=tree)
    assert any(match in e for e in HarnessTree(tree).validate())


def test_tool_budget_blocks_calls_but_not_the_submission(tmp_path: Path, tree: Path) -> None:
    _edit(tree / "loops" / "loops.yaml", lambda d: d["agents"]["analyst"].update(max_tool_calls=1))
    client = FakeChatClient([
        [Call("read_brief", {"name": "brief"})],
        [Call("list_documents", {})],
        [SUBMIT],
    ], default="UNEXPECTED")
    result = asyncio.run(run_analyst(_run_dir(tmp_path), client, harness_dir=tree))
    assert result.summary == "done"
    results = _tool_results(client)
    assert not any("budget" in r for r in results[:1]) and "tool-call budget (1) exhausted" in results[-1]


def test_turn_limit_ends_the_run(tmp_path: Path, tree: Path) -> None:
    _edit(tree / "loops" / "loops.yaml", lambda d: d["agents"]["analyst"].update(max_turns=1))
    client = FakeChatClient([
        [Call("read_brief", {"name": "brief"})],
        [SUBMIT],
    ], default="UNEXPECTED")
    with pytest.raises(MetaAgentError, match="without a valid submit_analysis"):
        asyncio.run(run_analyst(_run_dir(tmp_path), client, harness_dir=tree))
    assert len(client.requests) == 1
