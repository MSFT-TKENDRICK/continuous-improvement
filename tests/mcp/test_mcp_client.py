import asyncio
import json
import os
from pathlib import Path

import pytest

from ci_lab.mcp.client import Denial, McpDenied, McpHub, McpToolError, maf_tools
from ci_lab.mcp.registry import load_registry, parse_exposure

SRC = str(Path(__file__).resolve().parents[2] / "src")
ENV = {"PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}


@pytest.fixture
def roots(tmp_path):
    h, runs = tmp_path / "harness", tmp_path / "runs"
    (h / "prompts").mkdir(parents=True)
    runs.mkdir()
    (h / "prompts" / "common.md").write_text("hello harness\n", encoding="utf-8")
    return {"harness_root": h, "runs_root": runs}


def hub(roots, exposure=None, **kw):
    reg = load_registry()
    exp = parse_exposure({"format": "ci_lab.mcp.exposure.v1", **(exposure or {"servers": {"harness": {}}})}, reg)
    return McpHub(reg, exp, roots=roots, environ=ENV, **kw)


def test_stdio_round_trip(roots):
    async def go():
        async with hub(roots) as h:
            names = {t.name for t in await h.list_tools("harness")}
            comps = await h.call_tool("harness", "list_components")
            doc = await h.call_tool("harness", "read_component", {"path": "prompts/common.md"})
            with pytest.raises(McpToolError, match="escapes"):
                await h.call_tool("harness", "read_component", {"path": "../x"})
            return names, comps, doc
    names, comps, doc = asyncio.run(go())
    assert names == set(load_registry().servers["harness"].tools_allow)
    assert comps["components"] == {"prompts": ["prompts/common.md"]}
    assert doc["text"] == "hello harness\n"


def test_exposure_subset_and_description_override(roots):
    exp = {"servers": {"harness": {"tools": {"list_components": "Only this."}}}}

    async def go():
        async with hub(roots, exp) as h:
            tools = h.tools()
            with pytest.raises(McpDenied, match="not exposed"):
                await h.call_tool("harness", "read_component", {"path": "prompts/common.md"})
            return tools
    tools = asyncio.run(go())
    assert [(t.name, t.description) for t in tools] == [("list_components", "Only this.")]


def test_governance_hook_denies(roots):
    seen = []

    async def before_call(server, tool, args):
        seen.append((server, tool, dict(args)))
        if tool == "read_component":
            return Denial("reads need approval")
        if tool == "eval_summary":
            return "no evals"
        return None

    async def go():
        async with hub(roots, before_call=before_call) as h:
            with pytest.raises(McpDenied, match="reads need approval") as ei:
                await h.call_tool("harness", "read_component", {"path": "prompts/common.md"})
            with pytest.raises(McpDenied, match="no evals"):
                await h.call_tool("harness", "eval_summary", {"path": "x.json"})
            ok = await h.call_tool("harness", "list_components")
            return ei.value, ok
    denied, ok = asyncio.run(go())
    assert denied.reason == "reads need approval" and denied.tool == "read_component"
    assert ok["files"] == 1
    assert seen[0] == ("harness", "read_component", {"path": "prompts/common.md"})


def test_governance_hook_exception_blocks_call(roots):
    def before_call(server, tool, args):
        raise PermissionError("kill switch")

    async def go():
        async with hub(roots, before_call=before_call) as h:
            await h.call_tool("harness", "list_components")
    with pytest.raises(PermissionError, match="kill switch"):
        asyncio.run(go())


def test_maf_tools_direct_mode(roots):
    exp = {"servers": {"harness": {"mode": "direct", "tools": {"read_component": None, "list_components": None}}}}

    async def go():
        async with hub(roots, exp) as h:
            tools = {t.name: t for t in maf_tools(h)}
            assert maf_tools(h, mode="code") == []
            raw = await tools["harness__read_component"].invoke(arguments={"path": "prompts/common.md"},
                                                                skip_parsing=True)
            return tools, raw
    tools, raw = asyncio.run(go())
    assert set(tools) == {"harness__read_component", "harness__list_components"}
    assert "path" in tools["harness__read_component"].parameters()["properties"]
    assert json.loads(raw)["text"] == "hello harness\n"
