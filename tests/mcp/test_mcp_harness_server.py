import asyncio
import json
import sys
import types

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ci_lab.mcp.harness_server import build_server
from ci_lab.mcp.registry import load_registry

SECRET = "sk-raw-tool-output-should-never-leak"


@pytest.fixture
def roots(tmp_path):
    h, runs = tmp_path / "harness", tmp_path / "runs"
    (h / "agents").mkdir(parents=True)
    (h / "prompts").mkdir()
    (h / "agents" / "analyst.yaml").write_text("kind: Prompt\nname: analyst\n", encoding="utf-8")
    (h / "prompts" / "analyst.md").write_text("line one\nline two\n" + "x" * 100, encoding="utf-8")
    (h / "blob.bin").write_bytes(b"\xff\xfe\x00\x81")
    (h / ".hidden").write_text("no", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    run = runs / "r1"
    run.mkdir(parents=True)
    events = [{"type": "tool_call", "output": SECRET}, {"type": "tool_call"}, {"kind": "model_request"},
              {"type": "tool_call", "error": "boom " + SECRET}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n", encoding="utf-8")
    (run / "notes.txt").write_text(SECRET, encoding="utf-8")
    (runs / "eval.json").write_text(json.dumps({"scores": [
        {"case_id": "a", "suite": "s1", "score": 1.0, "violations": [], "tokens_in": 10, "wall_ms": 5},
        {"case_id": "b", "suite": "s1", "score": 0.0, "violations": [{"rule_id": "x.y", "detail": SECRET}]},
        {"case_id": "c", "suite": "s2", "score": None},
    ]}), encoding="utf-8")
    return h, runs


def call(server, tool, args=None):
    async def go():
        async with create_connected_server_and_client_session(server) as s:
            return await s.call_tool(tool, args or {})
    return asyncio.run(go())


def ok(server, tool, args=None):
    r = call(server, tool, args)
    assert not r.isError, r.content
    return r.structuredContent


def err(server, tool, args=None):
    r = call(server, tool, args)
    assert r.isError
    return r.content[0].text


def test_tools_match_frozen_allowlist(roots):
    async def go():
        async with create_connected_server_and_client_session(build_server(*roots)) as s:
            return {t.name for t in (await s.list_tools()).tools}
    assert asyncio.run(go()) == set(load_registry().servers["harness"].tools_allow)


def test_list_components(roots):
    out = ok(build_server(*roots), "list_components")
    assert out["components"]["agents"] == ["agents/analyst.yaml"]
    assert out["components"]["prompts"] == ["prompts/analyst.md"]
    assert ".hidden" not in json.dumps(out)


def test_list_components_uses_manifest_globs(roots):
    h, runs = roots
    (h / "harness.yaml").write_text("components:\n  prompt: [harness/prompts/*.md]\n", encoding="utf-8")
    out = ok(build_server(h, runs), "list_components")
    assert out["components"]["prompt"] == ["prompts/analyst.md"]


def test_read_component_contained_and_capped(roots):
    srv = build_server(*roots)
    out = ok(srv, "read_component", {"path": "prompts/analyst.md", "max_chars": 8})
    assert out["text"] == "line one" and out["truncated"] and out["chars"] > 8
    assert "escapes" in err(srv, "read_component", {"path": "../outside.txt"})
    assert "escapes" in err(srv, "read_component", {"path": str(roots[0].parent / "outside.txt")})
    assert "UTF-8" in err(srv, "read_component", {"path": "blob.bin"})
    assert "not a file" in err(srv, "read_component", {"path": "agents"})


def test_component_metrics_local_fallback(roots, monkeypatch):
    monkeypatch.setitem(sys.modules, "ci_lab.metrics.simplicity", None)
    out = ok(build_server(*roots), "component_metrics")
    assert out["source"] == "local"
    m = out["metrics"]
    assert m["files"] == 3 and m["component.prompts.lines"] == 3 and m["component.agents.files"] == 1


def test_component_metrics_prefers_ci_lab_metrics(roots, monkeypatch):
    fake = types.ModuleType("ci_lab.metrics.simplicity")
    fake.surface_metrics = lambda root: {"complexity": 7.0}
    monkeypatch.setitem(sys.modules, "ci_lab.metrics.simplicity", fake)
    assert ok(build_server(*roots), "component_metrics") == {"source": "ci_lab.metrics",
                                                             "metrics": {"complexity": 7.0}}


def test_trace_summary_typed_counts_only(roots):
    srv = build_server(*roots)
    out = ok(srv, "trace_summary", {"run_dir": "r1"})
    assert out["records"] == 4 and out["bad_lines"] == 1 and out["error_records"] == 1
    assert out["by_type"] == {"tool_call": 3, "model_request": 1}
    assert out["by_suffix"] == {".jsonl": 1, ".txt": 1}
    assert SECRET not in json.dumps(out)
    assert "escapes" in err(srv, "trace_summary", {"run_dir": str(roots[0])})
    assert "escapes" in err(srv, "trace_summary", {"run_dir": "../harness"})


def test_eval_summary(roots):
    srv = build_server(*roots)
    out = ok(srv, "eval_summary", {"path": "eval.json"})
    assert out["n"] == 3 and out["completed"] == 2 and out["mean_score"] == 0.5
    assert out["by_suite"]["s2"] == {"n": 0, "mean": None}
    assert out["violations"] == {"x.y": 1} and out["totals"]["tokens_in"] == 10.0
    assert SECRET not in json.dumps(out)
    assert "escapes" in err(srv, "eval_summary", {"path": "../outside.txt"})
