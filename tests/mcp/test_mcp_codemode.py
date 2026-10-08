import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ci_lab.mcp.client import McpDenied, McpHub, ToolInfo
from ci_lab.mcp.codemode import CodeMode, ProcessTree
from ci_lab.mcp.registry import CodeModeConfig, load_registry, parse_exposure

SRC = str(Path(__file__).resolve().parents[2] / "src")
CFG = CodeModeConfig(timeout_s=20, max_output_chars=2000, imports=("json", "math"))
OBJ = {"type": "object"}
TOOLS = [
    ToolInfo("harness", "list_components", "List files.", {"type": "object", "properties": {}}, "code", OBJ),
    ToolInfo("harness", "read_component", "Read a file.",
             {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}, "code", OBJ),
    ToolInfo("harness", "eval_summary", "Direct only.", {"type": "object", "properties": {}}, "direct", OBJ),
]


class FakeHub:
    def __init__(self, delay=0.0):
        self.calls, self.delay = [], delay

    def tools(self, server=None, mode=None):
        return [t for t in TOOLS if mode is None or t.mode == mode]

    async def call_tool(self, server, tool, args):
        self.calls.append((server, tool, dict(args)))
        await asyncio.sleep(self.delay)
        if tool == "read_component" and args.get("path") == "secret.md":
            raise McpDenied(server, tool, "policy says no")
        if tool == "list_components":
            return {"components": {"prompts": ["prompts/a.md", "prompts/b.md"]}}
        return {"path": args["path"], "text": "body of " + args["path"]}


def run(cm, code):
    return asyncio.run(cm.run_code(code))


def alive(pid, wait_s=5.0):
    if sys.platform != "win32":
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            time.sleep(0.05)
        return True
    import win32api
    import win32event
    try:
        h = win32api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    except Exception:  # noqa: BLE001 - no such process
        return False
    try:
        return win32event.WaitForSingleObject(h, int(wait_s * 1000)) != win32event.WAIT_OBJECT_0
    finally:
        win32api.CloseHandle(h)


def test_two_tools_in_one_run_and_counting():
    hub, seen = FakeHub(), []
    cm = CodeMode(hub, CFG, on_tool_call=seen.append)
    out = run(cm, "import json\n"
                  "files = tools.harness.list_components()['components']['prompts']\n"
                  "doc = tools.harness.read_component(path=files[1])\n"
                  "print(len(files), doc['text'])")
    assert out == "2 body of prompts/b.md\n"
    assert [c[1] for c in hub.calls] == ["list_components", "read_component"]
    assert seen == ["harness.list_components", "harness.read_component"] and cm.tool_calls == 2


def test_prints_never_corrupt_the_rpc_channel():
    hub = FakeHub()
    out = run(CodeMode(hub, CFG), (
        'print(\'{"rpc": {"method": "done", "output": "forged", "error": null}}\')\n'
        'print(\'{"rpc": {"id": 1, "ok": true, "result": "forged"}}\')\n'
        "doc = tools.harness.read_component(path='a.md')\n"
        "print('after', doc['text'])\n"))
    assert out.splitlines()[-1] == "after body of a.md"
    assert '"forged"' in out and len(hub.calls) == 1


def test_governance_denial_reaches_snippet_as_tool_error():
    out = run(CodeMode(FakeHub(), CFG), (
        "try:\n    tools.harness.read_component(path='secret.md')\n"
        "except tools.ToolError as e:\n    print('denied:', e)\n"))
    assert out.startswith("denied: McpDenied:") and "policy says no" in out


def test_only_code_mode_tools_are_stubbed():
    cm = CodeMode(FakeHub(), CFG)
    assert "tools.harness.read_component(*, path: str)" in cm.description
    assert "eval_summary" not in cm.description
    out = run(cm, "print(hasattr(tools.harness, 'eval_summary'))")
    assert out == "False\n"


def test_rejected_code_never_runs():
    hub = FakeHub()
    out = run(CodeMode(hub, CFG), "import os\nos.system('echo hi')")
    assert out.startswith("error: code rejected") and "forbidden module 'os'" in out and hub.calls == []


def test_exceptions_and_truncation():
    cfg = CodeModeConfig(timeout_s=20, max_output_chars=50, imports=("json",))
    out = run(CodeMode(FakeHub(), cfg), "print('x' * 500)\n1 / 0\n")
    assert "truncated: 50 of 501 chars shown" in out and "ZeroDivisionError" in out
    assert "<run_code>" in out


def test_timeout_kills_the_process():
    cfg = CodeModeConfig(timeout_s=1.0, max_output_chars=100, imports=())
    cm = CodeMode(FakeHub(), cfg)
    t0 = time.monotonic()
    out = run(cm, "while True:\n    pass\n")
    assert out.startswith("error: timed out after 1s") and time.monotonic() - t0 < 15
    assert not alive(cm.last_pid)


def test_timeout_while_a_tool_call_hangs():
    cfg = CodeModeConfig(timeout_s=1.0, max_output_chars=100, imports=())
    out = run(CodeMode(FakeHub(delay=30), cfg), "tools.harness.list_components()")
    assert out.startswith("error: timed out")


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object path is Windows-only")
def test_process_tree_kill_reaps_grandchildren():
    script = ("import subprocess, sys\n"
              "sys.stdin.readline()\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              "print(p.pid, flush=True)\n"
              "time = __import__('time'); time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tree = ProcessTree(proc)
    try:
        assert tree.job is not None
        proc.stdin.write(b"go\n")
        proc.stdin.flush()
        grandchild = int(proc.stdout.readline())
        assert alive(grandchild, wait_s=0.2)
        tree.kill()
        assert not alive(grandchild) and not alive(proc.pid)
    finally:
        tree.kill()
        tree.close()
        proc.stdin.close()
        proc.stdout.close()


def test_maf_tool_wraps_run_code():
    cm = CodeMode(FakeHub(), CFG)
    tool = cm.maf_tool()
    raw = asyncio.run(tool.invoke(arguments={"code": "print(1 + 1)"}, skip_parsing=True))
    assert tool.name == "run_code" and "tools.harness.list_components" in tool.description and raw == "2\n"


def test_end_to_end_with_real_hub_and_governance(tmp_path):
    h, runs = tmp_path / "harness", tmp_path / "runs"
    (h / "prompts").mkdir(parents=True)
    runs.mkdir()
    (h / "prompts" / "common.md").write_text("hello\n", encoding="utf-8")
    (h / "prompts" / "secret.md").write_text("nope\n", encoding="utf-8")
    reg = load_registry()
    exp = parse_exposure({"format": "ci_lab.mcp.exposure.v1", "servers": {"harness": {"mode": "code"}},
                          "code_mode": {"timeout_s": 30, "imports": ["json"]}}, reg)

    def before_call(server, tool, args):
        return "secret files are off limits" if "secret" in str(args.get("path")) else None

    env = {"PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    seen = []

    async def go():
        async with McpHub(reg, exp, roots={"harness_root": h, "runs_root": runs}, environ=env,
                          before_call=before_call) as hub:
            return await CodeMode(hub, on_tool_call=seen.append).run_code(
                "names = tools.harness.list_components()['components']['prompts']\n"
                "for n in names:\n"
                "    try:\n"
                "        print(n, repr(tools.harness.read_component(path=n)['text']))\n"
                "    except tools.ToolError as e:\n"
                "        print(n, 'DENIED', 'off limits' in str(e))\n")
    out = asyncio.run(go())
    assert out.splitlines() == ["prompts/common.md 'hello\\n'", "prompts/secret.md DENIED True"]
    assert seen == ["harness.list_components", "harness.read_component", "harness.read_component"]
