import ast
import sys
import types

import pytest

from ci_lab.mcp.client import ToolInfo
from ci_lab.mcp.codecheck import check_code
from ci_lab.mcp.stubs import bootstrap_source, describe, generate_stubs

ALLOWED = ("json", "math", "re")

READ = ToolInfo("harness", "read_component", "Read one file.",
                {"type": "object", "required": ["path"],
                 "properties": {"path": {"type": "string", "description": "relative path"},
                                "max_chars": {"type": "integer", "default": 20000}}},
                "code", {"type": "object"})
LIST = ToolInfo("harness", "list_components", "List files.", {"type": "object", "properties": {}}, "code",
                {"type": "object", "properties": {"result": {"type": "array"}}, "x-fastmcp-wrap-result": True})
ODD = ToolInfo("other", "weird", 'Has """quotes""" and \\ backslash.',
               {"type": "object", "properties": {"class": {"type": "string"}}}, "code")


@pytest.mark.parametrize("code", [
    "import json\nprint(json.dumps(tools.harness.list_components()))",
    "from tools import harness\nfrom math import sqrt\nx = harness.read_component(path='a')\nprint(sqrt(4))",
    "import tools\n_tmp = 1\nprint(getattr(tools, 'harness'))",
    "try:\n    tools.harness.read_component(path='x')\nexcept tools.ToolError as e:\n    print(e)",
    "print(f'{1+1}')",
])
def test_check_code_accepts(code):
    assert check_code(code, ALLOWED) == []


@pytest.mark.parametrize("code, needle", [
    ("import os", "forbidden module 'os'"),
    ("import subprocess", "forbidden module"),
    ("import socket", "forbidden module"),
    ("import ctypes", "forbidden module"),
    ("import importlib", "forbidden module"),
    ("from sys import modules", "forbidden module 'sys'"),
    ("import urllib.request", "forbidden module"),
    ("import collections", "not allowed"),
    ("from . import x", "relative"),
    ("from json import *", "star"),
    ("from tools import _rpc", "private name"),
    ("import tools._rpc", "private module"),
    ("tools.harness._call('a', 'b', {})", "starts with '_'"),
    ("x = ().__class__", "starts with '_'"),
    ("print(__builtins__)", "dunder name"),
    ("open('x')", "'open' is not allowed"),
    ("eval('1')", "'eval' is not allowed"),
    ("exec('1')", "'exec' is not allowed"),
    ("compile('1', 'f', 'exec')", "'compile' is not allowed"),
    ("__import__('os')", "not allowed"),
    ("globals()", "'globals' is not allowed"),
    ("n = 'x'\ngetattr(tools, n)", "literal attribute name"),
    ("getattr(tools, '_rpc')", "literal attribute name"),
    ("g = getattr", "may only be called directly"),
    ("'{0.__class__}'.format(1)", "'format' is not allowed"),
    ("match 1:\n    case object(__class__=c):\n        pass", "match attribute"),
    ("def f(:\n", "syntax error"),
])
def test_check_code_rejects(code, needle):
    errs = check_code(code, ALLOWED)
    assert any(needle in e for e in errs), errs


def test_check_code_cannot_allow_forbidden_modules():
    assert "forbidden module 'os'" in check_code("import os", ("os",))[0]


def test_generate_stubs_typed_and_parseable(monkeypatch):
    files = generate_stubs([READ, LIST, ODD])
    assert set(files) == {"tools/__init__.py", "tools/_rpc.py", "tools/harness.py", "tools/other.py"}
    src = files["tools/harness.py"]
    assert "def read_component(*, path: str, max_chars: int | None = None) -> dict:" in src
    assert "def list_components() -> list:" in src
    assert "def weird(**kwargs: Any) -> Any:" in files["tools/other.py"]  # 'class' is a keyword
    calls = []
    rpc = types.ModuleType("tools._rpc")
    rpc.call = lambda server, tool, args: calls.append((server, tool, args)) or "ok"
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    monkeypatch.setitem(sys.modules, "tools._rpc", rpc)
    ns: dict = {}
    exec(compile(src, "tools/harness.py", "exec"), ns)  # noqa: S102 - generated stub under test
    assert ns["read_component"](path="p") == "ok"
    assert calls == [("harness", "read_component", {"path": "p", "max_chars": None})]
    assert "path (str, required): relative path" in ns["read_component"].__doc__
    assert "max_chars (int, default 20000)" in ns["read_component"].__doc__


def test_describe_lists_tools_and_rules():
    text = describe([READ, LIST], timeout_s=5, max_output_chars=100, imports=["json"])
    assert "tools.harness.read_component(*, path: str" in text and "Read one file." in text
    assert "Allowed imports: json" in text and "100 chars" in text and "5s" in text


def test_bootstrap_is_valid_python(tmp_path):
    src = bootstrap_source(str(tmp_path / "stubs"), allowed_imports=["json"], max_output_chars=10)
    tree = ast.parse(src)
    assert any(isinstance(n, ast.Constant) and n.value == str(tmp_path / "stubs") for n in ast.walk(tree))
    assert "'tools'" in src and "'open'" in src and "MAX_CHARS = 10" in src
