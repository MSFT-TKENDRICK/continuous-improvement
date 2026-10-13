"""Code-mode stub and bootstrap generation.

``generate_stubs`` writes a self-contained ``tools`` package (``tools.<server>.<tool>(**kwargs)``, typed from
the MCP input schemas) whose functions talk to the parent over a JSON-lines RPC channel. ``bootstrap_source``
is the child entry point run as ``sys.executable -I bootstrap.py``. Nothing here imports ``ci_lab``.
"""

from __future__ import annotations

import ast
import keyword
from collections.abc import Mapping, Sequence
from typing import Any

from ci_lab.mcp.codecheck import BLOCKED_NAMES, STUB_MODULE

_JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list",
               "object": "dict", "null": "None"}

RPC_SOURCE = '''"""Code-mode RPC channel (generated)."""
import json

_chan = {}


class ToolError(Exception):
    """A tool call failed or was denied by governance."""


def bind(out, inp):
    _chan.update(out=out, inp=inp, seq=0)


def call(server, tool, args):
    _chan["seq"] += 1
    seq = _chan["seq"]
    msg = {"rpc": {"id": seq, "method": "call", "server": server, "tool": tool,
                   "args": {k: v for k, v in args.items() if v is not None}}}
    _chan["out"].write(json.dumps(msg, default=str) + "\\n")
    _chan["out"].flush()
    line = _chan["inp"].readline()
    if not line:
        raise ToolError("RPC channel closed")
    reply = json.loads(line).get("rpc") or {}
    if reply.get("id") != seq:
        raise ToolError("RPC channel out of sync")
    if reply.get("ok"):
        return reply.get("result")
    raise ToolError(reply.get("error") or "tool call failed")
'''

BOOTSTRAP_TEMPLATE = '''"""Code-mode bootstrap (generated). Run as: python -I bootstrap.py"""
import builtins
import io
import json
import linecache
import os
import sys
import traceback

STUB_DIR = {stub_dir!r}
ALLOWED = frozenset({allowed!r})
MAX_CHARS = {max_chars!r}
BLOCKED = frozenset({blocked!r})

sys.path.insert(0, STUB_DIR)
rpc_out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\\n")
rpc_in = os.fdopen(os.dup(0), "r", encoding="utf-8", newline="\\n")
null = os.open(os.devnull, os.O_RDWR)
os.dup2(null, 0)
os.dup2(null, 1)


class Capped(io.TextIOBase):
    def __init__(self):
        self.parts, self.kept, self.total = [], 0, 0

    def writable(self):
        return True

    def write(self, s):
        self.total += len(s)
        room = MAX_CHARS + 1 - self.kept
        if room > 0:
            self.parts.append(s[:room])
            self.kept += min(len(s), room)
        return len(s)

    def value(self):
        return "".join(self.parts)


buf = Capped()
sys.stdout = sys.stderr = buf
sys.stdin = io.StringIO("")

import tools  # noqa: E402
from tools import _rpc  # noqa: E402

_rpc.bind(rpc_out, rpc_in)
real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {{name!r}} is not allowed in code mode")
    return real_import(name, globals, locals, fromlist, level)


safe = {{k: v for k, v in vars(builtins).items() if k not in BLOCKED}}
safe["__import__"] = guarded_import
with open(os.path.join(os.path.dirname(STUB_DIR), "user_code.py"), encoding="utf-8") as f:
    src = f.read()
linecache.cache["<run_code>"] = (len(src), None, src.splitlines(True), "<run_code>")
error = None
try:
    exec(compile(src, "<run_code>", "exec"), {{"__builtins__": safe, "__name__": "__main__", "tools": tools}})
except SystemExit:
    pass
except BaseException as e:
    error = "".join(traceback.format_exception(type(e), e, e.__traceback__.tb_next))
rpc_out.write(json.dumps({{"rpc": {{"method": "done", "output": buf.value(), "output_chars": buf.total,
                                   "error": error}}}}) + "\\n")
rpc_out.flush()
'''


def _py_type(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return "Any"
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            parts = list(dict.fromkeys(_py_type(s) for s in schema[key]))
            return "Any" if "Any" in parts else " | ".join(parts)
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(dict.fromkeys(_JSON_TYPES.get(x, "Any") for x in t))
    return _JSON_TYPES.get(t, "Any") if isinstance(t, str) else "Any"


def _valid(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name) \
        and not name.startswith("_") and name not in BLOCKED_NAMES


def _signature(tool: Any) -> tuple[str, str, list[str]]:
    """(params source, call-args source, docstring arg lines) for a ToolInfo-like object."""
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    if not all(_valid(n) for n in props):
        return "**kwargs: Any", "kwargs", ["**kwargs: arguments per the tool's JSON schema"]
    params, args, docs = [], [], []
    for name in sorted(props, key=lambda n: (n not in required, n)):
        p = props[name] if isinstance(props[name], Mapping) else {}
        ty = _py_type(p)
        params.append(f"{name}: {ty}" if name in required else f"{name}: {ty} | None = None")
        args.append(f"{name!r}: {name}")
        extra = "required" if name in required else f"default {p['default']!r}" if "default" in p else "optional"
        desc = str(p.get("description") or p.get("title") or "").strip()
        docs.append(f"{name} ({ty}, {extra}){': ' + desc if desc else ''}")
    head = "*, " if params else ""
    return head + ", ".join(params), "{" + ", ".join(args) + "}", docs


def _return_type(tool: Any) -> str:
    out = tool.output_schema or {}
    if out.get("x-fastmcp-wrap-result"):
        return _py_type((out.get("properties") or {}).get("result"))
    return "dict" if out.get("type") == "object" else "Any"


def _doc(tool: Any, arg_docs: list[str]) -> str:
    doc = (tool.description or "").strip() or f"{tool.server}.{tool.name}"
    if arg_docs:
        doc += "\n\nArgs:\n" + "\n".join(f"    {d}" for d in arg_docs)
    return doc


def usable(tools: Sequence[Any]) -> list[Any]:
    return [t for t in tools if _valid(t.server) and _valid(t.name)]


def generate_stubs(tools: Sequence[Any]) -> dict[str, str]:
    """``{relative path: source}`` for the ``tools`` package; every source is parsed before return."""
    by_server: dict[str, list[Any]] = {}
    for t in usable(tools):
        by_server.setdefault(t.server, []).append(t)
    files = {f"{STUB_MODULE}/_rpc.py": RPC_SOURCE}
    init = ['"""MCP tool stubs for code mode (generated)."""', "from tools._rpc import ToolError",
            *(f"from tools import {s}" for s in sorted(by_server))]
    files[f"{STUB_MODULE}/__init__.py"] = "\n".join(init) + "\n"
    for server, ts in sorted(by_server.items()):
        lines = [f'"""Tools of MCP server {server!r} (generated)."""', "from typing import Any", "",
                 "from tools._rpc import call as _call", ""]
        for t in sorted(ts, key=lambda t: t.name):
            params, call_args, arg_docs = _signature(t)
            lines += ["", f"def {t.name}({params}) -> {_return_type(t)}:",
                      f"    return _call({server!r}, {t.name!r}, {call_args})", "",
                      f"{t.name}.__doc__ = {_doc(t, arg_docs)!r}", ""]
        files[f"{STUB_MODULE}/{server}.py"] = "\n".join(lines)
    for path, src in files.items():
        ast.parse(src, path)
    return files


def describe(tools: Sequence[Any], *, timeout_s: float, max_output_chars: int, imports: Sequence[str]) -> str:
    """The model-facing ``run_code`` description: rules plus the typed stub signatures and docstrings."""
    out = [
        ("Run a short Python snippet that calls MCP tools; chain several calls in ONE run and print only the "
         f"result you need (stdout is returned, capped at {max_output_chars} chars; timeout {timeout_s:g}s)."),
        ("`tools` is pre-imported; call tools as tools.<server>.<tool>(name=value, ...) and they return parsed "
         "JSON. Failed or denied calls raise tools.ToolError."),
        (f"Allowed imports: {', '.join(sorted(imports)) or 'none'}. No files, network, processes, "
         "open/exec/eval, getattr with computed names, or names/attributes starting with '_'."),
        "",
        "Available tools:",
    ]
    for t in sorted(usable(tools), key=lambda t: (t.server, t.name)):
        params, _, arg_docs = _signature(t)
        doc = _doc(t, arg_docs).replace("\n", "\n        ")
        out += [f"    def tools.{t.server}.{t.name}({params}) -> {_return_type(t)}:", f'        """{doc}"""', ""]
    return "\n".join(out).rstrip() + "\n"


def bootstrap_source(stub_dir: str, *, allowed_imports: Sequence[str], max_output_chars: int) -> str:
    src = BOOTSTRAP_TEMPLATE.format(stub_dir=str(stub_dir), allowed=sorted({*allowed_imports, STUB_MODULE}),
                                    max_chars=int(max_output_chars), blocked=sorted(BLOCKED_NAMES - {"__import__"}))
    ast.parse(src, "bootstrap.py")
    return src
