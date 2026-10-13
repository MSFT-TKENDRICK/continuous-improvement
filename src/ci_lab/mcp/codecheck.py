"""Static AST allowlist for code-mode snippets (first layer of the code-mode isolation; not an OS sandbox)."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ci_lab.mcp.registry import FORBIDDEN_MODULES

BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
    "setattr", "delattr", "help", "memoryview", "exit", "quit",
})
# str.format/format_map can reach attributes through the format spec ("{0.__class__}").
BLOCKED_ATTRS = frozenset({"format", "format_map"})
LITERAL_NAME_CALLS = frozenset({"getattr", "hasattr"})
STUB_MODULE = "tools"


def check_code(code: str, allowed_imports: Iterable[str]) -> list[str]:
    """Every rule ``code`` breaks (empty = accepted).

    Rules: imports only from ``allowed_imports`` (minus FORBIDDEN_MODULES) and the ``tools`` stubs, no
    relative or star imports; no name or attribute starting with ``_`` beyond plain ``_x`` locals (no
    dunders anywhere); no ``open``/``exec``/``eval``/``compile``/``globals``/...; ``getattr``/``hasattr``
    only as direct calls with a literal, non-underscore attribute name.
    """
    try:
        tree = ast.parse(code, "<run_code>")
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]
    allowed = (set(allowed_imports) - FORBIDDEN_MODULES) | {STUB_MODULE}
    errs: list[tuple[int, str]] = []

    def bad(node: ast.AST, msg: str) -> None:
        errs.append((getattr(node, "lineno", 0), msg))

    def check_module(node: ast.AST, name: str) -> None:
        parts = name.split(".")
        if any(p.startswith("_") for p in parts):
            bad(node, f"import of private module {name!r}")
        elif parts[0] in FORBIDDEN_MODULES:
            bad(node, f"import of forbidden module {name!r}")
        elif parts[0] not in allowed:
            bad(node, f"import of {name!r} is not allowed (allowed: {', '.join(sorted(allowed))})")

    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                check_module(node, a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                bad(node, "relative imports are not allowed")
                continue
            check_module(node, node.module)
            for a in node.names:
                if a.name == "*":
                    bad(node, "star imports are not allowed")
                elif a.name.startswith("_"):
                    bad(node, f"import of private name {a.name!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                bad(node, f"attribute {node.attr!r} starts with '_'")
            elif node.attr in BLOCKED_ATTRS:
                bad(node, f"attribute {node.attr!r} is not allowed (use f-strings)")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                bad(node, f"name {node.id!r} is not allowed")
            elif node.id.startswith("__"):
                bad(node, f"dunder name {node.id!r} is not allowed")
            elif node.id in LITERAL_NAME_CALLS and id(node) not in called:
                bad(node, f"{node.id!r} may only be called directly")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in LITERAL_NAME_CALLS:
            arg = node.args[1] if len(node.args) > 1 else None
            if node.keywords or not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                     and not arg.value.startswith("_")):
                bad(node, f"{node.func.id}() needs a literal attribute name not starting with '_'")
        elif isinstance(node, ast.MatchClass):
            for attr in node.kwd_attrs:
                if attr.startswith("_"):
                    bad(node, f"match attribute {attr!r} starts with '_'")
        elif isinstance(node, ast.alias | ast.arg | ast.keyword):
            name = getattr(node, "arg", None) or getattr(node, "asname", None)
            if name and name.startswith("__"):
                bad(node, f"dunder name {name!r} is not allowed")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name.startswith("__"):
            bad(node, f"dunder name {node.name!r} is not allowed")
    return [f"line {ln}: {msg}" for ln, msg in sorted(set(errs))]
