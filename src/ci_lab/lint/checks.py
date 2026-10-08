"""Frozen checks for each lint rule kind (AST, YAML, RE2 text). Pure functions over file text."""

from __future__ import annotations

import ast
import bisect
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import re2
import yaml

from ci_lab.lint.spec import (
    BannedAttrArg,
    BannedCall,
    BannedImport,
    BannedText,
    DeclarativeYamlExpressionFree,
    GhaBannedTrigger,
    GhaPinnedSha,
    LintRule,
    MaxLines,
)

MAX_TEXT_MATCHES_PER_FILE = 50


@dataclass(frozen=True)
class Hit:
    line: int
    detail: str = ""


class Source:
    """One file under lint; text/AST/YAML parsed lazily and cached across rules."""

    def __init__(self, path: str, text: str):
        self.path = path
        self.text = text
        self._nl: list[int] | None = None
        self._ast: ast.Module | None | bool = False
        self._yaml: Any = False
        self._nodes: list[yaml.Node] | None | bool = False

    def line_of(self, offset: int) -> int:
        if self._nl is None:
            self._nl = [i for i, c in enumerate(self.text) if c == "\n"]
        return bisect.bisect_left(self._nl, offset) + 1

    @property
    def tree(self) -> ast.Module | None:
        if self._ast is False:
            try:
                self._ast = ast.parse(self.text, filename=self.path)
            except (SyntaxError, ValueError):
                self._ast = None
        return self._ast  # type: ignore[return-value]

    @property
    def yaml_data(self) -> Any:
        if self._yaml is False:
            try:
                self._yaml = yaml.safe_load(self.text)
            except yaml.YAMLError:
                self._yaml = None
        return self._yaml

    @property
    def yaml_nodes(self) -> list[yaml.Node] | None:
        if self._nodes is False:
            try:
                self._nodes = [n for n in yaml.compose_all(self.text, Loader=yaml.SafeLoader) if n is not None]
            except yaml.YAMLError:
                self._nodes = None
        return self._nodes  # type: ignore[return-value]


# ---------------------------------------------------------------- python helpers


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _final_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def module_package(path: str) -> list[str]:
    parts = path.split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    return parts[:-1]


def _resolve_from(path: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    pkg = module_package(path)
    base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
    return ".".join([*base, *([node.module] if node.module else [])])


def _aliases(tree: ast.Module, path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    out[head] = head
        elif isinstance(n, ast.ImportFrom):
            mod = _resolve_from(path, n)
            for a in n.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
    return out


def check_banned_call(rule: BannedCall, src: Source) -> Iterator[Hit]:
    tree = src.tree
    if tree is None:
        return
    aliases = _aliases(tree, src.path)
    exact = {n for n in rule.names if not n.startswith("*.")}
    suffix = {n[2:] for n in rule.names if n.startswith("*.")}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted = _dotted(n.func)
        final = _final_name(n.func)
        resolved = None
        if dotted:
            head, _, rest = dotted.partition(".")
            resolved = aliases.get(head, head) + (f".{rest}" if rest else "")
        if (dotted in exact or resolved in exact
                or (isinstance(n.func, ast.Attribute) and final in suffix)):
            yield Hit(n.lineno, f"call {dotted or final}")


def _imported_modules(tree: ast.Module, path: str) -> Iterator[tuple[int, str]]:
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name
        elif isinstance(n, ast.ImportFrom):
            mod = _resolve_from(path, n)
            yield n.lineno, mod
            for a in n.names:
                if a.name != "*":
                    yield n.lineno, f"{mod}.{a.name}" if mod else a.name
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) \
                and isinstance(n.args[0].value, str):
            callee = _dotted(n.func) or ""
            if callee in ("importlib.import_module", "import_module", "__import__"):
                yield n.lineno, n.args[0].value


def _prefix_hit(mod: str, prefixes: list[str]) -> str | None:
    for p in prefixes:
        if mod == p or mod.startswith(p + "."):
            return p
    return None


def check_banned_import(rule: BannedImport, src: Source) -> Iterator[Hit]:
    tree = src.tree
    if tree is None:
        return
    seen: set[tuple[int, str]] = set()
    for line, mod in _imported_modules(tree, src.path):
        p = _prefix_hit(mod, rule.modules)
        if p and (line, p) not in seen:
            seen.add((line, p))
            yield Hit(line, f"imports {p}")


def _stringified(expr: ast.AST, banned: set[str]) -> str | None:
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name) and expr.func.id in ("str", "repr") and expr.func.id in banned:
            return f"{expr.func.id}()"
        if (isinstance(expr.func, ast.Attribute) and expr.func.attr == "format" and "format" in banned
                and isinstance(expr.func.value, ast.Constant) and isinstance(expr.func.value.value, str)):
            return "str.format()"
        return None
    if isinstance(expr, ast.JoinedStr) and "fstring" in banned:
        return "f-string" if any(isinstance(v, ast.FormattedValue) for v in expr.values) else None
    if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod) and "percent" in banned
            and isinstance(expr.left, ast.Constant) and isinstance(expr.left.value, str)):
        return "%-format"
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        for e in expr.elts:
            r = _stringified(e, banned)
            if r:
                return r
    if isinstance(expr, ast.Dict):
        for v in expr.values:
            r = _stringified(v, banned)
            if r:
                return r
    return None


def check_banned_attr_arg(rule: BannedAttrArg, src: Source) -> Iterator[Hit]:
    tree = src.tree
    if tree is None:
        return
    calls, banned = set(rule.calls), set(rule.banned)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or _final_name(n.func) not in calls:
            continue
        exprs = [a for i, a in enumerate(n.args) if i > 0 or isinstance(a, ast.Dict)]
        exprs += [k.value for k in n.keywords]
        for e in exprs:
            r = _stringified(e, banned)
            if r:
                yield Hit(getattr(e, "lineno", n.lineno), f"{_final_name(n.func)}(... {r} ...)")
                break


# ---------------------------------------------------------------- text / yaml


@lru_cache(maxsize=256)
def _re2(pattern: str) -> Any:
    return re2.compile(pattern)


def check_banned_text(rule: BannedText, src: Source) -> Iterator[Hit]:
    seen: set[int] = set()
    for m in _re2(rule.pattern).finditer(src.text):
        line = src.line_of(m.start())
        if line not in seen:
            seen.add(line)
            yield Hit(line)
            if len(seen) >= MAX_TEXT_MATCHES_PER_FILE:
                return


def _gha_on(data: Any) -> set[str]:
    if not isinstance(data, dict):
        return set()
    on = data.get("on", data.get(True))  # PyYAML 1.1 parses a bare `on` key as True
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return {x for x in on if isinstance(x, str)}
    if isinstance(on, dict):
        return {k for k in on if isinstance(k, str)}
    return set()


def _first_code_line(src: Source, word: str) -> int:
    rx = _re2(r"\b" + re2.escape(word) + r"\b")
    for i, line in enumerate(src.text.splitlines(), 1):
        code = line.split("#", 1)[0]
        if rx.search(code):
            return i
    return 1


def check_gha_banned_trigger(rule: GhaBannedTrigger, src: Source) -> Iterator[Hit]:
    data = src.yaml_data
    if data is None and src.text.strip():
        yield Hit(1, "workflow YAML does not parse")
        return
    for t in sorted(_gha_on(data) & set(rule.triggers)):
        yield Hit(_first_code_line(src, t), f"on: {t}")


_USES = r"""^\s*(?:-\s+)?uses:\s*["']?([^"'\s#]+)"""
_PINNED = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$"
_DOCKER = r"^docker://[^@\s]+@sha256:[0-9a-f]{64}$"


def check_gha_pinned_sha(rule: GhaPinnedSha, src: Source) -> Iterator[Hit]:
    uses, pinned, docker = _re2(_USES), _re2(_PINNED), _re2(_DOCKER)
    for i, line in enumerate(src.text.splitlines(), 1):
        m = uses.search(line)
        if not m:
            continue
        ref = m.group(1)
        if ref.startswith("./") or pinned.search(ref) or docker.search(ref):
            continue
        yield Hit(i, f"uses: {ref}")


def _walk_nodes(node: yaml.Node, seen: set[int]) -> Iterator[yaml.Node]:
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            yield from _walk_nodes(k, seen)
            yield from _walk_nodes(v, seen)
    elif isinstance(node, yaml.SequenceNode):
        for v in node.value:
            yield from _walk_nodes(v, seen)


def _root_kind(node: yaml.Node) -> str | None:
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            if isinstance(k, yaml.ScalarNode) and k.value == "kind" and isinstance(v, yaml.ScalarNode):
                return v.value
    return None


_STR_TAG = "tag:yaml.org,2002:str"


def check_declarative_yaml(rule: DeclarativeYamlExpressionFree, src: Source) -> Iterator[Hit]:
    docs = src.yaml_nodes
    if docs is None:
        if _re2(r"(?m)^kind:\s*\S").search(src.text):
            yield Hit(1, "declarative YAML does not parse")
        return
    roots, banned = set(rule.root_kinds), set(rule.banned_kinds)
    for doc in docs:
        if _root_kind(doc) not in roots:
            continue
        for n in _walk_nodes(doc, set()):
            if isinstance(n, yaml.ScalarNode) and n.tag == _STR_TAG and n.value.lstrip().startswith("="):
                yield Hit(n.start_mark.line + 1, "PowerFx expression ('=' prefix)")
            elif isinstance(n, yaml.MappingNode):
                for k, v in n.value:
                    if (isinstance(k, yaml.ScalarNode) and k.value == "kind"
                            and isinstance(v, yaml.ScalarNode) and v.value in banned):
                        yield Hit(v.start_mark.line + 1, f"kind: {v.value}")


def check_max_lines(rule: MaxLines, src: Source) -> Iterator[Hit]:
    n = src.text.count("\n") + (1 if src.text and not src.text.endswith("\n") else 0)
    if n > rule.max:
        yield Hit(n, f"{n} lines > {rule.max}")


CHECKS: dict[str, Any] = {
    "banned_call": check_banned_call,
    "banned_import": check_banned_import,
    "banned_attr_arg": check_banned_attr_arg,
    "banned_text": check_banned_text,
    "gha_banned_trigger": check_gha_banned_trigger,
    "gha_pinned_sha": check_gha_pinned_sha,
    "declarative_yaml_expression_free": check_declarative_yaml,
    "max_lines": check_max_lines,
}


def run_check(rule: LintRule, src: Source) -> Iterator[Hit]:
    return CHECKS[rule.kind](rule, src)
