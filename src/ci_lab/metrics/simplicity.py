"""Surface complexity of a harness tree and the simplicity score derived from it (pure, deterministic).

``surface_metrics`` walks a harness directory (never following symlinks; skipping ``__pycache__``,
dotfiles, binary files and the frozen ``exclude`` paths) and counts:

- ``files``: counted text files;
- ``prompt_tokens``: ``ceil(chars / 4)`` of every ``*.md`` file plus every YAML ``instructions:`` string;
- ``yaml_nodes``: values (mappings, sequences, scalars; keys are not counted) of every YAML document;
  an unparseable YAML file counts one node per non-blank line (no discount for breaking a file);
- ``agents``: YAML documents with ``kind: Prompt`` or ``kind: Agent``;
- ``tools``: entries of every ``tools`` list/mapping inside agent documents and in YAML files under a
  ``tools/`` directory (a ``tools/`` document without a ``tools`` key counts its top-level entries,
  excluding ``format``/``version``);
- ``workflow_steps``: items of every (nested) ``actions`` list inside ``kind: Workflow`` documents;
- ``mcp_servers``: entries of ``servers``/``mcpServers`` in YAML/JSON files under an ``mcp/`` directory;
- ``skill_lines``: non-blank lines of files under a ``skills/`` directory;
- ``py_loc``: non-blank, non-comment lines of ``*.py``;
- ``py_cyclomatic``: decision points (``if``/``elif``, loops, ``except``, ``match`` cases, boolean
  operators, conditional expressions, comprehension clauses) plus 1 per function; an unparseable
  file counts one per non-blank line;
- ``complexity``: the weighted composite of :data:`COMPLEXITY_WEIGHTS`::

      prompt_tokens/50 + yaml_nodes/20 + agents*5 + tools*2 + workflow_steps
      + mcp_servers*3 + skill_lines/10 + py_loc/10 + py_cyclomatic

- ``component.<name>.complexity`` for each component of ``component_globs`` (globs are matched against
  the path relative to ``harness_dir`` and against the same path prefixed with the directory name, so
  both ``prompts/**`` and repo-relative ``harness/prompts/**`` work).
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ci_lab.gitops.safe_path import match_globs

__all__ = ["COMPLEXITY_WEIGHTS", "COUNT_KEYS", "complexity", "relative_change", "simplicity_score",
           "surface_metrics"]

COMPLEXITY_WEIGHTS: Mapping[str, float] = {
    "prompt_tokens": 1 / 50,
    "yaml_nodes": 1 / 20,
    "agents": 5.0,
    "tools": 2.0,
    "workflow_steps": 1.0,
    "mcp_servers": 3.0,
    "skill_lines": 1 / 10,
    "py_loc": 1 / 10,
    "py_cyclomatic": 1.0,
}
COUNT_KEYS = ("files", "prompt_tokens", "yaml_nodes", "agents", "tools", "workflow_steps", "mcp_servers",
              "skill_lines", "py_loc", "py_cyclomatic")
_AGENT_KINDS = frozenset({"Prompt", "Agent"})
_YAML = (".yaml", ".yml")
_SNIFF = 8192


def complexity(counts: Mapping[str, float]) -> float:
    """The documented weighted composite of :data:`COMPLEXITY_WEIGHTS` over ``counts``."""
    return float(sum(w * float(counts.get(k, 0.0)) for k, w in COMPLEXITY_WEIGHTS.items()))


def relative_change(new: float, base: float) -> float:
    """``(new - base) / max(base, 1.0)``."""
    return (float(new) - float(base)) / max(float(base), 1.0)


def simplicity_score(new: Mapping[str, float], base: Mapping[str, float]) -> float:
    """Piecewise linear in ``d = relative_change(new.complexity, base.complexity)``:
    1.0 at ``d <= -0.10``, 0.5 at ``d == 0``, 0.0 at ``d >= +0.25``."""
    d = relative_change(new["complexity"], base["complexity"])
    if d <= -0.10:
        return 1.0
    if d <= 0.0:
        return 0.5 + 0.5 * (-d / 0.10)
    if d >= 0.25:
        return 0.0
    return 0.5 * (1.0 - d / 0.25)


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _nonblank(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:_SNIFF]:
        return None
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return None


def _walk(root: Path, exclude: Sequence[str]) -> Iterator[tuple[str, Path]]:
    """``(posix rel, path)`` of regular, non-hidden, non-excluded files; symlinks are never followed."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__"
                             and not (here / d).is_symlink())
        for name in sorted(filenames):
            p = here / name
            if name.startswith(".") or p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if exclude and match_globs(rel, exclude):
                continue
            yield rel, p


def _count_values(node: Any) -> int:
    if isinstance(node, Mapping):
        return 1 + sum(_count_values(v) for v in node.values())
    if isinstance(node, list):
        return 1 + sum(_count_values(v) for v in node)
    return 1


def _find(node: Any, key: str) -> Iterator[Any]:
    """Every value stored under ``key`` anywhere in ``node``."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _find(v, key)


def _size(v: Any) -> int:
    return len(v) if isinstance(v, (list, Mapping)) else 0


def _instructions(node: Any) -> Iterator[str]:
    yield from (v for v in _find(node, "instructions") if isinstance(v, str))


def _actions(node: Any) -> int:
    return sum(len(v) for v in _find(node, "actions") if isinstance(v, list))


class _Cyclomatic(ast.NodeVisitor):
    _ONE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.If, ast.For, ast.AsyncFor, ast.While,
            ast.IfExp, ast.ExceptHandler, ast.Assert, ast.match_case)

    def __init__(self) -> None:
        self.total = 0

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._ONE):
            self.total += 1
        elif isinstance(node, ast.BoolOp):
            self.total += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            self.total += 1 + len(node.ifs)
        super().generic_visit(node)


def _python(text: str) -> tuple[int, int]:
    lines = [ln for ln in _nonblank(text) if not ln.lstrip().startswith("#")]
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return len(lines), len(lines)
    v = _Cyclomatic()
    v.visit(tree)
    return len(lines), v.total


def _yaml_docs(text: str) -> list[Any] | None:
    try:
        return [d for d in yaml.safe_load_all(text) if d is not None]
    except yaml.YAMLError:
        return None


def _file_counts(rel: str, text: str) -> dict[str, float]:
    parts = rel.split("/")[:-1]
    suffix = Path(rel).suffix.lower()
    c = dict.fromkeys(COUNT_KEYS, 0.0)
    c["files"] = 1.0
    if "skills" in parts:
        c["skill_lines"] += len(_nonblank(text))
    if suffix == ".md":
        c["prompt_tokens"] += _tokens(text)
    elif suffix == ".py":
        loc, cyc = _python(text)
        c["py_loc"] += loc
        c["py_cyclomatic"] += cyc
    elif suffix in _YAML:
        docs = _yaml_docs(text)
        if docs is None:
            c["yaml_nodes"] += len(_nonblank(text))
            return c
        for doc in docs:
            c["yaml_nodes"] += _count_values(doc)
            c["prompt_tokens"] += sum(_tokens(s) for s in _instructions(doc))
            kind = doc.get("kind") if isinstance(doc, Mapping) else None
            in_tools_dir = "tools" in parts
            if kind in _AGENT_KINDS or in_tools_dir:
                found = list(_find(doc, "tools"))
                if found:
                    c["tools"] += sum(_size(v) for v in found)
                elif in_tools_dir and isinstance(doc, Mapping):
                    c["tools"] += sum(1 for k in doc if k not in ("format", "version"))
            if kind in _AGENT_KINDS:
                c["agents"] += 1
            if kind == "Workflow":
                c["workflow_steps"] += _actions(doc)
            if "mcp" in parts and isinstance(doc, Mapping):
                c["mcp_servers"] += sum(_size(doc.get(k)) for k in ("servers", "mcpServers"))
    elif suffix == ".json" and "mcp" in parts:
        try:
            doc = json.loads(text)
        except ValueError:
            return c
        if isinstance(doc, Mapping):
            c["mcp_servers"] += sum(_size(doc.get(k)) for k in ("servers", "mcpServers"))
    return c


def _sum(counts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    out = dict.fromkeys(COUNT_KEYS, 0.0)
    for c in counts:
        for k in COUNT_KEYS:
            out[k] += c[k]
    out["complexity"] = complexity(out)
    return out


def surface_metrics(harness_dir: Path, component_globs: Mapping[str, Sequence[str]] | None = None, *,
                    exclude: Sequence[str] = ("harness.yaml",)) -> dict[str, float]:
    """Surface metrics of ``harness_dir`` (see the module docstring). ``exclude`` holds globs relative to
    ``harness_dir`` of frozen files that are not candidate surface (default: the ``harness.yaml`` manifest)."""
    root = Path(harness_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"harness dir not found: {root}")
    per_file: dict[str, dict[str, float]] = {}
    for rel, path in _walk(root, tuple(exclude)):
        text = _read_text(path)
        if text is not None:
            per_file[rel] = _file_counts(rel, text)
    out = _sum(per_file.values())
    for name, globs in sorted((component_globs or {}).items()):
        globs = tuple(globs)
        prefixed = f"{root.name}/"
        hit = [c for rel, c in per_file.items() if match_globs(rel, globs) or match_globs(prefixed + rel, globs)]
        out[f"component.{name}.complexity"] = _sum(hit)["complexity"]
    return out
