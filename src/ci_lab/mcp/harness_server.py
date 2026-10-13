"""Read-only harness introspection MCP server (stdio, FastMCP).

``python -m ci_lab.mcp.harness_server --root <harness dir> --runs-root <runs dir>``

Every path argument is resolved and must stay under its configured root (``read_component`` under
``--root``; ``trace_summary``/``eval_summary`` under ``--runs-root``); symlinks are refused. Trace and eval
summaries return typed counts only, never raw tool output or transcripts.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

MAX_FILE_BYTES = 256_000
MAX_CHARS = 50_000
MAX_SCAN_FILES = 2_000
_LABEL = re.compile(r"[^A-Za-z0-9_.:\-]")


def _visible(rel: Path) -> bool:
    return not any(p.startswith(".") or p == "__pycache__" for p in rel.parts)


def contained(root: Path, path: str | Path) -> Path:
    """``path`` (relative to ``root`` or absolute) resolved, if it stays under ``root`` with no symlinks."""
    root = root.resolve()
    p = Path(path)
    cand = p if p.is_absolute() else root / p
    resolved = cand.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes its root: {path}")
    cur = root
    for part in resolved.relative_to(root).parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
    return resolved


def _files(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if _visible(rel) and p.is_file() and not p.is_symlink():
            out.append(p)
            if len(out) >= MAX_SCAN_FILES:
                break
    return out


def _label(v: Any) -> str:
    return _LABEL.sub("_", v)[:64] if isinstance(v, str) and v else "other"


def _component_globs(root: Path) -> dict[str, list[str]]:
    manifest = root / "harness.yaml"
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
    except (OSError, yaml.YAMLError):
        data = None
    comps = data.get("components") if isinstance(data, dict) else None
    if not isinstance(comps, dict):
        return {}
    return {str(k): [g for g in (v if isinstance(v, list) else [v]) if isinstance(g, str)] for k, v in comps.items()}


def _component_of(rel: str, globs: dict[str, list[str]]) -> str:
    for name, pats in globs.items():
        if any(fnmatch.fnmatch(rel, g.removeprefix("harness/")) for g in pats):
            return name
    return rel.split("/", 1)[0] if "/" in rel else "root"


def _local_metrics(root: Path) -> dict[str, float]:
    globs, out = _component_globs(root), Counter()
    for f in _files(root):
        rel = f.relative_to(root).as_posix()
        text = f.read_bytes().decode("utf-8", errors="replace")
        lines, comp = text.count("\n") + (1 if text and not text.endswith("\n") else 0), _component_of(rel, globs)
        for key, n in (("files", 1), ("lines", lines), ("chars", len(text))):
            out[key] += n
            out[f"component.{comp}.{key}"] += n
    return {k: float(v) for k, v in sorted(out.items())}


def _summarize_scores(scores: list[Any]) -> dict[str, Any]:
    by_suite: dict[str, list[float]] = {}
    violations: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    n = completed = 0
    for s in scores:
        if not isinstance(s, dict):
            continue
        n += 1
        suite = _label(s.get("suite"))
        score = s.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool):
            completed += 1
            by_suite.setdefault(suite, []).append(float(score))
        else:
            by_suite.setdefault(suite, [])
        for v in s.get("violations") or ():
            violations[_label(v.get("rule_id") if isinstance(v, dict) else v)] += 1
        for k in ("tokens_in", "tokens_out", "wall_ms", "llm_calls", "tool_calls"):
            if isinstance(s.get(k), int | float) and not isinstance(s.get(k), bool):
                totals[k] += s[k]
    allv = [x for xs in by_suite.values() for x in xs]
    return {
        "n": n, "completed": completed,
        "mean_score": sum(allv) / len(allv) if allv else None,
        "by_suite": {k: {"n": len(v), "mean": sum(v) / len(v) if v else None} for k, v in sorted(by_suite.items())},
        "violations": dict(violations.most_common(50)),
        "totals": {k: float(v) for k, v in sorted(totals.items())},
    }


def build_server(root: str | Path, runs_root: str | Path) -> FastMCP:
    root, runs_root = Path(root).resolve(), Path(runs_root).resolve()
    mcp = FastMCP("harness", instructions="Read-only introspection of the evolvable harness tree and run artifacts.")

    @mcp.tool()
    def list_components() -> dict[str, Any]:
        """List the harness tree's files grouped by component (paths relative to the harness root)."""
        globs, comps = _component_globs(root), {}
        files = _files(root) if root.is_dir() else []
        for f in files:
            rel = f.relative_to(root).as_posix()
            comps.setdefault(_component_of(rel, globs), []).append(rel)
        return {"files": len(files), "components": comps}

    @mcp.tool()
    def read_component(path: str, max_chars: int = 20_000) -> dict[str, Any]:
        """Read one UTF-8 text file of the harness tree. `path` is relative to the harness root."""
        p = contained(root, path)
        if not p.is_file():
            raise ValueError(f"not a file: {path}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes: {path}")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"not a UTF-8 text file: {path}") from e
        cap = max(1, min(int(max_chars), MAX_CHARS))
        return {"path": p.relative_to(root).as_posix(), "chars": len(text), "truncated": len(text) > cap,
                "text": text[:cap]}

    @mcp.tool()
    def component_metrics() -> dict[str, Any]:
        """Surface metrics of the harness tree (ci_lab.metrics when available, else file/line/char counts)."""
        try:
            from ci_lab.metrics.simplicity import surface_metrics
        except ImportError:
            return {"source": "local", "metrics": _local_metrics(root)}
        return {"source": "ci_lab.metrics", "metrics": dict(surface_metrics(root))}

    @mcp.tool()
    def trace_summary(run_dir: str) -> dict[str, Any]:
        """Typed counts over a run directory's JSON/JSONL records (never raw content). `run_dir` must be
        under the runs root."""
        d = contained(runs_root, run_dir)
        if not d.is_dir():
            raise ValueError(f"not a directory: {run_dir}")
        suffixes: Counter[str] = Counter()
        types: Counter[str] = Counter()
        records = errors = bad = 0
        files = _files(d)
        for f in files:
            suffixes[f.suffix.lower() or "none"] += 1
            if f.suffix.lower() not in (".jsonl", ".json"):
                continue
            raw = f.read_bytes().decode("utf-8", errors="replace")
            chunks = raw.splitlines() if f.suffix.lower() == ".jsonl" else [raw]
            for line in chunks:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                records += 1
                if isinstance(rec, dict):
                    types[_label(next((rec[k] for k in ("type", "kind", "event", "name") if k in rec), None))] += 1
                    errors += bool(rec.get("error")) or rec.get("status") in ("error", "failed")
        return {"files": len(files), "by_suffix": dict(sorted(suffixes.items())), "records": records,
                "by_type": dict(types.most_common(50)), "error_records": int(errors), "bad_lines": bad}

    @mcp.tool()
    def eval_summary(path: str) -> dict[str, Any]:
        """Aggregate an eval result JSON (``{"scores": [...]}`` or a list of task scores) under the runs
        root: counts, mean score per suite, violation rule ids and resource totals."""
        p = contained(runs_root, path)
        if not p.is_file():
            raise ValueError(f"not a file: {path}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"not a JSON file: {path}") from e
        scores = data.get("scores") if isinstance(data, dict) else data
        if not isinstance(scores, list):
            raise TypeError("expected {'scores': [...]} or a list of task scores")
        return _summarize_scores(scores)

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ci_lab.mcp.harness_server")
    ap.add_argument("--root", required=True, help="harness tree root")
    ap.add_argument("--runs-root", required=True, help="directory that trace/eval paths must stay under")
    args = ap.parse_args(argv)
    build_server(args.root, args.runs_root).run("stdio")


if __name__ == "__main__":
    main()
