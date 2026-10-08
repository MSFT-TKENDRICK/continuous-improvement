"""``ci-lab lint`` engine: discover files, apply rules, report in poteto lint-arch format."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from ci_lab import obs
from ci_lab.contracts import SPAN_LINT
from ci_lab.lint.checks import Source, run_check
from ci_lab.lint.spec import LintRule, load_rules, rule_files

MAX_FILE_BYTES = 2 * 1024 * 1024
_WALK_SKIP = {".git", ".venv", "node_modules", "__pycache__", "artifacts", ".pytest_cache", "dist", "build"}
Reader = Callable[[str], str | None]


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    path: str
    line: int
    message: str
    fix: str
    see: str
    detail: str = ""


@dataclass
class LintResult:
    findings: list[Finding] = field(default_factory=list)
    files: int = 0
    rules: int = 0
    elapsed_s: float = 0.0

    @property
    def errors(self) -> int:
        return sum(f.severity == "error" for f in self.findings)

    @property
    def warnings(self) -> int:
        return sum(f.severity == "warn" for f in self.findings)

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0


# ---------------------------------------------------------------- globs


@lru_cache(maxsize=1024)
def glob_regex(glob: str) -> re.Pattern[str]:
    """Posix glob → regex: ``**`` spans directories, ``*``/``?`` stay within one, ``{a,b}`` alternation."""
    out, i, n = [], 0, len(glob)
    while i < n:
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "{" and "}" in glob[i:]:
            j = glob.index("}", i)
            out.append("(?:" + "|".join(re.escape(x) for x in glob[i + 1:j].split(",")) + ")")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, globs: Sequence[str]) -> bool:
    return any(glob_regex(g).match(path) for g in globs)


def applies(rule: LintRule, path: str) -> bool:
    return glob_match(path, rule.include) and not glob_match(path, rule.exclude)


# ---------------------------------------------------------------- discovery


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def repo_root(start: Path | None = None) -> Path:
    start = Path(start or Path.cwd())
    out = _git(start, "rev-parse", "--show-toplevel")
    return Path(out.strip()) if out and out.strip() else start.resolve()


def discover_files(root: Path) -> list[str]:
    """Tracked + untracked-not-ignored files (git), else a filtered walk."""
    out = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if out is not None:
        return sorted({p for p in out.split("\0") if p and (root / p).is_file()})
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP]
        rel = Path(dirpath).relative_to(root)
        files += [(rel / f).as_posix() for f in filenames]
    return sorted(files)


def staged_files(root: Path) -> list[str]:
    out = _git(root, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    if out is None:
        raise RuntimeError("git diff --cached failed (not a git repository?)")
    return sorted(p for p in out.split("\0") if p)


def staged_reader(root: Path) -> Reader:
    """Read the *index* content (what will be committed), not the working tree."""

    def read(path: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(root), "show", f":{path}"], capture_output=True,
                               check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode != 0 or len(r.stdout) > MAX_FILE_BYTES:
            return None
        return _decode(r.stdout)

    return read


def _decode(data: bytes) -> str | None:
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fs_reader(root: Path) -> Reader:
    def read(path: str) -> str | None:
        p = root / path
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                return None
            return _decode(p.read_bytes())
        except OSError:
            return None

    return read


def normalize_paths(root: Path, paths: Sequence[str | Path], universe: Sequence[str]) -> list[str]:
    """Explicit ``--paths`` (files or dirs, relative or absolute) → repo-relative posix files."""
    rootr = root.resolve()
    out: set[str] = set()
    for p in paths:
        pp = Path(p)
        pp = (pp if pp.is_absolute() else Path.cwd() / pp).resolve()
        try:
            rel = pp.relative_to(rootr).as_posix()
        except ValueError:
            continue
        if pp.is_dir():
            prefix = "" if rel == "." else rel + "/"
            out.update(f for f in universe if f.startswith(prefix))
        elif pp.is_file():
            out.add(rel)
    return sorted(out)


# ---------------------------------------------------------------- run


def lint(root: Path, rules: Sequence[LintRule], files: Sequence[str], *, reader: Reader | None = None) -> LintResult:
    """Apply ``rules`` to repo-relative ``files``. Pure apart from reading file content."""
    t0 = time.perf_counter()
    read = reader or fs_reader(root)
    res = LintResult(rules=len(rules))
    for path in files:
        active = [r for r in rules if applies(r, path)]
        if not active:
            continue
        text = read(path)
        if text is None:
            continue
        res.files += 1
        src = Source(path, text)
        for rule in active:
            for hit in run_check(rule, src):
                res.findings.append(Finding(rule.id, rule.severity, path, hit.line, rule.message,
                                            rule.fix, rule.see, hit.detail))
    res.findings.sort(key=lambda f: (f.severity != "error", f.path, f.line, f.rule))
    res.elapsed_s = time.perf_counter() - t0
    return res


def run(root: Path, *, staged: bool = False, paths: Sequence[str | Path] = (),
        rule_paths: Sequence[Path] | None = None) -> LintResult:
    """Load ``lint/rules/*.yaml`` and lint the repo (or staged files / explicit paths) in a ci.lint span."""
    root = Path(root)
    with obs.span(SPAN_LINT, {"ci.lint.mode": "staged" if staged else ("paths" if paths else "repo")}):
        rules = load_rules(rule_paths if rule_paths is not None else rule_files(root))
        reader: Reader | None = None
        if staged:
            files = staged_files(root)
            reader = staged_reader(root)
        else:
            files = discover_files(root)
            if paths:
                files = normalize_paths(root, paths, files)
        res = lint(root, rules, files, reader=reader)
        obs.annotate({"ci.lint.files": res.files, "ci.lint.rules": res.rules,
                      "ci.lint.errors": res.errors, "ci.lint.warnings": res.warnings})
        return res


# ---------------------------------------------------------------- output


def format_text(res: LintResult) -> str:
    lines: list[str] = []
    for f in res.findings:
        level = "ERROR" if f.severity == "error" else "WARN"
        lines.append(f"[LINT][{level}] {f.path}:{f.line}")
        detail = f" ({f.detail})" if f.detail else ""
        lines.append(f"  Violation: [{f.rule}] {f.message}{detail}")
        lines.append(f"  Fix: {f.fix}")
        if f.see:
            lines.append(f"  See: {f.see}")
    stats = f"{res.files} file(s), {res.rules} rule(s), {res.elapsed_s:.2f}s"
    if res.errors:
        lines.append(f"[LINT] Failed with {res.errors} error(s), {res.warnings} warning(s) ({stats}).")
    else:
        lines.append(f"[LINT] Passed with {res.warnings} warning(s) ({stats}).")
    return "\n".join(lines)


def format_json(res: LintResult) -> str:
    return json.dumps({"schema_version": 1, "errors": res.errors, "warnings": res.warnings,
                       "files": res.files, "rules": res.rules, "elapsed_s": round(res.elapsed_s, 3),
                       "findings": [asdict(f) for f in res.findings]}, indent=2, sort_keys=True)
