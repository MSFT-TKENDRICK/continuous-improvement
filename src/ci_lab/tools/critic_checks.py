"""Deterministic critic checks run on an arm's diff BEFORE the LLM critic (design §5, §7).

Checks (each returns human-readable reasons, empty = pass):

* ``path``       every changed path is a valid relative path in the surface and not frozen
* ``component``  each commit's declared ``RRSI-Component`` matches the files it touched
* ``spec``       changed YAML/JSON parses; no ``=`` expressions (safe_mode); pluggable validator hook
* ``leak``       n-gram / identifier leak screen against the test set (case inputs, resource ids, names)
* ``denylist``   eval/judge-targeting vocabulary in added text
* ``tools``      no new tool names or bindings in spec files
* ``size``       file count, per-file and total added-size limits
"""

from __future__ import annotations

import difflib
import functools
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ci_lab.contracts import COMPONENTS
from ci_lab.tools.commit import git, parse_trailers
from ci_lab.tools.paths import PathRejected, components_for, matches_any, normalize_rel

__all__ = [
    "DENYLIST",
    "ArmDiff",
    "CommitInfo",
    "CriticConfig",
    "FileChange",
    "LeakCorpus",
    "case_leak_material",
    "check_components",
    "check_denylist",
    "check_leaks",
    "check_paths",
    "check_sizes",
    "check_specs",
    "check_tool_bindings",
    "collect_diff",
    "diff_text",
    "load_test_set_corpus",
    "run_checks",
    "tool_names",
]

DENYLIST: tuple[tuple[str, re.Pattern[str]], ...] = tuple((name, re.compile(rx, flags)) for name, rx, flags in [
    ("judge", r"\bjudg(?:e|es|ed|ing)\b", re.IGNORECASE),
    ("score", r"\bscor(?:e|es|ed|ing)\b", re.IGNORECASE),
    ("grader", r"\bgrad(?:er|ers|ing)\b", re.IGNORECASE),
    ("evaluator", r"\bevaluat(?:or|ors|ion|ions)\b", re.IGNORECASE),
    ("rubric", r"\brubrics?\b", re.IGNORECASE),
    ("ASSERT", r"\bASSERT\b|\bassert[-_]ai\b", 0),
    ("test case", r"\btest[ _-]?(?:cases?|sets?|suites?)\b", re.IGNORECASE),
    ("benchmark", r"\bbenchmarks?\b", re.IGNORECASE),
    ("judge dimension", r"\b(?:policy_violation|overrefusal)\b", re.IGNORECASE),
])
SPEC_SUFFIXES = (".yaml", ".yml", ".json")
WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


@dataclass(frozen=True)
class FileChange:
    path: str
    status: str  # A, M, D
    before: str | None
    after: str | None

    @property
    def added_lines(self) -> list[str]:
        a = (self.before or "").splitlines()
        b = (self.after or "").splitlines()
        return [ln[2:] for ln in difflib.ndiff(a, b) if ln.startswith("+ ")]

    @property
    def added_text(self) -> str:
        return "\n".join(self.added_lines)


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    component: str | None
    hypothesis: str | None
    files: tuple[str, ...]


@dataclass
class ArmDiff:
    base: str
    head: str
    files: list[FileChange] = field(default_factory=list)
    commits: list[CommitInfo] = field(default_factory=list)
    dirty: list[str] = field(default_factory=list)


def _show(worktree: Path, rev: str, path: str) -> str | None:
    try:
        return git(worktree, "show", f"{rev}:{path}")
    except RuntimeError:
        return None


def collect_diff(worktree: Path | str, base: str, head: str = "HEAD") -> ArmDiff:
    """Read the committed diff ``base..head`` (plus any uncommitted paths) from git."""
    wt = Path(worktree)
    head_sha = git(wt, "rev-parse", head).strip()
    base_sha = git(wt, "rev-parse", base).strip()
    files: list[FileChange] = []
    raw = [x for x in git(wt, "diff", "--name-status", "--no-renames", "-z", base_sha, head_sha).split("\x00") if x]
    for status, path in zip(raw[0::2], raw[1::2], strict=False):
        st = status[:1]
        files.append(FileChange(path, st, None if st == "A" else _show(wt, base_sha, path),
                                None if st == "D" else _show(wt, head_sha, path)))
    commits: list[CommitInfo] = []
    log = git(wt, "log", "--reverse", "--format=%H%x00%B%x1e", f"{base_sha}..{head_sha}")
    for rec in log.split("\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        sha, _, body = rec.partition("\x00")
        trailers = parse_trailers(body)
        touched = git(wt, "diff-tree", "--no-commit-id", "--name-only", "-r", "--no-renames", "-z", "--root",
                      sha).split("\x00")
        commits.append(CommitInfo(sha, (trailers.get("RRSI-Component") or [None])[-1],
                                  (trailers.get("RRSI-Hypothesis") or [None])[-1],
                                  tuple(t for t in touched if t)))
    dirty = [e[3:] for e in git(wt, "status", "--porcelain=v1", "-z", "--untracked-files=all").split("\x00")
             if len(e) > 3]
    return ArmDiff(base_sha, head_sha, files, commits, dirty)


def diff_text(worktree: Path | str, base: str, head: str = "HEAD", max_chars: int = 120_000) -> str:
    text = git(Path(worktree), "diff", "--no-color", "--no-ext-diff", base, head)
    return text if len(text) <= max_chars else text[:max_chars] + f"\n[diff truncated at {max_chars} chars]"


# ---------------------------------------------------------------- leak corpus


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


@dataclass
class LeakCorpus:
    """Test-set material that must not appear in a harness diff."""

    ngrams: set[tuple[str, ...]]
    literals: tuple[str, ...]
    n: int = 8

    @classmethod
    def build(cls, texts: Iterable[str], literals: Iterable[str] = (), n: int = 8) -> LeakCorpus:
        grams: set[tuple[str, ...]] = set()
        for t in texts:
            grams |= _ngrams(t, n)
        lits = tuple(sorted({s.strip() for s in literals if s and len(s.strip()) >= 4}, key=str.lower))
        return cls(grams, lits, n)

    def screen(self, added: str, baseline: str = "") -> list[str]:
        """Leaks in ``added`` that are not already present in ``baseline``."""
        hits: list[str] = []
        new = _ngrams(added, self.n) - _ngrams(baseline, self.n)
        shared = sorted(new & self.ngrams)
        if shared:
            hits.append(f"{len(shared)} {self.n}-gram(s) shared with test cases, e.g. {' '.join(shared[0])!r}")
        low_added, low_base = added.lower(), baseline.lower()
        for lit in self.literals:
            rx = re.compile(r"(?<![\w@.-])" + re.escape(lit.lower()) + r"(?![\w@-])")
            if rx.search(low_added) and not rx.search(low_base):
                hits.append(f"test-set identifier {lit!r}")
        return hits


LEAK_TITLE_MIN_WORDS = 6  # shorter seed titles ("Late boots, no status info") are ordinary phrases


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def case_leak_material(rows: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """``(texts, literals)`` a proposer must not copy from frozen ASSERT test-set rows.

    Texts are every string under ``seed`` plus any expected-output field (``expected*``,
    ``reference``, ``ideal``) for the n-gram screen. Literals are seed titles of at least
    :data:`LEAK_TITLE_MIN_WORDS` words, matched verbatim. ``dimensions`` labels are short
    category names and are not screened."""
    texts: list[str] = []
    literals: list[str] = []
    for row in rows:
        seed = row.get("seed")
        texts += [s for s in _strings(seed) if s.strip()]
        for key, value in row.items():
            if key.startswith("expected") or key in ("reference", "ideal", "ideal_response"):
                texts += [s for s in _strings(value) if s.strip()]
        title = seed.get("title") if isinstance(seed, Mapping) else None
        if isinstance(title, str) and len(_words(title)) >= LEAK_TITLE_MIN_WORDS:
            literals.append(title.strip())
    return texts, literals


@functools.lru_cache(maxsize=16)
def _test_set_corpus(files: tuple[tuple[str, int, int], ...], literals: tuple[str, ...], n: int) -> LeakCorpus:
    rows: list[Mapping[str, Any]] = []
    for path, _size, _mtime in files:
        rows += [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts, titles = case_leak_material(rows)
    return LeakCorpus.build(texts, [*literals, *titles], n)


def load_test_set_corpus(paths: Iterable[Path | str], literals: Iterable[str] = (), n: int = 8) -> LeakCorpus:
    """:class:`LeakCorpus` over frozen ``test_set.jsonl`` files plus extra ``literals``.

    Deterministic (paths and literals are sorted) and cached per (path, size, mtime), so
    repeated campaign wiring re-reads nothing until a test set changes."""
    files = []
    for p in {Path(p).resolve() for p in paths}:
        st = p.stat()
        files.append((str(p), st.st_size, st.st_mtime_ns))
    return _test_set_corpus(tuple(sorted(files)), tuple(sorted({s for s in literals if s})), n)


# ---------------------------------------------------------------- config


SpecValidator = Callable[[str, str], list[str]]  # (path, new_text) -> problems


@dataclass
class CriticConfig:
    surface_globs: Sequence[str]
    component_globs: Mapping[str, Sequence[str]]
    frozen_globs: Sequence[str] = ()
    leak_corpus: LeakCorpus | None = None
    baseline_text: str = ""  # whole incumbent surface text: pre-existing material is not a leak
    denylist: Sequence[tuple[str, re.Pattern[str]]] = DENYLIST
    spec_validator: SpecValidator | None = None
    max_files: int = 12
    max_file_bytes: int = 64 * 1024
    max_added_bytes: int = 24 * 1024
    max_commits: int = 6


# ---------------------------------------------------------------- checks


def check_paths(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    for p in sorted({f.path for f in diff.files} | set(diff.dirty)):
        try:
            normalize_rel(p)
        except PathRejected as exc:
            out.append(f"path: {exc}")
            continue
        if not matches_any(p, cfg.surface_globs):
            out.append(f"path: {p} is outside the editable surface")
        elif matches_any(p, cfg.frozen_globs):
            out.append(f"path: {p} is frozen")
    if diff.dirty:
        out.append(f"path: uncommitted changes left in the worktree: {', '.join(diff.dirty[:10])}")
    return out


def check_components(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    if len(diff.commits) > cfg.max_commits:
        out.append(f"component: {len(diff.commits)} commits exceed the limit of {cfg.max_commits}")
    for c in diff.commits:
        if c.component not in COMPONENTS:
            out.append(f"component: commit {c.sha[:12]} has no valid RRSI-Component trailer ({c.component!r})")
            continue
        if not c.hypothesis:
            out.append(f"component: commit {c.sha[:12]} has no RRSI-Hypothesis trailer")
        for p in c.files:
            found = components_for(p, cfg.component_globs)
            if c.component not in found:
                out.append(f"component: commit {c.sha[:12]} is tagged {c.component!r} but changes {p} "
                           f"({', '.join(sorted(found)) or 'no component'})")
    return out


def _expressions(node: Any, where: str = "") -> list[str]:
    if isinstance(node, str):
        return [where or "<root>"] if node.lstrip().startswith("=") else []
    if isinstance(node, Mapping):
        return [x for k, v in node.items() for x in _expressions(v, f"{where}.{k}" if where else str(k))]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _expressions(v, f"{where}[{i}]")]
    return []


def _parse(path: str, text: str) -> Any:
    return json.loads(text) if path.lower().endswith(".json") else yaml.safe_load(text)


def check_specs(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    for f in diff.files:
        if f.after is None or not f.path.lower().endswith(SPEC_SUFFIXES):
            continue
        try:
            doc = _parse(f.path, f.after)
        except Exception as exc:  # noqa: BLE001
            out.append(f"spec: {f.path} does not parse: {str(exc).splitlines()[0][:200]}")
            continue
        for where in _expressions(doc):
            out.append(f"spec: {f.path} has an expression value at {where} (safe_mode forbids '=')")
        if f.before is not None:
            try:
                before = _parse(f.path, f.before)
            except Exception:  # noqa: BLE001
                before = None
            if isinstance(before, Mapping) and isinstance(doc, Mapping):
                for key in ("kind", "name"):
                    if key in before and before.get(key) != doc.get(key):
                        out.append(f"spec: {f.path} changes {key!r}")
                bm, nm = before.get("model"), doc.get("model")
                if isinstance(bm, Mapping) and isinstance(nm, Mapping):
                    for key in ("id", "provider"):
                        if bm.get(key) != nm.get(key):
                            out.append(f"spec: {f.path} changes model.{key} (model allowlist is not evolvable)")
        if cfg.spec_validator is not None:
            out.extend(f"spec: {f.path}: {p}" for p in cfg.spec_validator(f.path, f.after))
    return out


def check_leaks(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    if cfg.leak_corpus is None:
        return []
    out: list[str] = []
    for f in diff.files:
        if f.after is None:
            continue
        hits = cfg.leak_corpus.screen(f.added_text, "\n".join([f.before or "", cfg.baseline_text]))
        out.extend(f"leak: {f.path}: {h}" for h in hits)
    return out


def check_denylist(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    for f in diff.files:
        if f.after is None:
            continue
        for name, rx in cfg.denylist:
            if len(rx.findall(f.after)) > len(rx.findall(f.before or "")):
                m = rx.search(f.added_text) or rx.search(f.after)
                out.append(f"denylist: {f.path} adds eval-targeting term {name!r} ({m.group(0) if m else name!r})")
    return out


def tool_names(doc: Any, path: str = "") -> set[str]:
    """Tool names and binding names declared anywhere in a spec document."""
    names: set[str] = set()
    if isinstance(doc, Mapping):
        for k, v in doc.items():
            if k == "tools":
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, Mapping):
                            if item.get("name"):
                                names.add(str(item["name"]))
                            names |= tool_names(item)
                        elif isinstance(item, str):
                            names.add(item)
                elif isinstance(v, Mapping):
                    names |= {str(x) for x in v}
            elif k == "bindings":
                for b in v if isinstance(v, list) else [v]:
                    if isinstance(b, Mapping) and b.get("name"):
                        names.add(f"binding:{b['name']}")
                    elif isinstance(b, str):
                        names.add(f"binding:{b}")
            else:
                names |= tool_names(v)
        if Path(path).name.lower().startswith("tool_specs") and "tools" not in doc:
            names |= {str(x) for x in doc}
    elif isinstance(doc, list):
        for item in doc:
            names |= tool_names(item)
    return names


def check_tool_bindings(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    for f in diff.files:
        if f.after is None or not f.path.lower().endswith(SPEC_SUFFIXES):
            continue
        try:
            new = tool_names(_parse(f.path, f.after), f.path)
            old = tool_names(_parse(f.path, f.before), f.path) if f.before is not None else set()
        except Exception:  # noqa: BLE001, S112 - reported by check_specs
            continue
        if added := sorted(new - old):
            out.append(f"tools: {f.path} adds tool(s)/binding(s) {', '.join(added)}; arms may not add tools")
    return out


def check_sizes(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    out: list[str] = []
    if len(diff.files) > cfg.max_files:
        out.append(f"size: {len(diff.files)} files changed; limit {cfg.max_files}")
    total = 0
    for f in diff.files:
        if f.after is not None and len(f.after.encode()) > cfg.max_file_bytes:
            out.append(f"size: {f.path} is {len(f.after.encode())} bytes; limit {cfg.max_file_bytes}")
        total += len(f.added_text.encode())
    if total > cfg.max_added_bytes:
        out.append(f"size: {total} bytes added; limit {cfg.max_added_bytes}")
    if not diff.files:
        out.append("size: empty diff (no committed edits)")
    return out


CHECKS: tuple[Callable[[ArmDiff, CriticConfig], list[str]], ...] = (
    check_paths, check_components, check_specs, check_leaks, check_denylist, check_tool_bindings, check_sizes)


def run_checks(diff: ArmDiff, cfg: CriticConfig) -> list[str]:
    """All deterministic checks; returns the list of rejection reasons (empty = pass)."""
    return [r for check in CHECKS for r in check(diff, cfg)]
