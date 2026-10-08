"""Prose-deletion proposer (design 13.6 N5).

Once a lesson is *enforced* by guards, prose in the skill/prompt that merely restates the guard's
trusted remediation is mechanically redundant. This module deletes **only** such sentences:

* a sentence is a candidate iff its normalized content tokens are a subset of the tokens of the
  lesson's enforced rules' rendered template ``message``/``fix`` (plus their slot values), and it has
  at least :data:`MIN_TOKENS` content tokens;
* sentences carrying policy intent or recovery guidance (:data:`KEEP_CUES`) are always kept, as are
  headings, code fences and at least one sentence per anchored section (the lesson's intent stays);
* the lesson stays in the registry (N5); nothing here retires it.

The output feeds a *deletion arm*, which (documented requirement, enforced by the campaign) must pass
on OOD cases and on >= :data:`MIN_MODEL_PINS` model pins (C4) with an automatic rollback canary.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ci_lab.contracts import Edit
from ci_lab.rulespec import LessonEntry, RuleSpec, TemplateSpec

from .strategy import COMMIT_TRAILER, Committer, git_commit
from .templates import render, trusted_templates

MIN_TOKENS = 3
MIN_MODEL_PINS = 2
DELETION_REQUIREMENTS = ("ood", f"model_pins>={MIN_MODEL_PINS}", "rollback_canary")
KEEP_CUES = frozenset({
    # policy intent
    "because", "why", "reason", "policy", "intent", "purpose", "so", "must", "never", "always", "only",
    "required", "prohibited", "forbidden", "legal", "compliance", "privacy", "fraud", "safety",
    # recovery guidance
    "if", "unless", "otherwise", "instead", "escalate", "apologize", "explain", "ask", "offer", "handoff",
    "transfer", "fallback", "recover", "when",
})
STOPWORDS = frozenset({
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "at", "by", "with", "this", "that", "these",
    "those", "it", "its", "is", "are", "be", "was", "were", "been", "first", "then", "before", "after", "you",
    "your", "please", "do", "does", "call", "calling", "called", "use", "same", "exact", "any",
})
_WORD = re.compile(r"[a-z0-9_]+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*_\[(])")
_BULLET = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")


class ProseError(ValueError):
    pass


def _stem(t: str) -> str:
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and "_" not in t:
        return t[:-1]
    return t


def tokens(text: str) -> set[str]:
    words = _WORD.findall(text.lower().replace("-", "_"))
    return {_stem(w) for w in words if w not in STOPWORDS}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@dataclass(frozen=True)
class Deletion:
    anchor: str
    line: int  # 1-based line in the original file
    sentence: str


@dataclass
class ProseProposal:
    lesson_id: str
    deletions: list[Deletion] = field(default_factory=list)
    new_text: dict[str, str] = field(default_factory=dict)  # rel path -> rewritten file
    diff: str = ""
    kept: int = 0

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted(self.new_text))

    def hypothesis(self) -> str:
        return (f"Lesson {self.lesson_id} is enforced by guards; delete {len(self.deletions)} sentence(s) that only "
                f"restate the guard's trusted remediation (N5). Requires: OOD pass, >= {MIN_MODEL_PINS} model "
                f"pins (C4), rollback canary. Lesson stays in the registry.")


def _section(lines: list[str], heading: str) -> tuple[int, int]:
    want = _slug(heading)
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m and _slug(m.group(2)) == want:
            level = len(m.group(1))
            end = len(lines)
            for j in range(i + 1, len(lines)):
                n = _HEADING.match(lines[j])
                if n and len(n.group(1)) <= level:
                    end = j
                    break
            return i + 1, end
    raise ProseError(f"heading {heading!r} not found")


def redundant_vocabulary(rules: Sequence[RuleSpec], catalog: Mapping[str, TemplateSpec] | None = None) -> set[str]:
    catalog = catalog if catalog is not None else trusted_templates()[0]
    vocab: set[str] = set()
    for r in rules:
        tpl = catalog.get(r.template)
        if tpl is None:
            raise ProseError(f"rule {r.id}: template {r.template!r} not in trusted catalog")
        msg, fix = render(tpl, r.slots)
        idents = " ".join([*(str(v) for v in r.slots.values()), r.target])
        vocab |= tokens(msg) | tokens(fix) | tokens(idents) | tokens(idents.replace("_", " "))
    return vocab


def is_redundant(sentence: str, vocab: set[str]) -> bool:
    toks = tokens(sentence)
    if len(toks) < MIN_TOKENS or not toks <= vocab:
        return False
    return not (set(_WORD.findall(sentence.lower())) & KEEP_CUES)


def _split_sentences(line: str) -> tuple[str, list[str]]:
    m = _BULLET.match(line)
    prefix = m.group(1) if m else ""
    body = line[len(prefix):]
    return prefix, [s for s in _SENT.split(body.strip()) if s]


def propose_deletions(entry: LessonEntry, rules: Sequence[RuleSpec], *, root: Path,
                      catalog: Mapping[str, TemplateSpec] | None = None) -> ProseProposal:
    """Deterministic N5 proposal for ``entry`` (must be ``enforced``; its rules must be enforce-mode)."""
    if entry.status != "enforced":
        raise ProseError(f"lesson {entry.lesson_id} is {entry.status}; only enforced lessons lose prose")
    mine = [r for r in rules if r.id in set(entry.rule_ids)]
    missing = set(entry.rule_ids) - {r.id for r in mine}
    if not mine or missing:
        raise ProseError(f"lesson {entry.lesson_id}: rules not found: {sorted(missing) or entry.rule_ids}")
    shadow = [r.id for r in mine if r.mode != "enforce"]
    if shadow:
        raise ProseError(f"lesson {entry.lesson_id}: rules still in shadow: {shadow}")
    vocab = redundant_vocabulary(mine, catalog)
    prop = ProseProposal(entry.lesson_id)
    by_file: dict[str, list[str]] = {}
    for anchor in entry.prose_anchors:
        rel, _, heading = anchor.partition("#")
        path = (Path(root) / rel).resolve()
        if Path(root).resolve() not in path.parents:
            raise ProseError(f"anchor {anchor!r} escapes the harness root")
        if rel not in by_file:
            by_file[rel] = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines = by_file[rel]
        start, end = _section(lines, heading) if heading else (0, len(lines))
        removed: list[tuple[int, str]] = []
        survivors = 0
        fenced = False
        plans: dict[int, str | None] = {}
        for i in range(start, end):
            raw = lines[i]
            text = raw.rstrip("\r\n")
            if text.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced or not text.strip() or _HEADING.match(text):
                continue
            prefix, sents = _split_sentences(text)
            keep = [s for s in sents if not is_redundant(s, vocab)]
            drop = [s for s in sents if s not in keep]
            survivors += len(keep)
            if drop:
                removed += [(i, s) for s in drop]
                eol = raw[len(text):]
                plans[i] = (prefix + " ".join(keep) + eol) if keep else None
        if survivors == 0 and removed:  # never empty a section: keep its first redundant sentence
            i0, _ = removed.pop(0)
            plans.pop(i0, None)
            removed = [(i, s) for i, s in removed if i != i0]
        for i, new in plans.items():
            lines[i] = "" if new is None else new
        prop.deletions += [Deletion(anchor, i + 1, s) for i, s in removed]
        prop.kept += survivors
    for rel, lines in by_file.items():
        old = (Path(root) / rel).read_text(encoding="utf-8")
        new = "".join(lines)
        if new != old:
            prop.new_text[rel] = new
            prop.diff += "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                                      fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return prop


def component_for(rel: str, component_globs: Mapping[str, Iterable[str]] | None = None) -> str:
    from fnmatch import fnmatch

    for comp, globs in (component_globs or {}).items():
        if any(fnmatch(rel, g) for g in globs):
            return comp
    return "prompt" if "prompt" in rel.lower() else "skill"


def apply_deletion(prop: ProseProposal, worktree: Path, *, component_globs: Mapping[str, Iterable[str]] | None = None,
                   committer: Committer = git_commit) -> Edit:
    """Write ``prop`` into ``worktree`` and commit it as one deletion-arm :class:`Edit`."""
    if not prop.new_text:
        raise ProseError(f"lesson {prop.lesson_id}: nothing to delete")
    for rel, text in prop.new_text.items():
        (Path(worktree) / rel).write_text(text, encoding="utf-8", newline="")
    comps = {component_for(rel, component_globs) for rel in prop.files}
    hypothesis = prop.hypothesis()
    sha = committer(Path(worktree), list(prop.files),
                    f"prose: N5 deletion for {prop.lesson_id}\n\n{hypothesis}\n\n{COMMIT_TRAILER}")
    return Edit(component=min(comps), hypothesis=hypothesis, files=prop.files, commit=sha)
