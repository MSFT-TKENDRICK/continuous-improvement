"""``commit_edit``: one tagged commit per proposer edit (RRSI component attribution)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ci_lab.contracts import TEXT_COMPONENTS, Edit
from ci_lab.tools.paths import PathRejected, components_for, matches_any, normalize_rel

__all__ = ["CO_AUTHOR", "changed_paths", "edits_since", "git", "make_commit_tool", "parse_trailers"]

CO_AUTHOR = "Copilot App <223556219+Copilot@users.noreply.github.com>"
MAX_HYPOTHESIS = 400
BOT_NAME = "ci-lab proposer"
BOT_EMAIL = "ci-lab-proposer@users.noreply.github.com"


def git(worktree: Path, *args: str, input: str | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(["git", "-C", str(worktree), *args], input=input, capture_output=True,
                          text=True, encoding="utf-8", env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def changed_paths(worktree: Path) -> list[str]:
    """Paths changed vs HEAD (staged, unstaged and untracked), posix, both sides of renames."""
    out = git(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
    paths: list[str] = []
    for entry in out.split("\x00"):
        if len(entry) > 3:
            paths.append(entry[3:])
    return sorted(set(paths))


def parse_trailers(message: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in message.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key and " " not in key:
            out.setdefault(key, []).append(value.strip())
    return out


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def make_commit_tool(worktree: Path | str, max_edits: int, *, component_globs: Mapping[str, Sequence[str]],
                     experiment_id: str, variant: str, surface_globs: Sequence[str] = (),
                     frozen_globs: Sequence[str] = (),
                     allowed_components: Sequence[str] | None = None) -> Callable[[str, str], str]:
    """Bind ``commit_edit(component, hypothesis)`` to an arm worktree.

    Every changed path must belong to ``component`` (per ``component_globs``), lie in the
    surface and not in the frozen globs; at most ``max_edits`` commits. The committed
    :class:`~ci_lab.contracts.Edit` records are kept on ``commit_edit.edits``.
    """
    wt = Path(worktree).resolve(strict=True)
    edits: list[Edit] = []

    def commit_edit(component: str, hypothesis: str) -> str:
        """Commit all current file changes as ONE edit to a single harness component.

        component: one of prompt, skill, client_tool, config, memory, context_mgmt.
        hypothesis: one sentence: what failure this edit should fix and why.
        """
        if component not in TEXT_COMPONENTS:
            return f"ERROR: unknown component {component!r}; expected one of {', '.join(TEXT_COMPONENTS)}"
        if allowed_components is not None and component not in allowed_components:
            return f"ERROR: this arm may only edit {', '.join(allowed_components)}, not {component!r}"
        hyp = _one_line(hypothesis)
        if not hyp:
            return "ERROR: hypothesis must be a non-empty sentence"
        if len(hyp) > MAX_HYPOTHESIS:
            return f"ERROR: hypothesis is {len(hyp)} chars; limit {MAX_HYPOTHESIS}"
        if len(edits) >= max_edits:
            return f"ERROR: edit budget exhausted ({max_edits} commits); call your submit tool"
        try:
            paths = changed_paths(wt)
        except RuntimeError as exc:
            return f"ERROR: {exc}"
        if not paths:
            return "ERROR: nothing to commit; write files first"
        bad: list[str] = []
        for p in paths:
            try:
                normalize_rel(p)
            except PathRejected as exc:
                bad.append(f"{p}: {exc}")
                continue
            if surface_globs and not matches_any(p, surface_globs):
                bad.append(f"{p}: outside the editable surface")
            elif matches_any(p, frozen_globs):
                bad.append(f"{p}: frozen")
            elif component not in (found := components_for(p, component_globs)):
                bad.append(f"{p}: belongs to {sorted(found) or 'no component'}, not {component!r}")
        if bad:
            return "ERROR: cannot commit as one " + component + " edit:\n" + "\n".join(bad)
        message = "\n".join([
            f"{component}: {hyp[:60]}{'...' if len(hyp) > 60 else ''}",
            "",
            f"RRSI-Component: {component}",
            f"RRSI-Hypothesis: {hyp}",
            f"OES-Experiment: {experiment_id}",
            f"OES-Variant: {variant}",
            f"Co-authored-by: {CO_AUTHOR}",
            "",
        ])
        try:
            git(wt, "add", "-A", "--", *paths)
            git(wt, "-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}", "-c", "commit.gpgsign=false",
                "commit", "-q", "--no-verify", "-F", "-", input=message)
            sha = git(wt, "rev-parse", "HEAD").strip()
        except RuntimeError as exc:
            return f"ERROR: {exc}"
        edits.append(Edit(component=component, hypothesis=hyp, files=tuple(paths), commit=sha))
        return f"committed {sha[:12]} ({component}; {len(paths)} file(s); {len(edits)}/{max_edits} edits)"

    commit_edit.edits = edits  # type: ignore[attr-defined]
    return commit_edit


def edits_since(worktree: Path | str, base: str, head: str = "HEAD") -> list[Edit]:
    """Edits recorded by ``commit_edit`` on ``base..head`` (oldest first), from commit trailers."""
    wt = Path(worktree)
    shas = git(wt, "rev-list", "--reverse", f"{base}..{head}").split()
    out: list[Edit] = []
    for sha in shas:
        trailers = parse_trailers(git(wt, "show", "-s", "--format=%B", sha))
        files = tuple(sorted(f for f in git(wt, "diff-tree", "--no-commit-id", "--name-only", "-r", "--no-renames",
                                            "-z", sha).split("\x00") if f))
        out.append(Edit(component=(trailers.get("RRSI-Component") or [""])[0],
                        hypothesis=(trailers.get("RRSI-Hypothesis") or [""])[0], files=files, commit=sha))
    return out
