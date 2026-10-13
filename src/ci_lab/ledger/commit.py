"""Ledger commits that never touch the user's index or checkout (design §3, C6).

``ledger_commit`` stages the given ``experiments/**`` paths from the working tree into a
private temporary index (``GIT_INDEX_FILE``) seeded from the target ref, writes a tree,
creates the commit with ``commit-tree`` and moves the ref with a compare-and-swap
``update-ref <ref> <new> <old>``. Anything outside ``experiments/**`` is refused, both
before staging (requested paths) and after (the tree diff against the parent).

Note: when the target ref is the branch checked out in ``repo``, its HEAD moves but its
index is left alone, so ``git status`` there shows the ledger files as changed relative
to the new HEAD until the user refreshes (by design: we never write the user's index).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ci_lab.gitops import git
from ci_lab.ledger.layout import LEDGER_ROOT
from ci_lab.ledger.lock import DEFAULT_TIMEOUT, ledger_lock


class LedgerPathError(ValueError):
    """A path outside ``experiments/**`` was requested or would be committed."""


class LedgerConflict(RuntimeError):
    """The target ref moved (compare-and-swap failed)."""

    def __init__(self, ref: str, expected: str | None, actual: str | None) -> None:
        self.ref, self.expected, self.actual = ref, expected, actual
        super().__init__(f"{ref} moved: expected {expected or '<absent>'}, found {actual or '<absent>'}")


@dataclass(frozen=True)
class LedgerCommit:
    ref: str
    commit: str
    parent: str | None
    tree: str
    created: bool  # False when nothing changed (no commit was made)


def _full_ref(ref: str) -> str:
    full = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
    git.check_rev(full)
    if git.run(None, "check-ref-format", full, check=False).returncode != 0:
        raise ValueError(f"invalid ref {ref!r}")
    return full


def is_ledger_path(rel: str, root: str = LEDGER_ROOT) -> bool:
    parts = rel.split("/")
    return len(parts) >= 2 and parts[0] == root and all(p not in ("", ".", "..") for p in parts)


def _rel_paths(top: Path, paths: Iterable[str | os.PathLike[str]], root: str) -> list[str]:
    rels: list[str] = []
    for p in paths:
        raw = Path(p)
        if raw.is_absolute():
            try:
                rel = raw.resolve().relative_to(top).as_posix()
            except ValueError as exc:
                raise LedgerPathError(f"{p} is outside the repository") from exc
        else:
            if ".." in os.fspath(p).replace("\\", "/").split("/"):
                raise LedgerPathError(f"refusing path with '..': {p}")
            rel = git.norm_relpath(p)
        if rel != root and not is_ledger_path(rel, root):
            raise LedgerPathError(f"refusing non-ledger path {rel!r} (only {root}/** may be committed)")
        rels.append(rel)
    if not rels:
        raise LedgerPathError("no paths to commit")
    return sorted(set(rels))


def ledger_commit(repo: str | os.PathLike[str], paths: Iterable[str | os.PathLike[str]], message: str, *,
                  ref: str = "refs/heads/main", expected_old: str | None = None, allow_empty: bool = False,
                  ledger_root: str = LEDGER_ROOT, lock_timeout: float = DEFAULT_TIMEOUT) -> LedgerCommit:
    """Commit working-tree ``paths`` (files or directories under ``experiments/``) onto ``ref``.

    ``expected_old`` pins the parent; when omitted the ref's current value is used. Either
    way the ref update is a CAS, so a concurrent writer yields :class:`LedgerConflict`.
    A missing path that exists in the parent tree is committed as a deletion."""
    if not message or not message.strip():
        raise ValueError("empty commit message")
    top = git.toplevel(repo)
    full_ref = _full_ref(ref)
    rels = _rel_paths(top, paths, ledger_root)

    with ledger_lock(top, timeout=lock_timeout):
        current = git.ref_value(top, full_ref)
        if expected_old is not None and current != expected_old:
            raise LedgerConflict(full_ref, expected_old, current)
        parent = current
        if parent is not None and git.rev_parse(top, parent, kind="commit") != parent:
            raise ValueError(f"{full_ref} does not point at a commit")

        index = git.common_dir(top) / "ci-lab" / f"index-{uuid.uuid4().hex}"
        index.parent.mkdir(parents=True, exist_ok=True)
        env = {"GIT_INDEX_FILE": str(index), "GIT_LITERAL_PATHSPECS": "1"}
        try:
            if parent:
                git.run(top, "read-tree", parent, env=env)
            else:
                git.run(top, "read-tree", "--empty", env=env)
            stage = [r for r in rels if (top / r).exists() or (
                parent and git.out(top, "ls-tree", "-r", "--name-only", parent, "--", r))]
            if stage:
                git.run(top, "add", "-A", "--", *stage, env=env)
            tree = git.out(top, "write-tree", env=env, read_only=False)
        finally:
            for leftover in (index, index.with_name(index.name + ".lock")):
                try:
                    leftover.unlink()
                except FileNotFoundError:
                    pass

        parent_tree = git.tree_hash(top, parent) if parent else None
        if parent:
            changed = git.out(top, "diff-tree", "-r", "--no-renames", "--name-only", "-z",
                              parent_tree, tree).split("\x00")
        else:
            changed = git.out(top, "ls-tree", "-r", "--name-only", "-z", tree).split("\x00")
        bad = [c for c in changed if c and not is_ledger_path(c, ledger_root)]
        if bad:
            raise LedgerPathError(f"refusing to commit non-ledger paths: {bad[:5]}")
        if tree == parent_tree and not allow_empty:
            return LedgerCommit(full_ref, parent, parent, tree, created=False)

        args = ["commit-tree", tree]
        if parent:
            args += ["-p", parent]
        new = git.out(top, *args, "-F", "-", input=message if message.endswith("\n") else message + "\n",
                      read_only=False)
        proc = git.run(top, "update-ref", "-m", f"ci-lab ledger: {message.splitlines()[0][:72]}",
                       full_ref, new, parent or git.ZERO_OID, check=False)
        if proc.returncode != 0:
            raise LedgerConflict(full_ref, parent, git.ref_value(top, full_ref))
        return LedgerCommit(full_ref, new, parent, tree, created=True)
