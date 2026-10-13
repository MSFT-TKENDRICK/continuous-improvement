"""Ref publication primitives: leased pushes, archive tags, ancestry checks (design §3)."""

from __future__ import annotations

import os
import re

from ci_lab.gitops import git
from ci_lab.gitops.names import archive_tag, check_ref_format

REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class PushRejected(RuntimeError):
    """The remote branch was not at the expected sha (lease broken) or the push was refused."""


class TagConflict(RuntimeError):
    pass


def _remote(name: str) -> str:
    if not REMOTE_RE.match(name or ""):
        raise ValueError(f"bad remote name {name!r}")
    return name


def remote_sha(repo: str | os.PathLike[str], remote: str, branch: str) -> str | None:
    """Current sha of ``refs/heads/<branch>`` on ``remote`` (``None`` if absent)."""
    check_ref_format(branch)
    lines = git.out(repo, "ls-remote", "--heads", "--", _remote(remote), f"refs/heads/{branch}").splitlines()
    for line in lines:
        sha, _, ref = line.partition("\t")
        if ref == f"refs/heads/{branch}":
            return sha
    return None


def push_with_lease(repo: str | os.PathLike[str], remote: str, branch: str, expected_remote_sha: str | None, *,
                    local: str | None = None, timeout: float = 300) -> str:
    """Push ``local`` (default ``refs/heads/<branch>``) to ``remote``'s ``branch`` only if the
    remote is still at ``expected_remote_sha`` (``None`` = must not exist). Idempotent: if
    the remote already has the commit, nothing is pushed. Returns the pushed sha."""
    check_ref_format(branch)
    _remote(remote)
    src = git.check_rev(local or f"refs/heads/{branch}")
    new = git.rev_parse(repo, src)
    if new is None:
        raise ValueError(f"cannot resolve {src!r}")
    if expected_remote_sha is not None:
        git.check_rev(expected_remote_sha)
    current = remote_sha(repo, remote, branch)
    if current == new:
        return new
    if current != expected_remote_sha:
        raise PushRejected(f"{remote}/{branch} is at {current or '<absent>'}, expected "
                           f"{expected_remote_sha or '<absent>'}")
    lease = f"--force-with-lease=refs/heads/{branch}:{expected_remote_sha or ''}"
    with git.write_lock(repo):
        proc = git.run(repo, "push", "--porcelain", "--no-verify", lease, "--", remote,
                       f"{new}:refs/heads/{branch}", check=False, timeout=timeout)
    if proc.returncode != 0:
        raise PushRejected(f"push of {branch} to {remote} rejected: {(proc.stderr or proc.stdout).strip()[:500]}")
    return new


def tag_archive(repo: str | os.PathLike[str], experiment_id: str, arm: str, commit: str, *,
                remote: str | None = None) -> str:
    """Create ``exp-archive/<eid>/<arm>`` at ``commit`` (CAS create; idempotent if it already
    points there) and optionally push it. Returns the tag name."""
    tag = archive_tag(experiment_id, arm)
    sha = git.rev_parse(repo, git.check_rev(commit))
    if sha is None:
        raise ValueError(f"unknown commit {commit!r}")
    full = f"refs/tags/{tag}"
    with git.write_lock(repo):
        existing = git.ref_value(repo, full)
        if existing is None:
            proc = git.run(repo, "update-ref", "-m", "ci-lab archive", full, sha, git.ZERO_OID, check=False)
            if proc.returncode != 0:
                existing = git.ref_value(repo, full)
        if existing is not None and existing != sha:
            raise TagConflict(f"{tag} already points at {existing}, not {sha}")
    if remote is not None:
        proc = git.run(repo, "push", "--porcelain", "--no-verify", "--", _remote(remote), f"{full}:{full}",
                       check=False)
        if proc.returncode != 0:
            raise PushRejected(f"push of {tag} rejected: {(proc.stderr or proc.stdout).strip()[:500]}")
    return tag


def is_ancestor(repo: str | os.PathLike[str], ancestor: str, descendant: str) -> bool:
    """``git merge-base --is-ancestor`` (fast-forward / acceptance check)."""
    proc = git.run(repo, "merge-base", "--is-ancestor", git.check_rev(ancestor), git.check_rev(descendant),
                   read_only=True, check=False)
    if proc.returncode in (0, 1):
        return proc.returncode == 0
    raise git.GitError(["merge-base", "--is-ancestor", ancestor, descendant], proc.returncode, proc.stdout,
                       proc.stderr)
