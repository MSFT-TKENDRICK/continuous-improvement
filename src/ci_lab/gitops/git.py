"""Thin, injection-safe ``git`` subprocess wrapper.

* argv lists only (never a shell); revisions that start with ``-`` are refused and path
  arguments always follow ``--``;
* ``GIT_TERMINAL_PROMPT=0`` always (never hang on a credential prompt);
* ``GIT_OPTIONAL_LOCKS=0`` for read-only calls (no opportunistic index refresh locks);
* inherited ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE``/... are dropped so a caller's
  environment (e.g. a git hook) cannot redirect us to another repository or index.
"""

from __future__ import annotations

import functools
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

ZERO_OID = "0" * 40

_SCRUB = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR",
          "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
          "GIT_CEILING_DIRECTORIES", "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
          "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS")


class GitError(RuntimeError):
    def __init__(self, args: Sequence[str], returncode: int, stdout: str, stderr: str) -> None:
        self.args_list = list(args)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"git {' '.join(args[:3])}... exited {returncode}: {stderr.strip()[:500]}")


def git_env(*, read_only: bool = False, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GCM_INTERACTIVE", "never")
    if read_only:
        env["GIT_OPTIONAL_LOCKS"] = "0"
    if extra:
        env.update(extra)
    return env


def check_rev(rev: str) -> str:
    """Refuse revision arguments that git could parse as options."""
    if not isinstance(rev, str) or not rev or rev.startswith("-") or "\x00" in rev or "\n" in rev:
        raise ValueError(f"unsafe git revision {rev!r}")
    return rev


def run(repo: str | os.PathLike[str] | None, *args: str, input: str | bytes | None = None,
        read_only: bool = False, env: Mapping[str, str] | None = None, check: bool = True,
        timeout: float | None = 300, text: bool = True) -> subprocess.CompletedProcess:
    """Run ``git [-C repo] <args>``; raises :class:`GitError` on non-zero exit when ``check``."""
    if any(not isinstance(a, str) for a in args):
        raise TypeError("git args must be strings")
    argv = ["git"]
    if repo is not None:
        argv += ["-C", os.fspath(repo)]
    argv += list(args)
    kwargs: dict = {"encoding": "utf-8", "errors": "replace"} if text else {}
    proc = subprocess.run(argv, input=input, capture_output=True, env=git_env(read_only=read_only, extra=env),
                          timeout=timeout, check=False, stdin=None if input is not None else subprocess.DEVNULL,
                          **kwargs)
    if check and proc.returncode != 0:
        out = proc.stdout if text else proc.stdout.decode("utf-8", "replace")
        err = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        raise GitError(list(args), proc.returncode, out, err)
    return proc


def out(repo: str | os.PathLike[str] | None, *args: str, read_only: bool = True, **kw) -> str:
    """Run and return stripped stdout (read-only by default)."""
    return run(repo, *args, read_only=read_only, **kw).stdout.strip()


def toplevel(repo: str | os.PathLike[str]) -> Path:
    return Path(out(repo, "rev-parse", "--show-toplevel")).resolve()


@functools.lru_cache(maxsize=256)
def _common_dir_cached(repo: str) -> Path:
    raw = out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(raw).resolve()


def common_dir(repo: str | os.PathLike[str]) -> Path:
    """Absolute git common dir (shared by all linked worktrees)."""
    return _common_dir_cached(str(Path(repo).resolve()))


def rev_parse(repo: str | os.PathLike[str], rev: str, *, kind: str = "commit") -> str | None:
    """Full object id of ``rev^{kind}`` or ``None`` when it does not resolve."""
    check_rev(rev)
    spec = f"{rev}^{{{kind}}}" if kind else rev
    proc = run(repo, "rev-parse", "--verify", "--quiet", "--end-of-options", spec, read_only=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def ref_value(repo: str | os.PathLike[str], ref: str) -> str | None:
    """Object id a ref points at (no peeling), or ``None`` when the ref does not exist."""
    return rev_parse(repo, ref, kind="")


def norm_relpath(path: str | os.PathLike[str]) -> str:
    """Repo-relative POSIX path (no leading ``./``, no trailing ``/``)."""
    s = os.fspath(path).replace("\\", "/")
    parts = [p for p in PurePosixPath(s).parts if p not in ("", ".")]
    return "/".join(parts)


def tree_hash(repo: str | os.PathLike[str], commit: str, path: str = "") -> str | None:
    """Tree (or blob) id of ``<commit>:<path>`` — the harness identity (design §3); ``None``
    when the path is absent in that commit."""
    check_rev(commit)
    rel = norm_relpath(path)
    if ".." in rel.split("/"):
        raise ValueError(f"unsafe tree path {path!r}")
    return rev_parse(repo, f"{commit}:{rel}", kind="")


def is_repo(path: str | os.PathLike[str]) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    proc = run(p, "rev-parse", "--git-dir", read_only=True, check=False)
    return proc.returncode == 0


def write_lock(repo: str | os.PathLike[str], *, timeout: float = 300.0):
    """Repository-wide single-writer mutex for ref/worktree-changing git operations."""
    from ci_lab.ledger.lock import FileLock

    return FileLock(common_dir(repo) / "ci-lab" / "git-write.lock", timeout=timeout)
