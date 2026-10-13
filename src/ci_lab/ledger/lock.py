"""Cross-process exclusive file locks.

``FileLock`` uses OS advisory locks (``msvcrt.locking`` on Windows, ``flock`` elsewhere),
so a crashed holder never leaves a stale lock. Locks are re-entrant for the same owner
(the current thread for ``acquire``/``with``; the current asyncio task for
``aacquire``/``async with``) and mutually exclusive between threads, tasks and processes.

Ledger locks live in the git common dir (shared by every linked worktree) so they never
appear in a working tree or a ledger commit.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

DEFAULT_TIMEOUT = 60.0


class LockTimeout(TimeoutError):
    pass


@dataclass
class _State:
    mutex: threading.Lock = field(default_factory=threading.Lock)
    fd: int | None = None
    owner: Any = None
    depth: int = 0


_registry: dict[str, _State] = {}
_registry_mutex = threading.Lock()


def _state_for(path: Path) -> _State:
    key = os.path.normcase(os.path.abspath(path))
    with _registry_mutex:
        return _registry.setdefault(key, _State())


def _os_trylock(fd: int) -> bool:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        return False
    return True


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """Exclusive lock on ``path`` (created if missing, never deleted)."""

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = DEFAULT_TIMEOUT,
                 poll: float = 0.02) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self._state = _state_for(self.path)

    def __repr__(self) -> str:
        return f"FileLock({str(self.path)!r})"

    # ------------------------------------------------------------ core

    def _try_enter(self, owner: Any) -> bool:
        st = self._state
        with st.mutex:
            if st.depth and st.owner == owner:
                st.depth += 1
                return True
            if st.depth:
                return False
            if st.fd is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                st.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            if not _os_trylock(st.fd):
                os.close(st.fd)
                st.fd = None
                return False
            st.owner, st.depth = owner, 1
            return True

    def _exit(self, owner: Any) -> None:
        st = self._state
        with st.mutex:
            if not st.depth or st.owner != owner:
                raise RuntimeError(f"{self!r} released by a non-owner")
            st.depth -= 1
            if st.depth:
                return
            fd, st.fd, st.owner = st.fd, None, None
            if fd is not None:
                try:
                    _os_unlock(fd)
                finally:
                    os.close(fd)

    def _timeout_error(self, timeout: float) -> LockTimeout:
        return LockTimeout(f"could not acquire {self.path} within {timeout:.1f}s")

    @property
    def held(self) -> bool:
        return self._state.depth > 0

    # ------------------------------------------------------------ sync (owner = thread)

    @staticmethod
    def _thread_owner() -> Any:
        return ("thread", threading.get_ident())

    def acquire(self, timeout: float | None = None) -> FileLock:
        limit = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        owner = self._thread_owner()
        delay = self.poll
        while not self._try_enter(owner):
            if time.monotonic() >= deadline:
                raise self._timeout_error(limit)
            time.sleep(delay)
            delay = min(delay * 1.5, 0.25)
        return self

    def release(self) -> None:
        self._exit(self._thread_owner())

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()

    # ------------------------------------------------------------ async (owner = task)

    @staticmethod
    def _task_owner() -> Any:
        task = asyncio.current_task()
        return ("task", id(task)) if task is not None else FileLock._thread_owner()

    async def aacquire(self, timeout: float | None = None) -> FileLock:
        limit = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        owner = self._task_owner()
        delay = self.poll
        while not self._try_enter(owner):
            if time.monotonic() >= deadline:
                raise self._timeout_error(limit)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 0.25)
        return self

    def arelease(self) -> None:
        self._exit(self._task_owner())

    async def __aenter__(self) -> FileLock:
        return await self.aacquire()

    async def __aexit__(self, *exc: object) -> None:
        self.arelease()


# ---------------------------------------------------------------- lock placement

def _existing_ancestor(path: Path) -> Path:
    p = path.absolute()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p if p.is_dir() else p.parent


def lock_dir_for(path: str | os.PathLike[str]) -> Path:
    """Directory for locks guarding ``path``: ``<git-common-dir>/ci-lab/locks`` when ``path``
    is inside a git repository, else a hidden ``.ci-lab-locks`` dir next to it."""
    from ci_lab.gitops import git

    anchor = _existing_ancestor(Path(path))
    if git.is_repo(anchor):
        return git.common_dir(anchor) / "ci-lab" / "locks"
    return Path(path).absolute().parent / ".ci-lab-locks"


def lock_for(path: str | os.PathLike[str], *, suffix: str = "", timeout: float = DEFAULT_TIMEOUT,
             lock_dir: str | os.PathLike[str] | None = None) -> FileLock:
    """A :class:`FileLock` dedicated to ``path`` (+ ``suffix``), keyed by its absolute path.
    Pass a precomputed ``lock_dir`` (from :func:`lock_dir_for`) to skip the git probe."""
    key = os.path.normcase(os.path.abspath(path)) + "|" + suffix
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    base = Path(lock_dir) if lock_dir is not None else lock_dir_for(path)
    return FileLock(base / f"{Path(path).name[:40]}-{digest}.lock", timeout=timeout)


def ledger_lock(repo: str | os.PathLike[str], *, timeout: float = DEFAULT_TIMEOUT) -> FileLock:
    """The single-coordinator ledger lock: ``<git-common-dir>/ci-lab/ledger.lock``."""
    from ci_lab.gitops import git

    return FileLock(git.common_dir(repo) / "ci-lab" / "ledger.lock", timeout=timeout)
