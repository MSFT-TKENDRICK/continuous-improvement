"""Worktree slot pool (design §4): ``<CI_WT_ROOT>/<cid>/s<n>`` reused across rounds.

Slots keep their warm ``.venv`` between arms: :meth:`SlotPool.reset` runs
``git switch --discard-changes -C <branch> <base>`` and ``git clean -fdx -e .venv``.
Leases are recorded in ``<root>/<cid>/pool.json`` under a pool lock (cross-process); a
lease whose owner process is gone (same host) is reclaimed. All ref/worktree-changing
git commands run under the repository's single-writer mutex (:func:`git.write_lock`).
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ci_lab.contracts import CAMPAIGN_RE
from ci_lab.gitops import git
from ci_lab.gitops.names import check_ref_format
from ci_lab.ledger.atomic import atomic_write_json, read_json
from ci_lab.ledger.lock import FileLock

KEEP = (".venv",)


class SlotPoolExhausted(RuntimeError):
    pass


def wt_root() -> Path:
    """``CI_WT_ROOT`` or ``C:\\x`` on Windows (short paths: MAX_PATH) / ``~/.ci-wt`` elsewhere."""
    env = os.environ.get("CI_WT_ROOT")
    if env:
        return Path(env)
    return Path("C:/x") if os.name == "nt" else Path.home() / ".ci-wt"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied => exists
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class Slot:
    campaign_id: str
    index: int
    path: Path
    branch: str | None
    base_commit: str | None


class SlotPool:
    def __init__(self, repo: str | os.PathLike[str], campaign_id: str, *,
                 root: str | os.PathLike[str] | None = None, max_slots: int | None = None,
                 lock_timeout: float = 300.0) -> None:
        if not CAMPAIGN_RE.match(campaign_id):
            raise ValueError(f"bad campaign id {campaign_id!r}")
        self.repo = git.toplevel(repo)
        self.campaign_id = campaign_id
        self.dir = Path(root if root is not None else wt_root()) / campaign_id
        self.max_slots = max_slots
        self.lock_timeout = lock_timeout
        self._state_path = self.dir / "pool.json"
        self._pool_lock = FileLock(self.dir / ".pool.lock", timeout=lock_timeout)

    # ------------------------------------------------------------ state

    def _load(self) -> dict[str, Any]:
        data = read_json(self._state_path, default=None) or {}
        data.setdefault("slots", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        atomic_write_json(self._state_path, data)

    def _slot(self, idx: int, entry: dict[str, Any]) -> Slot:
        return Slot(self.campaign_id, idx, self.dir / f"s{idx}", entry.get("branch"), entry.get("base"))

    def slots(self) -> list[Slot]:
        with self._pool_lock:
            data = self._load()
        return [self._slot(int(k), v) for k, v in sorted(data["slots"].items(), key=lambda kv: int(kv[0]))]

    def leased(self) -> list[int]:
        with self._pool_lock:
            return sorted(int(k) for k, v in self._load()["slots"].items() if v.get("lease"))

    @staticmethod
    def _lease_dead(lease: dict[str, Any]) -> bool:
        return lease.get("host") == socket.gethostname() and not pid_alive(int(lease.get("pid", -1)))

    def _write_lock(self) -> FileLock:
        return git.write_lock(self.repo, timeout=self.lock_timeout)

    # ------------------------------------------------------------ git ops

    def _is_worktree(self, path: Path) -> bool:
        return (path / ".git").is_file()

    def _create(self, path: Path, base: str) -> None:
        with self._write_lock():
            git.run(self.repo, "worktree", "prune")
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"slot dir {path} exists but is not a registered worktree")
            path.parent.mkdir(parents=True, exist_ok=True)
            git.run(self.repo, "worktree", "add", "--detach", "--", str(path), base)

    def _switch(self, path: Path, base: str, branch: str) -> None:
        with self._write_lock():
            git.run(path, "switch", "--discard-changes", "--no-guess", "-C", branch, base)
        args = ["clean", "-fdx"]
        for keep in KEEP:
            args += ["-e", keep]
        git.run(path, *args)

    def _resolve(self, base_commit: str) -> str:
        sha = git.rev_parse(self.repo, git.check_rev(base_commit))
        if sha is None:
            raise ValueError(f"unknown base commit {base_commit!r}")
        return sha

    # ------------------------------------------------------------ API

    def acquire(self, base_commit: str, branch: str) -> Slot:
        """Lease a free slot (creating ``s<n>`` if needed) checked out on a fresh ``branch``
        at ``base_commit``."""
        check_ref_format(branch)
        base = self._resolve(base_commit)
        with self._pool_lock:
            data = self._load()
            slots = data["slots"]
            for entry in slots.values():
                if entry.get("lease") and self._lease_dead(entry["lease"]):
                    entry["lease"] = None
            free = [int(k) for k, v in slots.items()
                    if not v.get("lease") and self._is_worktree(self.dir / f"s{k}")]
            if free:
                idx, new = min(free), False
            else:
                if self.max_slots is not None and len(slots) >= self.max_slots:
                    reusable = [int(k) for k, v in slots.items() if not v.get("lease")]
                    if not reusable:
                        raise SlotPoolExhausted(f"all {self.max_slots} slots of {self.campaign_id} are leased")
                    idx = min(reusable)
                else:
                    idx = next(i for i in range(len(slots) + 1) if str(i) not in slots)
                new = True
            slots[str(idx)] = {**slots.get(str(idx), {}), "lease": {
                "pid": os.getpid(), "host": socket.gethostname(), "ts": time.time()}}
            self._save(data)
        path = self.dir / f"s{idx}"
        try:
            if new and not self._is_worktree(path):
                self._create(path, base)
            self._switch(path, base, branch)
        except BaseException:
            self._update(idx, lease=None)
            raise
        self._update(idx, branch=branch, base=base)
        return Slot(self.campaign_id, idx, path, branch, base)

    def _update(self, idx: int, **fields: Any) -> None:
        with self._pool_lock:
            data = self._load()
            entry = data["slots"].setdefault(str(idx), {})
            entry.update(fields)
            self._save(data)

    def reset(self, slot: Slot, base_commit: str, branch: str | None = None) -> Slot:
        """Point a leased slot at ``base_commit`` on (re)created ``branch``; drop all changes
        and untracked files except ``.venv``."""
        branch = branch or slot.branch
        if not branch:
            raise ValueError("branch required")
        check_ref_format(branch)
        base = self._resolve(base_commit)
        self._switch(slot.path, base, branch)
        self._update(slot.index, branch=branch, base=base)
        return Slot(self.campaign_id, slot.index, slot.path, branch, base)

    def release(self, slot: Slot) -> None:
        """Return the slot to the pool (kept warm). HEAD is detached so the arm branch is no
        longer checked out anywhere (it can be pushed, moved or checked out elsewhere)."""
        if self._is_worktree(slot.path):
            with self._write_lock():
                git.run(slot.path, "switch", "--detach", "--discard-changes")
        self._update(slot.index, lease=None, branch=None)

    def remove(self, slot: Slot) -> None:
        """Delete the slot's worktree (branches are kept) and forget it."""
        with self._write_lock():
            if self._is_worktree(slot.path):
                for attempt in range(5):
                    proc = git.run(self.repo, "worktree", "remove", "--force", "--force", "--", str(slot.path),
                                   check=False)
                    if proc.returncode == 0:
                        break
                    time.sleep(0.2 * (attempt + 1))
            if slot.path.exists():
                shutil.rmtree(slot.path, ignore_errors=True)
            git.run(self.repo, "worktree", "prune")
        with self._pool_lock:
            data = self._load()
            data["slots"].pop(str(slot.index), None)
            self._save(data)

    def evaluator(self, commit: str) -> Path:
        """Read-only evaluator worktree ``<root>/<cid>/eval`` detached at ``commit``."""
        base = self._resolve(commit)
        path = self.dir / "eval"
        if not self._is_worktree(path):
            self._create(path, base)
        else:
            with self._write_lock():
                git.run(path, "switch", "--detach", "--discard-changes", base)
        git.run(path, "clean", "-fdx", "-e", ".venv")
        return path
