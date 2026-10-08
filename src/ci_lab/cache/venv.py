"""Provision a slot's ``.venv`` (design §4; research-cow §2).

``provision(worktree)`` runs ``uv sync`` with the shared cache env. If a ``golden`` venv
is given and the worktree volume supports CoW clones, the golden ``.venv`` is cloned first
(near-free on ReFS/APFS/btrfs) and ``uv sync`` then only fixes up the delta — notably the
editable install of the worktree's own project.

Caveat: console-script launchers/shebangs in a cloned venv still embed the golden path
until uv rewrites them; always invoke tools as ``python -m`` / ``uv run`` in slots.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ci_lab.cache.cow import cow_mode, reflink_file
from ci_lab.cache.env import with_shared_env

Runner = Callable[..., Any]


@dataclass(frozen=True)
class Provisioned:
    venv: Path
    method: Literal["clone", "sync"]
    command: tuple[str, ...]


def _clone_copy(src: str, dst: str) -> str:
    if not reflink_file(src, dst):
        shutil.copy2(src, dst)
    return dst


def clone_venv(golden: str | os.PathLike[str], dest: str | os.PathLike[str]) -> Path:
    """Copy ``golden`` to ``dest`` using per-file CoW clones where possible. On Windows,
    ``CopyFileW`` (used by ``shutil.copy2``) block-clones automatically on ReFS/Dev Drive."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(golden, dest, symlinks=True, copy_function=_clone_copy)
    return dest


def uv_sync_command(worktree: Path, *, frozen: bool | None = None, offline: bool = False,
                    python: str | None = None, extra: tuple[str, ...] = ()) -> list[str]:
    cmd = ["uv", "sync"]
    if os.environ.get("UV_NATIVE_TLS"):
        cmd.append("--native-tls")
    if frozen is None:
        frozen = (worktree / "uv.lock").is_file()
    if frozen:
        cmd.append("--frozen")
    if offline:
        cmd.append("--offline")
    if python:
        cmd += ["--python", python]
    return cmd + list(extra)


def provision(worktree: str | os.PathLike[str], golden: str | os.PathLike[str] | None = None, *,
              runner: Runner = subprocess.run, offline: bool = False, python: str | None = None,
              frozen: bool | None = None, extra: tuple[str, ...] = (), timeout: float = 1800) -> Provisioned:
    wt = Path(worktree)
    venv = wt / ".venv"
    method: Literal["clone", "sync"] = "sync"
    if golden is not None:
        g = Path(golden)
        if g.name != ".venv" and (g / ".venv").is_dir():
            g = g / ".venv"
        if (g / "pyvenv.cfg").is_file() and g.resolve() != venv.resolve() and cow_mode(wt, source=g) == "clone":
            clone_venv(g, venv)
            method = "clone"
    cmd = uv_sync_command(wt, frozen=frozen, offline=offline, python=python, extra=extra)
    env = with_shared_env(wt, no_sync=False)
    runner(cmd, cwd=str(wt), env=env, check=True, timeout=timeout)
    return Provisioned(venv, method, tuple(cmd))
