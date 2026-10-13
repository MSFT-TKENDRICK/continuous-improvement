"""Shared cache environment for slot worktrees (design §4; research-cow §2).

Every slot shares one uv cache and one bytecode cache so N worktrees cost ~1 install.
``UV_LINK_MODE`` is ``clone`` on CoW volumes (ReFS/Dev Drive, APFS, btrfs/xfs) and
``hardlink`` otherwise; uv itself falls back to copying when neither works.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from ci_lab.cache.cow import cow_mode
from ci_lab.gitops.slots import wt_root

_SCRUB = ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")


def cache_root() -> Path:
    """``CI_CACHE_DIR`` or ``<CI_WT_ROOT>/.cache`` (same volume as the slots, so links work)."""
    env = os.environ.get("CI_CACHE_DIR")
    return Path(env) if env else wt_root() / ".cache"


def link_mode(path: str | os.PathLike[str], *, source: str | os.PathLike[str] | None = None) -> str:
    """uv link mode for venvs materialized at ``path`` from cache ``source``."""
    return "clone" if cow_mode(path, source=source) == "clone" else "hardlink"


def shared_env(worktree: str | os.PathLike[str] | None = None, *, root: str | os.PathLike[str] | None = None,
               no_sync: bool | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overrides to add to a subprocess env. An explicit ``UV_CACHE_DIR`` in ``base`` wins.

    ``no_sync=True`` sets ``UV_NO_SYNC=1`` (use the provisioned venv as-is during eval runs);
    ``None`` leaves the caller's setting alone."""
    base = os.environ if base is None else base
    croot = Path(root) if root is not None else cache_root()
    uv_cache = Path(base["UV_CACHE_DIR"]) if base.get("UV_CACHE_DIR") else croot / "uv"
    target = Path(worktree) if worktree is not None else croot
    env = {
        "UV_CACHE_DIR": str(uv_cache),
        "UV_LINK_MODE": link_mode(target, source=uv_cache),
        "PYTHONPYCACHEPREFIX": str(croot / "pycache"),
    }
    if no_sync:
        env["UV_NO_SYNC"] = "1"
    return env


def with_shared_env(worktree: str | os.PathLike[str] | None = None, *, base: Mapping[str, str] | None = None,
                    root: str | os.PathLike[str] | None = None, no_sync: bool | None = None) -> dict[str, str]:
    """Full env (``base`` or ``os.environ``) + shared overrides; drops vars that would point uv
    at another project's venv. ``no_sync=False`` removes an inherited ``UV_NO_SYNC``."""
    src = dict(os.environ if base is None else base)
    for key in _SCRUB:
        src.pop(key, None)
    if no_sync is False:
        src.pop("UV_NO_SYNC", None)
    src.update(shared_env(worktree, root=root, no_sync=no_sync, base=src))
    return src
