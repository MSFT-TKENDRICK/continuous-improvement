"""Where a domain's harness lives inside a repo-root arm worktree (design §3).

Campaign slots are full repository checkouts; the evolvable harness is the directory prefix
shared by the domain's ``surface_globs`` (``src/order_support/harness`` for order-support,
``harness`` for the test stubs).
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from ci_lab.contracts import Domain
from ci_lab.rulespec import GUARDS_DIR as DEFAULT_GUARDS_DIR

_GLOB_CHARS = re.compile(r"[*?\[]")
GUARDS_SUBDIR = "guards"
"""Guard rule bundle directory inside a harness (``order_support.guarding.guards_dir``)."""


def harness_root(domain: Domain) -> str:
    """Directory prefix shared by the domain's surface globs (the harness tree, design §3)."""
    roots = set()
    for glob in domain.surface_globs:
        parts = []
        for part in glob.split("/"):
            if _GLOB_CHARS.search(part):
                break
            parts.append(part)
        roots.add("/".join(parts))
    if len(roots) != 1:
        raise ValueError(f"surface globs {list(domain.surface_globs)} do not share one harness root")
    return roots.pop()


def guards_rel(domain: Any = None) -> str:
    """Repo-relative posix dir of the guard rules the domain's agent loads from a repo-root worktree:
    ``<harness root>/guards`` (``src/order_support/harness/guards`` for order-support), or
    ``harness/guards`` when ``domain`` is ``None`` or declares no single harness root."""
    if domain is None:
        return DEFAULT_GUARDS_DIR
    try:
        root = harness_root(domain)
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_GUARDS_DIR
    return str(PurePosixPath(root) / GUARDS_SUBDIR) if root else GUARDS_SUBDIR


def guard_extractors(domain: Any = None) -> list[Path]:
    """Frozen extractor files the domain's guard bundle is loaded with (``Domain.guard_extractors``)."""
    if domain is None:
        return []
    return [Path(p) for p in getattr(domain, "guard_extractors", None) or ()]
