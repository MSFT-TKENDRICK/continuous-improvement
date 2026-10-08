"""Where a domain's harness lives inside a repo-root arm worktree (design §3).

Campaign slots are full repository checkouts; the evolvable harness is the directory prefix
shared by the domain's ``surface_globs`` (``src/order_support/harness`` for order-support,
``harness`` for the test stubs).
"""
from __future__ import annotations

import re

from ci_lab.contracts import Domain

_GLOB_CHARS = re.compile(r"[*?\[]")


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
