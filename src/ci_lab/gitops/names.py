"""Ref naming (design §3): validated arm branches, archive tags, sleep/ledger branches.

Every name produced here is also checked with ``git check-ref-format`` so nothing git
would reject (or interpret specially) can reach a ref-changing command.
"""

from __future__ import annotations

import functools
import re
import subprocess

from ci_lab.contracts import (
    ARM_BRANCH_RE,
    ARM_RE,
    CAMPAIGN_RE,
    SLEEP_BRANCH_RE,
    op_id,
    round_experiment_id,
)
from ci_lab.contracts import arm_branch as _arm_branch
from ci_lab.gitops.git import git_env

__all__ = ["ARM_BRANCH_RE", "ARM_RE", "ARCHIVE_TAG_RE", "CAMPAIGN_RE", "EXPERIMENT_RE", "SLEEP_BRANCH_RE",
           "archive_tag", "arm_branch", "check_ref_format", "ledger_branch", "op_id", "round_experiment_id",
           "sleep_branch"]

EXPERIMENT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,44}$")
ARCHIVE_TAG_RE = re.compile(r"^exp-archive/[a-z0-9][a-z0-9-]{2,44}/[a-z][a-z0-9-]{0,15}$")
LEDGER_BRANCH_RE = re.compile(r"^exp-ledger/[a-z0-9][a-z0-9-]{2,40}$")


@functools.lru_cache(maxsize=1024)
def _git_accepts(full_ref: str) -> bool:
    proc = subprocess.run(["git", "check-ref-format", full_ref], capture_output=True,
                          env=git_env(read_only=True), stdin=subprocess.DEVNULL, check=False)
    return proc.returncode == 0


def check_ref_format(ref: str, *, kind: str = "heads") -> str:
    """Validate ``ref`` (short name, e.g. ``exp/x-r01/v1``) as ``refs/<kind>/<ref>``."""
    if not isinstance(ref, str) or not ref or ref.startswith("-") or ref.startswith("refs/"):
        raise ValueError(f"bad ref name {ref!r}")
    if not _git_accepts(f"refs/{kind}/{ref}"):
        raise ValueError(f"git rejects ref name {ref!r}")
    return ref


def arm_branch(experiment_id: str, arm: str) -> str:
    """``exp/<eid>/<arm>`` (fixed depth 3)."""
    return check_ref_format(_arm_branch(experiment_id, arm))


def archive_tag(experiment_id: str, arm: str) -> str:
    """``exp-archive/<eid>/<arm>``: loser arms are preserved as tags (no branch D/F conflicts)."""
    tag = f"exp-archive/{experiment_id}/{arm}"
    if not ARM_RE.match(arm) or not ARCHIVE_TAG_RE.match(tag):
        raise ValueError(f"bad archive tag {tag!r}")
    return check_ref_format(tag, kind="tags")


def sleep_branch(yyyymmdd: str, attempt: int = 1) -> str:
    """``exp/sleep-<yyyymmdd>-<attempt>/cand`` (nightly candidate, C10)."""
    ref = f"exp/sleep-{yyyymmdd}-{int(attempt)}/cand"
    if not SLEEP_BRANCH_RE.match(ref):
        raise ValueError(f"bad sleep branch {ref!r}")
    return check_ref_format(ref)


def ledger_branch(campaign_id: str) -> str:
    """``exp-ledger/<cid>`` (``ledger.mode=pr``)."""
    ref = f"exp-ledger/{campaign_id}"
    if not CAMPAIGN_RE.match(campaign_id) or not LEDGER_BRANCH_RE.match(ref):
        raise ValueError(f"bad ledger branch {ref!r}")
    return check_ref_format(ref)
