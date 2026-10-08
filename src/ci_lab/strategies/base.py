"""Shared plumbing for arm strategies: optimizer span, edit-budget check, git committer,
evolve-case resolution and the optimizer cost report (design §11.2, C18, C20)."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_EXPERIMENT,
    ATTR_PROFILE,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_OPTIMIZER,
    ArmContext,
    Domain,
    Edit,
)

COMMIT_TRAILER = "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
FALLBACK_IDENTITY = ("ci-lab arm", "ci-lab-arm@localhost")
OPTIMIZER_DIR = "optimizer"
GUARD_GLOBS = ("**/harness/guards/**",)
"""v2.4 §13 (B2): guard rule bundles are writable only by the ``guard`` strategy."""

Committer = Callable[[Path, Sequence[str], str], str]
"""``(worktree, files, message) -> commit sha``."""


class EditBudgetExceeded(ValueError):
    """A strategy produced more Edits than ``ArmDirective.edit_budget``."""


@contextmanager
def optimizer_span(strategy: str, ctx: ArmContext) -> Iterator[Any]:
    """``ci.optimizer`` span around one strategy's proposal work."""
    with obs.span(SPAN_OPTIMIZER, {ATTR_STRATEGY: strategy, ATTR_EXPERIMENT: ctx.experiment_id,
                                   ATTR_VARIANT: ctx.directive.arm,
                                   ATTR_PROFILE: getattr(ctx.profile, "value", str(ctx.profile)),
                                   "ci.edit_budget": ctx.directive.edit_budget}) as s:
        yield s


def check_edit_budget(edits: Sequence[Any], ctx: ArmContext) -> list[Edit]:
    out = list(edits)
    if bad := [e for e in out if not isinstance(e, Edit)]:
        raise TypeError(f"strategies must return contracts.Edit, got {type(bad[0]).__name__}")
    if len(out) > ctx.directive.edit_budget:
        raise EditBudgetExceeded(f"{len(out)} edits > edit_budget {ctx.directive.edit_budget}")
    return out


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=worktree, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def git_commit(worktree: Path, files: Sequence[str], message: str) -> str:
    """Stage exactly ``files`` and commit them; returns the new HEAD sha."""
    worktree = Path(worktree)
    ident: list[str] = []
    try:
        _git(worktree, "config", "user.email")
    except subprocess.CalledProcessError:
        ident = ["-c", f"user.name={FALLBACK_IDENTITY[0]}", "-c", f"user.email={FALLBACK_IDENTITY[1]}"]
    _git(worktree, "add", "--", *files)
    _git(worktree, *ident, "commit", "-q", "-m", message, "-m", COMMIT_TRAILER, "--", *files)
    return _git(worktree, "rev-parse", "HEAD")


def evolve_cases_for(ctx: ArmContext, *, explicit: Sequence[str] | None = None, domain: Domain | None = None,
                     scorer: Any = None) -> list[str]:
    """Evolve case ids an optimizer may score: explicit > domain.splits()["evolve"] >
    scorer.evolve_cases() > case ids of ``ctx.failures`` (assumed evolve, C15)."""
    if explicit:
        return list(dict.fromkeys(explicit))
    if domain is not None:
        return list(domain.splits()["evolve"])
    if callable(getattr(scorer, "evolve_cases", None)):
        return list(scorer.evolve_cases())
    return sorted({f.case_id for f in ctx.failures})


def optimizer_dir(ctx: ArmContext) -> Path:
    return Path(ctx.run_dir) / OPTIMIZER_DIR


def write_report(ctx: ArmContext, strategy: str, payload: Mapping[str, Any]) -> Path:
    """``<run_dir>/optimizer/<arm>-<strategy>.json`` — consumed for ΔC accounting."""
    d = optimizer_dir(ctx)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ctx.directive.arm}-{strategy}.json"
    p.write_text(json.dumps({"experiment_id": ctx.experiment_id, "arm": ctx.directive.arm,
                             "strategy": strategy, **payload}, indent=2, default=str), encoding="utf-8")
    return p
