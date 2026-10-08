"""Packaged ACS policies: ``load_policy(name) -> AgentControl`` bundles manifest + adapters."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import cache
from importlib import resources
from typing import Any

from ci_lab.governance.acs import AcsRuntime, AgentControl, load_manifest
from ci_lab.governance.acs.runtime import MODES
from ci_lab.governance.adapters import DISPATCHER, Annotator

__all__ = ["MODE_ENV", "POLICIES", "governance_mode", "load_policy", "manifest_text", "target_mode"]

MODE_ENV = "CI_GOVERNANCE_MODE"
POLICIES = ("order_support", "meta_agents", "campaign")


def governance_mode(env: Mapping[str, str] | None = None) -> str:
    """``CI_GOVERNANCE_MODE`` (default ``enforce``); any other value is a hard startup error."""
    raw = (os.environ if env is None else env).get(MODE_ENV, "").strip() or "enforce"
    if raw not in MODES:
        raise RuntimeError(f"{MODE_ENV}={raw!r} is invalid; expected one of {sorted(MODES)}")
    return raw


def target_mode(env: Mapping[str, str] | None = None) -> str:
    """Mode for the order-support *target* agents: ``CI_GOVERNANCE_MODE`` when set, else
    ``evaluate_only``. The target is the experiment subject whose enforcement layer is the guard
    bundle (``CI_GUARDS``, shadow by default; paired guard-off/on trials), so ACS only records
    unless an operator opts in explicitly."""
    env = os.environ if env is None else env
    return governance_mode(env) if env.get(MODE_ENV, "").strip() else "evaluate_only"


def manifest_text(name: str) -> str:
    if name not in POLICIES:
        raise KeyError(f"unknown governance policy {name!r}; known: {POLICIES}")
    return resources.files(__package__).joinpath(f"{name}.acs.yaml").read_text("utf-8")


@cache
def _runtime(name: str) -> AcsRuntime:
    manifest = load_manifest(manifest_text(name))
    if not manifest.points:
        raise RuntimeError(f"governance policy {name!r} configures no intervention points")
    return AcsRuntime(manifest, dispatcher=DISPATCHER, annotator=Annotator())


def load_policy(name: str, *, mode: str | None = None,
                approval_resolver: Callable[..., Any] | Mapping[str, Callable[..., Any]] | None = None,
                decision_log: Callable[[dict[str, Any]], None] | None = None) -> AgentControl:
    """An :class:`AgentControl` over the packaged ``<name>.acs.yaml`` with the harness adapters."""
    return AgentControl(_runtime(name), approval_resolver, mode or governance_mode(),
                        decision_log=decision_log)
