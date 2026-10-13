"""Programmatic optimizers for arm strategies (design §11): DSPy LM routing, GEPA and
SkillOpt adapters over harness text components.

Importing this package never imports ``dspy``/``gepa``/``skillopt_sleep`` (C26); the
heavy libraries load inside the functions that need them.
"""
from __future__ import annotations

from ci_lab.optim.lm import cache_enabled, disable_dspy_cache, make_lm, resolve_endpoint, resolve_model
from ci_lab.optim.scoring import (
    BudgetExhausted,
    CaseOutcome,
    DomainEvolveScorer,
    EvolveGuard,
    EvolveScorer,
    HeldOutAccessError,
    MetricBudget,
    render_failure,
    stable_split,
)
from ci_lab.optim.targets import TargetError, TextTarget, resolve_targets

__all__ = [
    "BudgetExhausted", "CaseOutcome", "DomainEvolveScorer", "EvolveGuard", "EvolveScorer",
    "HeldOutAccessError", "MetricBudget", "TargetError", "TextTarget", "cache_enabled",
    "disable_dspy_cache", "make_lm", "render_failure", "resolve_endpoint", "resolve_model",
    "resolve_targets", "stable_split",
]
