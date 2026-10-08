"""Arm strategy registry (design §11.2): ``get_strategy(directive.strategy, **deps)``.

Strategies: ``agent`` (MAF meta-agent proposer), ``gepa`` (GEPA over text components),
``skillopt`` (SkillOpt-Sleep on skill markdown). Every strategy emits a ``ci.optimizer``
span, honours ``ArmDirective.edit_budget`` and returns plain :class:`~ci_lab.contracts.Edit`
commits that go through the same critic/ASSERT/RRSI/OES gates. Importing this package
does not import dspy/gepa/skillopt_sleep (C26).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ci_lab.contracts import STRATEGIES, ArmStrategy
from ci_lab.strategies.agent import AgentStrategy
from ci_lab.strategies.base import EditBudgetExceeded, git_commit
from ci_lab.strategies.gepa import GepaStrategy
from ci_lab.strategies.skillopt import SkillOptStrategy

_FACTORIES: dict[str, Callable[..., ArmStrategy]] = {
    "agent": AgentStrategy,
    "gepa": GepaStrategy,
    "skillopt": SkillOptStrategy,
}
assert set(_FACTORIES) == set(STRATEGIES)


class UnknownStrategy(KeyError):
    pass


def _accepted(factory: Callable[..., Any]) -> set[str] | None:
    params: dict[str, inspect.Parameter] = {}
    for cls in getattr(factory, "__mro__", (factory,)):
        init = cls.__dict__.get("__init__") if isinstance(factory, type) else factory
        if init is None:
            continue
        sig = inspect.signature(init)
        params.update({n: p for n, p in sig.parameters.items() if n not in params})
        if not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            break
    return {n for n, p in params.items() if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)} - {"self"}


def get_strategy(name: str, **deps: Any) -> ArmStrategy:
    """Build strategy ``name``. ``deps`` may be a shared superset (e.g. ``proposer``,
    ``domain``, ``scorer``, ``lm``, ``committer``); each strategy takes what it accepts.
    Missing required deps raise ``TypeError``."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise UnknownStrategy(f"unknown arm strategy {name!r}; expected one of {STRATEGIES}") from None
    accepted = _accepted(factory)
    return factory(**{k: v for k, v in deps.items() if accepted is None or k in accepted})


__all__ = ["AgentStrategy", "EditBudgetExceeded", "GepaStrategy", "STRATEGIES", "SkillOptStrategy",
           "UnknownStrategy", "get_strategy", "git_commit"]
