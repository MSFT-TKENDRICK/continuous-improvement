"""Arm strategy registry (design §11.2): ``get_strategy(directive.strategy, **deps)``.

Strategies: ``agent`` (MAF meta-agent proposer), ``gepa`` (DSPy GEPA over text components,
``ci_lab.optim``), ``skillopt`` (SkillOpt-Sleep on skill markdown) and ``guard`` (v2.4 §13,
``ci_lab.lessons_arm``; registered lazily on first use). Every strategy emits a
``ci.optimizer`` span, honours ``ArmDirective.edit_budget`` and returns plain
:class:`~ci_lab.contracts.Edit` commits that go through the same critic/ASSERT/RRSI/OES
gates. Importing this package does not import dspy/gepa/skillopt_sleep (C26) or the guard
arm.
"""
from __future__ import annotations

import importlib
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
# Strategies provided by another module: ``<module>.register()`` calls ``register_strategy``.
EXTERNAL = {"guard": "ci_lab.lessons_arm.strategy"}  # v2.4 §13 (writes harness/guards/*.yaml only)


def _agl_factory(*, journal: Any = None, store: Any = None, client: Any = None,
                 client_factory: Any = None, domain: Any = None, harness_dir: Any = None,
                 committer: Any = None) -> ArmStrategy:
    from ci_lab.agl.algorithm import LlmResourceAlgorithm

    return LlmResourceAlgorithm(journal=journal, store=store, client=client, client_factory=client_factory,
                                domain=domain, harness_dir=harness_dir, committer=committer)


_FACTORIES["agl"] = _agl_factory


class UnknownStrategy(KeyError):
    pass


def register_strategy(name: str, factory: Callable[..., ArmStrategy]) -> None:
    """Register a factory for a strategy named in ``contracts.STRATEGIES`` (e.g. ``guard``)."""
    if name not in STRATEGIES:
        raise UnknownStrategy(f"{name!r} is not in contracts.STRATEGIES {STRATEGIES}")
    _FACTORIES[name] = factory


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


def _load_external(name: str) -> None:
    module = EXTERNAL.get(name)
    if module is None or name in _FACTORIES:
        return
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise UnknownStrategy(f"arm strategy {name!r}: cannot import {module} ({exc})") from exc
    mod.register()


def available() -> tuple[str, ...]:
    """Strategy names that resolve now (registering every importable ``EXTERNAL`` one)."""
    for name in EXTERNAL:
        try:
            _load_external(name)
        except UnknownStrategy:
            continue
    return tuple(n for n in STRATEGIES if n in _FACTORIES)


def get_strategy(name: str, **deps: Any) -> ArmStrategy:
    """Build strategy ``name``. ``deps`` may be a shared superset (e.g. ``proposer``,
    ``domain``, ``scorer``, ``lm``, ``committer``); each strategy takes what it accepts.
    ``EXTERNAL`` strategies are imported and registered on first use. Missing required deps
    raise ``TypeError``."""
    if name in STRATEGIES:
        _load_external(name)
    try:
        factory = _FACTORIES[name]
    except KeyError:
        if name in STRATEGIES:
            raise UnknownStrategy(f"arm strategy {name!r} is not registered (provided by "
                                  f"{EXTERNAL.get(name, 'another module')} via register_strategy)") from None
        raise UnknownStrategy(f"unknown arm strategy {name!r}; expected one of {STRATEGIES}") from None
    accepted = _accepted(factory)
    return factory(**{k: v for k, v in deps.items() if accepted is None or k in accepted})


__all__ = [
    "EXTERNAL",
    "STRATEGIES",
    "AgentStrategy",
    "EditBudgetExceeded",
    "GepaStrategy",
    "SkillOptStrategy",
    "UnknownStrategy",
    "available",
    "get_strategy",
    "git_commit",
    "register_strategy",
]
