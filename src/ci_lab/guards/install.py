"""Frozen installer (B1/B2): the only place guard mode is chosen and ``CI_GUARDS`` is read.

    middleware = install_guards(bundle=bundle, tool_policies=TOOL_POLICIES,
                                sink=run_dir / "guards" / "decisions.jsonl")
    agent = Agent(client=client, tools=[...], middleware=middleware)

Mode: ``mode_override`` (explicit argument) wins, else env ``CI_GUARDS`` (off|shadow|enforce), else
each rule's own ``mode``. ``off`` still evaluates and records every attempt (shadow semantics, no
enforcement) so guard-off arms of paired runs measure attempted violations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ci_lab.guards.engine import GuardEngine, default_engine
from ci_lab.guards.middleware import (
    GuardAgentMiddleware,
    GuardFunctionMiddleware,
    GuardPreflightMiddleware,
)
from ci_lab.guards.recorder import DecisionSink, as_sink
from ci_lab.guards.runtime import (
    MAX_GUARD_BLOCKS_PER_CONVERSATION,
    MODES,
    GuardRuntime,
    Mode,
)
from ci_lab.rulespec import MAX_GUARD_BLOCKS_PER_TURN

GUARDS_ENV = "CI_GUARDS"
_UNSET: Any = object()


class GuardMiddleware(list[Any]):
    """``[agent, preflight(chat), function]`` middleware; pass as ``Agent(middleware=...)``."""

    def __init__(self, runtime: GuardRuntime) -> None:
        super().__init__([GuardAgentMiddleware(runtime), GuardPreflightMiddleware(runtime),
                          GuardFunctionMiddleware(runtime)])
        self.runtime = runtime


def resolve_mode(mode_override: str | None) -> Mode | None:
    mode = mode_override if mode_override is not None else (os.environ.get(GUARDS_ENV) or "").strip().lower() or None
    if mode is not None and mode not in MODES:
        raise ValueError(f"guard mode must be one of {MODES} (got {mode!r})")
    return mode  # type: ignore[return-value]


def install_guards(*, bundle: Any | None, tool_policies: Mapping[str, Any],
                   sink: DecisionSink | Path | str | None = None, mode_override: str | None = None,
                   degraded: bool = False, engine: GuardEngine | None = _UNSET,
                   max_blocks_per_turn: int = MAX_GUARD_BLOCKS_PER_TURN,
                   max_blocks_per_conversation: int = MAX_GUARD_BLOCKS_PER_CONVERSATION,
                   default_side_effect: bool = True, buffer_streams: bool = True) -> GuardMiddleware:
    """Build the guard middleware list for a MAF ``Agent``.

    ``bundle``/``degraded`` come from :func:`ci_lab.guards.load_guard_bundle` (``rules.load_with_lkg``);
    ``bundle=None`` means unavailable: side-effecting tools fail closed, the rest degrade (N1).
    ``tool_policies``: ``{tool: side_effect_bool}`` or tool_specs entries with ``side_effect``; tools
    not listed are treated as side-effecting unless ``default_side_effect=False``.
    """
    return GuardMiddleware(GuardRuntime(
        bundle=bundle, engine=default_engine() if engine is _UNSET else engine, tool_policies=tool_policies,
        mode_override=resolve_mode(mode_override), sink=as_sink(sink), degraded=degraded,
        max_blocks_per_turn=max_blocks_per_turn, max_blocks_per_conversation=max_blocks_per_conversation,
        default_side_effect=default_side_effect, buffer_streams=buffer_streams))
