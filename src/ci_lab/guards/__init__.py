"""Runtime trajectory correction (design §13, R2/R3): rules evaluated by the frozen engine
(``ci_lab.rules``) inside MAF middleware the model cannot bypass. See ``docs/guards.md``."""

from ci_lab.guards.engine import GuardEngine, GuardUnavailable, load_guard_bundle
from ci_lab.guards.install import (
    GUARDS_ENV,
    GuardMiddleware,
    install_guards,
    resolve_mode,
)
from ci_lab.guards.middleware import (
    GuardAgentMiddleware,
    GuardFunctionMiddleware,
    GuardPreflightMiddleware,
)
from ci_lab.guards.recorder import (
    STATE_KEY,
    JsonlDecisionSink,
    TrajectoryRecorder,
    read_decisions,
)
from ci_lab.guards.runtime import GuardRuntime
from ci_lab.guards.stream import GuardStreamingError

__all__ = [
    "GUARDS_ENV",
    "STATE_KEY",
    "GuardAgentMiddleware",
    "GuardEngine",
    "GuardFunctionMiddleware",
    "GuardMiddleware",
    "GuardPreflightMiddleware",
    "GuardRuntime",
    "GuardStreamingError",
    "GuardUnavailable",
    "JsonlDecisionSink",
    "TrajectoryRecorder",
    "install_guards",
    "load_guard_bundle",
    "read_decisions",
    "resolve_mode",
]
