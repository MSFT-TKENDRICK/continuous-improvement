"""MAF function bindings for the declarative order-support agent.

Each binding delegates to the frozen :func:`order_support.tools.execute`, so
every call still emits the same OpenInference TOOL span ASSERT reads, and the
model receives the same JSON tool message as the pre-MAF loop. The tools are
deterministic and idempotent (refund/ticket ids are hashes of the arguments),
which MAF's at-least-once checkpoint resume relies on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from order_support import tools


def _binding(name: str) -> Callable[..., str]:
    def call(**kwargs: Any) -> str:
        return json.dumps(tools.execute(name, kwargs), ensure_ascii=False)

    call.__name__ = name
    call.__doc__ = f"Run order_support.tools.{name} through tools.execute (emits a TOOL span)."
    return call


def bindings() -> dict[str, Callable[..., str]]:
    """``AgentFactory`` bindings keyed by the names agent.yaml's ``bindings`` refer to."""
    return {name: _binding(name) for name in tools.TOOLS}
