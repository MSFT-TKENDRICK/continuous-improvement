"""C28 seam: attach ci_lab's span processors to a non-global (e.g. AGL) tracer provider.

``ci_lab.telemetry.setup`` (M12) is the single owner of the *global* tracer provider; this
module never calls ``set_tracer_provider``. agentlightning 1.0.2's server/store/proxy install
no OpenTelemetry provider of their own, so in-process AGL spans (if any) and our ``ci.case``
spans already flow to the global provider. Should an AGL component ever own a separate
``TracerProvider``, call :func:`attach_telemetry` on it so its spans reach the same exporters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

log = logging.getLogger(__name__)

_ATTACHED_FLAG = "_ci_lab_processors_attached"


def _default_processors() -> Iterable[Any]:
    try:
        from ci_lab.telemetry import span_processors  # type: ignore[import-not-found,attr-defined]
    except ImportError:
        return ()
    return span_processors()


def attach_telemetry(provider: Any, processors: Iterable[Any] | Callable[[], Iterable[Any]] | None = None) -> int:
    """Add ci_lab span processors to ``provider`` once (idempotent per provider).

    ``processors`` defaults to ``ci_lab.telemetry.span_processors()`` when that module exists
    (no-op otherwise). Returns the number of processors attached by this call.
    """
    if provider is None or not hasattr(provider, "add_span_processor"):
        return 0
    if getattr(provider, _ATTACHED_FLAG, False):
        return 0
    procs = list(processors() if callable(processors) else processors if processors is not None
                 else _default_processors())
    for p in procs:
        provider.add_span_processor(p)
    if procs:
        try:
            setattr(provider, _ATTACHED_FLAG, True)
        except AttributeError:
            log.debug("cannot flag %r as attached", provider)
    return len(procs)
