"""Observability facade (design §12): OTel *API* only — safe to import from any module.

Modules emit harness spans and live run-status markers through this facade. Providers and
exporters (OTLP → Aspire dashboard, JSONL file) are installed by ``ci_lab.telemetry.setup``;
without it every span is a no-op, so tests and library use stay silent.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from ci_lab.contracts import RUN_STATUS_FILE

TRACER_NAME = "ci_lab"
TRACEPARENT_ENV = "TRACEPARENT"

_Scalar = str | bool | int | float


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def _clean(attrs: Mapping[str, Any]) -> dict[str, _Scalar]:
    out: dict[str, _Scalar] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, bool, int, float)) else str(v)
    return out


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None,
         links: list[trace.Link] | None = None) -> Iterator[trace.Span]:
    """Start a harness span (see ``contracts.SPAN_*`` / ``ATTR_*``); records exceptions."""
    with tracer().start_as_current_span(name, attributes=_clean(attributes or {}),
                                        links=links) as s:
        yield s


def current_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) hex of the current span, or None when not recording."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def link_to(trace_id: str, span_id: str) -> trace.Link:
    """Link a resumed round/night to the trace recorded before the interruption."""
    sc = trace.SpanContext(int(trace_id, 16), int(span_id, 16), is_remote=True,
                           trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED))
    return trace.Link(sc)


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a subprocess (ASSERT wrapper, sleep, copilot-serve) carrying the
    W3C trace context so its spans join the current trace."""
    out = dict(os.environ if env is None else env)
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    if "traceparent" in carrier:
        out[TRACEPARENT_ENV] = carrier["traceparent"]
        if "tracestate" in carrier:
            out["TRACESTATE"] = carrier["tracestate"]
    return out


def attach_from_env() -> object | None:
    """In a child process: make the parent's span (from ``TRACEPARENT``) current.
    Returns a token for ``otel_context.detach`` or None."""
    tp = os.environ.get(TRACEPARENT_ENV)
    if not tp:
        return None
    carrier = {"traceparent": tp}
    if os.environ.get("TRACESTATE"):
        carrier["tracestate"] = os.environ["TRACESTATE"]
    return otel_context.attach(propagate.extract(carrier))


def write_status(run_dir: Path | str, experiment_id: str, **fields: Any) -> Path:
    """Atomically merge ``fields`` into ``<run_dir>/<experiment_id>/status.json``.

    The dashboard canvas tails these markers for live progress (spans only export on end).
    ``arms`` is merged per arm. Values must be JSON-serializable and must not contain PII.
    """
    d = Path(run_dir) / experiment_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / RUN_STATUS_FILE
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    arms = fields.pop("arms", None)
    cur.update(fields)
    if arms:
        merged = dict(cur.get("arms") or {})
        for arm, val in arms.items():
            merged[arm] = {**merged.get(arm, {}), **val, "updated": time.time()}
        cur["arms"] = merged
    ids = current_ids()
    if ids and "trace_id" not in cur:
        cur["trace_id"] = ids[0]
    cur["experiment_id"] = experiment_id
    cur["updated"] = time.time()
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".status-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cur, f, sort_keys=True)
    for attempt in range(10):  # Windows: a reader holding the file blocks replace briefly
        try:
            os.replace(tmp, p)
            break
        except PermissionError:
            if attempt == 9:
                os.unlink(tmp)
                raise
            time.sleep(0.05 * (attempt + 1))
    return p
