"""Observability facade (design §12): OTel *API* only — safe to import from any module.

Modules emit harness spans and live run-status markers through this facade. Providers and
exporters (OTLP → Aspire dashboard, JSONL file) are installed by ``ci_lab.telemetry.setup``;
without it every span is a no-op, so tests and library use stay silent.

Context propagation rules (§12.5):
- in-process async tasks inherit context automatically;
- thread pools: submit ``wrap_ctx(fn)``;
- one-shot subprocesses: ``env=child_env()`` + ``attach_from_env()`` in the child;
- long-lived services (copilot-serve, AGL proxy): send ``carrier()`` per request and
  serve each request inside ``use_carrier(headers)`` — never attach once at startup;
- durable MAF messages / checkpoints: store ``carrier()`` in the payload.
"""
from __future__ import annotations

import contextvars
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from ci_lab.contracts import RUN_STATUS_DIR, RUN_STATUS_FILE

TRACER_NAME = "ci_lab"
TRACEPARENT_ENV = "TRACEPARENT"
MAX_ATTR_LEN = 1024
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_WRITER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_Scalar = str | bool | int | float
_lock = threading.Lock()


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def _scalar(v: Any) -> _Scalar | None:
    if isinstance(v, bool | int | float):
        return v
    if isinstance(v, str):
        return v if len(v) <= MAX_ATTR_LEN else v[:MAX_ATTR_LEN] + "…"
    return None


def _clean(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded attributes: scalars (strings truncated) and homogeneous scalar sequences.
    Anything else is replaced by ``<TypeName>`` — never ``str()``-ed (may hold PII)."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        s = _scalar(v)
        if s is not None:
            out[k] = s
            continue
        if isinstance(v, list | tuple) and v:
            items = [_scalar(x) for x in v[:64]]
            if all(x is not None for x in items) and len({type(x) for x in items}) == 1:
                out[k] = items
                continue
        out[k] = f"<{type(v).__name__}>"
    return out


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None,
         links: list[trace.Link] | None = None, *,
         new_trace: bool = False) -> Iterator[trace.Span]:
    """Start a harness span (``contracts.SPAN_*`` / ``ATTR_*``).

    ``new_trace=True`` starts a fresh root (one trace per round/night — C27).
    Exceptions mark the span ERROR with the exception *type only*; messages and stack
    traces are not recorded (may contain PII — C29)."""
    ctx = otel_context.Context() if new_trace else None
    with tracer().start_as_current_span(name, context=ctx, attributes=_clean(attributes or {}),
                                        links=links, record_exception=False,
                                        set_status_on_exception=False) as s:
        try:
            yield s
        except BaseException as e:
            s.set_status(trace.Status(trace.StatusCode.ERROR, type(e).__name__))
            s.add_event("exception", {"exception.type": type(e).__qualname__})
            raise


def annotate(attributes: Mapping[str, Any]) -> None:
    """Set bounded attributes on the current span (no-op when not recording)."""
    s = trace.get_current_span()
    if s.is_recording():
        s.set_attributes(_clean(attributes))


def current_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) hex of the current span, or None when not recording."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def link_to(trace_id: str, span_id: str) -> trace.Link | None:
    """Link a resumed round/night to a span of the trace recorded before the interruption.
    Returns None for malformed/zero ids (stale or tampered status files)."""
    t, s = str(trace_id).lower(), str(span_id).lower()
    if not (_HEX32.match(t) and _HEX16.match(s)) or int(t, 16) == 0 or int(s, 16) == 0:
        return None
    sc = trace.SpanContext(int(t, 16), int(s, 16), is_remote=True,
                           trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED))
    return trace.Link(sc, {"ci.link": "resume"})


def carrier() -> dict[str, str]:
    """W3C trace-context headers for the current span (per request / per message)."""
    c: dict[str, str] = {}
    propagate.inject(c)
    return c


@contextmanager
def use_carrier(headers: Mapping[str, str] | None) -> Iterator[None]:
    """Serve one request/work item inside the caller's trace context; always detaches."""
    token = otel_context.attach(
        propagate.extract(dict(headers or {}), context=otel_context.Context())
    )
    try:
        yield
    finally:
        otel_context.detach(token)


def wrap_ctx[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Bind the current contextvars (incl. OTel context) for thread-pool submission."""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.run(fn, *a, **kw)


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a ONE-SHOT subprocess (ASSERT wrapper, sleep night) carrying W3C trace
    context. Not for long-lived servers — use ``carrier()``/``use_carrier`` per request."""
    out = {k: v for k, v in (os.environ if env is None else env).items()
           if k not in (TRACEPARENT_ENV, "TRACESTATE")}
    c = carrier()
    if "traceparent" in c:
        out[TRACEPARENT_ENV] = c["traceparent"]
        if "tracestate" in c:
            out["TRACESTATE"] = c["tracestate"]
    return out


def attach_from_env() -> object | None:
    """In a one-shot child process: make the parent's span (from ``TRACEPARENT``) current.
    Returns a token for ``opentelemetry.context.detach`` or None."""
    tp = os.environ.get(TRACEPARENT_ENV)
    if not tp:
        return None
    c = {"traceparent": tp}
    if os.environ.get("TRACESTATE"):
        c["tracestate"] = os.environ["TRACESTATE"]
    return otel_context.attach(propagate.extract(c))


def _atomic_write(p: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".status-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True)
    for attempt in range(10):  # Windows: a reader holding the file blocks replace briefly
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 9:
                os.unlink(tmp)
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_json(p: Path) -> dict[str, Any]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write_status(run_dir: Path | str, experiment_id: str, *, writer: str | None = None,
                 **fields: Any) -> Path:
    """Merge ``fields`` into this writer's marker ``<run_dir>/<exp>/status.d/<writer>.json``.

    Each process/arm writes only its own file (default writer ``pid<pid>``), so concurrent
    writers never lose updates; ``read_status`` aggregates. ``arms`` merges per arm. The
    current span ids are recorded under ``trace`` for live focus and resume linking.
    Values must be JSON-serializable and must not contain PII/secrets."""
    writer = writer or f"pid{os.getpid()}"
    if not _WRITER.match(writer):
        raise ValueError(f"invalid status writer id: {writer!r}")
    d = Path(run_dir) / experiment_id / RUN_STATUS_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{writer}.json"
    with _lock:
        cur = _read_json(p)
        arms = fields.pop("arms", None)
        cur.update(fields)
        now = time.time()
        if arms:
            merged = dict(cur.get("arms") or {})
            for arm, val in arms.items():
                merged[arm] = {**merged.get(arm, {}), **val, "updated": now}
            cur["arms"] = merged
        ids = current_ids()
        if ids:
            cur["trace"] = {"trace_id": ids[0], "span_id": ids[1]}
        cur.update(experiment_id=experiment_id, writer=writer, updated=now,
                   seq=int(cur.get("seq", 0)) + 1)
        _atomic_write(p, cur)
    return p


def read_status(run_dir: Path | str, experiment_id: str) -> dict[str, Any]:
    """Aggregate all writers' markers (plus legacy ``status.json``): top-level fields from
    the most recently updated writer win; arms merge per arm by their ``updated`` time."""
    base = Path(run_dir) / experiment_id
    docs = [_read_json(f) for f in sorted((base / RUN_STATUS_DIR).glob("*.json"))]
    legacy = base / RUN_STATUS_FILE
    if legacy.is_file():
        docs.append(_read_json(legacy))
    docs = sorted((x for x in docs if x), key=lambda x: float(x.get("updated", 0) or 0))
    out: dict[str, Any] = {}
    arms: dict[str, dict[str, Any]] = {}
    for doc in docs:
        for arm, val in (doc.get("arms") or {}).items():
            if not isinstance(val, dict):
                continue
            if float(val.get("updated", 0) or 0) >= float(arms.get(arm, {}).get("updated", 0) or 0):
                arms[arm] = val
        out.update({k: v for k, v in doc.items() if k not in ("arms", "writer", "seq")})
    if arms:
        out["arms"] = arms
    out["writers"] = [doc.get("writer") for doc in docs if doc.get("writer")]
    return out


def previous_link(run_dir: Path | str, experiment_id: str) -> list[trace.Link]:
    """Links for a resumed round: [link to the last recorded span], or [] if none/invalid."""
    t = read_status(run_dir, experiment_id).get("trace") or {}
    link = link_to(t.get("trace_id", ""), t.get("span_id", "")) if isinstance(t, dict) else None
    return [link] if link else []
