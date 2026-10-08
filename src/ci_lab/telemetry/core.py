"""``ci_lab.telemetry.setup``: the single owner of the process's OTel TracerProvider (C28).

Installs exporters behind the OTel API used by ``ci_lab.obs`` and MAF's GenAI
instrumentation: OTLP/HTTP to a running local Aspire dashboard (discovered from the
dashboard state file) and/or the JSONL span file under the run dir.
"""
from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from ci_lab.contracts import ATTR_CAMPAIGN, ATTR_PROFILE
from ci_lab.telemetry import aspire as _aspire
from ci_lab.telemetry.jsonl import JsonlSpanExporter, redact_span

log = logging.getLogger(__name__)

AspireMode = Literal["auto", "on", "off"]
SERVICE_PREFIX = "ci-lab."
ATTR_SENSITIVE = "ci.telemetry.sensitive"  # resource flag: GenAI content capture was on (C29)


class _SafeExporter(SpanExporter):
    """Never raises into the app; redacts GenAI content unless sensitive; idempotent shutdown."""

    def __init__(self, inner: SpanExporter, *, sensitive: bool, label: str) -> None:
        self.inner = inner
        self.sensitive = sensitive
        self.label = label
        self._lock = threading.Lock()
        self._down = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            if not self.sensitive:
                spans = [redact_span(s) for s in spans]
            return self.inner.export(spans)
        except Exception:  # noqa: BLE001
            log.debug("telemetry exporter %s failed", self.label, exc_info=True)
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return self.inner.force_flush(timeout_millis)
        except Exception:  # noqa: BLE001
            return False

    def shutdown(self) -> None:
        with self._lock:
            if self._down:
                return
            self._down = True
        try:
            self.inner.shutdown()
        except Exception:  # noqa: BLE001
            log.debug("telemetry exporter %s shutdown failed", self.label, exc_info=True)


@dataclass
class TelemetryHandle:
    """Result of :func:`setup`. Holds no secrets."""

    component: str
    service_name: str
    resource: dict[str, Any]
    sensitive: bool
    run_dir: Path | None = None
    jsonl_path: Path | None = None
    otlp_url: str | None = None
    owns_provider: bool = False
    provider: TracerProvider | None = field(default=None, repr=False)
    exporters: list[_SafeExporter] = field(default_factory=list, repr=False)
    processors: list[SpanProcessor] = field(default_factory=list, repr=False)
    _extra: list[SpanProcessor] = field(default_factory=list, repr=False)

    @property
    def aspire(self) -> bool:
        return self.otlp_url is not None

    def span_processors(self) -> list[SpanProcessor]:
        """Fresh batch processors over this handle's exporters, for attaching to a foreign
        TracerProvider (e.g. one created by an Agent Lightning tracer) so its spans reach the
        same JSONL file / dashboard: ``for p in h.span_processors(): agl_tp.add_span_processor(p)``.
        They are flushed and shut down by :meth:`shutdown`."""
        procs = [BatchSpanProcessor(e) for e in self.exporters]
        self._extra.extend(procs)
        return procs

    def force_flush(self, timeout_millis: int = 10000) -> bool:
        ok = True
        procs: list[Any] = [*self.processors, *self._extra]
        if self.owns_provider and self.provider is not None and len(self.processors) < len(self.exporters):
            procs.append(self.provider)  # processors not located: flush the whole provider
        for p in procs:
            try:
                ok = p.force_flush(timeout_millis) and ok
            except Exception:  # noqa: BLE001
                ok = False
        return ok

    def shutdown(self) -> None:
        """Flush and stop our exporters. The global provider stays installed (OTel cannot
        replace it); a later :func:`setup` attaches fresh exporters to it."""
        global _HANDLE
        self.force_flush()
        for p in [*self.processors, *self._extra]:
            try:
                p.shutdown()
            except Exception:  # noqa: BLE001
                log.debug("span processor shutdown failed", exc_info=True)
        for e in self.exporters:
            e.shutdown()
        self.processors.clear()
        self._extra.clear()
        with _LOCK:
            if _HANDLE is self:
                _HANDLE = None


_LOCK = threading.RLock()
_HANDLE: TelemetryHandle | None = None
_ATEXIT = False


def current() -> TelemetryHandle | None:
    return _HANDLE


def git_ref(cwd: Path | str | None = None) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                           text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and len(out) in (40, 64) else None


def build_resource_attrs(component: str, *, profile: str | None, campaign_id: str | None,
                         sensitive: bool, cwd: Path | None = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "service.name": SERVICE_PREFIX + component,
        "service.instance.id": f"{socket.gethostname()}-{os.getpid()}",
        "process.pid": os.getpid(),
        ATTR_SENSITIVE: sensitive,
    }
    if profile:
        attrs[ATTR_PROFILE] = profile
    if campaign_id:
        attrs[ATTR_CAMPAIGN] = campaign_id
    ref = git_ref(cwd)
    if ref:
        attrs["vcs.ref"] = ref
    return attrs


def _otlp_exporter(state: dict[str, Any]) -> SpanExporter:
    import requests
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    session = requests.Session()
    session.trust_env = False  # loopback: never route through HTTP(S)_PROXY
    return OTLPSpanExporter(endpoint=f"{state['otlp_url'].rstrip('/')}/v1/traces",
                            headers={"x-otlp-api-key": state["otlp_key"]}, timeout=5,
                            session=session)


def _resolve_aspire(mode: AspireMode) -> dict[str, Any] | None:
    if mode == "off":
        return None
    st = _aspire.live_state(timeout=0.5)
    if st is None and mode == "on":
        raise RuntimeError("aspire='on' but no running dashboard found (run: ci-lab dashboard up)")
    return st


def _install_provider(handle: TelemetryHandle, exporters: list[_SafeExporter]) -> None:
    from agent_framework.observability import configure_otel_providers, enable_instrumentation

    tp = trace.get_tracer_provider()
    if not isinstance(tp, (TracerProvider, trace.ProxyTracerProvider)):
        # D4: someone installed a non-SDK provider first; we cannot attach processors and
        # OTel will not let us replace it, so spans would silently vanish.
        raise RuntimeError(
            f"a non-SDK global TracerProvider ({type(tp).__name__}) is already installed; "
            "call ci_lab.telemetry.setup() first at process entry")
    if isinstance(tp, TracerProvider):
        # C28: a provider already exists (an earlier setup, or a library such as an AGL
        # tracer). Never replace it — attach our exporters to it instead.
        handle.provider = tp
        handle.owns_provider = tp is _OWNED.get("provider")
        for e in exporters:
            p = BatchSpanProcessor(e)
            tp.add_span_processor(p)
            handle.processors.append(p)
        enable_instrumentation(enable_sensitive_data=handle.sensitive)
        return
    configure_otel_providers(
        service_name=handle.service_name,
        resource_attributes={k: v for k, v in handle.resource.items() if k != "service.name"},
        enable_sensitive_data=handle.sensitive,
        exporters=list(exporters),
    )
    enable_instrumentation(enable_sensitive_data=handle.sensitive)
    tp = trace.get_tracer_provider()
    if not isinstance(tp, TracerProvider):
        # No exporters at all: still own a provider so later setups/AGL can attach to it.
        from agent_framework.observability import create_resource

        tp = TracerProvider(resource=create_resource(
            service_name=handle.service_name,
            attributes={k: v for k, v in handle.resource.items() if k != "service.name"}))
        trace.set_tracer_provider(tp)
    handle.provider = tp
    handle.owns_provider = True
    _OWNED["provider"] = tp
    procs = getattr(getattr(tp, "_active_span_processor", None), "_span_processors", ())
    mine = {id(e) for e in exporters}
    handle.processors.extend(p for p in procs if id(getattr(p, "span_exporter", None)) in mine
                             or id(getattr(getattr(p, "_batch_processor", None), "_exporter", None)) in mine)


_OWNED: dict[str, Any] = {}


def setup(component: str, *, profile: str | None = None, run_dir: Path | str | None = None,
          aspire: AspireMode = "auto", jsonl: bool = True, sensitive: bool = False,
          campaign_id: str | None = None) -> TelemetryHandle:
    """Install telemetry for this process (idempotent; first call wins).

    * ``aspire="auto"`` exports to a running ``ci-lab dashboard`` if its state file points at a
      live instance, silently skipping otherwise; ``"on"`` requires it; ``"off"`` never.
    * ``jsonl`` writes ``<run_dir>/telemetry/spans-<pid>.jsonl`` (needs ``run_dir``).
    * ``sensitive`` enables MAF GenAI content capture — only for fake/local profiles (C29).

    A repeated call returns the existing handle; if it brings a ``run_dir`` and the handle has
    no JSONL exporter yet, one is attached.
    """
    global _HANDLE, _ATEXIT
    with _LOCK:
        if _HANDLE is not None:
            if jsonl and run_dir is not None and _HANDLE.jsonl_path is None:
                exp = _SafeExporter(JsonlSpanExporter(run_dir, sensitive=_HANDLE.sensitive),
                                    sensitive=_HANDLE.sensitive, label="jsonl")
                proc = BatchSpanProcessor(exp)
                if _HANDLE.provider is not None:
                    _HANDLE.provider.add_span_processor(proc)
                _HANDLE.exporters.append(exp)
                _HANDLE.processors.append(proc)
                _HANDLE.run_dir = Path(run_dir)
                _HANDLE.jsonl_path = exp.inner.path  # type: ignore[attr-defined]
            return _HANDLE
        rd = Path(run_dir) if run_dir is not None else None
        resource = build_resource_attrs(component, profile=profile, campaign_id=campaign_id,
                                        sensitive=sensitive, cwd=None)
        handle = TelemetryHandle(component=component, service_name=resource["service.name"],
                                 resource=resource, sensitive=sensitive, run_dir=rd)
        exporters: list[_SafeExporter] = []
        if jsonl and rd is not None:
            inner = JsonlSpanExporter(rd, sensitive=sensitive)
            exporters.append(_SafeExporter(inner, sensitive=sensitive, label="jsonl"))
            handle.jsonl_path = inner.path
        st = _resolve_aspire(aspire)
        if st is not None:
            exporters.append(_SafeExporter(_otlp_exporter(st), sensitive=sensitive, label="otlp"))
            handle.otlp_url = st["otlp_url"]
        handle.exporters = exporters
        _install_provider(handle, exporters)
        _HANDLE = handle
        if not _ATEXIT:
            atexit.register(_flush_at_exit)
            _ATEXIT = True
        return handle


def _flush_at_exit() -> None:
    h = _HANDLE
    if h is not None:
        h.shutdown()


def shutdown() -> None:
    """Flush and stop the current handle's exporters (no-op if not set up)."""
    h = _HANDLE
    if h is not None:
        h.shutdown()
