"""JSONL span exporter (design §12.2) and GenAI content redaction (C29).

Each line is one versioned span record (see :mod:`ci_lab.telemetry.record`); ``kind`` and
``status.code`` use OTLP enum numbers and nanosecond times are decimal strings, matching the
Aspire ``/api/telemetry`` OTLP-JSON except that attribute maps are flattened.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ci_lab.telemetry.record import SchemaError, from_readable_span, validate

log = logging.getLogger(__name__)

TELEMETRY_SUBDIR = "telemetry"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUPS = 20

# C29: GenAI content attributes/events that may carry prompts, completions, tool payloads.
_SENSITIVE_ATTRS = frozenset({
    "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
    "gen_ai.tool.call.arguments", "gen_ai.tool.call.result",
})
_SENSITIVE_RE = re.compile(r"^gen_ai\.(prompt|completion)(\.|$)|^gen_ai\..*\.(content|messages)$")


def is_sensitive_attr(key: str) -> bool:
    return key in _SENSITIVE_ATTRS or bool(_SENSITIVE_RE.match(key))


def is_sensitive_event(name: str) -> bool:
    return name.startswith("gen_ai.")


def redact_attrs(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (attrs or {}).items() if not is_sensitive_attr(k)}


def redact_span(span: ReadableSpan) -> ReadableSpan:
    """Copy of ``span`` without GenAI content attributes/events (no-op if none present)."""
    attrs = span.attributes or {}
    events = span.events or ()
    if not any(is_sensitive_attr(k) for k in attrs) and not any(
            is_sensitive_event(e.name) for e in events):
        return span
    return ReadableSpan(
        name=span.name, context=span.context, parent=span.parent, resource=span.resource,
        attributes=redact_attrs(attrs),
        events=[Event(e.name, redact_attrs(e.attributes), e.timestamp)
                for e in events if not is_sensitive_event(e.name)],
        links=span.links, kind=span.kind, status=span.status, start_time=span.start_time,
        end_time=span.end_time, instrumentation_scope=span.instrumentation_scope)


class JsonlSpanExporter(SpanExporter):
    """Append spans to ``<run_dir>/telemetry/spans-<pid>.jsonl``; size-rotated, thread-safe,
    never raises into the application. ``sensitive=False`` drops GenAI content (C29)."""

    def __init__(self, run_dir: Path | str, *, sensitive: bool = False,
                 max_bytes: int = DEFAULT_MAX_BYTES, backups: int = DEFAULT_BACKUPS) -> None:
        self.dir = Path(run_dir) / TELEMETRY_SUBDIR
        self.sensitive = sensitive
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self.dir / f"spans-{os.getpid()}.jsonl"

    def _rotate(self, p: Path) -> None:
        stem = p.name[: -len(".jsonl")]
        olds = sorted((int(m.group(1)), q) for q in self.dir.glob(f"{stem}.*.jsonl")
                      if (m := re.fullmatch(rf"{re.escape(stem)}\.(\d+)\.jsonl", q.name)))
        n = olds[-1][0] + 1 if olds else 1
        os.replace(p, self.dir / f"{stem}.{n}.jsonl")
        olds.append((n, None))
        for _, q in olds[: max(0, len(olds) - self.backups)]:
            if q is not None:
                q.unlink(missing_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        lines = []
        for s in spans:
            try:
                lines.append(json.dumps(from_readable_span(s if self.sensitive else redact_span(s)),
                                        separators=(",", ":"), ensure_ascii=False))
            except Exception:  # noqa: BLE001 - a bad span must not break the batch
                log.debug("jsonl: unserializable span dropped", exc_info=True)
        if not lines:
            return SpanExportResult.SUCCESS
        data = ("\n".join(lines) + "\n").encode("utf-8")
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                p = self.path
                try:
                    if p.stat().st_size and p.stat().st_size + len(data) > self.max_bytes:
                        self._rotate(p)
                except FileNotFoundError:
                    pass
                with open(p, "ab") as f:
                    f.write(data)
                    f.flush()
            return SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001 - telemetry must never raise into the app
            log.debug("jsonl: export failed", exc_info=True)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def read_jsonl(paths: Sequence[Path | str]) -> tuple[list[dict[str, Any]], int]:
    """Parse span records from JSONL files; returns (records, skipped_unparseable_lines).

    Unparseable lines (e.g. a torn tail after a crash) are skipped and counted; a well-formed
    line with a missing/unknown ``schemaVersion`` raises :class:`SchemaError` (D7)."""
    spans: list[dict[str, Any]] = []
    bad = 0
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                try:
                    spans.append(validate(obj))
                except SchemaError as exc:
                    raise SchemaError(f"{p}:{n}: {exc}") from None
    return spans, bad
