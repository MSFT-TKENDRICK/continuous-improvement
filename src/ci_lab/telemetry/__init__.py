"""Telemetry installation for the harness (design §12, M12).

``setup()`` owns the OTel TracerProvider; ``ci_lab.obs`` (OTel API only) emits harness spans
and MAF emits GenAI spans through it. Exporters: OTLP/HTTP to a local Aspire dashboard
(``ci-lab dashboard up``) and JSONL span records under the run dir.
"""
from __future__ import annotations

from ci_lab.telemetry.core import TelemetryHandle, current, setup, shutdown
from ci_lab.telemetry.jsonl import JsonlSpanExporter, read_jsonl, redact_span
from ci_lab.telemetry.record import SchemaError

__all__ = ["JsonlSpanExporter", "SchemaError", "TelemetryHandle", "current", "read_jsonl",
           "redact_span", "setup", "shutdown"]
