"""JsonlSpanExporter: shape, redaction (C29), rotation, thread-safety, never-raise."""
from __future__ import annotations

import json
import os
import threading

from opentelemetry.sdk.trace.export import SpanExportResult

from ci_lab.telemetry.jsonl import (
    JsonlSpanExporter,
    is_sensitive_attr,
    is_sensitive_event,
    read_jsonl,
)

KEYS = {"schemaVersion", "traceId", "spanId", "parentSpanId", "name", "kind", "startTimeUnixNano",
        "endTimeUnixNano", "status", "attributes", "events", "links", "resource", "scope"}


def _lines(p):
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x]


def test_writes_otlp_shaped_lines_per_pid(tmp_path, sdk_spans):
    exp = JsonlSpanExporter(tmp_path)
    assert exp.export(sdk_spans) is SpanExportResult.SUCCESS
    assert exp.path == tmp_path / "telemetry" / f"spans-{os.getpid()}.jsonl"
    rows = _lines(exp.path)
    assert len(rows) == 2 and all(set(r) == KEYS for r in rows)
    assert all(r["schemaVersion"] == 1 for r in rows)
    recs, bad = read_jsonl([exp.path])
    assert len(recs) == 2 and bad == 0


def test_redacts_genai_content_by_default(tmp_path, sdk_spans):
    exp = JsonlSpanExporter(tmp_path)
    exp.export(sdk_spans)
    text = exp.path.read_text("utf-8")
    assert "secret" not in text and "old-style" not in text and '"sys"' not in text
    child = next(r for r in _lines(exp.path) if r["name"] == "chat gpt-5")
    assert child["attributes"] == {"gen_ai.request.model": "gpt-5", "gen_ai.usage.input_tokens": 7}
    assert [e["name"] for e in child["events"]] == ["retry"]


def test_sensitive_true_keeps_content(tmp_path, sdk_spans):
    exp = JsonlSpanExporter(tmp_path, sensitive=True)
    exp.export(sdk_spans)
    child = next(r for r in _lines(exp.path) if r["name"] == "chat gpt-5")
    assert "gen_ai.input.messages" in child["attributes"]
    assert "gen_ai.content.prompt" in [e["name"] for e in child["events"]]


def test_sensitive_classification():
    for k in ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
              "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "gen_ai.prompt",
              "gen_ai.completion.0.content", "gen_ai.choice.message.content"):
        assert is_sensitive_attr(k), k
    for k in ("gen_ai.request.model", "gen_ai.usage.output_tokens", "gen_ai.operation.name",
              "ci.prompt_id", "gen_ai.tool.name"):
        assert not is_sensitive_attr(k), k
    assert is_sensitive_event("gen_ai.choice") and not is_sensitive_event("exception")


def test_rotation_keeps_bounded_backups(tmp_path, sdk_spans):
    exp = JsonlSpanExporter(tmp_path, max_bytes=1, backups=2)
    for _ in range(5):
        exp.export(sdk_spans[:1])
    d = tmp_path / "telemetry"
    stem = f"spans-{os.getpid()}"
    names = sorted(p.name for p in d.iterdir())
    assert names == sorted([f"{stem}.jsonl", f"{stem}.3.jsonl", f"{stem}.4.jsonl"])
    assert all(len(_lines(d / n)) == 1 for n in names)


def test_thread_safe_concurrent_exports(tmp_path, sdk_spans):
    exp = JsonlSpanExporter(tmp_path, max_bytes=64 * 1024, backups=1000)

    def work():
        for _ in range(10):
            exp.export(sdk_spans)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    files = sorted((tmp_path / "telemetry").glob("spans-*.jsonl"))
    recs, bad = read_jsonl(files)
    assert bad == 0 and len(recs) == 8 * 10 * 2 and len(files) > 1


def test_never_raises(tmp_path, sdk_spans, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    exp = JsonlSpanExporter(blocker)  # <file>/telemetry cannot be created
    assert exp.export(sdk_spans) is SpanExportResult.FAILURE
    exp2 = JsonlSpanExporter(tmp_path)
    exp2.shutdown()
    exp2.shutdown()
    assert exp2.export(sdk_spans) is SpanExportResult.FAILURE
    assert exp2.force_flush()
