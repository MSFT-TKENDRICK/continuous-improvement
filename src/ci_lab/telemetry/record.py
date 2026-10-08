"""Versioned internal span record (design §12.5 D7) and its adapters.

One record = one span, OTLP-JSON-like but with *flattened* attribute maps::

    {"schemaVersion": 1, "traceId": <32 hex>, "spanId": <16 hex>, "parentSpanId": <16 hex|"">,
     "name", "kind": 1..5 (OTLP SpanKind), "startTimeUnixNano": "<decimal>",
     "endTimeUnixNano": "<decimal>", "status": {"code": 0|1|2, "message"},
     "attributes": {k: scalar|[scalars]}, "events": [{"name", "timeUnixNano", "attributes"}],
     "links": [{"traceId", "spanId", "attributes"}], "resource": {k: v}, "scope": {"name", "version"}}

Adapters: SDK ``ReadableSpan`` → record (JSONL writer), Aspire ``/api/telemetry`` /
OTLP-JSON → records, records ↔ OTLP protobuf ``ExportTraceServiceRequest`` (replay).
Records with a missing/unknown ``schemaVersion`` are rejected (:class:`SchemaError`).
"""
from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import (
    AnyValue,
    InstrumentationScope,
    KeyValue,
)
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
    Status,
)
from opentelemetry.sdk.trace import ReadableSpan

from ci_lab.contracts import SPAN_SCHEMA_VERSION

SCHEMA_KEY = "schemaVersion"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_KINDS = {"SPAN_KIND_UNSPECIFIED": 0, "SPAN_KIND_INTERNAL": 1, "SPAN_KIND_SERVER": 2,
          "SPAN_KIND_CLIENT": 3, "SPAN_KIND_PRODUCER": 4, "SPAN_KIND_CONSUMER": 5}
_CODES = {"STATUS_CODE_UNSET": 0, "STATUS_CODE_OK": 1, "STATUS_CODE_ERROR": 2}


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------- values

def _value(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray)):
        return [_value(x) for x in v]
    return str(v)


def _flat(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _value(v) for k, v in (attrs or {}).items() if v is not None}


def _hex(i: int, width: int) -> str:
    return format(i, f"0{width}x")


# ---------------------------------------------------------------- validation

def validate(rec: Any) -> dict[str, Any]:
    """Check + normalize a record; raises :class:`SchemaError`."""
    if not isinstance(rec, dict):
        raise SchemaError("span record must be an object")
    ver = rec.get(SCHEMA_KEY)
    if ver != SPAN_SCHEMA_VERSION:
        raise SchemaError(f"unsupported span schemaVersion {ver!r} (expected {SPAN_SCHEMA_VERSION})")
    tid, sid = str(rec.get("traceId", "")).lower(), str(rec.get("spanId", "")).lower()
    pid = str(rec.get("parentSpanId") or "").lower()
    if not _HEX32.match(tid) or not _HEX16.match(sid) or (pid and not _HEX16.match(pid)):
        raise SchemaError("bad traceId/spanId/parentSpanId")
    try:
        start, end = int(rec["startTimeUnixNano"]), int(rec["endTimeUnixNano"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError("bad start/end time") from exc
    status = rec.get("status") or {}
    return {
        SCHEMA_KEY: SPAN_SCHEMA_VERSION, "traceId": tid, "spanId": sid, "parentSpanId": pid,
        "name": str(rec.get("name", "")), "kind": int(rec.get("kind") or 1),
        "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
        "status": {"code": int(status.get("code") or 0), "message": str(status.get("message") or "")},
        "attributes": dict(rec.get("attributes") or {}),
        "events": [{"name": str(e.get("name", "")), "timeUnixNano": str(int(e.get("timeUnixNano") or 0)),
                    "attributes": dict(e.get("attributes") or {})} for e in rec.get("events") or ()],
        "links": [{"traceId": str(lk["traceId"]).lower(), "spanId": str(lk["spanId"]).lower(),
                   "attributes": dict(lk.get("attributes") or {})} for lk in rec.get("links") or ()],
        "resource": dict(rec.get("resource") or {}),
        "scope": {"name": str((rec.get("scope") or {}).get("name") or ""),
                  "version": str((rec.get("scope") or {}).get("version") or "")},
    }


# ---------------------------------------------------------------- SDK span -> record

def from_readable_span(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.context
    scope = span.instrumentation_scope
    return {
        SCHEMA_KEY: SPAN_SCHEMA_VERSION,
        "traceId": _hex(ctx.trace_id, 32) if ctx else "",
        "spanId": _hex(ctx.span_id, 16) if ctx else "",
        "parentSpanId": _hex(span.parent.span_id, 16) if span.parent else "",
        "name": span.name,
        "kind": int(span.kind.value) + 1,  # SDK INTERNAL=0 -> OTLP 1
        "startTimeUnixNano": str(span.start_time or 0),
        "endTimeUnixNano": str(span.end_time or 0),
        "status": {"code": int(span.status.status_code.value),
                   "message": span.status.description or ""},
        "attributes": _flat(span.attributes),
        "events": [{"name": e.name, "timeUnixNano": str(e.timestamp), "attributes": _flat(e.attributes)}
                   for e in (span.events or ())],
        "links": [{"traceId": _hex(lk.context.trace_id, 32), "spanId": _hex(lk.context.span_id, 16),
                   "attributes": _flat(lk.attributes)} for lk in (span.links or ())],
        "resource": _flat(span.resource.attributes if span.resource else {}),
        "scope": {"name": scope.name if scope else "", "version": (scope.version or "") if scope else ""},
    }


# ---------------------------------------------------------------- OTLP-JSON (Aspire API) -> records

def _from_any_json(v: Mapping[str, Any] | None) -> Any:
    v = v or {}
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_from_any_json(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return json.dumps(_kv_json((v["kvlistValue"] or {}).get("values", [])), sort_keys=True)
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _kv_json(kvs: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    out = {}
    for kv in kvs or ():
        val = _from_any_json(kv.get("value"))
        if val is not None:
            out[kv["key"]] = val
    return out


def _enum(v: Any, table: Mapping[str, int], default: int) -> int:
    if isinstance(v, str):
        return table.get(v, int(v) if v.isdigit() else default)
    return int(v) if v is not None else default


def _json_id(v: Any) -> str:
    """OTLP-JSON ids are hex; tolerate base64 (protobuf-JSON mapping) too."""
    s = str(v or "")
    if not s or re.fullmatch(r"[0-9a-fA-F]+", s):
        return s.lower()
    return base64.b64decode(s).hex()


def from_otlp_json(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Records from an OTLP-JSON ``ExportTraceServiceRequest`` or an Aspire
    ``/api/telemetry/{traces,spans}`` response (``{"data": {"resourceSpans": [...]}}``)."""
    data = payload.get("data", payload) if isinstance(payload, Mapping) else {}
    out: list[dict[str, Any]] = []
    for rs in (data or {}).get("resourceSpans") or ():
        res = _kv_json((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans") or ():
            scope = ss.get("scope") or {}
            for s in ss.get("spans") or ():
                st = s.get("status") or {}
                out.append(validate({
                    SCHEMA_KEY: SPAN_SCHEMA_VERSION, "traceId": _json_id(s.get("traceId")),
                    "spanId": _json_id(s.get("spanId")), "parentSpanId": _json_id(s.get("parentSpanId")),
                    "name": s.get("name", ""), "kind": _enum(s.get("kind"), _KINDS, 1),
                    "startTimeUnixNano": s.get("startTimeUnixNano", "0"),
                    "endTimeUnixNano": s.get("endTimeUnixNano", "0"),
                    "status": {"code": _enum(st.get("code"), _CODES, 0), "message": st.get("message", "")},
                    "attributes": _kv_json(s.get("attributes")),
                    "events": [{"name": e.get("name", ""), "timeUnixNano": e.get("timeUnixNano", "0"),
                                "attributes": _kv_json(e.get("attributes"))} for e in s.get("events") or ()],
                    "links": [{"traceId": _json_id(lk.get("traceId")), "spanId": _json_id(lk.get("spanId")),
                               "attributes": _kv_json(lk.get("attributes"))} for lk in s.get("links") or ()],
                    "resource": res,
                    "scope": {"name": scope.get("name", ""), "version": scope.get("version", "")},
                }))
    return out


# ---------------------------------------------------------------- records <-> OTLP protobuf

def _any(v: Any) -> AnyValue:
    if isinstance(v, bool):
        return AnyValue(bool_value=v)
    if isinstance(v, int):
        return AnyValue(int_value=v)
    if isinstance(v, float):
        return AnyValue(double_value=v)
    if isinstance(v, str):
        return AnyValue(string_value=v)
    if isinstance(v, (list, tuple)):
        a = AnyValue()
        a.array_value.values.extend(_any(x) for x in v if x is not None)
        return a
    return AnyValue(string_value=json.dumps(v, sort_keys=True, default=str))


def _kvs(attrs: Mapping[str, Any]) -> list[KeyValue]:
    return [KeyValue(key=k, value=_any(v)) for k, v in attrs.items() if v is not None]


def _from_any(a: AnyValue) -> Any:
    which = a.WhichOneof("value")
    if which == "array_value":
        return [_from_any(x) for x in a.array_value.values]
    if which == "kvlist_value":
        return json.dumps({kv.key: _from_any(kv.value) for kv in a.kvlist_value.values}, sort_keys=True)
    if which == "bytes_value":
        return base64.b64encode(a.bytes_value).decode()
    return getattr(a, which) if which else None


def _attrs(kvs: Iterable[KeyValue]) -> dict[str, Any]:
    return {kv.key: _from_any(kv.value) for kv in kvs}


def to_otlp_request(records: Iterable[Mapping[str, Any]]) -> ExportTraceServiceRequest:
    """Group validated records by (resource, scope) into one protobuf request; ids and
    timestamps are preserved verbatim."""
    groups: dict[str, dict[str, list[Span]]] = {}
    res_of: dict[str, dict[str, Any]] = {}
    scope_of: dict[str, dict[str, str]] = {}
    for r in records:
        rk = json.dumps(r.get("resource") or {}, sort_keys=True, default=str)
        sk = json.dumps(r.get("scope") or {}, sort_keys=True)
        res_of[rk] = r.get("resource") or {}
        scope_of[sk] = r.get("scope") or {}
        span = Span(
            trace_id=bytes.fromhex(r["traceId"]), span_id=bytes.fromhex(r["spanId"]),
            parent_span_id=bytes.fromhex(r["parentSpanId"]) if r.get("parentSpanId") else b"",
            name=r["name"], kind=int(r.get("kind") or 1),
            start_time_unix_nano=int(r["startTimeUnixNano"]), end_time_unix_nano=int(r["endTimeUnixNano"]),
            attributes=_kvs(r.get("attributes") or {}),
            status=Status(code=int((r.get("status") or {}).get("code") or 0),
                          message=(r.get("status") or {}).get("message") or ""))
        for e in r.get("events") or ():
            span.events.add(time_unix_nano=int(e["timeUnixNano"]), name=e["name"],
                            attributes=_kvs(e.get("attributes") or {}))
        for lk in r.get("links") or ():
            span.links.add(trace_id=bytes.fromhex(lk["traceId"]), span_id=bytes.fromhex(lk["spanId"]),
                           attributes=_kvs(lk.get("attributes") or {}))
        groups.setdefault(rk, {}).setdefault(sk, []).append(span)
    req = ExportTraceServiceRequest()
    for rk, scopes in groups.items():
        rs = ResourceSpans(resource=Resource(attributes=_kvs(res_of[rk])))
        for sk, spans in scopes.items():
            sc = scope_of[sk]
            rs.scope_spans.append(ScopeSpans(
                scope=InstrumentationScope(name=sc.get("name", ""), version=sc.get("version", "")),
                spans=spans))
        req.resource_spans.append(rs)
    return req


def from_otlp_request(req: ExportTraceServiceRequest) -> list[dict[str, Any]]:
    out = []
    for rs in req.resource_spans:
        res = _attrs(rs.resource.attributes)
        for ss in rs.scope_spans:
            for s in ss.spans:
                out.append(validate({
                    SCHEMA_KEY: SPAN_SCHEMA_VERSION, "traceId": s.trace_id.hex(), "spanId": s.span_id.hex(),
                    "parentSpanId": s.parent_span_id.hex(), "name": s.name, "kind": s.kind,
                    "startTimeUnixNano": str(s.start_time_unix_nano),
                    "endTimeUnixNano": str(s.end_time_unix_nano),
                    "status": {"code": s.status.code, "message": s.status.message},
                    "attributes": _attrs(s.attributes),
                    "events": [{"name": e.name, "timeUnixNano": str(e.time_unix_nano),
                                "attributes": _attrs(e.attributes)} for e in s.events],
                    "links": [{"traceId": lk.trace_id.hex(), "spanId": lk.span_id.hex(),
                               "attributes": _attrs(lk.attributes)} for lk in s.links],
                    "resource": res, "scope": {"name": ss.scope.name, "version": ss.scope.version},
                }))
    return out
