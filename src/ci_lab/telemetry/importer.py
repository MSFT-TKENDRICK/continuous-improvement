"""Replay span records into a running Aspire dashboard via OTLP/HTTP (protobuf), preserving
trace/span ids and timestamps (e.g. JSONL artifacts from the GitHub Actions ``sleep-nightly``)."""
from __future__ import annotations

import ipaddress
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.telemetry import aspire as _aspire
from ci_lab.telemetry.jsonl import is_sensitive_event, read_jsonl, redact_attrs
from ci_lab.telemetry.record import to_otlp_request

DEFAULT_BATCH = 512


class TelemetryImportError(RuntimeError):
    pass


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _post(url: str, body: bytes, headers: Mapping[str, str], timeout: float = 30.0) -> int:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-protobuf", **headers})
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if _is_loopback(url) else urllib.request.build_opener())
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def redact_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {**rec, "attributes": redact_attrs(rec.get("attributes")),
            "events": [{**e, "attributes": redact_attrs(e.get("attributes"))}
                       for e in rec.get("events") or () if not is_sensitive_event(e.get("name", ""))]}


def import_records(records: Iterable[dict[str, Any]], *, otlp_url: str | None = None,
                   otlp_key: str | None = None, batch: int = DEFAULT_BATCH,
                   redact: bool = True) -> dict[str, int]:
    """POST records to ``<otlp_url>/v1/traces``; defaults to the running dashboard."""
    if otlp_url is None:
        st = _aspire.live_state()
        if not st:
            raise TelemetryImportError("no running dashboard (ci-lab dashboard up) and no --otlp-url given")
        otlp_url, otlp_key = st["otlp_url"], st["otlp_key"]
    url = otlp_url.rstrip("/") + "/v1/traces"
    headers = {"x-otlp-api-key": otlp_key} if otlp_key else {}
    recs = [redact_record(r) if redact else r for r in records]
    sent = batches = 0
    for i in range(0, len(recs), max(1, batch)):
        chunk = recs[i:i + batch]
        code = _post(url, to_otlp_request(chunk).SerializeToString(), headers)
        if code // 100 != 2:
            raise TelemetryImportError(f"OTLP endpoint returned HTTP {code} after {sent} spans")
        sent += len(chunk)
        batches += 1
    return {"spans": sent, "batches": batches}


def expand(paths: Sequence[Path | str]) -> list[Path]:
    """Files as given; directories expand to their ``**/spans-*.jsonl``."""
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.rglob("spans-*.jsonl")) if p.is_dir() else [p])
    return out


def import_files(paths: Sequence[Path | str], **kw: Any) -> dict[str, int]:
    files = expand(paths)
    records, bad = read_jsonl(files)
    res = import_records(records, **kw) if records else {"spans": 0, "batches": 0}
    return {**res, "files": len(files), "skipped_lines": bad}
