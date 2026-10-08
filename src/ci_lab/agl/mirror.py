"""Journal-first mirroring to ``agl-server`` (design C2) and the Copilot ``model_request`` adapter.

:class:`MirroringJournal` always writes the local :class:`FileRolloutJournal` first, then
best-effort mirrors to the server. Mirror failures are logged (never raised), mark the
rollout dirty and switch the mirror offline (so a dead server cannot stall agents); a
later :meth:`MirroringJournal.sync` reconciles each dirty rollout idempotently from the
journal: create (idempotent by id) -> post events missing on the server (matched by the
``ci_event_id`` we stamp into each event's data) -> advance state (409 reconciled).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import httpx

from ci_lab.agl.client import AglClient, AglError
from ci_lab.agl.journal import FileRolloutJournal, RolloutRecord
from ci_lab.contracts import RolloutKey, op_id

if TYPE_CHECKING:
    from ci_lab.agl.scope import RolloutScope

log = logging.getLogger(__name__)

EVENT_ID_FIELD = "ci_event_id"
MODEL_REQUEST_FIELDS = ("model", "model_version", "request", "response", "server", "latency_ms",
                        "http_status", "status", "retry_count", "usage", "finish_reason")
_MIRROR_ERRORS = (httpx.HTTPError, AglError, OSError, ValueError)


@dataclass
class SyncReport:
    rollouts: int = 0
    created: int = 0
    events_posted: int = 0
    state_patches: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _server_input(record: RolloutRecord) -> Any:
    inp: Any = dict(record.input)
    if record.key is not None and "data_id" not in inp:
        inp["data_id"] = record.key.case_id  # lets /api/rollouts/terminal report the case
    return inp


def _metadata(key: RolloutKey | None) -> dict[str, Any] | None:
    if key is None:
        return None
    return {"experiment_id": key.experiment_id, "variant": key.variant, "case_id": key.case_id,
            "trial": key.trial}


class MirroringJournal:
    """:class:`ci_lab.contracts.RolloutJournal` that mirrors a :class:`FileRolloutJournal` to AGL."""

    def __init__(self, journal: FileRolloutJournal, client: AglClient | None = None, *,
                 is_train: bool = False) -> None:
        self.journal = journal
        self.client = client
        self.is_train = is_train
        self.offline = False
        self._dirty: set[str] = set()
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"MirroringJournal({self.journal!r}, {self.client!r}, dirty={len(self._dirty)})"

    @property
    def dirty(self) -> frozenset[str]:
        return frozenset(self._dirty)

    # ------------------------------------------------------------ RolloutJournal

    def start(self, key: RolloutKey, input: Mapping[str, Any]) -> None:
        self.journal.start(key, input)
        self._mirror(key.rollout_id, lambda c: self._remote_start(c, key, input))

    def event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> None:
        if not self.journal.append_event(key, event_type, data, event_id=event_id):
            return  # duplicate: already journaled (and mirrored or pending)
        self._mirror(key.rollout_id, lambda c: c.post_event(
            key.rollout_id, key.attempt_id, event_type, {**data, EVENT_ID_FIELD: event_id}))

    def finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> None:
        self.journal.finish(key, status)
        record = self.journal.load(key.rollout_id)
        final = record.status if record is not None and record.terminal else status
        self._mirror(key.rollout_id, lambda c: c.patch_state(key.rollout_id, final))

    def events(self, key: RolloutKey) -> list[dict[str, Any]]:
        return self.journal.events(key)

    # ------------------------------------------------------------ mirroring

    def _remote_start(self, client: AglClient, key: RolloutKey, input: Mapping[str, Any]) -> None:
        record = self.journal.load(key.rollout_id)
        inp = _server_input(record) if record is not None else dict(input)
        client.create_rollout(key.rollout_id, inp, is_train=self.is_train, metadata=_metadata(key))
        client.patch_state(key.rollout_id, "running", attempt_id=key.attempt_id)

    def _mirror(self, rollout_id: str, op: Callable[[AglClient], Any]) -> None:
        client = self.client
        if client is None:
            return
        with self._lock:
            if self.offline or rollout_id in self._dirty:
                self._dirty.add(rollout_id)
                return
        try:
            op(client)
        except _MIRROR_ERRORS as exc:
            log.warning("AGL mirror failed for %s (%s: %s); will retry on sync()",
                        rollout_id, type(exc).__name__, exc)
            with self._lock:
                self._dirty.add(rollout_id)
                self.offline = True

    def sync(self, rollout_ids: Iterable[str] | None = None, *, all_rollouts: bool = False) -> SyncReport:
        """Reconcile dirty (or the given / all journaled) rollouts with the server.

        Use ``all_rollouts=True`` against a fresh (in-memory) server to replay the whole journal.
        """
        report = SyncReport()
        if self.client is None:
            return report
        with self._lock:
            if rollout_ids is not None:
                targets = list(rollout_ids)
            elif all_rollouts:
                targets = self.journal.rollout_ids()
            else:
                targets = sorted(self._dirty)
            self.offline = False
        for rid in targets:
            report.rollouts += 1
            try:
                self._reconcile(self.client, rid, report)
            except _MIRROR_ERRORS as exc:
                log.warning("AGL sync failed for %s (%s: %s)", rid, type(exc).__name__, exc)
                report.failed.append(rid)
                with self._lock:
                    self._dirty.add(rid)
                continue
            with self._lock:
                self._dirty.discard(rid)
        if report.failed:
            with self._lock:
                self.offline = True
        return report

    def _reconcile(self, client: AglClient, rollout_id: str, report: SyncReport) -> None:
        record = self.journal.load(rollout_id)
        if record is None or record.status is None:
            return
        detail = client.get_rollout(rollout_id)
        if detail is None:
            client.create_rollout(rollout_id, _server_input(record), is_train=self.is_train,
                                  metadata=_metadata(record.key))
            report.created += 1
            detail = client.get_rollout(rollout_id) or {}
        server = detail.get("rollout", {})
        state = server.get("status", {}).get("state", "queuing")
        server_last = server.get("status", {}).get("last_attempt_id")
        attempts = list(dict.fromkeys(record.attempts + [e["attempt_id"] for e in record.events]))
        latest = record.latest_attempt or (attempts[-1] if attempts else "0")
        if state == "queuing":
            client.patch_state(rollout_id, "running", attempt_id=latest)
            report.state_patches += 1
            server_last = latest
        for aid in attempts:
            mine = record.events_for(attempt_id=aid)
            if not mine:
                continue
            if server_last != aid:
                client.patch_status(rollout_id, last_attempt_id=aid)  # GET /events reads last_attempt_id
                server_last = aid
            have = {e.get("data", {}).get(EVENT_ID_FIELD) for e in client.get_events(rollout_id)}
            for ev in mine:
                if ev["event_id"] in have:
                    continue
                client.post_event(rollout_id, aid, ev["event_type"], {**ev["data"], EVENT_ID_FIELD: ev["event_id"]})
                report.events_posted += 1
        if server_last != latest:
            client.patch_status(rollout_id, last_attempt_id=latest)
        if record.terminal and state not in ("succeeded", "failed"):
            client.patch_state(rollout_id, str(record.status), attempt_id=latest)
            report.state_patches += 1

    def import_server_events(self, key: RolloutKey, *, event_type: str | None = "model_request") -> int:
        """Journal events that exist only on the server (e.g. ``model_request`` written by the AGL
        proxy in the offline profile). Ids are derived from (rollout, attempt, server timestamp,
        position) so repeated imports are idempotent. Returns the number newly journaled."""
        if self.client is None:
            return 0
        try:
            detail = self.client.get_rollout(key.rollout_id)
            if detail is None:
                return 0
            if (detail["rollout"]["status"].get("last_attempt_id") or "0") != key.attempt_id:
                self.client.patch_status(key.rollout_id, last_attempt_id=key.attempt_id)
            remote = self.client.get_events(key.rollout_id, event_type=event_type)
        except _MIRROR_ERRORS as exc:
            log.warning("AGL event import failed for %s (%s: %s)", key.rollout_id, type(exc).__name__, exc)
            return 0
        added = 0
        for i, ev in enumerate(remote):
            data = dict(ev.get("data") or {})
            if data.get(EVENT_ID_FIELD):
                continue  # ours, already journaled
            eid = op_id(key.rollout_id, key.attempt_id, "server", ev.get("event_type"), ev.get("timestamp"), i)
            data.pop("routed_experts", None)
            if self.journal.append_event(key, str(ev.get("event_type")), data, event_id=eid):
                added += 1
        return added


# ---------------------------------------------------------------- Copilot model_request adapter

def _first(d: Mapping[str, Any], *names: str) -> Any:
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


def model_request_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a provider callback dict into the AGL proxy's ``model_request`` field set.

    Accepts the proxy field names plus common aliases (``served_model``, ``endpoint``,
    ``error``, ``input_tokens``/``output_tokens``, ``messages``/``content``). Keys outside the
    proxy field set (and not consumed as aliases) are kept under ``data["ci"]``.
    """
    response = raw.get("response")
    if not isinstance(response, Mapping):
        content = _first(raw, "content", "text", "output")
        response = {} if content is None else {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": raw.get("finish_reason")}]}
    response = dict(response)
    request = raw.get("request")
    if not isinstance(request, Mapping):
        request = {k: raw[k] for k in ("messages", "tools", "options") if k in raw}
    model = _first(raw, "model", "served_model") or response.get("model") or "unknown"
    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else response.get("usage")
    if not isinstance(usage, Mapping):
        tin, tout = raw.get("input_tokens"), raw.get("output_tokens")
        usage = None if tin is None and tout is None else {
            "prompt_tokens": int(tin or 0), "completion_tokens": int(tout or 0),
            "total_tokens": int(tin or 0) + int(tout or 0)}
    finish = raw.get("finish_reason")
    if finish is None:
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish = choices[0].get("finish_reason") or None
    http_status = raw.get("http_status")
    error = raw.get("error")
    status = raw.get("status") or ("error" if error or (http_status or 0) >= 400 else "ok")
    if error is not None and "error" not in response:
        response["error"] = error if isinstance(error, (str, Mapping)) else str(error)
    version = raw.get("model_version")
    server = raw.get("server")
    if not isinstance(server, Mapping):
        server = {"model": model, "endpoint": _first(raw, "endpoint", "provider") or "copilot",
                  "version": version}
    data: dict[str, Any] = {
        "model": model,
        "model_version": version,
        "request": dict(request),
        "response": response,
        "server": dict(server),
        "latency_ms": raw.get("latency_ms"),
        "http_status": http_status if http_status is not None else (500 if status == "error" else 200),
        "status": status,
        "retry_count": int(raw.get("retry_count") or 0),
        "usage": dict(usage) if usage is not None else None,
        "finish_reason": finish,
    }
    consumed = set(MODEL_REQUEST_FIELDS) | {"served_model", "endpoint", "provider", "error", "input_tokens",
                                            "output_tokens", "messages", "tools", "options", "content",
                                            "text", "output", "name", "event_name"}
    extra = {k: v for k, v in raw.items() if k not in consumed}
    if raw.get("served_model") is not None:
        extra["served_model"] = raw["served_model"]
    if extra:
        data["ci"] = extra
    return data


def model_request_recorder(scope: RolloutScope | None = None) -> Callable[[dict[str, Any]], None]:
    """Adapter for ``CopilotChatClient(on_model_request=...)``.

    Records each callback dict as a ``model_request`` event on ``scope`` (or, when None, on
    the scope active in :data:`ci_lab.agl.scope.current_rollout` at call time; calls outside
    any scope are dropped). A ``name`` key (e.g. ``"turn-3"``) gives a stable event id;
    otherwise the scope's per-attempt sequence is used.
    """
    from ci_lab.agl.scope import current_rollout

    def record(raw: dict[str, Any]) -> None:
        target = scope or current_rollout.get()
        if target is None:
            log.debug("model_request outside any RolloutScope dropped")
            return
        target.record_model_request(model_request_data(raw), name=_first(raw, "name", "event_name"))

    return record
