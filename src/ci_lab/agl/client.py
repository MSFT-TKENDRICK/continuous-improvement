"""Thin synchronous REST client for the Agent Lightning 1.0.2 server (``agl-server``).

Shapes follow ``agentlightning.server.routes`` exactly: list bodies for
``POST /api/models`` and ``POST /api/rollouts``; ``RolloutDetail`` from
``GET /api/rollouts/{id}``; nested ``status`` patch; per-attempt event POST.
The API key travels only in the ``x-api-key`` header and is never logged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal
from urllib.parse import quote

import httpx

from ci_lab import obs
from ci_lab.contracts import RolloutKey

log = logging.getLogger(__name__)

STATES = ("queuing", "running", "succeeded", "failed")
TERMINAL_STATES = frozenset({"succeeded", "failed"})
DEFAULT_ATTEMPT_ID = "0"
ProxyMode = Literal["train", "val"]


class AglError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class AglConflict(AglError):
    """409 that could not be reconciled (e.g. a non-terminal state disagreement)."""


def _seg(value: str) -> str:
    return quote(str(value), safe="")


def _inject_trace_context(request: httpx.Request) -> None:
    """Per-request W3C trace context (design §12.5): long-lived servers get it per call."""
    for k, v in obs.carrier().items():
        request.headers[k] = v


def _error(resp: httpx.Response, what: str) -> AglError:
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001 - non-JSON error body
        detail = resp.text[:200]
    cls = AglConflict if resp.status_code == 409 else AglError
    return cls(f"{what} -> HTTP {resp.status_code}: {detail}", status_code=resp.status_code, detail=detail)


class AglClient:
    """Synchronous client for one ``agl-server``. Use as a context manager or call :meth:`close`."""

    def __init__(self, base_url: str, key: str | None = None, *, timeout: float = 10.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"x-api-key": key} if key else {}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout,
                                  transport=transport, event_hooks={"request": [_inject_trace_context]})

    def __repr__(self) -> str:
        return f"AglClient({self.base_url!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AglClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ plumbing

    def _call(self, method: str, path: str, *, ok: Iterable[int] = (200, 201), **kwargs: Any) -> httpx.Response:
        resp = self._http.request(method, path, **kwargs)
        if resp.status_code not in tuple(ok):
            raise _error(resp, f"{method} {path}")
        return resp

    # ------------------------------------------------------------ health / models

    def healthz(self) -> bool:
        try:
            resp = self._http.get("/healthz")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 and resp.json().get("status") == "ok"

    def register_models(self, models: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Upsert model endpoints ``[{model, endpoint, version}]`` (endpoint includes ``/v1``)."""
        body = [{"model": str(m["model"]), "endpoint": str(m["endpoint"]), "version": int(m.get("version", 0))}
                for m in models]
        return self._call("POST", "/api/models", json=body).json()

    def register_model(self, model: str, endpoint: str, version: int = 0) -> dict[str, Any]:
        return self.register_models([{"model": model, "endpoint": endpoint, "version": version}])[0]

    def delete_models(self) -> None:
        self._call("DELETE", "/api/models")

    # ------------------------------------------------------------ rollouts

    def create_rollout(self, rollout_id: str, input: Any, *, is_train: bool = False,
                       metadata: Mapping[str, Any] | None = None,
                       config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Idempotent create keyed by ``rollout_id``; returns the (possibly pre-existing) ``Rollout``."""
        item: dict[str, Any] = {"rollout_id": rollout_id, "input": input, "is_train": is_train}
        if metadata is not None:
            item["metadata"] = dict(metadata)
        if config is not None:
            item["config"] = dict(config)
        resp = self._http.post("/api/rollouts", json=[item])
        if resp.status_code == 409:
            existing = self.get_rollout(rollout_id)
            if existing is None:
                raise _error(resp, "POST /api/rollouts")
            return existing["rollout"]
        if resp.status_code not in (200, 201):
            raise _error(resp, "POST /api/rollouts")
        return resp.json()[0]

    def get_rollout(self, rollout_id: str) -> dict[str, Any] | None:
        """``RolloutDetail`` ``{"rollout": Rollout, "attempts": [...]}`` or None when unknown."""
        resp = self._http.get(f"/api/rollouts/{_seg(rollout_id)}")
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise _error(resp, "GET /api/rollouts/{id}")
        return resp.json()

    def list_rollouts(self, states: Iterable[str], *, limit: int = 500) -> list[dict[str, Any]]:
        params = [("state_in", s) for s in states] + [("limit", str(limit))]
        return self._call("GET", "/api/rollouts", params=params).json()

    def list_terminal(self, *, after: int = 0, limit: int = 1000) -> dict[str, Any]:
        return self._call("GET", "/api/rollouts/terminal", params={"after": after, "limit": limit}).json()

    def delete_rollout(self, rollout_id: str) -> None:
        self._call("DELETE", f"/api/rollouts/{_seg(rollout_id)}", ok=(200, 204))

    def patch_status(self, rollout_id: str, **status: Any) -> dict[str, Any]:
        """Raw ``PATCH {"status": {...}}``; raises :class:`AglConflict` on 409."""
        return self._call("PATCH", f"/api/rollouts/{_seg(rollout_id)}", json={"status": status}).json()

    def patch_state(self, rollout_id: str, state: str, *, attempt_id: str | None = None,
                    error_message: str | None = None) -> dict[str, Any]:
        """Transition ``queuing -> running -> succeeded|failed``; returns the server ``Rollout``.

        409 is reconciled: re-read and accept when the rollout is already in ``state`` or already
        terminal (first terminal write wins); ``queuing -> succeeded`` is advanced via ``running``.
        """
        if state not in STATES:
            raise ValueError(f"bad rollout state {state!r}")
        status: dict[str, Any] = {"state": state}
        if attempt_id is not None:
            status["last_attempt_id"] = attempt_id
        if error_message is not None:
            status["error_message"] = error_message
        try:
            return self.patch_status(rollout_id, **status)
        except AglConflict:
            detail = self.get_rollout(rollout_id)
            if detail is None:
                raise
            current = detail["rollout"]
            cur = current["status"]["state"]
            if cur == state or cur in TERMINAL_STATES:
                if cur != state:
                    log.warning("rollout %s already %s on server; not changing to %s", rollout_id, cur, state)
                return current
            if cur == "queuing" and state == "succeeded":
                self.patch_state(rollout_id, "running", attempt_id=attempt_id)
                return self.patch_state(rollout_id, state, attempt_id=attempt_id, error_message=error_message)
            raise

    # ------------------------------------------------------------ events

    def post_event(self, rollout_id: str, attempt_id: str, event_type: str,
                   data: Mapping[str, Any]) -> dict[str, Any]:
        path = f"/api/rollouts/{_seg(rollout_id)}/attempt/{_seg(attempt_id)}/events"
        return self._call("POST", path, json={"event_type": event_type, "data": dict(data)}).json()

    def get_events(self, rollout_id: str, *, event_type: str | None = None,
                   format: Literal["triplet"] | None = None) -> list[dict[str, Any]]:
        """Events of the rollout's ``status.last_attempt_id`` (server default ``"0"``)."""
        params: dict[str, str] = {}
        if event_type is not None:
            params["event_type"] = event_type
        if format is not None:
            params["format"] = format
        return self._call("GET", f"/api/rollouts/{_seg(rollout_id)}/events", params=params).json()

    # ------------------------------------------------------------ proxy

    def proxy_base_url(self, rollout: RolloutKey | str, mode: ProxyMode = "val", *,
                       attempt_id: str | None = None) -> str:
        """OpenAI-compatible base URL that routes through the AGL proxy for this rollout attempt."""
        return proxy_base_url(self.base_url, rollout, mode, attempt_id=attempt_id)


def proxy_base_url(base_url: str, rollout: RolloutKey | str, mode: ProxyMode = "val", *,
                   attempt_id: str | None = None) -> str:
    if mode not in ("train", "val"):
        raise ValueError(f"bad proxy mode {mode!r}")
    if isinstance(rollout, RolloutKey):
        rid, aid = rollout.rollout_id, attempt_id or rollout.attempt_id
    else:
        rid, aid = rollout, attempt_id or DEFAULT_ATTEMPT_ID
    return f"{base_url.rstrip('/')}/proxy/rollout/{_seg(rid)}/attempt/{_seg(aid)}/mode/{mode}/openai/v1"
