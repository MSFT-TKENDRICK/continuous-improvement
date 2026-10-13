"""Shared fixtures for ci_lab.agl tests: an httpx.MockTransport fake of agl-server 1.0.2.

Responses are built from the real ``agentlightning.schemas`` pydantic models so shapes
match the server (Rollout, RolloutDetail, Event, Model; 201/404/409 semantics).
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
from agentlightning.schemas import (
    VALID_TRANSITIONS,
    Event,
    Model,
    Rollout,
    RolloutConfig,
    RolloutCreate,
    RolloutLifecycleStatus,
    RolloutMetadata,
    RolloutPatch,
)

KEY = "test-key-not-secret"


class FakeAgl:
    def __init__(self, key: str = KEY) -> None:
        self.key = key
        self.rollouts: dict[str, Rollout] = {}
        self.events: dict[str, dict[str, list[Event]]] = {}
        self.models: dict[str, dict[str, Model]] = {}
        self.requests: list[httpx.Request] = []
        self.down = False
        self.create_conflict = False  # simulate a 409 on create for existing ids

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @staticmethod
    def _json(status: int, body: Any) -> httpx.Response:
        return httpx.Response(status, json=body)

    @staticmethod
    def _dump(obj: Any) -> Any:
        return obj.model_dump(mode="json")

    def state(self, rid: str) -> str:
        return str(self.rollouts[rid].status.state.value)

    def all_events(self, rid: str) -> list[Event]:
        return [e for evs in self.events.get(rid, {}).values() for e in evs]

    def add_proxy_event(self, rollout_id: str, attempt_id: str, data: dict[str, Any]) -> None:
        self.events[rollout_id].setdefault(attempt_id, []).append(Event(
            event_type="model_request", rollout_id=rollout_id, attempt_id=attempt_id,
            timestamp=time.time(), data=data))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = unquote(request.url.path)
        method = request.method
        if path == "/healthz":
            return self._json(200, {"status": "ok"})
        if request.headers.get("x-api-key") != self.key:
            return self._json(401, {"detail": "Invalid or missing API key"})
        body = json.loads(request.content) if request.content else None
        parts = path.strip("/").split("/")

        if parts == ["api", "models"] and method == "POST":
            out = []
            for item in body:
                m = Model.model_validate(item)
                self.models.setdefault(m.model, {})[m.endpoint] = m
                out.append(self._dump(m))
            return self._json(201, out)
        if parts == ["api", "rollouts"] and method == "POST":
            out = []
            for item in body:
                req = RolloutCreate.model_validate(item)
                if req.rollout_id in self.rollouts:
                    if self.create_conflict:
                        return self._json(409, {"detail": f"Rollout exists: {req.rollout_id}"})
                    out.append(self._dump(self.rollouts[req.rollout_id]))
                    continue
                now = time.time()
                meta = RolloutMetadata(**req.metadata) if isinstance(req.metadata, dict) else (
                    req.metadata or RolloutMetadata())
                r = Rollout(rollout_id=req.rollout_id or "gen", input=req.input, is_train=req.is_train,
                            config=req.config or RolloutConfig(), metadata=meta,
                            status=RolloutLifecycleStatus(created_at=now, updated_at=now))
                self.rollouts[r.rollout_id] = r
                self.events[r.rollout_id] = {}
                out.append(self._dump(r))
            return self._json(201, out)
        if len(parts) == 3 and parts[:2] == ["api", "rollouts"]:
            rid = parts[2]
            if rid not in self.rollouts:
                return self._json(404, {"detail": f"Rollout not found: {rid}"})
            r = self.rollouts[rid]
            if method == "GET":
                return self._json(200, {"rollout": self._dump(r), "attempts": sorted(self.events[rid])})
            if method == "PATCH":
                patch = RolloutPatch.model_validate(body)
                updates = patch.status.model_dump(exclude_unset=True) if patch.status else {}
                if "state" in updates and updates["state"] not in VALID_TRANSITIONS[r.status.state]:
                    return self._json(409, {"detail": f"Rollout {rid}: cannot transition "
                                                      f"{r.status.state} -> {updates['state']}"})
                status = r.status.model_copy(update={**updates, "version": r.status.version + 1,
                                                     "updated_at": time.time()})
                self.rollouts[rid] = r = r.model_copy(update={"status": status})
                return self._json(200, self._dump(r))
        if len(parts) == 6 and parts[:2] == ["api", "rollouts"] and parts[3] == "attempt" and method == "POST":
            rid, aid = parts[2], parts[4]
            if rid not in self.rollouts:
                return self._json(404, {"detail": f"Rollout not found: {rid}"})
            ev = Event(event_type=body["event_type"], rollout_id=rid, attempt_id=aid, timestamp=time.time(),
                       data=body.get("data") or {})
            self.events[rid].setdefault(aid, []).append(ev)
            return self._json(200, self._dump(ev))
        if len(parts) == 4 and parts[3] == "events" and method == "GET":
            rid = parts[2]
            if rid not in self.rollouts:
                return self._json(404, {"detail": f"Rollout not found: {rid}"})
            aid = self.rollouts[rid].status.last_attempt_id or "0"
            evs = self.events[rid].get(aid, [])
            et = request.url.params.get("event_type")
            if et is not None:
                evs = [e for e in evs if e.event_type == et]
            return self._json(200, [self._dump(e) for e in evs])
        return self._json(404, {"detail": "Not Found"})


@pytest.fixture
def fake_agl() -> FakeAgl:
    return FakeAgl()
