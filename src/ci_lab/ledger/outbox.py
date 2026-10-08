"""Durable effect outbox (C6): run each logical operation at most once to completion.

Journal = append-only JSONL of op state transitions::

    {"op": ..., "state": "started",   "attempt": n, "ts": ..., "pid": ...}
    {"op": ..., "state": "succeeded", "attempt": n, "result": <json>, "reconciled": bool}
    {"op": ..., "state": "failed",    "attempt": n, "error": "Type: message"}

``run_once(op, fn, reconcile=)``:

* latest state ``succeeded`` -> return the stored result (``fn`` is not called);
* otherwise (never run, failed, or ``started`` without an outcome = crashed mid-effect)
  call ``reconcile()`` first; a non-``None`` answer is recorded as the result
  (``reconciled: true``) and ``fn`` is not called; only when it returns ``None`` is
  ``fn`` (re-)run.

Each op holds a dedicated cross-process lock for its whole execution, so concurrent
callers of the same op (threads, asyncio tasks, processes) run it once; different ops
run concurrently. Results must be JSON-serializable (dataclasses/tuples/paths are
converted) and the JSON form is returned on the first run as well as on replays, so
callers behave identically before and after a crash.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ci_lab.ledger.atomic import append_jsonl, read_jsonl, to_jsonable
from ci_lab.ledger.lock import FileLock, lock_dir_for, lock_for

STARTED, SUCCEEDED, FAILED = "started", "succeeded", "failed"
_MAX_ERROR = 2000


class FileOutbox:
    """File-backed :class:`ci_lab.contracts.Outbox`."""

    def __init__(self, path: str | os.PathLike[str], *, lock_timeout: float = 3600.0) -> None:
        self.path = Path(path)
        self.lock_timeout = lock_timeout
        self._lock_dir = lock_dir_for(self.path)
        self._journal_lock = lock_for(self.path, suffix="journal", timeout=60.0, lock_dir=self._lock_dir)

    # ------------------------------------------------------------ inspection

    def records(self) -> list[dict[str, Any]]:
        return [r for r in read_jsonl(self.path) if isinstance(r.get("op"), str)]

    def latest(self, op: str) -> dict[str, Any] | None:
        last = None
        for rec in self.records():
            if rec["op"] == op:
                last = rec
        return last

    def status(self, op: str) -> str | None:
        rec = self.latest(op)
        return rec.get("state") if rec else None

    def pending(self) -> list[str]:
        """Ops whose latest state is ``started`` (crashed or still running elsewhere)."""
        state: dict[str, str] = {}
        for rec in self.records():
            state[rec["op"]] = rec.get("state", "")
        return sorted(op for op, s in state.items() if s == STARTED)

    # ------------------------------------------------------------ journal

    def _append(self, op: str, state: str, attempt: int, **extra: Any) -> None:
        rec = {"op": op, "state": state, "attempt": attempt, "ts": time.time(),
               "pid": os.getpid(), "host": socket.gethostname(), **extra}
        with self._journal_lock:
            append_jsonl(self.path, rec)

    def _op_lock(self, op: str) -> FileLock:
        return lock_for(self.path, suffix=f"op|{op}", timeout=self.lock_timeout, lock_dir=self._lock_dir)

    @staticmethod
    def _error(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]

    def _begin(self, op: str) -> tuple[bool, Any, int]:
        rec = self.latest(op)
        if rec and rec.get("state") == SUCCEEDED:
            return True, rec.get("result"), int(rec.get("attempt", 0))
        return False, None, int(rec.get("attempt", 0)) + 1 if rec else 1

    def _finish(self, op: str, attempt: int, result: Any, *, reconciled: bool) -> Any:
        """Record success and return the JSON form (identical to what a replay returns)."""
        try:
            encoded = json.loads(json.dumps(result, default=to_jsonable))
        except (TypeError, ValueError) as exc:
            # The effect happened: record it (so it is never repeated) but surface the bug.
            self._append(op, SUCCEEDED, attempt, result=None, reconciled=reconciled,
                         unserializable=repr(result)[:_MAX_ERROR])
            raise TypeError(f"outbox op {op!r} returned a non-JSON-serializable result") from exc
        self._append(op, SUCCEEDED, attempt, result=encoded, reconciled=reconciled)
        return encoded

    # ------------------------------------------------------------ API

    def run_once(self, op: str, fn: Callable[[], Any], *,
                 reconcile: Callable[[], Any | None] | None = None) -> Any:
        with self._op_lock(op):
            done, result, attempt = self._begin(op)
            if done:
                return result
            if reconcile is not None:
                found = reconcile()
                if found is not None:
                    return self._finish(op, attempt, found, reconciled=True)
            self._append(op, STARTED, attempt)
            try:
                result = fn()
            except BaseException as exc:
                self._append(op, FAILED, attempt, error=self._error(exc))
                raise
            return self._finish(op, attempt, result, reconciled=False)

    async def arun_once(self, op: str, fn: Callable[[], Awaitable[Any]], *,
                        reconcile: Callable[[], Awaitable[Any | None]] | None = None) -> Any:
        lock = self._op_lock(op)
        await lock.aacquire()
        try:
            done, result, attempt = self._begin(op)
            if done:
                return result
            if reconcile is not None:
                found = await reconcile()
                if found is not None:
                    return self._finish(op, attempt, found, reconciled=True)
            self._append(op, STARTED, attempt)
            try:
                result = await fn()
            except BaseException as exc:
                self._append(op, FAILED, attempt, error=self._error(exc))
                raise
            return self._finish(op, attempt, result, reconciled=False)
        finally:
            lock.arelease()
