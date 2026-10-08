"""Local append-only rollout journal (design C2: journal-first, then mirror).

One JSONL file per rollout under ``root/<rollout_id>.jsonl``. Every line is a
record ``{"v": 1, "kind": "start"|"event"|"finish", ...}``. Writers take a
cross-process lock (``root/.journal.lock``), catch up on records appended by
other processes, dedupe, then append + flush (+ fsync). A truncated last line
left by a crash is tolerated: it is skipped when unparsable and the next append
starts on a fresh line.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

from ci_lab.contracts import RolloutKey

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TERMINAL = ("succeeded", "failed")
_ROLLOUT_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
LOCK_NAME = ".journal.lock"

RolloutStatus = Literal["running", "succeeded", "failed"]


def _check_rollout_id(rollout_id: str) -> str:
    if not _ROLLOUT_ID_RE.match(rollout_id):
        raise ValueError(f"unsafe rollout id {rollout_id!r}")
    return rollout_id


class FileLock:
    """Exclusive inter-process lock on a sidecar file (msvcrt on Windows, flock on POSIX)."""

    def __init__(self, path: Path, timeout: float = 60.0) -> None:
        self.path = path
        self.timeout = timeout
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115 - closed in __exit__
        try:
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + self.timeout
                while True:
                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"journal lock busy: {self.path}") from None
                        time.sleep(0.005)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, *exc: object) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


@dataclass
class RolloutRecord:
    """Materialised view of one rollout's journal file."""

    rollout_id: str
    key: RolloutKey | None = None
    input: dict[str, Any] = field(default_factory=dict)
    status: RolloutStatus | None = None  # None = no start record yet
    attempts: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    finished_attempt: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def latest_attempt(self) -> str | None:
        return self.attempts[-1] if self.attempts else None

    def events_for(self, attempt_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events
                if (attempt_id is None or e["attempt_id"] == attempt_id)
                and (event_type is None or e["event_type"] == event_type)]


@dataclass
class _Cache:
    offset: int = 0
    record: RolloutRecord | None = None
    event_ids: set[str] = field(default_factory=set)
    ends_with_newline: bool = True


def _apply(cache: _Cache, rollout_id: str, rec: Mapping[str, Any]) -> None:
    """Fold one parsed journal line into the cached view (first write wins)."""
    r = cache.record or RolloutRecord(rollout_id=rollout_id)
    cache.record = r
    kind = rec.get("kind")
    attempt = str(rec.get("attempt_id", "0"))
    if kind == "start":
        if attempt not in r.attempts:
            r.attempts.append(attempt)
        if r.key is None and isinstance(rec.get("key"), dict):
            try:
                r.key = RolloutKey(**rec["key"])
            except TypeError:
                pass
        if r.started_at is None:
            r.input = dict(rec.get("input") or {})
            r.started_at = rec.get("ts")
        if r.status is None:
            r.status = "running"
    elif kind == "event":
        eid = str(rec.get("event_id"))
        if eid in cache.event_ids:
            return
        cache.event_ids.add(eid)
        r.events.append({"event_id": eid, "event_type": rec.get("event_type"), "attempt_id": attempt,
                         "data": rec.get("data") or {}, "ts": rec.get("ts")})
    elif kind == "finish":
        if r.status in TERMINAL:
            return
        status = rec.get("status")
        if status in TERMINAL:
            r.status = status
            r.finished_at = rec.get("ts")
            r.finished_attempt = attempt


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


class FileRolloutJournal:
    """Durable :class:`ci_lab.contracts.RolloutJournal` backed by per-rollout JSONL files."""

    def __init__(self, root: Path | str, *, fsync: bool = True, lock_timeout: float = 60.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = FileLock(self.root / LOCK_NAME, timeout=lock_timeout)
        self._tlock = threading.RLock()
        self._caches: dict[str, _Cache] = {}

    def __repr__(self) -> str:
        return f"FileRolloutJournal({str(self.root)!r})"

    # ------------------------------------------------------------ paths / io

    def path(self, rollout_id: str) -> Path:
        return self.root / f"{_check_rollout_id(rollout_id)}.jsonl"

    def _catch_up(self, rollout_id: str) -> _Cache:
        cache = self._caches.setdefault(rollout_id, _Cache())
        path = self.path(rollout_id)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return cache
        if size < cache.offset:  # file replaced/truncated externally: rebuild
            cache = self._caches[rollout_id] = _Cache()
        if size == cache.offset:
            return cache
        with open(path, "rb") as fh:
            fh.seek(cache.offset)
            blob = fh.read()
        lines = blob.split(b"\n")
        tail = lines.pop()  # b"" when blob ends with a newline
        for raw in lines:
            self._parse_into(cache, rollout_id, raw)
        if tail:
            # Unterminated last line = crash remnant (writes happen under the lock).
            self._parse_into(cache, rollout_id, tail, truncated=True)
        cache.offset += len(blob)
        cache.ends_with_newline = not tail
        return cache

    def _parse_into(self, cache: _Cache, rollout_id: str, raw: bytes, *, truncated: bool = False) -> None:
        if not raw.strip():
            return
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("skipping %s journal line in %s", "truncated" if truncated else "corrupt", rollout_id)
            return
        if isinstance(rec, dict):
            _apply(cache, rollout_id, rec)

    def _append(self, rollout_id: str, rec: dict[str, Any], cache: _Cache) -> None:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        payload = (b"" if cache.ends_with_newline else b"\n") + line.encode("utf-8") + b"\n"
        with open(self.path(rollout_id), "ab") as fh:
            fh.write(payload)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        cache.offset += len(payload)
        cache.ends_with_newline = True
        _apply(cache, rollout_id, json.loads(line))

    def _write(self, rollout_id: str, decide: Any) -> bool:
        """Under both locks: catch up, ask ``decide(cache)`` for a record (or None), append it."""
        with self._tlock, self._lock:
            cache = self._catch_up(rollout_id)
            rec = decide(cache)
            if rec is None:
                return False
            rec = {"v": SCHEMA_VERSION, "rollout_id": rollout_id, "ts": time.time(), **rec}
            self._append(rollout_id, rec, cache)
            return True

    # ------------------------------------------------------------ RolloutJournal

    def start(self, key: RolloutKey, input: Mapping[str, Any]) -> None:
        self.append_start(key, input)

    def event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> None:
        self.append_event(key, event_type, data, event_id=event_id)

    def finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> None:
        self.append_finish(key, status)

    def events(self, key: RolloutKey) -> list[dict[str, Any]]:
        rec = self.load(key.rollout_id)
        return list(rec.events) if rec else []

    # ------------------------------------------------------------ bool-returning variants

    def append_start(self, key: RolloutKey, input: Mapping[str, Any]) -> bool:
        """Record the start of ``key.attempt``; returns False if already recorded."""
        aid = key.attempt_id

        def decide(cache: _Cache) -> dict[str, Any] | None:
            if cache.record is not None and aid in cache.record.attempts:
                return None
            return {"kind": "start", "attempt_id": aid, "key": asdict(key), "input": dict(input)}

        return self._write(key.rollout_id, decide)

    def append_event(self, key: RolloutKey, event_type: str, data: Mapping[str, Any], *, event_id: str) -> bool:
        """Append an event unless ``event_id`` is already journaled; returns whether it was written."""
        if not event_id:
            raise ValueError("event_id is required (use contracts.op_id)")

        def decide(cache: _Cache) -> dict[str, Any] | None:
            if event_id in cache.event_ids:
                return None
            return {"kind": "event", "attempt_id": key.attempt_id, "event_id": event_id,
                    "event_type": event_type, "data": dict(data)}

        return self._write(key.rollout_id, decide)

    def append_finish(self, key: RolloutKey, status: Literal["succeeded", "failed"]) -> bool:
        """Record the terminal state once per rollout; later calls are no-ops (returns False)."""
        if status not in TERMINAL:
            raise ValueError(f"bad terminal status {status!r}")

        def decide(cache: _Cache) -> dict[str, Any] | None:
            if cache.record is not None and cache.record.status in TERMINAL:
                if cache.record.status != status:
                    log.warning("rollout %s already %s; ignoring %s", key.rollout_id, cache.record.status, status)
                return None
            return {"kind": "finish", "attempt_id": key.attempt_id, "status": status}

        return self._write(key.rollout_id, decide)

    # ------------------------------------------------------------ readers

    def load(self, rollout_id: str) -> RolloutRecord | None:
        """Current view of one rollout (None when it has no journal file / records)."""
        with self._tlock, self._lock:
            cache = self._catch_up(rollout_id)
            if cache.record is None:
                return None
            return _copy_record(cache.record)

    def rollout_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl") if _ROLLOUT_ID_RE.match(p.stem))

    def iter_rollouts(self) -> Iterator[RolloutRecord]:
        for rid in self.rollout_ids():
            rec = self.load(rid)
            if rec is not None:
                yield rec


def _copy_record(r: RolloutRecord) -> RolloutRecord:
    return RolloutRecord(rollout_id=r.rollout_id, key=r.key, input=copy.deepcopy(r.input), status=r.status,
                         attempts=list(r.attempts), events=copy.deepcopy(r.events),
                         started_at=r.started_at, finished_at=r.finished_at,
                         finished_attempt=r.finished_attempt)
