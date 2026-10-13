"""Write-ahead agent bus storage (bus contract v2 §3).

One append-only JSONL file per topic (``<root>/<topic>.wal.jsonl``), hash-chained (``prev`` =
previous ``hash``; seq 0 chains to ``GENESIS``) and fsynced per entry. Appends are serialized
per topic by an asyncio lock plus a cross-process ``ledger.lock.FileLock`` and validated by
``state.check`` before anything is written. Readers never modify a WAL: a torn final line (a
writer crashed mid-line) is ignored by readers and truncated by the next append, which first
records ``note{"text": "torn_tail_repaired"}``. Any other damage raises :class:`BusCorrupt`.
The read cache is keyed by (verified size, last hash) only and re-verifies the last line.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace

from ci_lab.bus import ids
from ci_lab.bus.state import BusInvariantError, BusState, check, is_run_topic
from ci_lab.bus.types import (
    BODY_TYPES,
    GENESIS,
    ArtifactRef,
    Author,
    Entry,
    Kind,
    NoteBody,
    canonical_json,
)
from ci_lab.ledger.atomic import atomic_write_bytes
from ci_lab.ledger.lock import DEFAULT_TIMEOUT, lock_dir_for, lock_for

__all__ = ["EVENT_APPEND", "SUFFIX", "TORN_TAIL_NOTE", "AgentBus", "BusCorrupt"]

SUFFIX = ".wal.jsonl"
TORN_TAIL_NOTE = "torn_tail_repaired"
EVENT_APPEND = "ci.bus.append"
_REPAIRER = Author(role="orchestrator", name="bus")


class BusCorrupt(RuntimeError):
    """A topic WAL or artifact failed verification (anything but a torn final WAL line)."""


@dataclass(frozen=True)
class _Scan:
    entries: tuple[Entry, ...]
    tail: int  # byte offset of the last entry's line
    size: int  # end of the verified prefix
    torn: bool  # bytes after ``size`` are a torn final line


def _parse(topic: str, data: bytes, offset: int, prior: tuple[Entry, ...]) -> _Scan:
    entries, pos, tail = list(prior), offset, offset
    lines = data.split(b"\n")
    for i, raw in enumerate(lines[:-1]):
        try:
            obj = json.loads(raw)
        except ValueError:
            if i == len(lines) - 2 and not lines[-1]:
                return _Scan(tuple(entries), tail, pos, True)
            raise BusCorrupt(f"{topic}: unparseable line at byte {pos}") from None
        try:
            e = Entry.from_json(obj)
        except (ValueError, TypeError) as exc:
            raise BusCorrupt(f"{topic}: invalid entry at byte {pos}: {exc}") from None
        prev = entries[-1].hash if entries else GENESIS
        if e.seq != len(entries) or e.topic != topic or e.prev != prev or not e.hash_ok():
            raise BusCorrupt(f"{topic}: broken seq/hash chain at seq {len(entries)} (byte {pos})")
        entries.append(e)
        tail, pos = pos, pos + len(raw) + 1
    return _Scan(tuple(entries), tail, pos, bool(lines[-1]))


def _artifact_path(sha: str) -> str:
    return f"{sha[:2]}/{sha}"


def _put_file(dst: Path, raw: bytes, sha: str) -> None:
    if dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        return
    atomic_write_bytes(dst, raw)


async def _finish_then_cancel(coro: Any) -> Any:
    """Await ``coro`` to completion even if the caller is cancelled, then re-raise the cancellation.

    A worker-thread write keeps running after its awaiting task is cancelled; releasing the
    WAL lock before it finishes would let the next append interleave and break the hash chain."""
    fut = asyncio.ensure_future(coro)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(fut)
            break
        except asyncio.CancelledError:
            if fut.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class AgentBus:
    """The bus: the sole business authority (state = fold(WAL))."""

    def __init__(self, root: str | os.PathLike[str], *, lock_timeout: float = DEFAULT_TIMEOUT) -> None:
        self.root = Path(root)
        self.lock_dir = lock_dir_for(self.root)
        self.lock_timeout = lock_timeout
        self._alocks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
            weakref.WeakKeyDictionary())
        self._cache: dict[str, _Scan] = {}

    # ------------------------------------------------------------ paths

    def wal_path(self, topic: str) -> Path:
        return self.root / f"{ids.topic_relpath(topic)}{SUFFIX}"

    def artifact_dir(self, topic: str) -> Path:
        return self.root / ids.topic_relpath(topic) / "artifacts"

    # ------------------------------------------------------------ reads

    def _scan(self, topic: str) -> _Scan:
        path, cached = self.wal_path(topic), self._cache.get(topic)
        try:
            with path.open("rb") as fh:
                if cached is not None and cached.entries:
                    fh.seek(cached.tail)
                    n = len(cached.entries)
                    try:
                        scan = _parse(topic, fh.read(), cached.tail, cached.entries[:-1])
                    except BusCorrupt:
                        scan = None
                    if scan is not None and len(scan.entries) >= n and scan.entries[n - 1] == cached.entries[-1]:
                        self._cache[topic] = scan
                        return scan
                    fh.seek(0)
                data = fh.read()
        except FileNotFoundError:
            return _Scan((), 0, 0, False)
        scan = _parse(topic, data, 0, ())
        self._cache[topic] = scan
        return scan

    def read(self, topic: str) -> list[Entry]:
        """Verified entries (dense seq, intact hash chain); a torn final line is ignored."""
        return list(self._scan(topic).entries)

    def state(self, topic: str) -> BusState:
        return BusState(topic=ids.validate_topic(topic), entries=self._scan(topic).entries)

    def head(self, topic: str) -> tuple[int, str]:
        """``(seq, hash)`` of the last entry; ``(-1, GENESIS)`` for an empty topic."""
        entries = self._scan(topic).entries
        return (entries[-1].seq, entries[-1].hash) if entries else (-1, GENESIS)

    def topics(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            return []
        found = (p.relative_to(self.root).as_posix()[: -len(SUFFIX)] for p in self.root.rglob(f"*{SUFFIX}"))
        return sorted(t for t in found if t.startswith(prefix))

    def heads(self, run: str) -> dict[str, str]:
        """Topic -> head hash for every non-empty topic of ``run`` (anchored into OES)."""
        heads = {t: self.head(t) for t in self.topics(f"{ids.run_id(run)}/")}
        return {t: h for t, (seq, h) in heads.items() if seq >= 0}

    def read_artifact(self, topic: str, ref: ArtifactRef) -> bytes:
        if ref.path != _artifact_path(ref.sha256):
            raise BusCorrupt(f"{topic}: artifact path {ref.path!r} is not content-addressed")
        data = (self.artifact_dir(topic) / ref.path).read_bytes()
        if len(data) != ref.bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise BusCorrupt(f"{topic}: artifact {ref.sha256} failed verification")
        return data

    # ------------------------------------------------------------ writes

    def effect(self, topic: str, action: str, key: str, author: Author, detail: Any = None,
               **kw: Any) -> Any:
        """``async with bus.effect(...)``: delegate to :func:`ci_lab.bus.effects.effect`."""
        from ci_lab.bus.effects import effect
        return effect(self, topic, action, key, author, detail, **kw)

    async def put_artifact(self, topic: str, data: bytes | str) -> ArtifactRef:
        """Store content-addressed ``data`` (atomic tmp+replace; idempotent)."""
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        sha = hashlib.sha256(raw).hexdigest()
        ref = ArtifactRef(path=_artifact_path(sha), sha256=sha, bytes=len(raw))
        await _finish_then_cancel(asyncio.to_thread(_put_file, self.artifact_dir(topic) / ref.path, raw, sha))
        return ref

    async def append(self, topic: str, kind: Kind, author: Author, body: Any, *, ref: int | None = None) -> Entry:
        """Validate (``state.check``), seal and durably append one entry; return it.

        ``body`` is the typed body for ``kind`` or its JSON mapping."""
        path = self.wal_path(topic)
        alock = self._alocks.setdefault(asyncio.get_running_loop(), {}).setdefault(topic, asyncio.Lock())
        async with alock, lock_for(path, suffix="wal", timeout=self.lock_timeout, lock_dir=self.lock_dir):
            written = await _finish_then_cancel(
                asyncio.to_thread(self._append_locked, topic, path, kind, author, body, ref))
        span = trace.get_current_span()
        for e in written:
            span.add_event(EVENT_APPEND, attributes={
                "ci.bus.topic": e.topic, "ci.bus.seq": e.seq, "ci.bus.kind": e.kind, "ci.bus.role": e.author.role})
        return written[-1]

    def _run_manifest(self, topic: str) -> Any:
        if is_run_topic(topic):
            return None
        try:
            run_topic = ids.run_topic(ids.topic_run(topic))
        except ids.IdError:
            return None
        return self.state(run_topic).manifest

    def _append_locked(self, topic: str, path: Path, kind: Kind, author: Author, body: Any,
                       ref: int | None) -> list[Entry]:
        scan, written = self._scan(topic), []
        if scan.torn:
            with path.open("r+b") as fh:
                fh.truncate(scan.size)
                fh.flush()
                os.fsync(fh.fileno())
            scan = _Scan(scan.entries, scan.tail, scan.size, False)
            note = self._entry(scan, topic, "note", _REPAIRER, NoteBody(text=TORN_TAIL_NOTE), None)
            try:
                check(BusState(topic=topic, entries=scan.entries), note)
            except BusInvariantError:
                pass  # an empty run topic must start with its manifest
            else:
                scan = self._write(path, scan, note.sealed())
                written.append(scan.entries[-1])
        if isinstance(body, Mapping) and kind in BODY_TYPES:
            body = BODY_TYPES[kind].from_json(body)
        entry = self._entry(scan, topic, kind, author, body, ref)
        check(BusState(topic=topic, entries=scan.entries), entry, manifest=self._run_manifest(topic))
        scan = self._write(path, scan, entry.sealed())
        return [*written, scan.entries[-1]]

    @staticmethod
    def _entry(scan: _Scan, topic: str, kind: Kind, author: Author, body: Any, ref: int | None) -> Entry:
        prev = scan.entries[-1].hash if scan.entries else GENESIS
        return Entry(seq=len(scan.entries), topic=topic, kind=kind, author=author, ref=ref, body=body,
                     ts=datetime.now(UTC).isoformat(timespec="microseconds"), prev=prev)

    def _write(self, path: Path, scan: _Scan, entry: Entry) -> _Scan:
        line = (canonical_json(entry.to_json()) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        new = _Scan((*scan.entries, entry), scan.size, scan.size + len(line), False)
        self._cache[entry.topic] = new
        return new
