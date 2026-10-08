"""Crash-safe file writes (temp file in the same directory + fsync + ``os.replace``).

Windows-safe: ``os.replace`` can transiently fail with ``PermissionError`` while another
process (indexer, antivirus, a reader) holds the destination open, so replaces are
retried with a short backoff. Bytes are written verbatim (``\\n`` newlines, UTF-8).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

log = logging.getLogger(__name__)

_REPLACE_RETRIES = 40
_REPLACE_BACKOFF = 0.025


def _fsync_dir(directory: Path) -> None:
    if os.name == "nt":  # directories cannot be opened/fsynced on Windows; NTFS journals metadata
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace(src: str, dst: Path) -> None:
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF * (1 + attempt // 4))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes, *, fsync: bool = True) -> Path:
    """Atomically replace ``path`` with ``data``; readers see the old or new file, never a mix."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        _replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(dst.parent)
    return dst


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8",
                      fsync: bool = True) -> Path:
    return atomic_write_bytes(path, text.encode(encoding), fsync=fsync)


def to_jsonable(obj: Any) -> Any:
    """``json.dumps`` default hook: dataclasses, enums, paths, tuples/sets."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj: Any, *, indent: int | None = None, sort_keys: bool = True) -> str:
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=to_jsonable)


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2,
                      sort_keys: bool = True, fsync: bool = True) -> Path:
    return atomic_write_text(path, dumps(obj, indent=indent, sort_keys=sort_keys) + "\n", fsync=fsync)


def read_json(path: str | os.PathLike[str], default: Any = ...) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is ...:
            raise
        return default


def append_jsonl(path: str | os.PathLike[str], records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
                 *, fsync: bool = True) -> None:
    """Append one record (or several) as JSON lines. Not atomic across processes on its own:
    callers that share the file must hold a :class:`ci_lab.ledger.lock.FileLock`.

    A torn final line from an earlier crash is terminated first so the new record is never
    glued onto it (the torn line is then skipped by :func:`read_jsonl`)."""
    items = [records] if isinstance(records, Mapping) else list(records)
    if not items:
        return
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(dumps(r) + "\n" for r in items).encode("utf-8")
    with open(dst, "a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                payload = b"\n" + payload
            fh.seek(0, os.SEEK_END)
        fh.write(payload)
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read JSON lines, skipping blank and unparseable (torn) lines."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            log.warning("skipping unparseable line %d in %s", lineno, path)
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out
