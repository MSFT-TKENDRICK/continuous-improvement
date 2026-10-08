"""Local durable defaults: a file ledger and a JSON-journal outbox.

Stand-ins for ``ci_lab.ledger`` (M6) so the ``fake``/dry-run profiles work across
CLI invocations. Single-process; the outbox journal is append-only JSONL."""

from __future__ import annotations

import json
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.campaign import records


class FileLedger:
    """``LedgerStore`` over a directory (normally ``<repo>/experiments``)."""

    def __init__(self, root: Path, *, committer: Callable[[str, Sequence[str]], str | None] | None = None) -> None:
        self.root = Path(root)
        self._committer = committer
        self._lock = threading.Lock()
        self.commits: list[tuple[str, tuple[str, ...]]] = []

    def _path(self, rel: str) -> Path:
        parts = rel.split("/")
        if not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in parts) or "\\" in rel:
            raise ValueError(f"bad ledger path {rel!r}")
        return self.root.joinpath(*parts)

    def read_json(self, rel: str) -> Any | None:
        return records.read_json(self._path(rel))

    def write_json(self, rel: str, obj: Any) -> None:
        records.write_json(self._path(rel), obj)

    def read_jsonl(self, rel: str) -> list[dict[str, Any]]:
        return records.read_jsonl(self._path(rel))

    def append_jsonl(self, rel: str, obj: Mapping[str, Any], *, key: str) -> bool:
        return records.append_jsonl(self._path(rel), obj, key=key)

    def cas_json(self, rel: str, expected: Any | None, new: Any) -> bool:
        with self._lock:
            if self.read_json(rel) != expected:
                return False
            self.write_json(rel, new)
            return True

    def commit(self, message: str, paths: Sequence[str]) -> str | None:
        self.commits.append((message, tuple(paths)))
        return self._committer(message, paths) if self._committer else None


class FileOutbox:
    """Durable :class:`ci_lab.contracts.Outbox`: results journaled as JSONL by op id."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.done: dict[str, Any] = {r["op"]: r["result"] for r in records.read_jsonl(self.path)}
        self.calls: list[str] = []

    def _record(self, op: str, result: Any) -> Any:
        with self._lock:
            self.done[op] = result
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"op": op, "result": result}, sort_keys=True) + "\n")
        return result

    def run_once(self, op: str, fn: Callable[[], Any], *, reconcile: Callable[[], Any | None] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := reconcile()) is not None:
            return self._record(op, found)
        self.calls.append(op)
        return self._record(op, fn())

    async def arun_once(self, op: str, fn: Callable[[], Awaitable[Any]], *,
                        reconcile: Callable[[], Awaitable[Any | None]] | None = None) -> Any:
        if op in self.done:
            return self.done[op]
        if reconcile is not None and (found := await reconcile()) is not None:
            return self._record(op, found)
        self.calls.append(op)
        return self._record(op, await fn())
