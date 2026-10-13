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
from ci_lab.ledger import decisions as ledger_decisions
from ci_lab.ledger import looks as ledger_looks
from ci_lab.ledger.layout import Layout


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

    def record_decisions(self, campaign_id: str, experiment_id: str, decisions: Mapping[str, Any]) -> str:
        """Decision-time write of ``campaigns/<cid>/rounds/<eid>/decisions.json`` through
        :func:`ci_lab.ledger.decisions.record_decisions` (verdict check + ``ci.step{record}`` span).
        Returns the ledger-relative path."""
        path = ledger_decisions.record_decisions(_RootedLayout(self.root), campaign_id, experiment_id, decisions)
        return path.relative_to(self.root).as_posix()

    def record_look(self, dataset_hash: str, *, experiment_id: str, planned: int, campaign_id: str | None = None,
                    split: str = "heldout") -> dict[str, Any]:
        """Reserve one held-out look in the global C15 ledger ``holdout-looks.jsonl`` via
        :func:`ci_lab.ledger.looks.record_look` (idempotent per experiment; raises
        :class:`ci_lab.ledger.looks.LookBudgetExceeded` past the planned looks)."""
        return ledger_looks.record_look(_RootedLayout(self.root).holdout_looks(), dataset_hash,
                                        experiment_id=experiment_id, planned=planned, campaign_id=campaign_id,
                                        split=split)


class _RootedLayout(Layout):
    """:class:`ci_lab.ledger.layout.Layout` anchored at an explicit ledger root (FileLedger's root
    need not be ``<repo>/experiments``)."""

    def __init__(self, root: Path) -> None:
        super().__init__(Path(root).parent)
        object.__setattr__(self, "_root", Path(root))

    @property
    def root(self) -> Path:
        return self._root  # type: ignore[attr-defined]


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
