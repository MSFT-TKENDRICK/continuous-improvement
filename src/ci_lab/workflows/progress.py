"""Tracing + live progress markers for campaign workflows (design §12, C27, C36, D1, D3).

* One OTel trace per round / calibration / confirm: :func:`root_span` starts a new
  root (``obs.span(new_trace=True)``) linked to the span recorded by an interrupted
  earlier attempt (``obs.previous_link``), so a resume is its own trace (C27).
* :class:`Progress` writes phase / arm-state fields through
  :func:`ci_lab.obs.write_status` into the per-writer marker
  ``<run_dir>/<experiment_id>/status.d/<writer>.json`` (orchestrators write as
  ``round`` / ``campaign``, arm workers as their arm name) and heartbeats them while
  long steps run. Readers aggregate with :func:`ci_lab.obs.read_status`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace

from ci_lab import obs
from ci_lab.contracts import ATTR_PHASE, SPAN_STEP

HEARTBEAT_S = 30.0  # dashboard marks a run stale after 2x heartbeat_s (C36)


@contextlib.contextmanager
def root_span(name: str, attributes: Mapping[str, Any], run_root: Path,
              experiment_id: str) -> Iterator[trace.Span]:
    """Start ``name`` as the root of a new trace, linked to the previous attempt's trace."""
    links = obs.previous_link(run_root, experiment_id) or None
    with obs.span(name, attributes, links=links, new_trace=True) as span:
        yield span


class Progress:
    """Live status writer for one experiment run dir and one writer id."""

    def __init__(self, run_root: Path, experiment_id: str, *, writer: str,
                 heartbeat_s: float = HEARTBEAT_S, **base: Any) -> None:
        self.run_root = Path(run_root)
        self.experiment_id = experiment_id
        self.writer = writer
        self.heartbeat_s = float(heartbeat_s)
        self.base = base

    def as_writer(self, writer: str) -> Progress:
        return Progress(self.run_root, self.experiment_id, writer=writer, heartbeat_s=self.heartbeat_s,
                        **self.base)

    def read(self) -> dict[str, Any]:
        """Aggregated view over all writers of this experiment."""
        return obs.read_status(self.run_root, self.experiment_id)

    def write(self, **fields: Any) -> None:
        """Best effort: a dashboard reader holding the file must never fail a step."""
        with contextlib.suppress(OSError):
            obs.write_status(self.run_root, self.experiment_id, writer=self.writer,
                             heartbeat_s=self.heartbeat_s, **self.base, **fields)

    @contextlib.asynccontextmanager
    async def heartbeat(self, fields: Callable[[], Mapping[str, Any]]) -> AsyncIterator[None]:
        """Rewrite ``fields()`` every ``heartbeat_s`` seconds while the block runs."""

        async def beat() -> None:
            while True:
                await asyncio.sleep(self.heartbeat_s)
                self.write(**fields())

        task = asyncio.create_task(beat())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _phase_fields(phase: str) -> Mapping[str, Any]:
    return {"phase": phase}


@dataclass
class Tracker:
    """Span + status marker + heartbeat around one workflow phase (step or agent)."""

    progress: Progress | None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    fields: Callable[[str], Mapping[str, Any]] = _phase_fields

    @contextlib.asynccontextmanager
    async def phase(self, phase: str) -> AsyncIterator[None]:
        with obs.span(SPAN_STEP, {ATTR_PHASE: phase, **self.attrs}):
            if self.progress is None:
                yield
                return
            self.progress.write(**self.fields(phase))
            async with self.progress.heartbeat(lambda: self.fields(phase)):
                yield
