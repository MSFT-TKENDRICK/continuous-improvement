"""MAF workflow engine for the task graph (bus contract v2 §14): a conformance adapter.

:func:`run_graph_maf` runs the same :func:`~ci_lab.taskgraph.scheduler.run_deliverable` coroutine as
:func:`~ci_lab.taskgraph.scheduler.run_graph`, one MAF :class:`Executor` per deliverable. Edges mirror the
dependency structure: a start executor fans out to the roots, a single dependency is a plain edge and
several are a fan-in edge group. Every deliverable executor sends exactly one :class:`Settled` message when
its task is terminal — :class:`Committed`, or :class:`Blocked` when it failed, was rejected or aborted —
so a fan-in always fires (dependents then record ``abort{reason: dependency_blocked}``). Independent
deliverables reached in the same superstep run concurrently (bounded by ``max_parallel``).

The bus stays the sole authority: the manifest (I0) and the epoch adoption are written once before the
workflow, the result is re-derived with :meth:`GraphResult.from_bus`, and checkpoints (``<run_dir>/ckpt``,
private, pickle bearing) are disposable — a rerun after a crash starts a fresh workflow whose executors skip
terminal tasks and reuse in-flight attempts from the bus.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from agent_framework import (
    Executor,
    FileCheckpointStorage,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from opentelemetry import trace

from ci_lab.adversary.challenger import Challenger
from ci_lab.adversary.harden import Hardener, Scorer
from ci_lab.bus.judge import Escalator
from ci_lab.bus.pools import ResourcePools
from ci_lab.bus.wal import AgentBus
from ci_lab.ledger.layout import run_dir
from ci_lab.maf.workflows import _assert_checkpoints_written, secure_dir
from ci_lab.taskgraph.model import TaskGraph
from ci_lab.taskgraph.scheduler import (
    GraphResult,
    GraphRun,
    StudentFactory,
    VotersFor,
    adopt_hardened,
    run_deliverable,
    write_manifest,
)
from ci_lab.taskgraph.vault import RubricVault

__all__ = ["CHECKPOINT_TYPES", "Blocked", "Committed", "DeliverableExecutor", "Settled", "Start", "build_workflow",
           "run_graph_maf"]

_tracer = trace.get_tracer("ci_lab.taskgraph.maf_engine")


@dataclass(frozen=True)
class Start:
    run: str


@dataclass(frozen=True)
class Settled:
    task: str


@dataclass(frozen=True)
class Committed(Settled):
    pass


@dataclass(frozen=True)
class Blocked(Settled):
    """``task`` ended without a commit; its dependents must not run."""

    status: str


CHECKPOINT_TYPES = [f"{__name__}:{c.__qualname__}" for c in (Start, Settled, Committed, Blocked)]


class _StartExecutor(Executor):
    @handler(input=Start, output=Start)
    async def start(self, msg: Start, ctx: WorkflowContext[Start]) -> None:
        await ctx.send_message(msg)


class DeliverableExecutor(Executor):
    """Runs one deliverable once every dependency has settled; always emits one :class:`Settled`."""

    def __init__(self, run: GraphRun, task: str) -> None:
        super().__init__(id=f"task:{task}")
        self._run, self.task = run, task

    async def _settle(self, ctx: WorkflowContext[Settled]) -> None:
        await run_deliverable(self._run, self.task)
        term = self._run.bus.state(self._run.topic(self.task)).terminal
        await ctx.send_message(Committed(self.task) if term is not None and term.kind == "commit"
                               else Blocked(self.task, "pending" if term is None else term.kind))

    @handler(input=Start, output=Settled)
    async def on_start(self, _msg: Start, ctx: WorkflowContext[Settled]) -> None:
        await self._settle(ctx)

    @handler(input=Settled, output=Settled)
    async def on_dependency(self, _msg: Settled, ctx: WorkflowContext[Settled]) -> None:
        await self._settle(ctx)

    @handler(input=list[Settled], output=Settled)
    async def on_dependencies(self, _msgs: list[Settled], ctx: WorkflowContext[Settled]) -> None:
        await self._settle(ctx)


def build_workflow(run: GraphRun, storage: FileCheckpointStorage | None = None, *, name: str | None = None) -> Workflow:
    """The MAF workflow for ``run.graph``: start → roots (fan-out), dependency → dependent (edge / fan-in)."""
    graph = run.graph
    start = _StartExecutor(id="graph:start")
    execs = {t: DeliverableExecutor(run, t) for t in graph.topo_order()}
    builder = WorkflowBuilder(max_iterations=len(execs) + 2, name=name or f"taskgraph-{run.run_id}",
                              start_executor=start, checkpoint_storage=storage)
    roots = [execs[d.id] for d in graph.deliverables if not d.depends_on]
    if len(roots) == 1:
        builder.add_edge(start, roots[0])
    else:
        builder.add_fan_out_edges(start, roots)
    for d in graph.deliverables:
        if len(d.depends_on) == 1:
            builder.add_edge(execs[d.depends_on[0]], execs[d.id])
        elif d.depends_on:
            builder.add_fan_in_edges([execs[x] for x in d.depends_on], execs[d.id])
    return builder.build()


async def run_graph_maf(graph: TaskGraph, *, bus: AgentBus, vault: RubricVault, voters_for: VotersFor,
                        student_factory: StudentFactory, run_id: str, pools: ResourcePools,
                        max_parallel: int | None = None, challenger: Challenger | None = None,
                        hardener: Hardener | None = None, quorum: int = 1, escalator: Escalator | None = None,
                        timeout_s: float = 120.0, hardener_scorer: Scorer | None = None,
                        checkpoint_dir: str | Path | None = None, adopt_epoch_patches: bool = True) -> GraphResult:
    """:func:`~ci_lab.taskgraph.scheduler.run_graph` on the MAF workflow engine (same arguments and
    result). ``checkpoint_dir`` defaults to ``<run_dir(run_id)>/ckpt``; stale checkpoints of an earlier
    attempt at this run are discarded — resume always comes from the bus."""
    started = time.perf_counter()
    run = GraphRun(graph, bus, vault, voters_for, student_factory, run_id, pools,
                   max_parallel or len(graph.deliverables), quorum, escalator, timeout_s, challenger, hardener,
                   hardener_scorer)
    run.quorum = (await write_manifest(run)).quorum
    await adopt_hardened(run, adopt_epoch_patches)
    where = secure_dir(checkpoint_dir if checkpoint_dir is not None else run_dir(run_id) / "ckpt")
    storage = FileCheckpointStorage(where, allowed_checkpoint_types=CHECKPOINT_TYPES)
    workflow = build_workflow(run, storage)
    for cp in await storage.list_checkpoints(workflow_name=workflow.name):
        await storage.delete(cp.checkpoint_id)
    with _tracer.start_as_current_span("ci.taskgraph.run",
                                       attributes={"ci.run": run_id, "ci.graph": graph.id, "ci.engine": "maf"}):
        await workflow.run(Start(run_id))
        await asyncio.gather(*run.hardening.values())
    written = {cp.iteration_count for cp in await storage.list_checkpoints(workflow_name=workflow.name)}
    _assert_checkpoints_written(workflow.name, written, 0, where)
    return GraphResult.from_bus(bus, graph, run_id, wall_s=time.perf_counter() - started)
