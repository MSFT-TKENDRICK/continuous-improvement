"""Declarative, expression-free, checkpointed MAF workflows.

* :func:`assert_expression_free` — workflow YAML may contain no PowerFx (no string
  starting with ``=``) and none of the actions that need an expression engine or reach
  the network. Branching lives in Python function tools.
* :func:`build_workflow` — ``WorkflowFactory`` with pre-built agents, registered function
  tools and a ``FileCheckpointStorage`` whose allowlist includes every declarative
  internal type (without it MAF drops checkpoints with only a warning).
* :func:`run_or_resume` — resume from the latest checkpoint of this workflow if one
  exists, else run fresh; raise :class:`CheckpointNotWrittenError` if the run did not
  persist a checkpoint for every superstep it executed.

Trust boundary: checkpoints embed **pickle** data. A checkpoint dir must be private to one
run on one machine (``CI_RUN_DIR``), created ``0700`` where the OS supports it, and never
cached, uploaded, committed or restored across jobs/branches. Resume is at-least-once per
superstep, so side-effecting tools must be idempotent.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Collection, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from agent_framework import FileCheckpointStorage, Workflow, WorkflowCheckpoint

from ci_lab.maf.loader import experimental_features
from ci_lab.maf.specs import iter_strings

FORBIDDEN_ACTIONS: frozenset[str] = frozenset({
    "If", "ConditionGroup", "Foreach", "GotoAction", "BreakLoop", "ContinueLoop",
    "HttpRequestAction", "InvokeMcpTool",
})
DEFAULT_MAX_ITERATIONS = 100


class WorkflowSpecError(ValueError):
    """Workflow YAML is not expression-free or references unknown agents/tools."""


class CheckpointNotWrittenError(RuntimeError):
    """A workflow run finished without persisting a single checkpoint."""


def declarative_allowlist(extra: Collection[str] = ()) -> list[str]:
    """``module:Class`` for every class defined in the declarative base module, plus ``extra``."""
    from agent_framework_declarative._workflows import _declarative_base as base

    names = [f"{base.__name__}:{name}" for name, obj in vars(base).items()
             if inspect.isclass(obj) and obj.__module__ == base.__name__]
    return sorted({*names, *extra})


def _iter_actions(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        if isinstance(obj.get("kind"), str):
            yield obj
        for value in obj.values():
            yield from _iter_actions(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_actions(value)


def assert_expression_free(yaml_text: str) -> dict[str, Any]:
    """Parse workflow YAML and reject ``=`` expressions and forbidden action kinds.
    Returns the parsed definition."""
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise WorkflowSpecError(f"invalid workflow YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkflowSpecError("workflow YAML must be a mapping")
    if found := [p for p, s in iter_strings(data) if s.startswith("=")]:
        raise WorkflowSpecError(f"workflow contains PowerFx '=' expressions at {', '.join(found[:5])}")
    if bad := sorted({a["kind"] for a in _iter_actions(data) if a["kind"] in FORBIDDEN_ACTIONS}):
        raise WorkflowSpecError(f"workflow uses forbidden actions: {bad}")
    return data


def _check_references(data: Mapping[str, Any], agents: Mapping[str, Any], tools: Mapping[str, Any]) -> None:
    if data.get("kind") != "Workflow":
        raise WorkflowSpecError("workflow YAML must have kind: Workflow")
    if "agents" in data:
        raise WorkflowSpecError("inline agent definitions are not allowed; pass pre-built agents")
    for action in _iter_actions(data):
        kind = action["kind"]
        if kind == "InvokeAzureAgent":
            ref = action.get("agent")
            name = ref.get("name") if isinstance(ref, Mapping) else ref
            if name not in agents:
                raise WorkflowSpecError(f"action {action.get('id')!r}: unknown agent {name!r}")
        elif kind == "InvokeFunctionTool" and action.get("functionName") not in tools:
            raise WorkflowSpecError(f"action {action.get('id')!r}: unknown tool {action.get('functionName')!r}")


def secure_dir(path: str | Path) -> Path:
    """Create ``path`` (0700 on POSIX; Windows inherits the parent's ACL) and refuse symlinks."""
    path = Path(path)
    if path.is_symlink():
        raise WorkflowSpecError(f"checkpoint dir may not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(path, 0o700)
    return path


def build_workflow(yaml_path: str | Path, *, agents: Mapping[str, Any], tools: Mapping[str, Callable[..., Any]],
                   checkpoint_dir: str | Path, max_iterations: int = DEFAULT_MAX_ITERATIONS,
                   checkpoint_types: Collection[str] = ()) -> tuple[Workflow, FileCheckpointStorage]:
    """Validate and build a checkpointed declarative workflow.

    ``checkpoint_types`` adds ``module:Class`` entries for our own types that may appear
    in workflow state (tool results)."""
    text = Path(yaml_path).read_text(encoding="utf-8")
    data = assert_expression_free(text)
    _check_references(data, agents, tools)
    storage = FileCheckpointStorage(secure_dir(checkpoint_dir),
                                    allowed_checkpoint_types=declarative_allowlist(checkpoint_types))
    with experimental_features("DECLARATIVE_AGENTS"):
        from agent_framework_declarative import AgentFactory, WorkflowFactory

        # WorkflowFactory builds a default AgentFactory (which loads .env); inline agents are
        # rejected above, so this one is never used to create clients.
        factory = WorkflowFactory(agent_factory=AgentFactory(safe_mode=True, env_file_path=os.devnull),
                                  agents=dict(agents), checkpoint_storage=storage, max_iterations=max_iterations)
    for name, fn in tools.items():
        factory.register_tool(name, fn)
    workflow = factory.create_workflow_from_definition(data, base_path=Path(yaml_path).parent)
    return workflow, storage


def _checkpoint_order(cp: WorkflowCheckpoint) -> tuple[datetime, int]:
    return datetime.fromisoformat(cp.timestamp), cp.iteration_count


async def latest_checkpoint(storage: FileCheckpointStorage, workflow_name: str) -> WorkflowCheckpoint | None:
    """Most recent checkpoint (by timestamp, then iteration) for ``workflow_name``."""
    checkpoints = await storage.list_checkpoints(workflow_name=workflow_name)
    return max(checkpoints, key=_checkpoint_order) if checkpoints else None


async def run_or_resume(yaml_path: str | Path, message: Any, *, agents: Mapping[str, Any],
                        tools: Mapping[str, Callable[..., Any]], checkpoint_dir: str | Path,
                        max_iterations: int = DEFAULT_MAX_ITERATIONS,
                        checkpoint_types: Collection[str] = ()) -> list[Any]:
    """Resume ``yaml_path`` from its latest checkpoint in ``checkpoint_dir`` if any, else run
    it fresh with ``message``. Returns the workflow outputs of this invocation."""
    workflow, storage = build_workflow(yaml_path, agents=agents, tools=tools, checkpoint_dir=checkpoint_dir,
                                       max_iterations=max_iterations, checkpoint_types=checkpoint_types)
    before = {cp.checkpoint_id for cp in await storage.list_checkpoints(workflow_name=workflow.name)}
    latest = await latest_checkpoint(storage, workflow.name)
    if latest is not None:
        result = await workflow.run(checkpoint_id=latest.checkpoint_id, checkpoint_storage=storage)
    else:
        result = await workflow.run(message)
    written = {cp.iteration_count for cp in await storage.list_checkpoints(workflow_name=workflow.name)
               if cp.checkpoint_id not in before}
    _assert_checkpoints_written(workflow.name, written, 0 if latest is None else latest.iteration_count + 1,
                                checkpoint_dir)
    return list(result.get_outputs())


def _assert_checkpoints_written(name: str, iterations: set[int], first: int, where: str | Path) -> None:
    # MAF only logs a warning when a checkpoint fails to save (e.g. a state type missing from
    # allowed_checkpoint_types) and carries on, so a run can write the first/last checkpoint
    # but silently skip the ones in between. Require one checkpoint per superstep.
    if not iterations:
        raise CheckpointNotWrittenError(f"workflow {name!r} wrote no checkpoint to {where}")
    if missing := sorted(set(range(first, max(iterations) + 1)) - iterations):
        raise CheckpointNotWrittenError(
            f"workflow {name!r} skipped checkpoints for supersteps {missing} in {where} "
            "(state type missing from allowed_checkpoint_types?)")
