"""Workflow runners for ``sleep.yaml``.

``run_workflow(yaml_path, tools, ckpt_dir) -> dict[action_id, result]`` is injected into
:func:`ci_lab.sleep.night.run_night`:

* :func:`maf_runner` — the production runner: MAF ``agent_framework_declarative``
  ``WorkflowFactory`` with a :class:`FileCheckpointStorage` (uses ``ci_lab.maf.workflows``
  when that module provides ``run_workflow``).
* :func:`sequential_runner` — a dependency-free interpreter of the same YAML for tests.

Both refuse anything that is not expression-free (no PowerFx, hence no .NET).
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

Tool = Callable[..., Any]
RunWorkflow = Callable[[Path, Mapping[str, Tool], Path], Mapping[str, Any]]

FORBIDDEN_KINDS = frozenset({"If", "ConditionGroup", "Foreach", "GotoAction", "Goto", "BreakLoop",
                             "ContinueLoop", "SetVariable", "SetTextVariable", "ParseValue"})
ALLOWED_ACTIONS = frozenset({"InvokeFunctionTool"})


class WorkflowError(RuntimeError):
    pass


def _walk(node: Any, path: str = "$") -> None:
    if isinstance(node, str):
        if node.lstrip().startswith("="):
            raise WorkflowError(f"{path}: PowerFx expression {node!r} not allowed")
    elif isinstance(node, Mapping):
        kind = node.get("kind")
        if isinstance(kind, str) and kind in FORBIDDEN_KINDS:
            raise WorkflowError(f"{path}: action kind {kind} not allowed (expression-free)")
        for k, v in node.items():
            _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, f"{path}[{i}]")


def load_actions(yaml_text: str) -> list[dict[str, Any]]:
    """Parse + validate: Workflow / OnConversationStart / literal InvokeFunctionTool actions only."""
    doc = yaml.safe_load(yaml_text)
    if not isinstance(doc, dict) or doc.get("kind") != "Workflow":
        raise WorkflowError("not a declarative Workflow")
    _walk(doc)
    trigger = doc.get("trigger") or {}
    if trigger.get("kind") != "OnConversationStart":
        raise WorkflowError("trigger must be OnConversationStart")
    actions = trigger.get("actions") or []
    if not actions:
        raise WorkflowError("workflow has no actions")
    for a in actions:
        if a.get("kind") not in ALLOWED_ACTIONS:
            raise WorkflowError(f"action kind {a.get('kind')!r} not allowed")
        for k, v in (a.get("arguments") or {}).items():
            if not isinstance(v, (str, int, float, bool)) or v is None:
                raise WorkflowError(f"action {a.get('id')}: argument {k} must be a literal scalar")
    return actions


def sequential_runner(yaml_path: Path, tools: Mapping[str, Tool], ckpt_dir: Path) -> dict[str, Any]:
    actions = load_actions(Path(yaml_path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for a in actions:
        fn = tools.get(a["functionName"])
        if fn is None:
            raise WorkflowError(f"tool {a['functionName']} not registered")
        out[a["id"]] = fn(**dict(a.get("arguments") or {}))
    return out


def _allowed_checkpoint_types() -> list[str]:
    import agent_framework_declarative._workflows._declarative_base as db

    return [f"{db.__name__}:{n}" for n, o in vars(db).items()
            if inspect.isclass(o) and o.__module__ == db.__name__]


def _offloaded(fn: Tool, pool: ThreadPoolExecutor) -> Tool:
    """Run a sync step in a worker thread: steps may call ``asyncio.run`` themselves
    (MAF agents), which is illegal on the workflow's event-loop thread."""

    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> Any:
        return pool.submit(fn, **kwargs).result()

    return wrapper


def maf_runner(yaml_path: Path, tools: Mapping[str, Tool], ckpt_dir: Path) -> dict[str, Any]:
    text = Path(yaml_path).read_text(encoding="utf-8")
    actions = load_actions(text)  # validate before MAF sees it
    try:
        shared = importlib.import_module("ci_lab.maf.workflows")
    except ImportError:
        shared = None
    if shared is not None and callable(getattr(shared, "run_workflow", None)):
        return dict(shared.run_workflow(Path(yaml_path), dict(tools), Path(ckpt_dir)))

    from agent_framework import FileCheckpointStorage
    from agent_framework_declarative import WorkflowFactory

    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    storage = FileCheckpointStorage(str(ckpt), allowed_checkpoint_types=_allowed_checkpoint_types())
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sleep-step") as pool:
        factory = WorkflowFactory(checkpoint_storage=storage)
        by_name = {a["functionName"]: a["id"] for a in actions}
        for name, fn in tools.items():
            step = _offloaded(fn, pool)

            def recording(_step: Tool = step, _id: str = by_name.get(name, name), **kw: Any) -> Any:
                results[_id] = _step(**kw)
                return results[_id]

            functools.update_wrapper(recording, fn)
            factory = factory.register_tool(name, recording)
        workflow = factory.create_workflow_from_yaml(text)

        def _run() -> None:
            asyncio.run(workflow.run("sleep"))

        # run the event loop in its own thread so callers may already be inside a loop
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sleep-wf") as loop_pool:
            loop_pool.submit(_run).result()
    missing = [a["id"] for a in actions if a["id"] not in results]
    if missing:
        raise WorkflowError(f"workflow did not run steps: {missing}")
    return results


def default_runner() -> RunWorkflow:
    try:
        importlib.import_module("agent_framework_declarative")
    except ImportError:
        return sequential_runner
    return maf_runner
