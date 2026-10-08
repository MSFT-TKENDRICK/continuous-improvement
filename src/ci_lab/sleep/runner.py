"""Workflow runners for ``sleep.yaml``.

``run_workflow(yaml_path, tools, ckpt_dir) -> dict[action_id, result]`` is injected into
:func:`ci_lab.sleep.night.run_night`:

* :func:`maf_runner` — the production runner: MAF ``agent_framework_declarative``
  ``WorkflowFactory`` with a :class:`FileCheckpointStorage` (built by
  :func:`ci_lab.maf.workflows.build_workflow`); ``resume=True`` continues an interrupted run
  from its latest checkpoint.
* :func:`sequential_runner` — a dependency-free interpreter of the same YAML, used only when
  explicitly requested (tests, the ``fake`` profile without MAF, or ``SLEEP_RUNNER=sequential``).

:func:`default_runner` fails closed: for the ``copilot``/``offline`` profiles it never falls
back from MAF to the sequential interpreter. Both refuse anything that is not
expression-free (no PowerFx, hence no .NET).
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from ci_lab import obs

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


def _offloaded(fn: Tool, pool: ThreadPoolExecutor) -> Tool:
    """Run a sync step in a worker thread: steps may call ``asyncio.run`` themselves
    (MAF agents), which is illegal on the workflow's event-loop thread."""

    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> Any:
        return pool.submit(fn, **kwargs).result()

    return wrapper


STATUS_FILE = "run.status"  # not *.json: FileCheckpointStorage treats every *.json as a checkpoint
STEPS_FILE = "steps.done"
RUNNER_ENV = "SLEEP_RUNNER"
RUNNERS = ("maf", "sequential")


def _done_steps(ckpt: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        lines = (ckpt / STEPS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # torn last line of a crashed run
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            out[row["id"]] = row.get("result")
    return out


def maf_runner(yaml_path: Path, tools: Mapping[str, Tool], ckpt_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run ``sleep.yaml`` with MAF declarative workflows, checkpointing every superstep to
    ``ckpt_dir`` (:class:`FileCheckpointStorage`).

    With ``resume=True`` a previous run in ``ckpt_dir`` that did not finish is continued from
    its latest checkpoint: completed steps are not re-executed and their results are read back
    from ``steps.done`` (JSON; non-JSON values come back as ``repr`` strings). Otherwise the
    workflow runs fresh.
    """
    from ci_lab.maf.workflows import build_workflow, latest_checkpoint

    text = Path(yaml_path).read_text(encoding="utf-8")
    actions = load_actions(text)  # validate before MAF sees it
    ckpt = Path(ckpt_dir)
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sleep-step") as pool:
        by_name = {a["functionName"]: a["id"] for a in actions}
        wrapped: dict[str, Tool] = {}
        for name, fn in tools.items():
            step = _offloaded(fn, pool)

            def recording(_step: Tool = step, _id: str = by_name.get(name, name), **kw: Any) -> Any:
                results[_id] = _step(**kw)
                with (ckpt / STEPS_FILE).open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": _id, "result": results[_id]}, default=repr) + "\n")
                return results[_id]

            functools.update_wrapper(recording, fn)
            wrapped[name] = recording
        workflow, storage = build_workflow(yaml_path, agents={}, tools=wrapped, checkpoint_dir=ckpt)
        status = ckpt / STATUS_FILE

        async def _run() -> int:
            before = {c.checkpoint_id for c in await storage.list_checkpoints(workflow_name=workflow.name)}
            latest = await latest_checkpoint(storage, workflow.name) if resume else None
            running = status.is_file() and status.read_text(encoding="utf-8").strip() == "running"
            if latest is not None and running:
                results.update(_done_steps(ckpt))
                await workflow.run(checkpoint_id=latest.checkpoint_id, checkpoint_storage=storage)
            else:
                (ckpt / STEPS_FILE).unlink(missing_ok=True)
                status.write_text("running", encoding="utf-8")
                await workflow.run("sleep")
            after = {c.checkpoint_id for c in await storage.list_checkpoints(workflow_name=workflow.name)}
            return len(after - before)

        # run the event loop in its own thread so callers may already be inside a loop;
        # wrap_ctx keeps MAF's own spans inside the night trace
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sleep-wf") as loop_pool:
            written = loop_pool.submit(obs.wrap_ctx(lambda: asyncio.run(_run()))).result()
    missing = [a["id"] for a in actions if a["id"] not in results]
    if missing:
        raise WorkflowError(f"workflow did not run steps: {missing}")
    if not written:
        raise WorkflowError(f"workflow wrote no checkpoint to {ckpt}")
    status.write_text("done", encoding="utf-8")
    return results


def _maf_missing() -> str | None:
    for mod in ("agent_framework", "agent_framework_declarative", "ci_lab.maf.workflows"):
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            return f"{mod}: {exc}"
    return None


def default_runner(profile: str | None = None) -> RunWorkflow:
    """The runner for ``profile``: always :func:`maf_runner` for production profiles.

    :func:`sequential_runner` is only returned when explicitly requested
    (``SLEEP_RUNNER=sequential``) or for the ``fake`` profile when MAF is not installed. A
    production profile without MAF declarative workflows raises :class:`WorkflowError`
    instead of silently running outside MAF.
    """
    choice = os.environ.get(RUNNER_ENV, "").strip().lower()
    if choice and choice not in RUNNERS:
        raise WorkflowError(f"{RUNNER_ENV}={choice!r}: expected one of {', '.join(RUNNERS)}")
    if choice == "sequential":
        return sequential_runner
    missing = _maf_missing()
    if missing is None:
        return maf_runner
    if profile == "fake" and not choice:
        return sequential_runner
    raise WorkflowError(f"sleep needs MAF declarative workflows (agent-framework-declarative), which failed to "
                        f"import ({missing}); refusing to fall back to the sequential interpreter for profile "
                        f"{profile or 'default'!r}. Install the 'maf' dependencies or set {RUNNER_ENV}=sequential "
                        "for tests.")
