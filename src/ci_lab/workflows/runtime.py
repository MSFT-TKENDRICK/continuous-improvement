"""Default MAF workflow runtime: build + checkpointed run/resume (design §5, C14).

``ci_lab.maf.workflows`` (M1) owns the canonical implementation; these defaults
keep the campaign driver runnable on their own and are injected through
:class:`ci_lab.campaign.deps.CampaignDeps` so integration can swap them.

Failure model: declarative ``InvokeFunctionTool`` swallows ``Exception``s, so step
functions convert errors into :class:`StepAborted` (a ``BaseException``) which
stops the workflow at the failing superstep. ``run_or_resume`` leaves the run
marked ``running`` (``<ckpt>/run.status``); the next call resumes from the latest checkpoint and only
re-executes the in-flight superstep (steps are idempotent via run-dir markers).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

STATUS_FILE = "run.status"  # not *.json: FileCheckpointStorage scans those


class StepAborted(BaseException):  # noqa: N818 - control-flow signal, not an error type
    """Raised by a step function to abort the enclosing workflow run."""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"step {step!r} failed: {type(cause).__name__}: {cause}")
        self.step = step
        self.cause = cause


class StepFailed(RuntimeError):
    """Raised to callers when a workflow aborted; rerunning resumes it."""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"workflow step {step!r} failed: {type(cause).__name__}: {cause}")
        self.step = step


def declarative_allowlist() -> list[str]:
    """``module:qualname`` of declarative internals that must be unpickled (silent-failure guard)."""
    import agent_framework_declarative._workflows._declarative_base as base

    return sorted(f"{base.__name__}:{name}" for name, obj in vars(base).items()
                  if inspect.isclass(obj) and obj.__module__ == base.__name__)


def _storage(ckpt_dir: Path) -> Any:
    from agent_framework import FileCheckpointStorage

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(ckpt_dir, 0o700)
    return FileCheckpointStorage(ckpt_dir, allowed_checkpoint_types=declarative_allowlist())


def build_workflow(path: Path, agents: Mapping[str, Any], tools: Mapping[str, Callable[..., Any]],
                   ckpt_dir: Path) -> Any:
    """WorkflowFactory + FileCheckpointStorage; tools are registered by name."""
    from agent_framework_declarative import WorkflowFactory

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # AgentFactory ExperimentalWarning (unused: agents are prebuilt)
        factory = WorkflowFactory(agents=dict(agents), checkpoint_storage=_storage(ckpt_dir))
        for name, fn in tools.items():
            factory.register_tool(name, fn)
        return factory.create_workflow_from_yaml_path(path)


def _status(ckpt_dir: Path) -> str | None:
    try:
        return json.loads((ckpt_dir / STATUS_FILE).read_text(encoding="utf-8"))["state"]
    except (OSError, ValueError, KeyError):
        return None


def _set_status(ckpt_dir: Path, state: str, **extra: Any) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f".{STATUS_FILE}.tmp"
    tmp.write_text(json.dumps({"state": state, **extra}), encoding="utf-8")
    os.replace(tmp, ckpt_dir / STATUS_FILE)


async def run_or_resume(workflow: Any, ckpt_dir: Path, message: str = "start") -> dict[str, Any]:
    """Run ``workflow`` or resume it from its latest checkpoint if a previous run
    in ``ckpt_dir`` did not complete. Raises :class:`StepFailed` on step abort."""
    state = _status(ckpt_dir)
    if state == "completed":
        return {"status": "completed", "resumed": False, "skipped": True}
    storage = _storage(ckpt_dir)
    latest = await storage.get_latest(workflow_name=workflow.name) if state == "running" else None
    _set_status(ckpt_dir, "running", workflow=workflow.name)
    try:
        if latest is not None:
            result = await workflow.run(checkpoint_id=latest.checkpoint_id, checkpoint_storage=storage)
        else:
            result = await workflow.run(message)
    except StepAborted as exc:
        raise StepFailed(exc.step, exc.cause) from exc.cause
    if not await storage.list_checkpoints(workflow_name=workflow.name):
        raise RuntimeError(f"workflow {workflow.name!r} wrote no checkpoints (allowlist?)")
    _set_status(ckpt_dir, "completed", workflow=workflow.name)
    outputs = result.get_outputs() if hasattr(result, "get_outputs") else []
    return {"status": "completed", "resumed": latest is not None, "outputs": [str(o) for o in outputs]}


class GatedAgent:
    """Wraps a MAF agent so a re-run of its workflow action is a no-op once the
    agent's durable output (written by its terminal ``submit_*`` tool) exists.
    ``wrap`` (optional) returns an async context manager entered around each real
    run, e.g. a tracing/progress phase."""

    def __init__(self, agent: Any, done: Path,
                 wrap: Callable[[], contextlib.AbstractAsyncContextManager[Any]] | None = None) -> None:
        self._agent = agent
        self._done = done
        self._wrap = wrap

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    async def run(self, messages: Any = None, **kwargs: Any) -> Any:
        if self._done.exists():
            return f"already submitted: {self._done.name}"
        async with (self._wrap() if self._wrap is not None else contextlib.nullcontext()):
            return await self._agent.run(messages, **kwargs)
