"""Student factories for ``ci-lab graph run``: offline ``fake_student`` and the MAF ``AgentStudentFactory``."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ci_lab.bus.voters.local import artifact_path
from ci_lab.taskgraph.model import StudentSpec

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def fake_student(spec: StudentSpec, middleware: Any) -> Any:
    """Deterministic echo for offline smoke runs: the first fenced block of the instructions (else them)."""

    async def run(message: str) -> Any:
        m = _FENCE.search(spec.instructions)
        return SimpleNamespace(text=m.group(1) if m else spec.instructions)

    return SimpleNamespace(run=run)


class AgentStudentFactory:
    """``<harness_dir>/agents/student.yaml`` on ``profile``: file tools over a fresh per-attempt workspace
    seeded with the task's ``file`` context refs (relative to ``context_root``); returns the written output
    artifact. ``harness_dir=None`` uses the repo-root ``harness/``."""

    def __init__(self, *, context_root: Path, work_root: Path, profile: str = "copilot",
                 client: Any = None, builder: Callable[..., Any] | None = None,
                 harness_dir: Path | None = None) -> None:
        from ci_lab.meta.spec_loader import load_spec

        self.meta = load_spec("student", harness_dir=harness_dir)
        self.context_root, self.work_root, self.profile = Path(context_root), Path(work_root), profile
        self.client, self.builder = client, builder

    def __call__(self, spec: StudentSpec, middleware: Any) -> Any:
        return SimpleNamespace(run=lambda message: self._run(spec, list(middleware or ()), message))

    async def _run(self, spec: StudentSpec, middleware: list[Any], message: str) -> str:
        from ci_lab.meta.spec_loader import TerminalSubmitMiddleware, default_builder
        from ci_lab.providers.factory import make_chat_client
        from ci_lab.tools.arm_fs import make_arm_fs

        self.work_root.mkdir(parents=True, exist_ok=True)
        ws = Path(tempfile.mkdtemp(prefix=f"{spec.id}-", dir=self.work_root))
        for c in spec.context:
            if c.kind == "file" and (src := self.context_root / c.ref).is_file():
                (ws / c.ref).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, ws / c.ref)
        out = artifact_path(spec)
        done: list[str] = []

        def submit_output(summary: str) -> str:
            if not (ws / out).is_file():
                return f"ERROR: write {out} first"
            done.append(summary)
            return "submitted"

        tools = {**make_arm_fs(ws, ["**"], writable_globs=[out]), "submit_output": submit_output}
        client = self.client or make_chat_client(profile=self.profile, model=self.meta.model, purpose="target")
        agent = (self.builder or default_builder())(
            self.meta, client=client, bindings={t: tools[t] for t in self.meta.tools},
            middleware=[TerminalSubmitMiddleware(self.meta.terminal_tool), *middleware],
            loop_should_continue=lambda **_: not done,
            loop_next_message=lambda **_: f"Write {out} with write_file, then call submit_output.")
        session = agent.create_session() if hasattr(agent, "create_session") else None
        await agent.run(message, session=session)
        return (ws / out).read_text(encoding="utf-8") if (ws / out).is_file() else ""
