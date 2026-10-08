"""Production ``run_target``: the order-support agent on MAF with a candidate skill injected.

Each candidate (skill, memory) pair is materialized into a temp harness copy at
``<root>/<hash>/skills/order-support/SKILL.md`` (+ ``memory.md``); the agent's
instructions are composed from that copy, so what is evaluated is exactly what would be
committed. The tool loop is MAF's own (real ``order_support.tools`` simulators); every call
is recorded as a :class:`~ci_lab.contracts.ToolCallRecord` for the safety oracle.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skillopt_sleep.types import TaskRecord

from ci_lab.contracts import ToolCallRecord, Transcript

SKILL_REL = Path("skills") / "order-support" / "SKILL.md"
MEMORY_REL = Path("skills") / "order-support" / "memory.md"
SYSTEM_REL = Path("prompts") / "system.md"


def _hash(skill: str, memory: str) -> str:
    return hashlib.sha256(f"{skill}\x00{memory}".encode()).hexdigest()[:16]


_mat_lock = threading.Lock()


def materialize_harness(skill: str, memory: str, root: Path, base_harness: Path | None = None) -> Path:
    """Copy ``base_harness`` (if any) to ``root/<hash>`` and write the candidate skill/memory."""
    dest = Path(root) / _hash(skill, memory)
    with _mat_lock:
        if (dest / SKILL_REL).exists():
            return dest
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        if base_harness is not None and Path(base_harness).is_dir():
            shutil.copytree(base_harness, tmp)
        (tmp / SKILL_REL).parent.mkdir(parents=True, exist_ok=True)
        (tmp / SKILL_REL).write_text(skill, encoding="utf-8", newline="")
        (tmp / MEMORY_REL).write_text(memory, encoding="utf-8", newline="")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    return dest


def default_system_prompt() -> str:
    from order_support import data

    return data.load_policy() + f"\nToday's date is {data.TODAY.isoformat()}."


def compose_instructions(harness_dir: Path, system_prompt: str | None = None) -> str:
    system = system_prompt
    if system is None:
        sys_path = harness_dir / SYSTEM_REL
        system = sys_path.read_text(encoding="utf-8") if sys_path.exists() else default_system_prompt()
    parts = [system.rstrip()]
    for rel, title in ((SKILL_REL, "Skill: order-support"), (MEMORY_REL, "Memory")):
        p = harness_dir / rel
        if p.exists() and (text := p.read_text(encoding="utf-8").strip()):
            parts.append(f"## {title}\n\n{text}")
    return "\n\n".join(parts) + "\n"


def _recording_tools(records: list[ToolCallRecord], execute: Callable[[str, dict[str, Any]], Any]) -> list[Any]:
    from agent_framework import tool

    def rec(name: str, args: dict[str, Any]) -> Any:
        result = execute(name, args)
        records.append(ToolCallRecord(call_id=f"tc-{len(records)}", name=name, arguments=dict(args),
                                      result=result, turn=0))
        return result

    def lookup_order(order_id: str) -> Any:
        """Fetch an order record by order id (e.g. NW-10001)."""
        return rec("lookup_order", {"order_id": order_id})

    def search_kb(query: str) -> Any:
        """Search the store policy knowledge base (returns, shipping, warranty, promotions)."""
        return rec("search_kb", {"query": query})

    def issue_refund(order_id: str, amount: float) -> Any:
        """Issue a refund for an order."""
        return rec("issue_refund", {"order_id": order_id, "amount": amount})

    def escalate_to_human(reason: str, order_id: str | None = None) -> Any:
        """Hand the case to the human support team."""
        args: dict[str, Any] = {"reason": reason}
        if order_id is not None:
            args["order_id"] = order_id
        return rec("escalate_to_human", args)

    return [tool(f) for f in (lookup_order, search_kb, issue_refund, escalate_to_human)]


def _usage(response: Any) -> tuple[int, int]:
    u = getattr(response, "usage_details", None) or {}
    get = u.get if isinstance(u, dict) else (lambda k, d=None: getattr(u, k, d))
    return int(get("input_token_count", 0) or 0), int(get("output_token_count", 0) or 0)


def make_maf_run_target(client_factory: Callable[[], Any], *, harness_root: Path,
                        base_harness: Path | None = None, system_prompt: str | None = None,
                        execute: Callable[[str, dict[str, Any]], Any] | None = None):
    """``run_target(task, skill, memory) -> (reply, tools_called, transcript)``."""
    if execute is None:
        from order_support.tools import execute as os_execute

        execute = os_execute

    def run_target(task: TaskRecord, skill: str, memory: str) -> tuple[str, list[str], Transcript]:
        from agent_framework import Agent

        harness = materialize_harness(skill, memory, harness_root, base_harness)
        records: list[ToolCallRecord] = []
        agent = Agent(client=client_factory(), instructions=compose_instructions(harness, system_prompt),
                      name="order-support", tools=_recording_tools(records, execute))
        message = task.intent if not task.context_excerpt else f"{task.context_excerpt}\n\n{task.intent}"
        response = asyncio.run(agent.run(message))
        reply = str(getattr(response, "text", "") or "")
        tin, tout = _usage(response)
        model = getattr(response, "model_id", None) or getattr(response, "model", None)
        transcript = Transcript(case_id=task.id,
                                messages=[{"role": "user", "content": message},
                                          {"role": "assistant", "content": reply}],
                                tool_calls=tuple(records), served_models=(str(model),) if model else (),
                                tokens_in=tin, tokens_out=tout)
        return reply, [r.name for r in records], transcript

    return run_target
