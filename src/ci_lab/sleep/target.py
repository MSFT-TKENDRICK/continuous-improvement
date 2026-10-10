"""Production sleep targets with a candidate harness skill injected.

Each candidate is materialized into a content-addressed harness copy. The default
``make_harness_run_target`` loads the owning frozen declarative agent from that copy, so its
``skills_paths`` sees exactly the candidate that the acceptance gate evaluates. The legacy
``make_maf_run_target`` order-support adapter remains available explicitly until L9.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from skillopt_sleep.types import TaskRecord

from ci_lab.contracts import ToolCallRecord, Transcript

SKILL_REL = Path("skills") / "order-support" / "SKILL.md"
MEMORY_REL = Path("skills") / "order-support" / "memory.md"
SYSTEM_REL = Path("prompts") / "system.md"
AGENT_YAML = "agent.yaml"


def _hash(skill: str, memory: str, skill_rel: Path, memory_rel: Path | None) -> str:
    material = f"{skill_rel.as_posix()}\x00{memory_rel}\x00{skill}\x00{memory}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


_mat_lock = threading.Lock()


def materialize_harness(skill: str, memory: str, root: Path, base_harness: Path | None = None, *,
                        skill_rel: Path = SKILL_REL, memory_rel: Path | None = MEMORY_REL) -> Path:
    """Copy ``base_harness`` (if any) to ``root/<hash>`` and write the candidate skill/memory."""
    dest = Path(root) / _hash(skill, memory, skill_rel, memory_rel)
    with _mat_lock:
        if (dest / skill_rel).exists():
            return dest
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        if base_harness is not None and Path(base_harness).is_dir():
            shutil.copytree(base_harness, tmp)
        (tmp / skill_rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / skill_rel).write_text(skill, encoding="utf-8", newline="")
        if memory_rel is not None:
            (tmp / memory_rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / memory_rel).write_text(memory, encoding="utf-8", newline="")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    return dest


def default_system_prompt() -> str:
    from order_support import data

    return data.load_policy() + f"\nToday's date is {data.TODAY.isoformat()}."


def instruction_files(harness_dir: Path) -> list[Path]:
    """Relative instruction files, in order, from ``agent.yaml``'s ``x-ci.instructions_files``.

    Mirrors ``order_support.agent`` so a new file in that list (e.g. ``prompts/identity.md``) is
    evaluated without code changes here. Falls back to ``system.md`` + the skill when the harness
    copy has no ``agent.yaml``, no list, or an unreadable one. Entries escaping the harness are
    rejected, as in production.
    """
    default = [SYSTEM_REL, SKILL_REL]
    spec_path = Path(harness_dir) / AGENT_YAML
    if not spec_path.is_file():
        return default
    import yaml

    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return default
    x_ci = spec.get("x-ci") if isinstance(spec, dict) else None
    files = x_ci.get("instructions_files") if isinstance(x_ci, dict) else None
    if not isinstance(files, list) or not files:
        return default
    root = Path(harness_dir).resolve()
    out: list[Path] = []
    for f in files:
        rel = Path(str(f))
        if rel.is_absolute() or not (root / rel).resolve().is_relative_to(root):
            raise ValueError(f"harness instructions file {str(f)!r} escapes {root}")
        out.append(rel)
    return out


def compose_instructions(harness_dir: Path, system_prompt: str | None = None) -> str:
    """System prompt, then every other listed instruction file, then the candidate memory.

    ``system_prompt`` replaces ``prompts/system.md``; when neither is available the frozen
    policy is used. The skill keeps its ``## Skill`` heading; other files are included verbatim.
    """
    harness_dir = Path(harness_dir)
    system = system_prompt
    if system is None:
        sys_path = harness_dir / SYSTEM_REL
        system = sys_path.read_text(encoding="utf-8") if sys_path.exists() else default_system_prompt()
    parts = [system.rstrip()]
    files = [rel for rel in instruction_files(harness_dir) if rel != SYSTEM_REL]
    if SKILL_REL not in files:
        files.append(SKILL_REL)
    for rel in [*files, MEMORY_REL]:
        p = harness_dir / rel
        if not (p.is_file() and (text := p.read_text(encoding="utf-8").strip())):
            continue
        title = {SKILL_REL: "Skill: order-support", MEMORY_REL: "Memory"}.get(rel)
        parts.append(f"## {title}\n\n{text}" if title else text)
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


def _harness_rel(path: str) -> Path:
    rel = Path(path.replace("\\", "/"))
    if rel.parts[:1] == ("harness",):
        rel = Path(*rel.parts[1:])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"invalid harness target path {path!r}")
    return rel


def make_harness_run_target(client_factory: Callable[[], Any], *, harness_root: Path,
                            base_harness: Path, owner_agent: str, skill_path: str,
                            memory_path: str | None = None):
    """Run a frozen harness agent with the candidate skill exposed from its copied tree."""
    skill_rel = _harness_rel(skill_path)
    memory_rel = _harness_rel(memory_path) if memory_path else None

    def run_target(task: TaskRecord, skill: str, memory: str) -> tuple[str, list[str], Transcript]:
        from agent_framework import FunctionTool

        from ci_lab.governance.maf import governed_harness_agent
        from ci_lab.meta.spec_loader import load_spec

        harness = materialize_harness(skill, memory, harness_root, base_harness,
                                      skill_rel=skill_rel, memory_rel=memory_rel)
        spec = load_spec(owner_agent, harness_dir=harness)
        records: list[ToolCallRecord] = []
        submitted: list[Mapping[str, Any]] = []

        def bind(name: str) -> FunctionTool:
            async def invoke(**kwargs: Any) -> Any:
                result: Any
                if name.startswith("submit_"):
                    submitted.append(dict(kwargs))
                    result = {"accepted": True}
                elif name == "list_files":
                    result = [p.relative_to(harness).as_posix() for p in sorted(harness.rglob("*"))
                              if p.is_file()][:200]
                elif name == "read_file":
                    rel = _harness_rel(str(kwargs.get("path") or ""))
                    path = (harness / rel).resolve()
                    result = (path.read_text(encoding="utf-8")[:20_000]
                              if path.is_relative_to(harness.resolve()) and path.is_file()
                              else "ERROR: file unavailable")
                elif name == "search_text":
                    needle = str(kwargs.get("query") or kwargs.get("pattern") or "").lower()
                    result = [p.relative_to(harness).as_posix() for p in sorted(harness.rglob("*"))
                              if p.is_file() and needle in p.read_text(encoding="utf-8", errors="ignore").lower()][:50]
                else:
                    result = "UNAVAILABLE in sleep replay; inspect the supplied task evidence"
                records.append(ToolCallRecord(call_id=f"tc-{len(records)}", name=name,
                                              arguments=dict(kwargs), result=result, turn=0))
                return result

            return FunctionTool(name=name, description=f"Sleep replay binding for {name}.", func=invoke)

        client = client_factory()
        skill_dirs = list(dict.fromkeys([*(str(path) for path in spec.skills_paths), str(harness / "skills")]))
        agent = governed_harness_agent(
            client,
            name=spec.name,
            description=spec.description,
            agent_instructions=spec.instructions,
            tools=[bind(name) for name in spec.tools],
            loop_max_iterations=spec.max_turns or 8,
            skills_paths=skill_dirs,
            governance={"agent_name": spec.name, "model": spec.model},
        )
        message = task.intent if not task.context_excerpt else f"{task.context_excerpt}\n\n{task.intent}"
        response = asyncio.run(agent.run(message))
        reply = str(getattr(response, "text", "") or "")
        if not reply and submitted:
            reply = json.dumps(submitted[-1], sort_keys=True)
        tin, tout = _usage(response)
        model = getattr(response, "model_id", None) or getattr(response, "model", None) or spec.model
        transcript = Transcript(
            case_id=task.id,
            messages=[{"role": "user", "content": message}, {"role": "assistant", "content": reply}],
            tool_calls=tuple(records), served_models=(str(model),), tokens_in=tin, tokens_out=tout,
        )
        return reply, [record.name for record in records], transcript

    return run_target


def make_maf_run_target(client_factory: Callable[[], Any], *, harness_root: Path,
                        base_harness: Path | None = None, system_prompt: str | None = None,
                        execute: Callable[[str, dict[str, Any]], Any] | None = None):
    """``run_target(task, skill, memory) -> (reply, tools_called, transcript)``."""
    if execute is None:
        from order_support.tools import execute as os_execute

        execute = os_execute

    def run_target(task: TaskRecord, skill: str, memory: str) -> tuple[str, list[str], Transcript]:
        from ci_lab.governance.maf import governed_agent
        from ci_lab.governance.policies import target_mode

        harness = materialize_harness(skill, memory, harness_root, base_harness)
        records: list[ToolCallRecord] = []
        agent = governed_agent(client=client_factory(), instructions=compose_instructions(harness, system_prompt),
                               name="order-support", tools=_recording_tools(records, execute),
                               policy="order_support", governance={"mode": target_mode()})
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
