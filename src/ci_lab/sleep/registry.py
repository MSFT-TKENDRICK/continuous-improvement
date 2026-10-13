"""Multi-target skills registry (``targets.yaml``): which skills SkillOpt-Sleep evolves.

Each target names the skill (and optional memory) file, the agent that owns it, the ASSERT eval
suite gating it and its reviewed tasks file. Paths are confined to the prefixes the publisher
accepts, so a registry edit can never widen what a nightly PR may touch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FORMAT = "ci_lab.sleep.targets.v1"
DEFAULT_REGISTRY = Path(__file__).with_name("targets.yaml")
SKILL_PREFIX = "src/order_support/harness/skills/"
TASKS_PREFIX = "experiments/sleep/"
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_KEYS = frozenset({"name", "skill", "memory", "owner_agent", "eval_suite", "tasks", "enabled"})


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class SkillTarget:
    name: str
    skill_path: str
    owner_agent: str
    eval_suite: str
    tasks_file: str = "experiments/sleep/tasks.jsonl"
    memory_path: str | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "skill": self.skill_path, "memory": self.memory_path,
                "owner_agent": self.owner_agent, "eval_suite": self.eval_suite, "tasks": self.tasks_file}


ORDER_SUPPORT = SkillTarget(
    name="order-support",
    skill_path="src/order_support/harness/skills/order-support/SKILL.md",
    memory_path="src/order_support/harness/skills/order-support/memory.md",
    owner_agent="order-support", eval_suite="order_support")


def _rel(value: Any, prefix: str, what: str, name: str) -> str:
    path = str(value or "")
    parts = path.split("/")
    if (not path.startswith(prefix) or "\\" in path or ":" in path
            or any(p in ("", ".", "..", ".git") for p in parts)):
        raise RegistryError(f"target {name}: {what} {path!r} must be a plain path under {prefix}")
    return path


def parse_target(raw: Any) -> SkillTarget:
    if not isinstance(raw, dict):
        raise RegistryError("each target must be a mapping")
    unknown = set(raw) - _KEYS
    if unknown:
        raise RegistryError(f"unknown target keys: {sorted(unknown)}")
    name = str(raw.get("name") or "")
    if not _NAME_RE.fullmatch(name):
        raise RegistryError(f"bad target name {name!r}")
    skill = _rel(raw.get("skill"), SKILL_PREFIX, "skill", name)
    if not skill.endswith("/SKILL.md"):
        raise RegistryError(f"target {name}: skill must point at a SKILL.md")
    memory = raw.get("memory")
    memory = _rel(memory, SKILL_PREFIX, "memory", name) if memory else None
    tasks = _rel(raw.get("tasks") or "experiments/sleep/tasks.jsonl", TASKS_PREFIX, "tasks", name)
    if not tasks.endswith(".jsonl") or tasks.endswith(".pending.jsonl"):
        raise RegistryError(f"target {name}: tasks must be a reviewed .jsonl file (not pending)")
    owner, suite = str(raw.get("owner_agent") or ""), str(raw.get("eval_suite") or "")
    if not owner or not suite:
        raise RegistryError(f"target {name}: owner_agent and eval_suite are required")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise RegistryError(f"target {name}: enabled must be a bool")
    return SkillTarget(name=name, skill_path=skill, memory_path=memory, owner_agent=owner,
                       eval_suite=suite, tasks_file=tasks, enabled=enabled)


def validate_targets(targets: Sequence[SkillTarget]) -> list[SkillTarget]:
    out = list(targets)
    if not out:
        raise RegistryError("no enabled skill targets")
    for attr in ("name", "skill_path"):
        values = [getattr(t, attr) for t in out]
        if len(set(values)) != len(values):
            raise RegistryError(f"duplicate target {attr}")
    mems = [t.memory_path for t in out if t.memory_path]
    if len(set(mems)) != len(mems) or set(mems) & {t.skill_path for t in out}:
        raise RegistryError("memory paths must be unique and distinct from skill paths")
    return out


def load_targets(path: Path | None = None, *, include_disabled: bool = False) -> list[SkillTarget]:
    p = Path(path) if path is not None else DEFAULT_REGISTRY
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"cannot read skills registry {p}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("format") != REGISTRY_FORMAT:
        raise RegistryError(f"{p}: expected format {REGISTRY_FORMAT!r}")
    raw = doc.get("targets")
    if not isinstance(raw, list):
        raise RegistryError(f"{p}: 'targets' must be a list")
    targets = [parse_target(t) for t in raw]
    validate_targets(targets)
    return targets if include_disabled else validate_targets([t for t in targets if t.enabled])
