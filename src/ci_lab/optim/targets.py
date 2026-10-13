"""Text components an optimizer may rewrite (design §3, §11.2).

A *target id* is a worktree-relative POSIX path (``prompts/system.md``) optionally
followed by ``#dotted.key`` addressing a string inside a YAML mapping
(``agent.yaml#instructions``). ``resolve_targets`` maps ``ArmDirective.component_focus``
entries — component names (``prompt``/``skill``/``memory``) or explicit target ids —
to :class:`TextTarget` s, refusing anything outside the evolvable surface.
"""
from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

HARNESS_ROOT = "src/order_support/harness"
# Text components only: client_tool/config/context_mgmt are structured and belong to
# the agent strategy (allowlisted keys, critic-checked bindings).
DEFAULT_COMPONENT_GLOBS: Mapping[str, Sequence[str]] = {
    "prompt": (f"{HARNESS_ROOT}/prompts/*.md",),
    "skill": (f"{HARNESS_ROOT}/skills/**/SKILL.md",),
    "memory": (f"{HARNESS_ROOT}/skills/**/memory.md",),
}
TEXT_COMPONENTS = frozenset(DEFAULT_COMPONENT_GLOBS)


class TargetError(ValueError):
    """Target id is invalid or outside the evolvable surface."""


def _match(path: str, pattern: str) -> bool:
    if fnmatch.fnmatchcase(path, pattern):
        return True
    # let "**/" also match zero directories
    return "**/" in pattern and fnmatch.fnmatchcase(path, pattern.replace("**/", ""))


@dataclass(frozen=True)
class TextTarget:
    path: str
    key: str | None = None

    @property
    def id(self) -> str:
        return self.path if self.key is None else f"{self.path}#{self.key}"

    @classmethod
    def parse(cls, target_id: str) -> TextTarget:
        path, _, key = target_id.partition("#")
        path = path.replace("\\", "/").strip()
        p = PurePosixPath(path)
        if not path or p.is_absolute() or ":" in p.parts[0] or ".." in p.parts \
                or ".git" in p.parts or path.startswith("/"):
            raise TargetError(f"invalid target path {target_id!r}")
        return cls(str(p), key.strip() or None)

    def file(self, root: Path) -> Path:
        root = Path(root).resolve()
        f = root.joinpath(*PurePosixPath(self.path).parts)
        for parent in [f, *f.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise TargetError(f"symlink in target path {self.path!r}")
        if not f.resolve().is_relative_to(root):
            raise TargetError(f"target escapes worktree: {self.path!r}")
        return f

    def read(self, root: Path) -> str:
        text = self.file(root).read_text(encoding="utf-8")
        if self.key is None:
            return text
        value = _get_key(_load_yaml(text), self.key)
        if not isinstance(value, str):
            raise TargetError(f"{self.id} is not a string")
        return value

    def write(self, root: Path, new_text: str) -> None:
        f = self.file(root)
        if self.key is None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(new_text, encoding="utf-8", newline="\n")
            return
        import yaml

        data = _load_yaml(f.read_text(encoding="utf-8"))
        _set_key(data, self.key, new_text)
        f.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=10**6),
                     encoding="utf-8", newline="\n")


def _load_yaml(text: str) -> Any:
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TargetError("keyed target requires a YAML mapping")
    return data


def _get_key(data: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            raise TargetError(f"missing key {key!r}")
        data = data[part]
    return data


def _set_key(data: dict, key: str, value: str) -> None:
    *head, last = key.split(".")
    for part in head:
        data = data[part]
    if not isinstance(data.get(last), str):
        raise TargetError(f"{key!r} must be an existing string")
    data[last] = value


def on_surface(path: str, surface_globs: Sequence[str] | None, frozen_globs: Sequence[str] = ()) -> bool:
    if any(_match(path, g) for g in frozen_globs):
        return False
    return surface_globs is None or any(_match(path, g) for g in surface_globs)


def resolve_targets(worktree: Path, focus: Iterable[str], *, default_focus: Sequence[str] = ("prompt",),
                    component_globs: Mapping[str, Sequence[str]] | None = None,
                    surface_globs: Sequence[str] | None = None, frozen_globs: Sequence[str] = (),
                    ) -> list[tuple[str, TextTarget]]:
    """Return ``[(component, target)]`` (deduplicated, sorted per entry) for the focus.

    Component names expand via ``component_globs`` (domain) falling back to
    :data:`DEFAULT_COMPONENT_GLOBS`; non-text components are skipped. Explicit target
    ids are attributed to the first component whose globs match (else ``"prompt"``).
    """
    worktree = Path(worktree)
    globs = {**DEFAULT_COMPONENT_GLOBS, **{k: v for k, v in (component_globs or {}).items()
                                           if k in TEXT_COMPONENTS}}
    out: dict[str, tuple[str, TextTarget]] = {}
    for entry in (tuple(focus) or tuple(default_focus)):
        if entry in globs:
            paths = sorted({p.relative_to(worktree).as_posix() for g in globs[entry]
                            for p in worktree.glob(g) if p.is_file()})
            found = [(entry, TextTarget.parse(p)) for p in paths]
        elif "/" in entry or "." in entry:
            t = TextTarget.parse(entry)
            comp = next((c for c, gs in globs.items() if any(_match(t.path, g) for g in gs)), "prompt")
            found = [(comp, t)]
        else:
            continue  # non-text component (client_tool/config/context_mgmt): not ours
        for comp, t in found:
            if not on_surface(t.path, surface_globs, frozen_globs):
                raise TargetError(f"{t.path} is outside the evolvable surface")
            if not t.file(worktree).is_file():
                raise TargetError(f"target not found: {t.path}")
            out.setdefault(t.id, (comp, t))
    return list(out.values())
