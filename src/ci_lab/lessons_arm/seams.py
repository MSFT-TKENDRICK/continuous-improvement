"""Seams to M16 ``ci_lab.lessons``: lesson registry (N4) and the replay rejection filter (B4).

Both import ``ci_lab.lessons`` lazily and degrade gracefully while it is being built in
parallel: the registry falls back to parsing ``{schema_version: 1, lessons: [LessonEntry]}``;
replay returns ``None`` (caller records ``replay: unavailable`` — HOOK(M16)).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ci_lab.rulespec import GUARDS_DIR, LessonCluster, LessonEntry

ReplayFn = Callable[[Any, LessonCluster, Path], tuple[bool, list[str]]]
"""``(rules_or_bundle, cluster, work_dir) -> (ok, reasons)``; ``ok`` means *pass to closed loop*, never
accept. The first argument is a ``ci_lab.rules.Bundle`` (candidate rule + the frozen extractors) when
the engine is installed, else a ``list[RuleSpec]``."""


def load_registry(path: Path | None) -> list[LessonEntry]:
    """Registry entries (``lessons/registry.yaml``); missing file ⇒ ``[]``."""
    if path is None or not Path(path).is_file():
        return []
    try:
        from ci_lab.lessons.registry import Registry  # type: ignore[import-not-found]
    except ImportError:
        Registry = None
    if Registry is not None:
        reg = Registry.load(Path(path))
        return [LessonEntry.model_validate(e) if not isinstance(e, LessonEntry) else e
                for e in getattr(reg, "entries", None) or getattr(reg, "lessons", [])]
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [LessonEntry.model_validate(e) for e in doc.get("lessons") or []]


def _norm(path: str) -> str:
    return str(PurePosixPath(path.replace("\\", "/"))).lstrip("./")


def lessons_touching(files: Iterable[str], registry: Sequence[LessonEntry]) -> set[str]:
    """Lesson ids whose guard file or prose anchors (``path#heading``) are in ``files`` (N4)."""
    out: set[str] = set()
    norm = {_norm(f) for f in files}
    for f in norm:
        p = PurePosixPath(f)
        if str(p.parent) == GUARDS_DIR and p.suffix == ".yaml":
            out.add(p.stem)
    for e in registry:
        if any(_norm(a.split("#", 1)[0]) in norm for a in e.prose_anchors):
            out.add(e.lesson_id)
    return out


def default_replay(trajectories: Any = None, *, dataset_texts: Sequence[str] = ()) -> ReplayFn | None:
    """M16 ``ci_lab.lessons.replay.validate_ok`` bound to an evolve trajectory corpus, or ``None``
    when the module / corpus is unavailable. ``trajectories``: a sequence of ``Trajectory`` or a
    path accepted by ``ci_lab.lessons.common.load_trajectories``."""
    try:
        from ci_lab.lessons.replay import validate_ok  # type: ignore[import-not-found]
    except ImportError:
        return None
    if trajectories is None:
        return None
    corpus: list[Any] | None = None

    def replay(rules: Any, cluster: LessonCluster, work_dir: Path) -> tuple[bool, list[str]]:
        nonlocal corpus
        if corpus is None:
            if isinstance(trajectories, (str, Path)):
                from ci_lab.lessons.common import load_trajectories  # type: ignore[import-not-found]

                corpus = list(load_trajectories(Path(trajectories)))
            else:
                corpus = list(trajectories)
        subject = rules if hasattr(rules, "rules") and not isinstance(rules, (list, tuple)) else list(rules)
        ok, reasons = validate_ok(subject, corpus, cluster=cluster, dataset_texts=list(dataset_texts),
                                  work_dir=work_dir)
        return bool(ok), [str(r) for r in reasons]

    return replay
