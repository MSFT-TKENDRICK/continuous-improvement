"""Lesson registry ``lessons/registry.yaml`` (N4).

Format::

    schema_version: 1
    lessons:
      - {lesson_id, cluster_id, rule_ids: [...], prose_anchors: ["path#heading"], status}

Reads also accept a bare list or a ``{lesson_id: entry}`` mapping. ``conflicts`` lets campaign refuse
concurrent adoption of arms touching the same lesson unless their interaction was evaluated.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from ci_lab.rulespec import LessonEntry

REGISTRY_FILE = "registry.yaml"
SCHEMA_VERSION = 1


class Registry:
    def __init__(self, entries: Iterable[LessonEntry] = (), path: Path | None = None) -> None:
        self.path = path
        self._by_id: dict[str, LessonEntry] = {}
        for e in entries:
            self._by_id[e.lesson_id] = e

    # ------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path) -> Registry:
        if path.is_dir():
            path = path / REGISTRY_FILE
        if not path.exists():
            return cls(path=path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(parse_entries(raw), path=path)

    def save(self, path: Path | None = None) -> Path:
        p = path or self.path
        if p is None:
            raise ValueError("no registry path")
        if p.is_dir():
            p = p / REGISTRY_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema_version": SCHEMA_VERSION,
                "lessons": [self._by_id[k].model_dump(mode="json") for k in sorted(self._by_id)]}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
        os.replace(tmp, p)
        self.path = p
        return p

    # ------------------------------------------------------------ access
    @property
    def entries(self) -> list[LessonEntry]:
        return [self._by_id[k] for k in sorted(self._by_id)]

    def get(self, lesson_id: str) -> LessonEntry | None:
        return self._by_id.get(lesson_id)

    def upsert(self, entry: LessonEntry) -> None:
        self._by_id[entry.lesson_id] = entry

    def by_cluster(self, cluster_id: str) -> list[LessonEntry]:
        return [e for e in self.entries if e.cluster_id == cluster_id]

    def prose_fix_count(self, cluster_id: str) -> int:
        """How many times this fingerprint (cluster id) was 'fixed' by prose only (no rule ids)."""
        return sum(1 for e in self.by_cluster(cluster_id) if e.prose_anchors and not e.rule_ids)

    def touched_lessons(self, rule_ids: Iterable[str] = (), prose_anchors: Iterable[str] = ()) -> set[str]:
        """Lesson ids whose rules or prose anchors an arm touches (anchors match on path#heading or path)."""
        rids = set(rule_ids)
        anchors = set(prose_anchors)
        paths = {a.split("#", 1)[0] for a in anchors}
        out = set()
        for e in self.entries:
            if rids & set(e.rule_ids) or any(a in anchors or a.split("#", 1)[0] in paths for a in e.prose_anchors):
                out.add(e.lesson_id)
        return out

    def conflicts(self, lesson_ids_touched_by_arms: Mapping[str, Iterable[str]],
                  interaction_evaluated: Iterable[Iterable[str]] = ()) -> dict[str, list[str]]:
        """``{lesson_id: [arms]}`` for lessons touched by ≥2 arms (N4).

        ``interaction_evaluated`` lists arm groups whose joint effect was measured together; a conflict
        whose arms are all inside one such group is cleared.
        """
        return conflicts(lesson_ids_touched_by_arms, interaction_evaluated)


def conflicts(lesson_ids_touched_by_arms: Mapping[str, Iterable[str]],
              interaction_evaluated: Iterable[Iterable[str]] = ()) -> dict[str, list[str]]:
    by_lesson: dict[str, set[str]] = {}
    for arm, lessons in lesson_ids_touched_by_arms.items():
        for lid in lessons:
            by_lesson.setdefault(str(lid), set()).add(str(arm))
    groups = [set(g) for g in interaction_evaluated]
    out = {}
    for lid in sorted(by_lesson):
        arms = by_lesson[lid]
        if len(arms) >= 2 and not any(arms <= g for g in groups):
            out[lid] = sorted(arms)
    return out


def parse_entries(raw: Any) -> list[LessonEntry]:
    if raw is None:
        return []
    if isinstance(raw, Mapping) and "lessons" in raw:
        raw = raw["lessons"]
    if isinstance(raw, Mapping):
        items = []
        for k, v in raw.items():
            if k == "schema_version":
                continue
            d = dict(v or {})
            d.setdefault("lesson_id", str(k))
            items.append(d)
        raw = items
    if not isinstance(raw, list):
        raise TypeError("registry must be {lessons: [...]}, a list, or {lesson_id: entry}")
    return [LessonEntry.model_validate(x) for x in raw]
