from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ci_lab.lessons.registry import Registry, conflicts, parse_entries
from ci_lab.rulespec import LessonEntry


def _reg() -> Registry:
    return Registry([
        LessonEntry(lesson_id="L-refund", cluster_id="lc-a", rule_ids=["refund.needs_status"]),
        LessonEntry(lesson_id="L-tone", cluster_id="lc-b", prose_anchors=["AGENTS.md#tone"]),
        LessonEntry(lesson_id="L-tone2", cluster_id="lc-b", prose_anchors=["AGENTS.md#tone-again"]),
    ])


def test_roundtrip(tmp_path: Path) -> None:
    reg = _reg()
    p = reg.save(tmp_path / "lessons" / "registry.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and [e["lesson_id"] for e in raw["lessons"]] == ["L-refund", "L-tone",
                                                                                       "L-tone2"]
    back = Registry.load(tmp_path / "lessons")
    assert back.entries == reg.entries
    assert Registry.load(tmp_path / "missing.yaml").entries == []
    back.upsert(LessonEntry(lesson_id="L-new", cluster_id="lc-c"))
    back.save()
    assert Registry.load(p).get("L-new") is not None


def test_parse_shapes() -> None:
    e = {"lesson_id": "L1", "cluster_id": "lc-1"}
    assert parse_entries([e])[0].lesson_id == "L1"
    assert parse_entries({"lessons": [e]})[0].cluster_id == "lc-1"
    assert parse_entries({"schema_version": 1, "L2": {"cluster_id": "lc-2"}})[0].lesson_id == "L2"
    assert parse_entries(None) == []
    with pytest.raises(TypeError):
        parse_entries("nope")


def test_prose_fix_count_and_touch() -> None:
    reg = _reg()
    assert reg.prose_fix_count("lc-b") == 2 and reg.prose_fix_count("lc-a") == 0
    assert reg.touched_lessons(rule_ids=["refund.needs_status"]) == {"L-refund"}
    assert reg.touched_lessons(prose_anchors=["AGENTS.md#other"]) == {"L-tone", "L-tone2"}
    assert reg.touched_lessons(prose_anchors=["README.md"]) == set()


def test_conflicts() -> None:
    touched = {"arm-a": ["L-refund", "L-tone"], "arm-b": ["L-refund"], "arm-c": ["L-tone"], "arm-d": []}
    assert conflicts(touched) == {"L-refund": ["arm-a", "arm-b"], "L-tone": ["arm-a", "arm-c"]}
    assert conflicts(touched, interaction_evaluated=[["arm-a", "arm-b"]]) == {"L-tone": ["arm-a", "arm-c"]}
    assert _reg().conflicts({"x": ["L1"], "y": ["L2"]}) == {}
