from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from ci_lab.bus import ids
from ci_lab.bus.ids import IdError


def test_run_and_task_ids() -> None:
    assert ids.run_id("r1.x_y-z") == "r1.x_y-z"
    assert ids.task_id("t_1-a") == "t_1-a"
    for bad in ("", "R1", "-x", "a" * 65, "a/b", None):
        with pytest.raises(IdError):
            ids.run_id(bad)  # type: ignore[arg-type]
    with pytest.raises(IdError):
        ids.task_id("a.b")


def test_attempt_round_trip() -> None:
    a = ids.attempt_id("t1", 3)
    assert a == "t1@3"
    assert ids.parse_attempt(a) == ("t1", 3)
    for bad in (0, -1, True, "1"):
        with pytest.raises(IdError):
            ids.attempt_id("t1", bad)  # type: ignore[arg-type]
    for bad in ("t1", "t1@0", "t1@01", "t1@x", "T@1", "t1@1@2"):
        with pytest.raises(IdError):
            ids.parse_attempt(bad)


def test_proposal_round_trip() -> None:
    p = ids.proposal_id("t1@2", "student", "Stu.1")
    assert p == "t1@2/student:Stu.1"
    assert ids.parse_proposal(p) == ("t1@2", "student", "Stu.1")
    for bad in ("t1@2/student", "t1@2:student:x", "t1@0/student:x", "t1@2/Student:x", "t1@2/student:"):
        with pytest.raises(IdError):
            ids.parse_proposal(bad)
    with pytest.raises(IdError):
        ids.proposal_id("t1@2", "student", "a/b")


def test_rubric_version_round_trip() -> None:
    v = ids.rubric_version("rub-1", 4)
    assert v == "rub-1@v4"
    assert ids.parse_rubric_version(v) == ("rub-1", 4)
    for bad in ("rub", "rub@4", "rub@v-1", "rub@v01", "@v1"):
        with pytest.raises(IdError):
            ids.parse_rubric_version(bad)
    with pytest.raises(IdError):
        ids.rubric_version("rub", -1)


def test_topics() -> None:
    assert ids.run_topic("r1") == "r1/_run"
    assert ids.task_topic("r1", "t1") == "r1/t1"
    assert ids.topic_run("r1/t1") == "r1"
    assert ids.topic_task("r1/t1") == "t1"
    assert ids.topic_task("r1/_run") is None
    assert ids.validate_topic("a@b:c/d.e-f_g") == "a@b:c/d.e-f_g"
    for bad in ("", "/r1/t1", "r1/../t", "r1\\t1", "r1 t1", "x" * 201, "r1/t*"):
        with pytest.raises(IdError):
            ids.validate_topic(bad)


def test_topic_relpath_uses_path_rules() -> None:
    assert ids.topic_relpath("r1/t1") == PurePosixPath("r1/t1")
    for bad in ("r1/con", "r1/NUL.txt", "r1/aux/x", "r1/t:1", "r1//t", "r1/t.", "r1/.git", "r1/./t"):
        with pytest.raises(IdError):
            ids.topic_relpath(bad)
