"""Validated bus identifiers and topic -> relative path mapping (bus contract v2 §1).

Every helper raises :class:`IdError` (a ``ValueError``) on malformed input and returns the
validated string typed as the matching ``NewType``.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import NewType

from ci_lab.tools.paths import PathRejected, normalize_rel

__all__ = [
    "RUN_TOPIC", "AttemptId", "IdError", "ProposalId", "RubricVersion", "RunId", "TaskId", "Topic",
    "attempt_id", "parse_attempt", "parse_proposal", "parse_rubric_version", "proposal_id", "rubric_version",
    "run_id", "run_topic", "task_id", "task_topic", "topic_relpath", "topic_run", "topic_task",
    "validate_name", "validate_topic",
]

RunId = NewType("RunId", str)
TaskId = NewType("TaskId", str)
AttemptId = NewType("AttemptId", str)
ProposalId = NewType("ProposalId", str)
RubricVersion = NewType("RubricVersion", str)
Topic = NewType("Topic", str)

RUN_TOPIC = "_run"

_RUN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_TASK = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ROLE = re.compile(r"[a-z]{1,32}")
_RUBRIC = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_NUM = re.compile(r"0|[1-9][0-9]{0,8}")
_TOPIC = re.compile(r"[A-Za-z0-9._/@:-]{1,200}")


class IdError(ValueError):
    """A bus identifier or topic failed validation."""


def _match(rx: re.Pattern[str], value: object, what: str) -> str:
    if not isinstance(value, str) or not rx.fullmatch(value):
        raise IdError(f"bad {what}: {value!r}")
    return value


def run_id(value: str) -> RunId:
    return RunId(_match(_RUN, value, "run id"))


def task_id(value: str) -> TaskId:
    return TaskId(_match(_TASK, value, "task id"))


def validate_name(value: str) -> str:
    """Author / voter name usable inside a proposal id."""
    return _match(_NAME, value, "name")


def _num(text: str, what: str, *, minimum: int) -> int:
    n = int(_match(_NUM, text, what))
    if n < minimum:
        raise IdError(f"{what} must be >= {minimum}: {text!r}")
    return n


def attempt_id(task: str, n: int) -> AttemptId:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise IdError(f"attempt number must be an int >= 1: {n!r}")
    return AttemptId(f"{task_id(task)}@{n}")


def parse_attempt(value: str) -> tuple[TaskId, int]:
    task, sep, n = value.partition("@") if isinstance(value, str) else ("", "", "")
    if not sep:
        raise IdError(f"bad attempt id: {value!r}")
    return task_id(task), _num(n, "attempt number", minimum=1)


def proposal_id(attempt: str, role: str, name: str) -> ProposalId:
    parse_attempt(attempt)
    return ProposalId(f"{attempt}/{_match(_ROLE, role, 'role')}:{validate_name(name)}")


def parse_proposal(value: str) -> tuple[AttemptId, str, str]:
    """``task@n/role:name`` -> ``(attempt, role, name)``."""
    attempt, sep, rest = value.partition("/") if isinstance(value, str) else ("", "", "")
    role, sep2, name = rest.partition(":")
    if not (sep and sep2):
        raise IdError(f"bad proposal id: {value!r}")
    parse_attempt(attempt)
    return AttemptId(attempt), _match(_ROLE, role, "role"), validate_name(name)


def rubric_version(rubric_id: str, version: int) -> RubricVersion:
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise IdError(f"rubric version must be an int >= 0: {version!r}")
    return RubricVersion(f"{_match(_RUBRIC, rubric_id, 'rubric id')}@v{version}")


def parse_rubric_version(value: str) -> tuple[str, int]:
    rid, sep, v = value.partition("@v") if isinstance(value, str) else ("", "", "")
    if not sep:
        raise IdError(f"bad rubric version: {value!r}")
    return _match(_RUBRIC, rid, "rubric id"), _num(v, "rubric version", minimum=0)


def validate_topic(value: str) -> Topic:
    """Charset ``[A-Za-z0-9._/@:-]{1,200}``; no ``..``, no leading ``/``, no ``\\``."""
    _match(_TOPIC, value, "topic")
    if ".." in value or value.startswith("/") or "\\" in value:
        raise IdError(f"bad topic: {value!r}")
    return Topic(value)


def run_topic(run: str) -> Topic:
    return Topic(f"{run_id(run)}/{RUN_TOPIC}")


def task_topic(run: str, task: str) -> Topic:
    return Topic(f"{run_id(run)}/{task_id(task)}")


def topic_run(topic: str) -> str:
    """First segment of a topic (the run / experiment id)."""
    return validate_topic(topic).split("/", 1)[0]


def topic_task(topic: str) -> str | None:
    """Last segment of a task topic, ``None`` for the run-level ``<run>/_run`` topic."""
    last = validate_topic(topic).rsplit("/", 1)[-1]
    return None if last == RUN_TOPIC or "/" not in topic else last


def topic_relpath(topic: str) -> PurePosixPath:
    """Relative storage path for a topic, validated with ``tools.paths.normalize_rel`` rules
    (no ``:``/ADS, reserved Windows device names, trailing dot/space, ``.git``, short names)."""
    try:
        return PurePosixPath(normalize_rel(validate_topic(topic)))
    except PathRejected as exc:
        raise IdError(f"topic not mappable to a path: {topic!r}: {exc}") from exc
