"""ACS shared primitives: canonical serialization and action identity (spec §8, §13.1),
fail-closed errors with reserved reasons (§16) and resource limits (§15)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

JsonValue = Any


class AcsError(Exception):
    """A fail-closed ACS error; ``reason`` is always a reserved ``runtime_error:``/``host_error:`` id."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Limits:
    """Finite evaluation and loading limits (spec §15); breaches fail closed."""

    max_snapshot_bytes: int = 1 << 20
    max_policy_input_depth: int = 64
    max_annotators_per_point: int = 32
    max_annotator_output_bytes: int = 64 << 10
    max_policy_output_bytes: int = 64 << 10
    max_extends_depth: int = 16
    max_merged_manifest_bytes: int = 1 << 20


DEFAULT_LIMITS = Limits()


def canonical_json(value: JsonValue) -> str:
    """Members sorted by name at every level, array order kept, scalars unchanged (spec §8).

    Raises ``ValueError``/``TypeError`` for non-JSON values (NaN, sets, objects).
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: JsonValue) -> bytes:
    return canonical_json(value).encode("utf-8")


def action_identity(policy_input: JsonValue) -> str:
    """``sha256:<hex>`` of the canonical policy input (spec §13.1)."""
    return "sha256:" + hashlib.sha256(canonical_bytes(policy_input)).hexdigest()


def json_depth(value: JsonValue) -> int:
    """Nesting depth of containers (a scalar is 0)."""
    depth, level = 0, [value]
    while level := [v for v in level if isinstance(v, dict | list)]:
        depth += 1
        level = [c for v in level for c in (v.values() if isinstance(v, dict) else v)]
    return depth


@cache
def reserved_reasons() -> frozenset[str]:
    """The closed set of reserved reasons from the vendored ``reserved-reasons.json``."""
    text = (
        resources.files(__package__)
        .joinpath("schema/reserved-reasons.json")
        .read_text("utf-8")
    )
    return frozenset(r["reason"] for r in json.loads(text)["reasons"])
