"""ACS canonical serialization and action identity (spec §8, §13.1)."""

from __future__ import annotations

import hashlib
import math

import pytest

from ci_lab.governance.acs import action_identity, canonical_json, reserved_reasons
from ci_lab.governance.acs.canonical import json_depth


def test_canonical_sorts_members_at_every_level_and_keeps_array_order() -> None:
    value = {"b": [3, {"z": 1, "a": 2}], "a": "é", "c": None}
    assert canonical_json(value) == '{"a":"é","b":[3,{"a":2,"z":1}],"c":null}'


def test_action_identity_is_sha256_of_utf8_canonical_json() -> None:
    value = {"x": "🔥", "a": [True, 1.5]}
    digest = hashlib.sha256('{"a":[true,1.5],"x":"🔥"}'.encode()).hexdigest()
    assert action_identity(value) == f"sha256:{digest}"
    assert action_identity({"a": [True, 1.5], "x": "🔥"}) == action_identity(value)


@pytest.mark.parametrize("bad", [math.nan, {"a": math.inf}, {1, 2}])
def test_non_json_values_are_rejected(bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        canonical_json(bad)


def test_json_depth() -> None:
    assert json_depth("x") == 0
    assert json_depth({}) == 1
    assert json_depth({"a": [1, {"b": []}]}) == 4


def test_reserved_reasons_cover_both_namespaces() -> None:
    reasons = reserved_reasons()
    assert "runtime_error:manifest_invalid" in reasons
    assert "host_error:approval_identity_mismatch" in reasons
    assert len([r for r in reasons if r.startswith("runtime_error:")]) == 16
    assert len([r for r in reasons if r.startswith("host_error:")]) == 8
