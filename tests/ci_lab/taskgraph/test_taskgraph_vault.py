from __future__ import annotations

import hashlib
import json
import random

import pytest

from ci_lab.taskgraph.model import canonical_json
from ci_lab.taskgraph.vault import (
    SEALED_DIRNAME,
    RubricVault,
    VaultDenied,
    VaultMissing,
    VaultTampered,
    new_canary,
)
from ci_lab.tools.paths import PathRejected, normalize_rel


def test_seal_is_content_addressed_and_idempotent(tmp_path, rubric_factory) -> None:
    vault = RubricVault.for_run(tmp_path)
    assert vault.root == tmp_path / SEALED_DIRNAME == tmp_path / "sealed"
    r = rubric_factory()
    c = vault.seal(r)
    assert c == r.commitment()
    data = (vault.root / f"{c}.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == c and data.decode() == canonical_json(r.to_json())
    assert vault.seal(r) == c
    assert sorted(p.name for p in vault.root.iterdir()) == [f"{c}.json"]
    assert vault.open(c, role="judge") == r


@pytest.mark.parametrize("role", ["student", "intern", ""])
def test_open_denies_students_and_unknown_roles(tmp_path, rubric_factory, role: str) -> None:
    vault = RubricVault(tmp_path)
    c = vault.seal(rubric_factory())
    with pytest.raises(VaultDenied):
        vault.open(c, role=role)


def test_open_missing_and_bad_commitments(tmp_path) -> None:
    vault = RubricVault(tmp_path)
    for bad in ("0" * 64, "../" + "a" * 61, "A" * 64, "abc"):
        with pytest.raises(VaultMissing):
            vault.open(bad, role="orchestrator")


def test_tampering_is_detected(tmp_path, rubric_factory) -> None:
    vault = RubricVault(tmp_path)
    r = rubric_factory()
    c = vault.seal(r)
    path = tmp_path / f"{c}.json"
    path.write_text(path.read_text().replace('"pass_score":0.7', '"pass_score":0.1'), encoding="utf-8")
    with pytest.raises(VaultTampered):
        vault.open(c, role="judge")
    pretty = json.dumps(r.to_json(), indent=1).encode()
    forged = hashlib.sha256(pretty).hexdigest()
    (tmp_path / f"{forged}.json").write_bytes(pretty)
    with pytest.raises(VaultTampered, match="canonical"):
        vault.open(forged, role="judge")
    junk = b"not json"
    (tmp_path / f"{hashlib.sha256(junk).hexdigest()}.json").write_bytes(junk)
    with pytest.raises(VaultTampered, match="malformed"):
        vault.open(hashlib.sha256(junk).hexdigest(), role="judge")


def test_versions_sorted_and_fail_closed(tmp_path, rubric_factory) -> None:
    vault = RubricVault(tmp_path)
    assert vault.versions("summary-rubric") == []
    v3, v1, other = rubric_factory(version=3), rubric_factory(version=1), rubric_factory("reply")
    for r in (v3, other, v1):
        vault.seal(r)
    assert vault.versions("summary-rubric") == [v1, v3]
    assert vault.versions("reply-rubric") == [other]
    (tmp_path / f"{v3.commitment()}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VaultTampered):
        vault.versions("summary-rubric")


def test_new_canary() -> None:
    canaries = {new_canary() for _ in range(32)}
    assert len(canaries) == 32 and all(len(c) == 16 and int(c, 16) >= 0 and c == c.lower() for c in canaries)
    assert new_canary(random.Random(7)) == new_canary(random.Random(7))
    assert len(new_canary(random.Random(0))) == 16


def test_vault_files_are_not_reachable_by_relative_paths(tmp_path, rubric_factory) -> None:
    c = RubricVault.for_run(tmp_path / "run").seal(rubric_factory())
    with pytest.raises(PathRejected):
        normalize_rel(f"run/{SEALED_DIRNAME}/{c}.json")
