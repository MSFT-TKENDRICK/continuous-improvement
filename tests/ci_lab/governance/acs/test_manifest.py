"""ACS manifest validation, path language and local ``extends`` resolution (spec §2-§4, §9, §10, §12)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ci_lab.governance.acs import AcsError, Limits, load_manifest, parse_path, resolve

BASE = """\
agent_control_specification_version: 0.4.0-alpha.1
policies:
  p: {type: rego, query: data.acs.verdict}
intervention_points:
  pre_tool_call:
    tool_name_from: $snap.tool_call.name
    policy: {id: p}
    policy_target: $.tool_call.args
    policy_target_kind: tool_args
    annotations:
      beta: {from: $target.q}
      alpha: {from: $pi.tool}
annotators:
  alpha: {type: classifier}
  beta: {type: llm}
tools:
  search: {type: Tool}
"""


def _edit(path: str, value: object) -> dict:
    data = yaml.safe_load(BASE)
    *parents, leaf = path.split("/")
    node = data
    for key in parents:
        node = node[key]
    if value is None:
        del node[leaf]
    else:
        node[leaf] = value
    return data


def test_valid_manifest_compiles_points() -> None:
    manifest = load_manifest(BASE)
    point = manifest.points["pre_tool_call"]
    assert point.policy_target.root == "snap" and point.policy_target.segments == (
        "tool_call",
        "args",
    )
    assert point.policy_target_kind == "tool_args"
    assert [a.name for a in point.annotations] == ["alpha", "beta"]
    assert point.policy_id == "p" and manifest.tools == {"search": {"type": "Tool"}}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("intervention_points", {}),
        ("policies", None),
        ("surprise", 1),
        (
            "intervention_points/on_message",
            {"policy": {"id": "p"}, "policy_target": "$snap.x"},
        ),
        ("intervention_points/pre_tool_call/extra", 1),
        ("intervention_points/pre_tool_call/policy_target", None),
        ("intervention_points/pre_tool_call/policy", {"id": "nope"}),
        ("intervention_points/pre_tool_call/policy_target", "$target.x"),
        ("intervention_points/pre_tool_call/policy_target", "snap.x"),
        ("intervention_points/pre_tool_call/policy_target", "$snap.a[-1]"),
        (
            "intervention_points/input",
            {"policy": {"id": "p"}, "policy_target": "$snap", "tool_name_from": "$.n"},
        ),
        (
            "intervention_points/pre_tool_call/annotations",
            {"gamma": {"from": "$target"}},
        ),
        (
            "intervention_points/pre_tool_call/annotations",
            {"alpha": {"from": "$pi.annotations.beta"}},
        ),
        (
            "intervention_points/pre_tool_call/annotations",
            {"alpha": {"from": "$target", "system_prompt_file": "x"}},
        ),
        ("policies/p", {"type": "rego"}),
        ("policies/p", {"type": "test", "bundle_url": "https://x"}),
        ("policies/p", {"type": "cedar", "policy_set": "a", "policy_path": "b"}),
        ("policies/p", {"type": "custom"}),
        ("annotators/alpha", {"type": "classifier", "system_prompt_url": "https://x"}),
        ("extends", ["./base.yaml"]),
        ("tools/search", "Tool"),
    ],
)
def test_invalid_manifests_fail_closed(path: str, value: object) -> None:
    with pytest.raises(AcsError) as info:
        load_manifest(_edit(path, value))
    assert info.value.reason == "runtime_error:manifest_invalid"


def test_tool_entry_members_are_unconstrained() -> None:
    labels = {"sink": "retrieval"}
    manifest = load_manifest(_edit("tools/search/security_labels", labels))
    assert manifest.tools["search"]["security_labels"] == labels


def test_unparseable_and_non_object_text() -> None:
    for text in ("a: [", "- 1", "x: 2024-01-01"):
        with pytest.raises(AcsError, match="manifest_invalid"):
            load_manifest(text)


def test_paths_parse_and_resolve() -> None:
    path = parse_path('$snap.a[1]["x.y"]')
    assert (path.root, path.segments) == ("snap", ("a", 1, "x.y"))
    assert parse_path("$").segments == () and parse_path("$.a").root == "snap"
    assert resolve({"a": [0, {"x.y": 5}]}, path.segments) == 5
    for bad in ("$snapx", "$a", "$.a..b", "x", "$snap[x]"):
        with pytest.raises(ValueError):
            parse_path(bad)
    for value, reason in (
        ({"a": []}, "path_missing"),
        ({"a": {}}, "path_type_mismatch"),
        ({}, "path_missing"),
    ):
        with pytest.raises(AcsError, match=reason):
            resolve(value, path.segments)


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


VERSION = "agent_control_specification_version: 0.4.0-alpha.1\n"
PARENT = VERSION + "policies:\n  p: {type: test}\n"
POINT = "intervention_points:\n  input:\n    policy: {id: p}\n    policy_target: $snap.input\n"


def test_file_extends_merges_parents_first(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "lib/base.yaml",
        PARENT
        + "metadata: {a: {x: 1}}\n"
        + POINT
        + "    annotations: {c: {from: $target}}\n",
    )
    child = _write(
        tmp_path,
        "agent.yaml",
        VERSION
        + "extends: [lib/base.yaml]\nmetadata: {a: {y: 2}}\n"
        + POINT
        + "    annotations: {d: {from: $target}}\nannotators: {c: {type: classifier}, d: {type: llm}}\n",
    )
    manifest = load_manifest(child)
    assert manifest.data["metadata"] == {"a": {"x": 1, "y": 2}}
    assert [a.name for a in manifest.points["input"].annotations] == ["c", "d"]
    assert "extends" not in manifest.data


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        (
            {"a.yaml": "extends: [b.yaml]\n", "b.yaml": "extends: [a.yaml]\n"},
            "resolution_cycle",
        ),
        ({"a.yaml": "extends: [../outside.yaml]\n"}, "resolution_path_traversal"),
        (
            {
                "a.yaml": "extends: [b.yaml]\n",
                "b.yaml": "policies: {p: {type: custom, adapter: x}}\n",
            },
            "resolution_merge_conflict",
        ),
        (
            {
                "a.yaml": "extends: [b.yaml]\n",
                "b.yaml": "agent_control_specification_version: '9'\n",
            },
            "resolution_merge_conflict",
        ),
        ({"a.yaml": "extends: ['https://example.com/m.yaml']\n"}, "manifest_invalid"),
        ({"a.yaml": "extends: [missing.yaml]\n"}, "manifest_invalid"),
    ],
)
def test_file_extends_failures(
    tmp_path: Path, files: dict[str, str], reason: str
) -> None:
    root = tmp_path / "root"
    _write(tmp_path, "outside.yaml", PARENT)
    for name, text in files.items():
        _write(root, name, (PARENT + POINT if name == "a.yaml" else "") + text)
    with pytest.raises(AcsError) as info:
        load_manifest(root / "a.yaml")
    assert info.value.reason == f"runtime_error:{reason}"


def test_extends_depth_limit(tmp_path: Path) -> None:

    for i in range(3):
        _write(tmp_path, f"m{i}.yaml", VERSION + f"extends: [m{i + 1}.yaml]\n")
    _write(tmp_path, "m3.yaml", PARENT + POINT)
    assert load_manifest(tmp_path / "m0.yaml").points["input"].policy_id == "p"
    with pytest.raises(AcsError, match="resource_limit_exceeded"):
        load_manifest(tmp_path / "m0.yaml", limits=Limits(max_extends_depth=2))
