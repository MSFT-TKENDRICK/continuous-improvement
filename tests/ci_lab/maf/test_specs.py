from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from ci_lab.maf.specs import (
    SpecError,
    load_agent_spec,
    load_manifest,
    parse_agent_spec,
    resolve_contained,
    strip_doc_keys,
)

MODELS = ["gpt-5-mini"]


def _load(path: Path, **kw):
    kw.setdefault("allowed_models", MODELS)
    return load_agent_spec(path, **kw)


def _parse(data, base: Path, **kw):
    kw.setdefault("allowed_models", MODELS)
    return parse_agent_spec(data, base_dir=base, **kw)


def _data(agent_text: str) -> dict:
    return yaml.safe_load(agent_text)


# ---------------------------------------------------------------- valid specs

def test_valid_spec_loads(agent_yaml: Path) -> None:
    loaded = _load(agent_yaml, allowed_bindings={"read_file"})
    spec = loaded.spec
    assert spec.name == "CiStudent"
    assert spec.model.provider == "GitHubCopilot"
    assert spec.binding_names() == {"read_file"}
    assert loaded.path == agent_yaml.resolve()
    assert len(loaded.source_sha256) == 64
    assert spec.digest() == _load(agent_yaml).spec.digest()


def test_declarative_dict_drops_model_identity(agent_yaml: Path) -> None:
    d = _load(agent_yaml).spec.declarative_dict()
    assert d["model"] == {"options": {"maxOutputTokens": 256, "reasoningEffort": "low"}}
    full = _load(agent_yaml).spec.declarative_dict(include_model_identity=True)
    assert full["model"]["id"] == "gpt-5-mini" and full["model"]["provider"] == "GitHubCopilot"


def test_dict_form_bindings_normalised(tmp_path: Path, agent_text: str) -> None:
    data = _data(agent_text)
    data["tools"][0]["bindings"] = {"read_file": "x"}
    assert _parse(data, tmp_path).spec.tools[0].bindings[0].name == "read_file"


def test_explicitly_allowed_provider(tmp_path: Path, agent_text: str) -> None:
    data = _data(agent_text)
    data["model"]["provider"] = "OpenAI"
    assert _parse(data, tmp_path, allowed_providers={"OpenAI"}).spec.model.provider == "OpenAI"


# ---------------------------------------------------------------- rejections

@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["model"].__setitem__("id", "gpt-4o"), "allowlist"),
    (lambda d: d["model"].__setitem__("provider", "OpenAI"), "provider"),
    (lambda d: d["model"]["options"].__setitem__("temperature", 0.1), "options not allowed"),
    (lambda d: d.__setitem__("kind", "Workflow"), "invalid agent spec"),
    (lambda d: d.__setitem__("unexpected", 1), "invalid agent spec"),
    (lambda d: d.pop("description"), "invalid agent spec"),
    (lambda d: d.__setitem__("instructions", ""), "invalid agent spec"),
    (lambda d: d.__setitem__("name", "bad name!"), "invalid agent spec"),
    (lambda d: d["model"].pop("id"), "invalid agent spec"),
    (lambda d: d["tools"][0].__setitem__("kind", "mcp"), "invalid agent spec"),
    (lambda d: d["tools"][0].__setitem__("bindings", []), "invalid agent spec"),
    (lambda d: d["tools"].append(dict(d["tools"][0])), "duplicate tool"),
    (lambda d: d["tools"][0]["bindings"].append({"name": "delete_everything"}), "bindings not allowed"),
    (lambda d: d.__setitem__("x-ci", {"bogus": 1}), "x-ci"),
])
def test_rejections(tmp_path: Path, agent_text: str, mutate, match: str) -> None:
    data = _data(agent_text)
    mutate(data)
    with pytest.raises(SpecError, match=match):
        _parse(data, tmp_path, allowed_bindings={"read_file"})


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("instructions", "=Env.SECRET"),
    lambda d: d["model"].__setitem__("id", "=Env.MODEL"),
    lambda d: d["model"]["options"].__setitem__("reasoningEffort", "=Concat('a','b')"),
    lambda d: d["tools"][0].__setitem__("description", "=1+1"),
    lambda d: d.__setitem__("metadata", {"tags": ["ok", "=Env.X"]}),
    lambda d: d.__setitem__("metadata", {"=Env.KEY": "v"}),
    lambda d: d.__setitem__("x-ci", {"append_text": "=Env.TOKEN"}),
])
def test_rejects_powerfx_strings_anywhere(tmp_path: Path, agent_text: str, mutate) -> None:
    data = _data(agent_text)
    mutate(data)
    with pytest.raises(SpecError, match="'=' expressions"):
        _parse(data, tmp_path)


def test_rejects_expression_composed_from_instructions_file(tmp_path: Path, write, agent_text: str) -> None:
    write("prompts/evil.md", "=Env.GITHUB_TOKEN\n")
    data = _data(agent_text)
    del data["instructions"]
    data["x-ci"] = {"instructions_files": ["prompts/evil.md"]}
    with pytest.raises(SpecError, match="composed agent spec"):
        _parse(data, tmp_path)


def test_binding_allowlist_none_means_unchecked(tmp_path: Path, agent_text: str) -> None:
    data = _data(agent_text)
    data["tools"][0]["bindings"] = [{"name": "anything"}]
    assert _parse(data, tmp_path).spec.binding_names() == {"anything"}
    with pytest.raises(SpecError, match="bindings not allowed"):
        _parse(data, tmp_path, allowed_bindings=set())


def test_custom_option_allowlist(tmp_path: Path, agent_text: str) -> None:
    with pytest.raises(SpecError, match="options not allowed"):
        _parse(_data(agent_text), tmp_path, allowed_options={"maxOutputTokens"})


def test_spec_file_errors(write, tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="cannot read"):
        _load(tmp_path / "missing.yaml")
    with pytest.raises(SpecError, match="invalid YAML"):
        _load(write("bad.yaml", "kind: [unclosed\n"))
    with pytest.raises(SpecError, match="mapping"):
        _load(write("list.yaml", "- a\n- b\n"))


# ---------------------------------------------------------------- frozen tool schemas

def test_frozen_schema_allows_description_changes(tmp_path: Path, agent_text: str) -> None:
    frozen = _parse(_data(agent_text), tmp_path).spec.tool_schemas()
    assert frozen["read_file"]["properties"]["resource_id"]["type"] == "string"
    evolved = _data(agent_text)
    evolved["tools"][0]["description"] = "A much better description."
    evolved["tools"][0]["parameters"]["properties"]["resource_id"]["description"] = "Better param doc."
    _parse(evolved, tmp_path, frozen_tool_schemas=frozen)


@pytest.mark.parametrize("mutate, match", [
    (lambda p: p["properties"]["resource_id"].__setitem__("kind", "integer"), "differs"),
    (lambda p: p["properties"]["resource_id"].__setitem__("required", False), "differs"),
    (lambda p: p["properties"].__setitem__("extra", {"kind": "string"}), "differs"),
    (lambda p: p["properties"].__setitem__("description", {"kind": "string"}), "differs"),
])
def test_frozen_schema_rejects_parameter_changes(tmp_path: Path, agent_text: str, mutate, match: str) -> None:
    frozen = _parse(_data(agent_text), tmp_path).spec.tool_schemas()
    evolved = _data(agent_text)
    mutate(evolved["tools"][0]["parameters"])
    with pytest.raises(SpecError, match=match):
        _parse(evolved, tmp_path, frozen_tool_schemas=frozen)


def test_frozen_schema_rejects_tool_set_change(tmp_path: Path, agent_text: str) -> None:
    frozen = _parse(_data(agent_text), tmp_path).spec.tool_schemas()
    data = _data(agent_text)
    data["tools"][0]["name"] = "read_file_v2"
    with pytest.raises(SpecError, match="tool set changed"):
        _parse(data, tmp_path, frozen_tool_schemas=frozen)
    data = _data(agent_text)
    data["tools"] = []
    with pytest.raises(SpecError, match="tool set changed"):
        _parse(data, tmp_path, frozen_tool_schemas=frozen)


def test_strip_doc_keys_keeps_property_names() -> None:
    schema = {"type": "object", "description": "d", "properties": {"title": {"type": "string", "title": "T"}}}
    assert strip_doc_keys(schema) == {"type": "object", "properties": {"title": {"type": "string"}}}


# ---------------------------------------------------------------- x-ci composition

def test_xci_composes_instructions_in_order(write, agent_text: str) -> None:
    write("agents/prompts/system.md", "# System\nBe precise.\n")
    write("agents/skills/changes.md", "Changes need a receipt.\n")
    data = _data(agent_text)
    data["x-ci"] = {"instructions_files": ["prompts/system.md", "skills/changes.md"], "append_text": "Sign off."}
    path = write("agents/spec.yaml", yaml.safe_dump(data))
    loaded = _load(path, extra_instructions="Run 42.")
    assert loaded.spec.instructions == (
        "You are the harness student agent.\n\n# System\nBe precise.\n\nChanges need a receipt.\n\nSign off.\n\nRun 42."
    )
    assert [p.name for p in loaded.instruction_files] == ["system.md", "changes.md"]
    assert "x-ci" not in loaded.spec.declarative_dict()


def test_xci_files_only_instructions(write, agent_text: str) -> None:
    write("agents/prompts/system.md", "Only from file.")
    data = _data(agent_text)
    del data["instructions"]
    data["x-ci"] = {"instructions_files": ["prompts/system.md"]}
    assert _load(write("agents/spec.yaml", yaml.safe_dump(data))).spec.instructions == "Only from file."


@pytest.mark.parametrize("rel", [
    "../outside.md", "prompts/../../outside.md", "/etc/passwd", "C:/Windows/win.ini", "C:outside.md",
    "\\\\server\\share\\x.md", "prompts/system.md:ads", "missing.md", "prompts",
])
def test_xci_rejects_escaping_or_bad_paths(write, agent_text: str, rel: str) -> None:
    write("outside.md", "secret")
    write("agents/prompts/system.md", "ok")
    data = _data(agent_text)
    data["x-ci"] = {"instructions_files": [rel]}
    with pytest.raises(SpecError):
        _load(write("agents/spec.yaml", yaml.safe_dump(data)))


def test_xci_rejects_symlinks(tmp_path: Path, write, agent_text: str) -> None:
    target = write("outside.md", "secret")
    link = tmp_path / "agents" / "link.md"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    data = _data(agent_text)
    data["x-ci"] = {"instructions_files": ["link.md"]}
    with pytest.raises(SpecError, match="symlink"):
        _load(write("agents/spec.yaml", yaml.safe_dump(data)))


@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions")
def test_xci_rejects_junctions(tmp_path: Path, write, agent_text: str) -> None:
    import _winapi

    write("outside/secret.md", "secret")
    (tmp_path / "agents").mkdir(exist_ok=True)
    _winapi.CreateJunction(str(tmp_path / "outside"), str(tmp_path / "agents" / "jdir"))
    data = _data(agent_text)
    data["x-ci"] = {"instructions_files": ["jdir/secret.md"]}
    with pytest.raises(SpecError, match="symlink"):
        _load(write("agents/spec.yaml", yaml.safe_dump(data)))


def test_xci_requires_base_dir(agent_text: str) -> None:
    data = _data(agent_text)
    data["x-ci"] = {"instructions_files": ["a.md"]}
    with pytest.raises(SpecError, match="base directory"):
        parse_agent_spec(data, base_dir=None, allowed_models=MODELS)


def test_resolve_contained_ok(tmp_path: Path) -> None:
    (tmp_path / "a" / "b.md").parent.mkdir()
    (tmp_path / "a" / "b.md").write_text("x")
    assert resolve_contained(tmp_path, "a/b.md") == (tmp_path / "a" / "b.md").resolve()
    assert resolve_contained(tmp_path, "./a\\b.md") == (tmp_path / "a" / "b.md").resolve()
    for bad in ("", " a", ".", "a/.."):
        with pytest.raises(SpecError):
            resolve_contained(tmp_path, bad)


# ---------------------------------------------------------------- manifest

MANIFEST = """\
agents:
  CiStudent:
    spec: agents/harness_agent.yaml
    runtime: harness
    purpose: target
    bindings: [read_file]
    skills_paths: [skills]
"""


def test_manifest_loads(write, agent_yaml: Path, tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    m = load_manifest(write("manifest.yaml", MANIFEST))
    e = m.entries["CiStudent"]
    assert e.spec_path == agent_yaml.resolve()
    assert e.entry.runtime == "harness" and e.entry.purpose == "target"
    assert e.skills_paths == ((tmp_path / "skills").resolve(),)


def test_manifest_runtime_defaults_to_prompt(write, agent_yaml: Path) -> None:
    m = load_manifest(write("manifest.yaml", "agents:\n  CiStudent: {spec: agents/harness_agent.yaml, purpose: judge}\n"))
    assert m.entries["CiStudent"].entry.runtime == "prompt"


@pytest.mark.parametrize("text, match", [
    ("agents: {}\n", "invalid manifest"),
    ("agents:\n  A: {spec: agents/harness_agent.yaml, purpose: villain}\n", "invalid manifest"),
    ("agents:\n  A: {spec: agents/harness_agent.yaml, purpose: target, runtime: shell}\n", "invalid manifest"),
    ("agents:\n  A: {spec: agents/harness_agent.yaml, purpose: target, extra: 1}\n", "invalid manifest"),
    ("agents:\n  'bad name': {spec: agents/harness_agent.yaml, purpose: target}\n", "invalid manifest"),
    ("agents:\n  A: {spec: ../outside.yaml, purpose: target}\n", "escapes"),
    ("agents:\n  A: {spec: agents/missing.yaml, purpose: target}\n", "not found"),
    ("agents:\n  A: {spec: agents, purpose: target}\n", "not a file"),
    ("agents:\n  A: {spec: agents/harness_agent.yaml, purpose: target, skills_paths: [../x]}\n", "escapes"),
    ("agents:\n  A: {spec: agents/harness_agent.yaml, purpose: target, skills_paths: [manifest.yaml]}\n",
     "not a directory"),
    ("agents:\n  A: {spec: '=Env.SPEC', purpose: target}\n", "'=' expressions"),
])
def test_manifest_rejections(write, agent_yaml: Path, text: str, match: str) -> None:
    write("outside.yaml", "x: 1")
    with pytest.raises(SpecError, match=match):
        load_manifest(write("manifest.yaml", text))
