"""CI_META_MODEL / CI_ALLOWED_MODELS: operator model selection that keeps provenance honest."""

from __future__ import annotations

import shutil

import pytest
import yaml

from ci_lab.contracts import Profile
from ci_lab.maf.models import (
    ModelEnvError,
    extra_allowed_models,
    meta_model_override,
    with_extra_allowed,
)
from ci_lab.meta.spec_loader import (
    AGENTS,
    SpecError,
    default_builder,
    load_manifest,
    load_spec,
    loader_builder,
    manifest_allowed_models,
    subagent_specs,
)
from ci_lab.testing import FakeChatClient


def _build(spec, builder=None):
    bindings = {t: (lambda: "x") for t in spec.tools}
    subs = {s.key: {t: (lambda: "x") for t in s.tools} for s in subagent_specs(spec)}
    return (builder or default_builder())(spec, client=FakeChatClient(), bindings=bindings,
                                          **({"subagent_bindings": subs} if subs else {}))


def test_env_parsing():
    assert meta_model_override({}) is None and meta_model_override({"CI_META_MODEL": "  "}) is None
    assert meta_model_override({"CI_META_MODEL": " gpt-5.5 "}) == "gpt-5.5"
    assert extra_allowed_models({"CI_ALLOWED_MODELS": "a-1, b.2,,"}) == ("a-1", "b.2")
    assert with_extra_allowed(["a", "b"], {"CI_ALLOWED_MODELS": "b,c"}) == ("a", "b", "c")
    for bad in ("=Env.X", "gpt 5", "a\nb", "../x" * 50):
        with pytest.raises(ModelEnvError):
            meta_model_override({"CI_META_MODEL": bad})
    with pytest.raises(ModelEnvError):
        extra_allowed_models({"CI_ALLOWED_MODELS": "ok,=bad"})


def test_default_model_is_allowed_and_used_by_every_agent():
    m = load_manifest()
    assert m["model"] == "claude-sonnet-5" and m["model"] in manifest_allowed_models(m)
    assert {load_spec(k).model for k in AGENTS} == {"claude-sonnet-5"}


@pytest.mark.parametrize("key", sorted(AGENTS))
def test_override_is_applied_before_validation_and_hashing(monkeypatch, key):
    default = _build(load_spec(key)).additional_properties["ci_lab"]
    monkeypatch.setenv("CI_META_MODEL", "gpt-5.5")
    spec = load_spec(key)
    assert spec.model == "gpt-5.5" and all(s.model == "gpt-5.5" for s in subagent_specs(spec))
    props = _build(spec).additional_properties["ci_lab"]
    assert props["model"] == "gpt-5.5"
    assert props["spec_digest"] != default["spec_digest"]  # the hash reflects the real model
    for sub, digest in props.get("subagents", {}).items():
        assert digest != default["subagents"][sub]


def test_override_must_be_in_the_allowlist(monkeypatch):
    monkeypatch.setenv("CI_META_MODEL", "gpt-4o")
    with pytest.raises(SpecError, match="CI_META_MODEL='gpt-4o' is not in the allowlist"):
        load_spec("critic")


def test_malformed_override_fails_closed(monkeypatch):
    monkeypatch.setenv("CI_META_MODEL", "=Env.SECRET")
    with pytest.raises(SpecError, match="not a valid model id"):
        load_spec("critic")


def test_allowlist_is_extended_only_by_env(monkeypatch):
    assert "gpt-4o" not in manifest_allowed_models()
    monkeypatch.setenv("CI_ALLOWED_MODELS", "gpt-4o,o4-mini")
    assert manifest_allowed_models()[-2:] == ("gpt-4o", "o4-mini")
    monkeypatch.setenv("CI_META_MODEL", "gpt-4o")
    spec = load_spec("critic")
    assert _build(spec).additional_properties["ci_lab"]["model"] == "gpt-4o"
    # An explicit allowlist passed by a caller stays exact.
    with pytest.raises(SpecError, match="allowlist"):
        _build(spec, default_builder(allowed_models=["claude-sonnet-5"]))


def test_spec_file_cannot_pick_an_unlisted_model(tmp_path):
    from ci_lab.meta.spec_loader import SPECS_DIR

    shutil.copytree(SPECS_DIR, tmp_path / "specs")
    path = tmp_path / "specs" / "critic.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["model"]["id"] = "gpt-4o"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    spec = load_spec(path)
    assert spec.model == "gpt-4o"
    with pytest.raises(SpecError, match="allowlist"):
        _build(spec)


def test_loader_builder_refuses_an_override_it_cannot_apply(monkeypatch):
    monkeypatch.setenv("CI_META_MODEL", "gpt-5.5")
    spec = load_spec("critic")
    with pytest.raises(SpecError, match="validated_harness_builder"):
        loader_builder(lambda spec_path, **kw: "agent")(spec, client="c", bindings={t: print for t in spec.tools})


def test_proposer_strategy_uses_the_spec_model(monkeypatch):
    from ci_lab.meta.run import MetaAgentError, ProposerStrategy

    calls = []

    def factory(*, profile, model, purpose, rollout=None):
        calls.append(model)
        return "client"

    class Ctx:
        profile = Profile.FAKE

    monkeypatch.setenv("CI_META_MODEL", "claude-opus-5")
    assert ProposerStrategy(object(), client_factory=factory).client_for(Ctx()) == "client"
    assert calls == ["claude-opus-5"]
    with pytest.raises(MetaAgentError, match="CI_META_MODEL"):
        ProposerStrategy(object(), client_factory=factory, model="gpt-5.5").client_for(Ctx())


def test_lesson_synthesizer_override(monkeypatch):
    from ci_lab.lessons_arm.agent import LessonSynthesizer, SynthesizerError
    from ci_lab.lessons_arm.agent import default_builder as la_builder

    seen = []

    def factory(*, profile, model, purpose):
        seen.append(model)
        return FakeChatClient(["done"])

    monkeypatch.setenv("CI_META_MODEL", "claude-sonnet-5.5")
    syn = LessonSynthesizer(client_factory=factory, profile=Profile.FAKE)
    assert syn.model == syn.spec["model"]["id"] == "claude-sonnet-5.5"
    syn.client()
    assert seen == ["claude-sonnet-5.5"]
    explicit = LessonSynthesizer(client=FakeChatClient(), model="gpt-5.5")
    assert explicit.spec["model"]["id"] == "gpt-5.5"  # validated + hashed spec names the client's model
    monkeypatch.setenv("CI_META_MODEL", "gpt-4o")
    syn = LessonSynthesizer(client=FakeChatClient())
    with pytest.raises(SynthesizerError, match="allowlist"):
        la_builder()(syn.spec, syn.client(), {"submit_rule": lambda **_: "ok"})
    monkeypatch.setenv("CI_META_MODEL", "bad model")
    with pytest.raises(SynthesizerError, match="not a valid model id"):
        LessonSynthesizer(client=FakeChatClient())
