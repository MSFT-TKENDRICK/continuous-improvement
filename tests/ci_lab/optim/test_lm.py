import subprocess
import sys

import pytest

from ci_lab.contracts import Profile
from ci_lab.optim import lm as lmmod


def test_import_is_lazy():
    code = ("import sys, ci_lab.optim, ci_lab.optim.lm, ci_lab.optim.scoring, ci_lab.optim.targets; "
            "bad=[m for m in ('dspy','gepa','skillopt_sleep','litellm') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("profile,purpose,cache,expected", [
    (Profile.COPILOT, "optimizer", None, False),
    (Profile.COPILOT, "optimizer", True, False),
    (Profile.OFFLINE, "judge", True, False),
    (Profile.OFFLINE, "optimizer", None, True),
    (Profile.OFFLINE, "reflector", False, False),
    (Profile.FAKE, "optimizer", True, False),
])
def test_cache_rules(profile, purpose, cache, expected):
    assert lmmod.cache_enabled(profile, purpose, cache) is expected


def test_make_lm_copilot_no_cache(monkeypatch):
    monkeypatch.delenv(lmmod.AGL_BASE_URL_ENV, raising=False)
    env = {lmmod.COPILOT_SERVE_URL_ENV: "http://127.0.0.1:9999/v1/", lmmod.COPILOT_SERVE_KEY_ENV: "k1"}
    lm = lmmod.make_lm("copilot", model="gpt-x", cache=True, env=env)
    assert lm.model == "openai/gpt-x"
    assert lm.cache is False
    assert lm.kwargs["api_base"] == "http://127.0.0.1:9999/v1"
    assert lm.kwargs["api_key"] == "k1"


def test_make_lm_offline_judge_and_optimizer():
    env = {"OPENAI_BASE_URL": "http://127.0.0.1:8081/v1", "CI_LAB_JUDGE_MODEL": "openai/qwen"}
    judge = lmmod.make_lm(Profile.OFFLINE, "judge", env=env)
    assert judge.cache is False and judge.model == "openai/qwen"
    opt = lmmod.make_lm(Profile.OFFLINE, env=env)
    assert opt.cache is True and opt.model == "openai/local"
    assert opt.kwargs["api_key"] == "local"


def test_agl_proxy_wins_and_key_file(tmp_path):
    kf = tmp_path / "key"
    kf.write_text("secret\n")
    env = {lmmod.AGL_BASE_URL_ENV: "http://127.0.0.1:4747/rollout/r1/v1",
           lmmod.COPILOT_SERVE_URL_ENV: "http://127.0.0.1:9999/v1",
           lmmod.COPILOT_SERVE_KEY_FILE_ENV: str(kf)}
    assert lmmod.resolve_endpoint("copilot", env) == ("http://127.0.0.1:4747/rollout/r1/v1", "secret")
    assert lmmod.resolve_endpoint("offline", {})[0] == lmmod.DEFAULT_OFFLINE_BASE


def test_fake_profile_is_dummy_lm():
    from dspy.utils.dummies import DummyLM

    lm = lmmod.make_lm("fake", fake_answers=[{"answer": "hi"}])
    assert isinstance(lm, DummyLM) and lm.cache is False
    out = lm("question?")
    assert "hi" in out[0]
    with pytest.raises(ValueError):
        lmmod.resolve_endpoint("fake", {})


def test_lm_usage_tokens():
    class L:
        history = [{"usage": {"total_tokens": 5}}, {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}, {}]
    assert lmmod.lm_usage_tokens(L()) == 10
