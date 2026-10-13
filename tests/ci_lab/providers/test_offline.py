"""``offline`` network policy: chat and s1 judge endpoints must be loopback (fail closed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ci_lab.contracts import Profile
from ci_lab.providers.factory import make_chat_client
from ci_lab.providers.offline import (
    NetworkPolicyError,
    check_loopback,
    check_offline_endpoints,
    is_loopback_host,
    offline_endpoint_envs,
)

ENDPOINT_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AGL_OPENAI_BASE_URL", "CI_S1_LLAMA_URL",
                 "CI_S1_SYSTEMONE_URL", "CI_S1_EXTRA_URL", "CI_LAB_AGL_URL")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENDPOINT_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]", "127.8.9.10"])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["", "0.0.0.0", "10.0.0.5", "localhost.example.com", "api.openai.com"])
def test_non_loopback_hosts(host: str) -> None:
    assert not is_loopback_host(host)


@pytest.mark.parametrize("name", ["CI_S1_LLAMA_URL", "CI_S1_SYSTEMONE_URL"])
def test_s1_judge_urls_must_be_loopback(name: str) -> None:
    check_offline_endpoints({name: "http://127.0.0.1:8081", "OPENAI_API_BASE": "http://localhost:8080/v1"})
    check_offline_endpoints({name: "http://[::1]:8081/v1"})
    check_offline_endpoints({name: ""})  # unset/empty: the backend default (127.0.0.1:8081) applies
    with pytest.raises(NetworkPolicyError, match=rf"\${name} must point at a loopback host, not 'judge.example.com'"):
        check_offline_endpoints({name: "https://judge.example.com/v1"})


def test_other_s1_backend_urls_are_checked() -> None:
    env = {"CI_S1_NEWBACKEND_URL": "http://10.1.2.3:9000", "CI_S1_API_KEY": "secret", "CI_S1_RUBRICS": "r.yaml"}
    assert "CI_S1_NEWBACKEND_URL" in offline_endpoint_envs(env)
    assert "CI_S1_API_KEY" not in offline_endpoint_envs(env)
    with pytest.raises(NetworkPolicyError, match="CI_S1_NEWBACKEND_URL"):
        check_offline_endpoints(env)


def test_url_without_scheme_fails_closed() -> None:
    with pytest.raises(NetworkPolicyError, match="judge.example.com:8081"):
        check_loopback("$CI_S1_LLAMA_URL", "judge.example.com:8081")


def test_offline_chat_client_fails_closed_on_off_host_judge(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENAI_API_BASE", "http://127.0.0.1:9999/v1")
    clean_env.setenv("CI_S1_LLAMA_URL", "http://127.0.0.1:8081")
    assert make_chat_client(profile=Profile.OFFLINE, model="qwen", purpose="target") is not None
    clean_env.setenv("CI_S1_SYSTEMONE_URL", "https://systemone.example.com")
    with pytest.raises(NetworkPolicyError, match="CI_S1_SYSTEMONE_URL"):
        make_chat_client(profile=Profile.OFFLINE, model="qwen", purpose="target")


def test_offline_chat_client_base_url_must_be_loopback(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(NetworkPolicyError, match="base_url"):
        make_chat_client(profile=Profile.OFFLINE, model="qwen", purpose="target", base_url="https://api.openai.com/v1")


def test_copilot_profile_is_not_restricted(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("CI_S1_LLAMA_URL", "https://judge.example.com")
    assert make_chat_client(profile=Profile.COPILOT, model="m", purpose="judge") is not None


def test_offline_campaign_wiring_fails_closed(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ci_lab.campaign.wiring import wired_deps

    clean_env.setenv("CI_S1_LLAMA_URL", "http://judge.example.com:8081")
    with pytest.raises(NetworkPolicyError, match="CI_S1_LLAMA_URL"):
        wired_deps("offline", run_root=tmp_path / "runs", repo_root=tmp_path)


def test_offline_sleep_wiring_fails_closed(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ci_lab.sleep.night import SleepConfig
    from ci_lab.sleep.wiring import build_deps

    clean_env.setenv("CI_S1_SYSTEMONE_URL", "https://systemone.example.com")
    cfg = SleepConfig(repo_root=tmp_path, out_dir=tmp_path / "out", work_dir=tmp_path / "work")
    with pytest.raises(NetworkPolicyError, match="CI_S1_SYSTEMONE_URL"):
        build_deps(Profile.OFFLINE, cfg)
