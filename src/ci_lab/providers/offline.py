"""Network policy for the ``offline`` profile: every model endpoint must be on this host.

``offline`` is network-free. The chat endpoints (``$OPENAI_API_BASE`` / ``$OPENAI_BASE_URL`` /
``$AGL_OPENAI_BASE_URL``) and the s1 judge backends the ASSERT child talks to
(``$CI_S1_LLAMA_URL``, ``$CI_S1_SYSTEMONE_URL`` and any other ``CI_S1_*_URL``) must all point
at a loopback host. :func:`check_offline_endpoints` raises :class:`NetworkPolicyError`
otherwise, so an offline run fails closed before it opens a connection.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

__all__ = [
    "OFFLINE_ENDPOINT_ENVS",
    "S1_JUDGE_URL_ENVS",
    "NetworkPolicyError",
    "check_loopback",
    "check_offline_endpoints",
    "is_loopback_host",
    "offline_endpoint_envs",
]

OFFLINE_ENDPOINT_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AGL_OPENAI_BASE_URL")
# s1 judge backends (ci_lab.judge.backends): llama-server logprob judge and SystemOne.
S1_JUDGE_URL_ENVS = ("CI_S1_LLAMA_URL", "CI_S1_SYSTEMONE_URL")
_S1_URL_ENV = re.compile(r"^CI_S1_\w*(?:URL|BASE)$")  # future s1 backends follow the same naming


class NetworkPolicyError(ValueError):
    """An ``offline`` endpoint is not on this host."""


def is_loopback_host(host: str) -> bool:
    host = host.strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_loopback(name: str, url: str | None) -> None:
    """Raise :class:`NetworkPolicyError` unless ``url`` (when set) has a loopback host."""
    url = (url or "").strip()
    if not url:
        return
    host = urlsplit(url).hostname or ""
    if not is_loopback_host(host):
        raise NetworkPolicyError(f"offline profile is network-free: {name} must point at a loopback host, "
                                 f"not {host or url!r}")


def offline_endpoint_envs(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The endpoint variables :func:`check_offline_endpoints` checks, in a stable order."""
    env = os.environ if env is None else env
    extra = sorted(k for k in env if _S1_URL_ENV.match(k) and k not in S1_JUDGE_URL_ENVS)
    return (*OFFLINE_ENDPOINT_ENVS, *S1_JUDGE_URL_ENVS, *extra)


def check_offline_endpoints(env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`NetworkPolicyError` unless every configured model endpoint is loopback,
    including the s1 judge backend URLs."""
    env = os.environ if env is None else env
    for name in offline_endpoint_envs(env):
        check_loopback(f"${name}", env.get(name))
