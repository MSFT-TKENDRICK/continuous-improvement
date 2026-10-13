"""Shared fixtures for the real-library integration tests (``test_reallib_*``).

Everything is offline: model calls go to :class:`ci_lab.testing.LoopbackLLM`, a deterministic
OpenAI-compatible server on 127.0.0.1, so the real client libraries (litellm, openai, httpx)
and their callers run unpatched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from ci_lab.testing import LoopbackLLM


@pytest.fixture
def loopback() -> Iterator[Callable[..., LoopbackLLM]]:
    """Factory: ``loopback(respond, judge=...)`` starts a :class:`LoopbackLLM`, stopped at teardown."""
    servers: list[LoopbackLLM] = []

    def make(respond=None, **kw) -> LoopbackLLM:  # type: ignore[no-untyped-def]
        srv = LoopbackLLM(respond, **kw).start()
        servers.append(srv)
        return srv

    yield make
    for srv in servers:
        srv.stop()
