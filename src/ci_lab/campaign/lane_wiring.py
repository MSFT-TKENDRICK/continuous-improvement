"""Production collaborators of the challenger lane (:mod:`ci_lab.campaign.challenger_lane`).

``lane_voters`` is the System-1 rubric judge (:class:`~ci_lab.bus.voters.remote.S1RubricVoter`,
model :data:`~ci_lab.bus.voters.remote.DEFAULT_S1_MODEL`) at ``$CI_S1_LLAMA_URL`` when that is set,
else ``None`` (the lane then notes that it is inert). ``adversary_complete`` drives the
``LLMAdversary`` with the ``adversary`` meta spec's model through the campaign's chat-client
factory, as ``ci-bus run --challenger llm`` does; the client is built on first use, so building the
deps opens no connection. ``offline`` has no LLM adversary.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ci_lab.contracts import Profile

__all__ = ["adversary_complete", "lane_voters", "s1_judge"]


def s1_judge(environ: Mapping[str, str] | None = None) -> Any:
    """The System-1 judge voter when ``$CI_S1_LLAMA_URL`` configures one, else ``None``."""
    from ci_lab.bus.voters.remote import DEFAULT_S1_MODEL, S1RubricVoter
    from ci_lab.judge.backends import LLAMA_URL_ENV

    url = ((os.environ if environ is None else environ).get(LLAMA_URL_ENV) or "").strip()
    return S1RubricVoter(DEFAULT_S1_MODEL, api_base=url) if url else None


def lane_voters(environ: Mapping[str, str] | None = None) -> Callable[[Any], Sequence[Any]] | None:
    judge = s1_judge(environ)
    return None if judge is None else (lambda _arm: [judge])


def adversary_complete(profile: Profile | str, client_factory: Callable[..., Any]
                       ) -> Callable[[str], Awaitable[str]] | None:
    if Profile(profile) is not Profile.COPILOT:
        return None
    client: Any = None

    async def complete(prompt: str) -> str:
        nonlocal client
        if client is None:
            from ci_lab.adversary.challenger import ADVERSARY_SPEC
            from ci_lab.meta.spec_loader import load_spec

            client = client_factory(profile=Profile(profile), model=load_spec(ADVERSARY_SPEC).model,
                                    purpose="proposer")
        return str((await client.get_response(prompt)).text)

    return complete
