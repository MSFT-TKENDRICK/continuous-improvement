"""DSPy LM routing for optimizers (design §11.4 C19, C24, C26).

``make_lm`` builds a ``dspy.LM`` speaking LiteLLM ``openai/<model>`` to an
OpenAI-compatible endpoint resolved from the inference profile:

1. ``api_base=`` argument;
2. ``AGL_OPENAI_BASE_URL`` — the Agent Lightning per-rollout proxy URL, when the
   process runs inside an AGL rollout (every LLM call becomes an AGL event);
3. profile endpoint — ``copilot``: ``ci-lab copilot-serve`` (``CI_COPILOT_SERVE_URL``,
   key from ``CI_COPILOT_SERVE_KEY`` or the file in ``CI_COPILOT_SERVE_KEY_FILE``);
   ``offline``: llama-server (``OPENAI_API_BASE`` / ``OPENAI_BASE_URL``, default
   ``http://127.0.0.1:8081/v1``; key ``OPENAI_API_KEY`` or ``"local"``).

The ``fake`` profile returns a deterministic ``dspy.utils.DummyLM`` (no network).

DSPy is imported lazily (C26): importing this module never imports ``dspy``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ci_lab.contracts import Profile, Purpose

if TYPE_CHECKING:  # pragma: no cover
    import dspy

AGL_BASE_URL_ENV = "AGL_OPENAI_BASE_URL"
COPILOT_SERVE_URL_ENV = "CI_COPILOT_SERVE_URL"
COPILOT_SERVE_KEY_ENV = "CI_COPILOT_SERVE_KEY"
COPILOT_SERVE_KEY_FILE_ENV = "CI_COPILOT_SERVE_KEY_FILE"
OFFLINE_BASE_URL_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL")
OFFLINE_KEY_ENV = "OPENAI_API_KEY"
MODEL_ENV_FMT = "CI_LAB_{purpose}_MODEL"  # e.g. CI_LAB_OPTIMIZER_MODEL

DEFAULT_OFFLINE_BASE = "http://127.0.0.1:8081/v1"
DEFAULT_COPILOT_BASE = "http://127.0.0.1:8765/v1"
DEFAULT_MODELS: Mapping[Profile, str] = {
    Profile.COPILOT: "gpt-5-mini",
    Profile.OFFLINE: "local",
    Profile.FAKE: "fake-model",
}


def cache_enabled(profile: Profile | str, purpose: Purpose = "optimizer",
                  cache: bool | None = None) -> bool:
    """C19: never cache Copilot calls (temperature is ignored, so a cache would fake
    determinism and poison A/A) nor judging; the fake LM never caches. An explicit
    ``cache=True`` cannot override these rules; ``cache=False`` always disables."""
    profile = Profile(profile)
    if profile in (Profile.COPILOT, Profile.FAKE) or purpose == "judge":
        return False
    return True if cache is None else bool(cache)


def resolve_model(profile: Profile | str, purpose: Purpose = "optimizer",
                  model: str | None = None, env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    profile = Profile(profile)
    name = model or env.get(MODEL_ENV_FMT.format(purpose=purpose.upper()), "").strip() \
        or DEFAULT_MODELS[profile]
    return name.split("/", 1)[1] if name.startswith("openai/") else name


def resolve_endpoint(profile: Profile | str, env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(api_base, api_key) for an OpenAI-compatible profile. Never logs the key."""
    env = os.environ if env is None else env
    profile = Profile(profile)
    if profile is Profile.FAKE:
        raise ValueError("fake profile has no endpoint")
    if profile is Profile.COPILOT:
        key = env.get(COPILOT_SERVE_KEY_ENV, "").strip()
        if not key and (kf := env.get(COPILOT_SERVE_KEY_FILE_ENV, "").strip()):
            key = Path(kf).read_text(encoding="utf-8").strip()
        default_base = env.get(COPILOT_SERVE_URL_ENV, "").strip() or DEFAULT_COPILOT_BASE
    else:
        key = env.get(OFFLINE_KEY_ENV, "").strip()
        default_base = next((env[v].strip() for v in OFFLINE_BASE_URL_ENVS if env.get(v, "").strip()),
                            DEFAULT_OFFLINE_BASE)
    base = env.get(AGL_BASE_URL_ENV, "").strip() or default_base
    return base.rstrip("/"), key or "local"


def make_lm(profile: Profile | str, purpose: Purpose = "optimizer", *, model: str | None = None,
            cache: bool | None = None, api_base: str | None = None, api_key: str | None = None,
            fake_answers: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
            env: Mapping[str, str] | None = None, **lm_kwargs: Any) -> dspy.LM:
    """Build the DSPy LM for ``purpose`` under ``profile`` (see module doc).

    ``fake_answers`` scripts the ``fake`` profile's ``DummyLM`` (list = in order,
    dict = keyed by a substring of the last message); each answer's values are
    returned verbatim (raw completion text; pass ``fake_adapter=dspy.ChatAdapter()``
    to get DSPy field formatting instead). Exhausted scripts answer
    ``"No more responses"`` deterministically.
    """
    import dspy  # C26: lazy

    profile = Profile(profile)
    use_cache = cache_enabled(profile, purpose, cache)
    if profile is Profile.FAKE:
        from dspy.utils.dummies import DummyLM

        answers = list(fake_answers) if isinstance(fake_answers, Sequence) else dict(fake_answers or {})
        lm = DummyLM(answers, adapter=lm_kwargs.pop("fake_adapter", None) or _RawTextAdapter())
        lm.cache = False
        return lm
    base, key = resolve_endpoint(profile, env)
    return dspy.LM(
        f"openai/{resolve_model(profile, purpose, model, env)}",
        api_base=api_base or base,
        api_key=api_key or key,
        cache=use_cache,
        engine="litellm",  # C24: LiteLLM -> AGL proxy -> copilot-serve | llama-server
        **lm_kwargs,
    )


class _RawTextAdapter:
    """DummyLM output formatter emitting answer values verbatim (raw LM text)."""

    def format_field_with_value(self, fields_with_values: Mapping[Any, Any], role: str = "assistant") -> str:
        return "\n\n".join(str(v) for v in fields_with_values.values())


def disable_dspy_cache() -> None:
    """Process-wide C19 switch for campaign/judging processes (disk + memory)."""
    import dspy

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def lm_usage_tokens(lm: Any) -> int:
    """Best-effort total tokens recorded in a DSPy LM's history (GEPA cost, C18)."""
    total = 0
    for entry in getattr(lm, "history", None) or []:
        usage = entry.get("usage") if isinstance(entry, Mapping) else None
        if isinstance(usage, Mapping):
            total += int(usage.get("total_tokens") or 0) or \
                int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return total
