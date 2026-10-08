"""Operator model selection for the meta agents (env only; never from arm data).

The model id in a meta-agent spec is experiment provenance: the composed spec is validated
against an allowlist and hashed (``AgentSpec.digest``) into the agent's ``ci_lab`` properties.
A default baked into the YAML goes stale when the Copilot account's model list changes, so an
operator may pick another model per run, without editing specs:

* ``CI_META_MODEL``: replaces ``model.id`` of every meta agent, subagent and the lesson
  synthesizer **before** validation and hashing, so the recorded spec, digest and client all
  name the model that really ran. It must still be in the allowlist.
* ``CI_ALLOWED_MODELS``: comma-separated ids appended to the manifest ``allowed_models``.
  This is the only way to extend the allowlist; arms are data-only and cannot set env vars,
  so they can never choose a model (design §7/C13).

Ids are checked fail-closed: a malformed value raises :class:`ModelEnvError` rather than
being ignored. The chat-client factory never substitutes models.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

__all__ = [
    "ALLOWED_MODELS_ENV",
    "META_MODEL_ENV",
    "ModelEnvError",
    "check_model_id",
    "extra_allowed_models",
    "meta_model_override",
    "with_extra_allowed",
]

META_MODEL_ENV = "CI_META_MODEL"
ALLOWED_MODELS_ENV = "CI_ALLOWED_MODELS"

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")


class ModelEnvError(ValueError):
    """A model id taken from the environment is malformed."""


def check_model_id(value: str, *, source: str) -> str:
    """Return ``value`` stripped, or raise if it is not a plain model id (no ``=`` expressions,
    whitespace or control characters)."""
    model = value.strip()
    if not _MODEL_ID_RE.fullmatch(model):
        raise ModelEnvError(f"{source}: {value!r} is not a valid model id")
    return model


def meta_model_override(env: Mapping[str, str] | None = None) -> str | None:
    """The ``CI_META_MODEL`` model id, or ``None`` when unset or empty."""
    raw = (os.environ if env is None else env).get(META_MODEL_ENV, "")
    return check_model_id(raw, source=META_MODEL_ENV) if raw.strip() else None


def extra_allowed_models(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Model ids from ``CI_ALLOWED_MODELS`` (comma-separated; empty items are ignored)."""
    raw = (os.environ if env is None else env).get(ALLOWED_MODELS_ENV, "")
    return tuple(check_model_id(m, source=ALLOWED_MODELS_ENV) for m in raw.split(",") if m.strip())


def with_extra_allowed(models: Iterable[str], env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """``models`` followed by the ``CI_ALLOWED_MODELS`` extras, de-duplicated, order kept."""
    return tuple(dict.fromkeys([*(str(m) for m in models), *extra_allowed_models(env)]))
