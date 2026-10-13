"""Model availability preflight for the ``copilot`` profile (fail fast, before spending budget).

A model id baked into a spec, harness or default goes stale when the Copilot account's model
list changes, and the SDK only says so when the first session is created, deep inside a run.
These helpers check up front that every model a command will use is served:

* :func:`check_copilot_models`: ids against the Copilot SDK's ``list_models()``;
* :func:`check_served_models`: OpenAI-compatible endpoints (``ci-lab copilot-serve``,
  llama-server) against their ``GET /models``.

Each requirement is a :class:`ModelUse` that names who uses the model and how to change it,
so the error says what is missing, what is available and which override to set. Nothing
here substitutes a model: picking another one is always an explicit operator override, so
the recorded provenance stays true. Listing is injectable (``list_models=``/``fetch=``) so
tests never need the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ListModels",
    "ModelPreflightError",
    "ModelUse",
    "ServedModels",
    "check_copilot_models",
    "check_served_models",
    "copilot_model_ids",
    "model_ids",
    "served_model_ids",
]

log = logging.getLogger(__name__)

ListModels = Callable[[], Awaitable[Iterable[Any]]]
ServedModels = Callable[[str, str], Iterable[str]]
LIST_TIMEOUT_S = 120.0


class ModelPreflightError(RuntimeError):
    """A model a command needs is not available (or availability could not be determined)."""


@dataclass(frozen=True)
class ModelUse:
    """``model`` is used by ``user``; ``override`` says how an operator selects another one."""

    model: str
    user: str
    override: str = ""


def model_ids(models: Iterable[Any]) -> tuple[str, ...]:
    """Sorted unique ids from SDK ``ModelInfo`` objects (``.id``), dicts (``["id"]``) or strings."""
    out: set[str] = set()
    for m in models:
        mid = m if isinstance(m, str) else (m.get("id") if isinstance(m, dict) else getattr(m, "id", None))
        if mid:
            out.add(str(mid))
    return tuple(sorted(out))


async def copilot_model_ids() -> tuple[str, ...]:
    """Model ids the signed-in Copilot account can use (starts and stops a Copilot CLI)."""
    from copilot import CopilotClient

    from ci_lab.providers.copilot import service_env

    base = tempfile.mkdtemp(prefix="ci-lab-models-")
    client = CopilotClient(mode="empty", base_directory=base, log_level="error", env=service_env())
    try:
        await client.start()
        return model_ids(await client.list_models())
    finally:
        with contextlib.suppress(Exception):
            await client.stop()
        shutil.rmtree(base, ignore_errors=True)


def _missing_message(what: str, missing: Sequence[ModelUse], available: Sequence[str]) -> str:
    by_model: dict[str, list[ModelUse]] = {}
    for use in missing:
        by_model.setdefault(use.model, []).append(use)
    lines = [f"{what}: {len(by_model)} model(s) needed by this command are not available."]
    for model, uses in by_model.items():
        lines.append(f"  - {model!r} used by {', '.join(dict.fromkeys(u.user for u in uses))}")
        for hint in dict.fromkeys(u.override for u in uses if u.override):
            lines.append(f"      override: {hint}")
    lines.append(f"Available: {', '.join(available) if available else '(none)'}")
    return "\n".join(lines)


def _unique(uses: Iterable[ModelUse]) -> list[ModelUse]:
    return list(dict.fromkeys(u for u in uses if u.model))


async def check_copilot_models(uses: Iterable[ModelUse], *, list_models: ListModels | None = None,
                               timeout_s: float = LIST_TIMEOUT_S) -> tuple[str, ...]:
    """Raise :class:`ModelPreflightError` unless every ``uses`` model is in the Copilot SDK's
    ``list_models()``. Returns the available ids. No requirement -> no SDK call."""
    wanted = _unique(uses)
    if not wanted:
        return ()
    try:
        available = model_ids(await asyncio.wait_for((list_models or copilot_model_ids)(), timeout_s))
    except ModelPreflightError:
        raise
    except Exception as exc:
        raise ModelPreflightError(f"Copilot model preflight: could not list models ({type(exc).__name__}: {exc}); "
                                  "is the Copilot CLI signed in?") from exc
    if missing := [u for u in wanted if u.model not in available]:
        raise ModelPreflightError(_missing_message("Copilot model preflight", missing, available))
    return available


def served_model_ids(base_url: str, api_key: str, *, timeout_s: float = 10.0) -> tuple[str, ...]:
    """Ids from an OpenAI-compatible ``GET {base_url}/models`` (copilot-serve, llama-server)."""
    import httpx

    r = httpx.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"},
                  timeout=timeout_s)
    r.raise_for_status()
    return model_ids(r.json().get("data") or [])


def check_served_models(base_url: str, api_key: str, uses: Iterable[ModelUse], *,
                        fetch: ServedModels | None = None) -> tuple[str, ...]:
    """Raise :class:`ModelPreflightError` unless the endpoint serves every ``uses`` model
    (LiteLLM ``openai/<id>`` prefixes are stripped). Returns the served ids."""
    wanted = [ModelUse(u.model.removeprefix("openai/"), u.user, u.override) for u in _unique(uses)]
    if not wanted:
        return ()
    what = f"Model preflight for {base_url}"
    try:
        served = model_ids((fetch or served_model_ids)(base_url, api_key))
    except Exception as exc:
        raise ModelPreflightError(f"{what}: could not list served models ({type(exc).__name__}: {exc}); "
                                  "is the endpoint running and the key right?") from exc
    if missing := [u for u in wanted if u.model not in served]:
        raise ModelPreflightError(_missing_message(what, missing, served))
    return served
