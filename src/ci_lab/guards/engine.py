"""Seam to the frozen rule engine (``ci_lab.rules``, M14) and bundle loading (N1).

Guards depend only on the pinned engine API (FLEET.md Wave 3): ``evaluate(bundle, view, *, on)``,
``redact(text, matches, bundle)`` and ``load_with_lkg(guards_dir, lock, extractor_paths)``. The
engine is resolved lazily so guards import (and fail closed) even when the engine is missing.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from ci_lab.rulespec import GuardView, TemplateSpec

log = logging.getLogger(__name__)

On = Literal["tool_call", "response", "trajectory"]
ENGINE_MODULE = "ci_lab.rules"


class GuardEngine(Protocol):
    """The subset of ``ci_lab.rules`` guards call at runtime (pure, no I/O)."""

    def evaluate(self, bundle: Any, view: GuardView, *, on: On) -> list[Any]: ...

    def redact(self, text: str, matches: Sequence[Any], bundle: Any) -> str: ...


class GuardUnavailable(RuntimeError):
    """No compiled bundle or no engine: guards cannot evaluate (fail closed / degrade, N1)."""


def default_engine() -> GuardEngine | None:
    """``ci_lab.rules`` when importable, else None (callers treat None as unavailable)."""
    try:
        return importlib.import_module(ENGINE_MODULE)  # type: ignore[return-value]
    except ImportError:
        log.warning("guard engine %s is not importable; guards run unavailable", ENGINE_MODULE)
        return None


def load_guard_bundle(guards_dir: Path | str, lock: Path | str | None = None,
                      extractor_paths: Sequence[Path | str] = ()) -> tuple[Any | None, bool]:
    """Load ``guards_dir`` transactionally via ``rules.load_with_lkg`` → ``(bundle, degraded)``.

    ``degraded`` is True when the engine rolled back to the last-known-good bundle. If neither the
    bundle nor an LKG copy loads, returns ``(None, True)``: guards then fail closed on side-effecting
    tools and degrade read-only tools / responses to warn-only (N1).
    """
    guards_dir = Path(guards_dir)
    lock_path = Path(lock) if lock is not None else guards_dir / "BUNDLE.lock"
    try:
        rules = importlib.import_module(ENGINE_MODULE)
        bundle, degraded = rules.load_with_lkg(guards_dir, lock_path, tuple(Path(p) for p in extractor_paths))
    except Exception as exc:  # noqa: BLE001 - any loader failure is availability, not a crash
        log.error("guard bundle load failed (%s); guards degraded", type(exc).__name__)
        for problem in getattr(exc, "details", None) or ():  # rules.RuleLoadError: list[Problem]
            log.error("[GUARDS][ERROR] %s: %s Fix: %s", getattr(problem, "file", "?"),
                      getattr(problem, "violation", problem), getattr(problem, "fix", ""))
        return None, True
    return bundle, bool(degraded)


def bundle_templates(bundle: Any) -> Mapping[str, TemplateSpec]:
    templates = getattr(bundle, "templates", None)
    return templates if isinstance(templates, Mapping) else {}


def bundle_digest(bundle: Any) -> str:
    return str(getattr(bundle, "digest", "") or "none")
