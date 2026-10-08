"""Judge tooling: ``s1/`` LiteLLM provider for ASSERT, judge-vs-human audit, DSPy judge alignment.

Submodules import heavy dependencies (litellm, dspy) lazily; importing this package is cheap.

* :mod:`ci_lab.judge.provider` - ``register()`` the ``s1/<backend>/<model>`` provider (C21, C25)
* :mod:`ci_lab.judge.audit`    - per-dimension agreement, kappa, bootstrap CIs, diagnostic flags (C21)
* :mod:`ci_lab.judge.align`    - rubric alignment -> OES evaluator-experiment proposal (C17)
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(**kwargs: Any) -> Any:
    """Idempotently register the ``s1`` LiteLLM provider (see :func:`ci_lab.judge.provider.register`)."""
    from ci_lab.judge.provider import register as _register

    return _register(**kwargs)
