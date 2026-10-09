
"""Domain registry for the self-hosted harness."""

from __future__ import annotations

from typing import Any

DOMAIN_CHOICES = ("harness",)
DEFAULT_DOMAIN = "harness"


def get_domain(name: str = DEFAULT_DOMAIN, **kwargs: Any) -> Any:
    if name == "harness":
        from ci_lab.domain.harness import HarnessDomain

        return HarnessDomain(**kwargs)
    raise ValueError(f"unknown domain {name!r}; choose one of {', '.join(DOMAIN_CHOICES)}")


def harness_domain(**kwargs: Any) -> Any:
    return get_domain("harness", **kwargs)


__all__ = ["DEFAULT_DOMAIN", "DOMAIN_CHOICES", "get_domain", "harness_domain"]
