
"""Domain registry.  L7 adds ``harness`` without changing the ``order_support`` default."""

from __future__ import annotations

from typing import Any

DOMAIN_CHOICES = ("order_support", "harness")
DEFAULT_DOMAIN = "order_support"


def get_domain(name: str = DEFAULT_DOMAIN, **kwargs: Any) -> Any:
    if name == "order_support":
        from ci_lab.domain.order_support import OrderSupportDomain

        return OrderSupportDomain(**kwargs)
    if name == "harness":
        from ci_lab.domain.harness import HarnessDomain

        return HarnessDomain(**kwargs)
    raise ValueError(f"unknown domain {name!r}; choose one of {', '.join(DOMAIN_CHOICES)}")


def order_support_domain(**kwargs: Any) -> Any:
    return get_domain("order_support", **kwargs)


def harness_domain(**kwargs: Any) -> Any:
    return get_domain("harness", **kwargs)


__all__ = ["DEFAULT_DOMAIN", "DOMAIN_CHOICES", "get_domain", "harness_domain", "order_support_domain"]
