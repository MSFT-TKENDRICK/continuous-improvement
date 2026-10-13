"""Frozen deterministic identity verification for order support (design §13.7).

The model passes what the *customer* stated; the tool compares it against the order's customer
record and returns only ``{"verified": bool, "order_id": <normalized>}``. It never echoes PII, so
its output cannot be used to learn customer details. The ``identity_verified`` flag is derived
from this result by the frozen extractor (``extractors.yaml``), never from model text.
"""

from __future__ import annotations

from typing import Any

from order_support.data import ORDERS


def _norm_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _contact_matches(stated: str, email: str, phone: str) -> bool:
    stated = stated.strip()
    if not stated:
        return False
    if "@" in stated:
        return stated.casefold() == email.strip().casefold()
    got, want = _digits(stated), _digits(phone)
    if not got or not want:
        return False
    if len(got) == 4:
        return want.endswith(got)  # last-4
    return len(got) >= 7 and want.endswith(got)  # full number, with or without country code


def verify_identity(order_id: str, full_name: str, email_or_phone: str) -> dict[str, Any]:
    """Verify the customer for ``order_id``: full name (case/space-insensitive) AND the order's
    email (exact, case-insensitive) or phone (last 4 digits or full number)."""
    normalized = str(order_id or "").strip().upper()
    order = ORDERS.get(normalized)
    verified = False
    if order is not None and isinstance(full_name, str) and isinstance(email_or_phone, str):
        customer = order["customer"]
        verified = (_norm_name(full_name) == _norm_name(customer["name"])
                    and _contact_matches(email_or_phone, customer["email"], customer["phone"]))
    return {"verified": verified, "order_id": normalized}
