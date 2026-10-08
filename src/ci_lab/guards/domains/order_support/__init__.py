"""Order-support guard domain pack (design §13.7): frozen ``verify_identity`` tool, the
``identity_verified`` extractor and side-effect tool policies. Seed rules live (arm-visible, all
``mode: shadow``) in ``src/order_support/harness/guards/order_support.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ci_lab.guards.domains.order_support.verify_identity import verify_identity

HERE = Path(__file__).parent
EXTRACTORS = HERE / "extractors.yaml"
# src/ci_lab/guards/domains/order_support -> src/order_support/harness/guards
GUARDS_DIR = HERE.parents[3] / "order_support" / "harness" / "guards"
SEED_RULES = GUARDS_DIR / "order_support.yaml"

# B5: side-effecting tools are serialized per conversation, batch-preflighted and fail closed.
# HOOK(M3): mirror as `side_effect: true` in order_support tool_specs and pass those specs instead.
TOOL_POLICIES: dict[str, bool] = {
    "issue_refund": True,
    "escalate_to_human": True,
    "lookup_order": False,
    "search_kb": False,
    "verify_identity": False,
}

VERIFY_IDENTITY_DESCRIPTION = (
    "Verify the customer's identity for an order. Pass exactly the full name and the email address "
    "or phone number the CUSTOMER stated in the conversation (never values from lookup_order). "
    "Returns only {verified, order_id}; refunds require verified == true for the same order."
)


def verify_identity_tool() -> Any:
    """``verify_identity`` as a MAF ``FunctionTool`` returning canonical JSON text."""
    from agent_framework import tool

    from ci_lab.rulespec import canonical_json

    def _verify_identity(order_id: str, full_name: str, email_or_phone: str) -> str:
        return canonical_json(verify_identity(order_id, full_name, email_or_phone))

    return tool(_verify_identity, name="verify_identity", description=VERIFY_IDENTITY_DESCRIPTION)


def load_order_support_bundle(guards_dir: Path | str = GUARDS_DIR) -> tuple[Any | None, bool]:
    """``(bundle, degraded)`` for the order-support guards dir plus the frozen extractors."""
    from ci_lab.guards.engine import load_guard_bundle

    return load_guard_bundle(guards_dir, extractor_paths=(EXTRACTORS,))


__all__ = ["EXTRACTORS", "GUARDS_DIR", "SEED_RULES", "TOOL_POLICIES", "load_order_support_bundle",
           "verify_identity", "verify_identity_tool"]
