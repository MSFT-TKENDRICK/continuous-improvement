"""Round decision records: ``rounds/<eid>/decisions.json`` (design v1 §3, step 7; §12.3 spans)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.contracts import ATTR_CAMPAIGN, ATTR_DECISION, ATTR_EXPERIMENT, ATTR_PHASE, SPAN_STEP
from ci_lab.ledger.atomic import atomic_write_json, read_json
from ci_lab.ledger.layout import Layout

DECISIONS = ("ship", "do_not_ship", "rerun")


def record_decisions(repo: str | os.PathLike[str] | Layout, campaign_id: str, experiment_id: str,
                     decisions: Mapping[str, Any]) -> Path:
    """Atomically write ``decisions.json`` for a round. ``decisions["decision"]`` (if present)
    must be an OES verdict (``ship`` / ``do_not_ship`` / ``rerun``). Emits a
    ``ci.step{ci.phase=record}`` span carrying ``oes.experiment_id`` and ``oes.decision``."""
    layout = repo if isinstance(repo, Layout) else Layout(repo)
    decision = decisions.get("decision")
    if decision is not None and decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    path = layout.decisions_json(campaign_id, experiment_id)
    obs.annotate({ATTR_EXPERIMENT: experiment_id, ATTR_DECISION: decision})  # caller's round span
    with obs.span(SPAN_STEP, {ATTR_PHASE: "record", ATTR_CAMPAIGN: campaign_id, ATTR_EXPERIMENT: experiment_id,
                              ATTR_DECISION: decision}):
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, dict(decisions))
    return path


def read_decisions(repo: str | os.PathLike[str] | Layout, campaign_id: str, experiment_id: str) -> dict[str, Any] | None:
    layout = repo if isinstance(repo, Layout) else Layout(repo)
    return read_json(layout.decisions_json(campaign_id, experiment_id), default=None)
