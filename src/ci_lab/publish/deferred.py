"""Deferred publishing: split model-driven rounds from privileged GitHub writes.

``campaign-scheduled.yml`` runs rounds in a job that holds a read-only token (plus
``copilot-requests``). That job uses :class:`DeferredPublisher`, which records each
``publish_round`` request to a JSONL file and leaves the ledger stack untouched (a
dry-run publisher would write synthetic PR numbers into ``stack.json``). A separate,
environment-gated job, which runs no model, replays the requests with the real
:class:`~ci_lab.publish.github.GitHubPublisher` (:func:`replay_deferred`, called by
``ci-lab campaign publish``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ci_lab.publish.github import PublishError

PUBLISHED = "published.jsonl"


class DeferredPublisher:
    """``Publisher`` that queues ``publish_round`` calls instead of executing them."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def publish_round(self, *, eid: str, winner: str | None, heads: Mapping[str, str],
                      stack: Mapping[str, Any], title: str, body: str) -> dict[str, Any]:
        row = {"eid": eid, "winner": winner, "heads": dict(heads), "title": title, "body": body}
        if not any(r.get("eid") == eid for r in read_requests(self.path)):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return {"stack": dict(stack), "deferred": True}

    def land(self, stack: Mapping[str, Any]) -> dict[str, Any]:
        raise PublishError("land is never deferred; run `ci-lab campaign land` with a real publisher")


def read_requests(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("eid"), str) or \
                not isinstance(row.get("heads"), dict) or not isinstance(row.get("title"), str) or \
                not isinstance(row.get("body"), str) or not (row.get("winner") is None or
                                                             isinstance(row["winner"], str)):
            raise PublishError(f"{path}:{n}: malformed deferred publish request")
        rows.append(row)
    return rows


def replay_deferred(path: Path | str, *, cid: str, ledger: Any, publisher: Any) -> list[dict[str, Any]]:
    """Publish queued rounds in order. Idempotent: rounds already listed in the campaign's
    ``published.jsonl`` are skipped, and the publisher's own ops reconcile against GitHub."""
    base = f"campaigns/{cid}"
    done = {r.get("eid") for r in ledger.read_jsonl(f"{base}/{PUBLISHED}")}
    prefix = f"{cid}-r"
    out = []
    for req in read_requests(path):
        eid = req["eid"]
        if not eid.startswith(prefix):
            raise PublishError(f"request {eid!r} does not belong to campaign {cid!r}")
        if eid in done:
            out.append({"eid": eid, "skipped": True})
            continue
        stack = ledger.read_json(f"{base}/stack.json") or {"layers": [], "stack_number": None}
        result = publisher.publish_round(eid=eid, winner=req["winner"], heads=req["heads"], stack=stack,
                                         title=req["title"], body=req["body"])
        paths = []
        if result.get("stack") != stack:
            ledger.write_json(f"{base}/stack.json", result["stack"])
            paths.append(f"{base}/stack.json")
        ledger.append_jsonl(f"{base}/{PUBLISHED}", {"eid": eid, "winner": req["winner"],
                                                     "layers": len(result["stack"]["layers"])}, key="eid")
        paths.append(f"{base}/{PUBLISHED}")
        ledger.commit(f"Publish {eid}", paths)
        done.add(eid)
        out.append({"eid": eid, "winner": req["winner"], "layers": len(result["stack"]["layers"])})
    return out
