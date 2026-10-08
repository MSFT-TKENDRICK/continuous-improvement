"""Canonical JSON + hash-lock for OES envelopes.

``contentHash`` = ``"sha256:" + sha256(canonical_json(envelope minus contentHash))``.
Canonical JSON: UTF-8, keys sorted, no insignificant whitespace, no NaN/Infinity,
non-ASCII kept literal. Python's float repr is shortest-round-trip and platform
independent, so the hash is stable across Windows/Linux.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

HASH_FIELD = "contentHash"
PREFIX = "sha256:"


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(obj: Any) -> str:
    return PREFIX + hashlib.sha256(canonical_json(obj)).hexdigest()


def content_hash(doc: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in doc.items() if k != HASH_FIELD})


def seal(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``doc`` with ``contentHash`` (re)computed."""
    out = {k: v for k, v in doc.items() if k != HASH_FIELD}
    out[HASH_FIELD] = content_hash(out)
    return out


def verify(doc: Mapping[str, Any]) -> bool:
    return doc.get(HASH_FIELD) == content_hash(doc)
