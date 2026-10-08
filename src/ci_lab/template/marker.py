"""Template marker ``.github/template.yml`` (STDLIB ONLY).

The file is YAML restricted to one ``key: <JSON value>`` per line (plus ``#`` comments), so it is read
here without PyYAML and still loads with ``yaml.safe_load``. In the template repository ``role`` is
``"template"``; ``ci-lab template init --apply`` rewrites it in a derived repository with
``role: "derived"`` and ``initialized: true``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MARKER_REL = ".github/template.yml"
TEMPLATE_REPO = "MSFT-TKENDRICK/continuous-improvement"
TEMPLATE_OWNERS = ("@MSFT-TKENDRICK",)
FORMAT = "ci-lab.template/1"
ROLES = ("template", "derived")

_KEY_RE = re.compile(r"^([a-z][a-z0-9_]*):\s*(.*)$")
_ORDER = ("format", "role", "initialized", "repository", "owners", "template_repository", "template_owners",
          "template_commit", "template_version", "source_commit", "initialized_on", "updated_on", "reset_state")
_HEADER = {
    "template": (
        "# Template marker read by `ci-lab template init|doctor` (docs/template.md).",
        "# This is the template repository itself: `ci-lab template init` refuses to run here.",
        "# In a repository created from it, `ci-lab template init --apply` rewrites this file.",
        "# One `key: <JSON value>` per line, so stdlib tools can parse it.",
    ),
    "derived": (
        "# Template marker written by `ci-lab template init --apply` (docs/template.md).",
        "# This repository was created from the template below; `ci-lab template doctor` checks it.",
        "# One `key: <JSON value>` per line, so stdlib tools can parse it.",
    ),
}


class MarkerError(ValueError):
    pass


def parse_marker(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _KEY_RE.match(line)
        if not m:
            raise MarkerError(f"{MARKER_REL}:{n}: expected `key: <JSON value>`")
        key, value = m.groups()
        try:
            out[key] = json.loads(value) if value else None
        except json.JSONDecodeError as e:
            raise MarkerError(f"{MARKER_REL}:{n}: value of {key!r} is not JSON ({e.msg})") from None
    role = out.get("role")
    if role is not None and role not in ROLES:
        raise MarkerError(f"{MARKER_REL}: role must be one of {ROLES}, got {role!r}")
    return out


def read_marker(root: Path) -> dict[str, Any] | None:
    path = root / MARKER_REL
    if not path.is_file():
        return None
    return parse_marker(path.read_text(encoding="utf-8"))


def render_marker(data: Mapping[str, Any]) -> str:
    role = str(data.get("role") or "derived")
    lines = list(_HEADER.get(role, _HEADER["derived"]))
    keys = [k for k in _ORDER if k in data] + sorted(k for k in data if k not in _ORDER)
    for key in keys:
        lines.append(f"{key}: {json.dumps(data[key], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def template_identity(marker: Mapping[str, Any] | None) -> tuple[str, tuple[str, ...]]:
    """``(template repo, template owners)``: the marker's values, else the built-in defaults."""
    marker = marker or {}
    repo = marker.get("template_repository") or TEMPLATE_REPO
    owners = marker.get("template_owners") or list(TEMPLATE_OWNERS)
    return str(repo), tuple(str(o) for o in owners)


def expected_owners(marker: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Owners every CODEOWNERS rule must carry: the derived repo's owners, else the template's."""
    if marker and marker.get("role") == "derived" and marker.get("owners"):
        return tuple(str(o) for o in marker["owners"])
    return template_identity(marker)[1]
