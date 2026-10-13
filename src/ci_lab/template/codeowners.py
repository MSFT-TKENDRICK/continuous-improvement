"""Deterministic CODEOWNERS owner rewrite (STDLIB ONLY)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

CODEOWNERS_REL = ".github/CODEOWNERS"
# @user, @org/team, or an email address (the three owner forms GitHub accepts)
OWNER_RE = re.compile(
    r"^(?:@[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,99})?"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})$")


class OwnersError(ValueError):
    pass


def parse_owners(values: Iterable[str]) -> tuple[str, ...]:
    """Split on whitespace/commas, validate each owner, drop duplicates (first wins)."""
    out: list[str] = []
    for value in values:
        for tok in re.split(r"[\s,]+", value.strip()):
            if not tok:
                continue
            if not OWNER_RE.match(tok):
                raise OwnersError(f"invalid CODEOWNERS owner {tok!r}: use @user, @org/team or an email")
            if tok not in out:
                out.append(tok)
    if not out:
        raise OwnersError("no owners given")
    return tuple(out)


def _split_rule(body: str) -> tuple[str, list[str], str]:
    """``pattern owner... [# comment]`` -> (pattern, owners, trailing comment incl. leading space)."""
    comment = ""
    m = re.search(r"\s+#", body)
    if m:
        body, comment = body[:m.start()], body[m.start():]
    parts = body.split()
    return parts[0], parts[1:], comment


def rules(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            pattern, owners, _ = _split_rule(s)
            out.append((pattern, owners))
    return out


@dataclass
class Rewrite:
    text: str
    changed: list[str] = field(default_factory=list)      # patterns whose owners were replaced
    kept_custom: list[str] = field(default_factory=list)  # patterns with owners outside the replaceable set


def rewrite_owners(text: str, replaceable: Iterable[str], owners: Sequence[str]) -> Rewrite:
    """Replace the owners of every rule whose owners all belong to ``replaceable`` with ``owners``.

    Rules that already carry other owners (a human customized them) are left untouched and reported.
    Column alignment, comments, blank lines and the file's newline style are preserved.
    """
    replace = set(replaceable)
    new_owners = list(owners)
    res = Rewrite(text="")
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        eol = line[len(body):]
        s = body.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue
        pattern, cur, comment = _split_rule(body.strip())
        if not cur or cur == new_owners:
            out.append(line)
            continue
        if not set(cur) <= replace:
            res.kept_custom.append(pattern)
            out.append(line)
            continue
        indent = body[:len(body) - len(body.lstrip())]
        col = body.find(cur[0], body.find(pattern) + len(pattern))
        pad = max(col - len(indent) - len(pattern), 1)
        out.append(f"{indent}{pattern}{' ' * pad}{' '.join(new_owners)}{comment}{eol}")
        res.changed.append(pattern)
    res.text = "".join(out)
    return res
