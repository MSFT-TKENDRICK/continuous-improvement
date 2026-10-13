"""Sealed rubric vault (bus contract v2 §5): content-addressed, tamper-evident, student-denied.

Rubrics live at ``<run_dir>/sealed/<commitment>.json`` as their canonical JSON, so the file's
sha256 *is* the commitment a :class:`~ci_lab.taskgraph.model.Deliverable` carries. The
``sealed`` directory is on ``ci_lab.tools.paths.DENIED_GLOBS`` so arm FS tools cannot read it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import tempfile
from pathlib import Path

from ci_lab.taskgraph.model import Rubric, canonical_json

__all__ = [
    "SEALED_DIRNAME",
    "RubricVault",
    "VaultDenied",
    "VaultError",
    "VaultMissing",
    "VaultTampered",
    "new_canary",
]

SEALED_DIRNAME = "sealed"
READER_ROLES = frozenset({"orchestrator", "planner", "examiner", "voter", "judge", "adversary", "hardener"})
_COMMITMENT = re.compile(r"^[0-9a-f]{64}$")


class VaultError(Exception):
    """Base class for vault failures."""


class VaultDenied(VaultError):
    """The caller's role may not read sealed rubrics (students, unknown roles)."""


class VaultMissing(VaultError):
    """No sealed rubric for the commitment."""


class VaultTampered(VaultError):
    """A sealed file no longer hashes to its commitment."""


def new_canary(rng: random.Random | None = None) -> str:
    """16 lowercase hex chars; ``secrets`` unless a seeded ``rng`` is given (tests)."""
    return f"{rng.getrandbits(64):016x}" if rng is not None else secrets.token_hex(8)


class RubricVault:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def for_run(cls, run_dir: Path | str) -> RubricVault:
        return cls(Path(run_dir) / SEALED_DIRNAME)

    def _path(self, commitment: str) -> Path:
        if not isinstance(commitment, str) or not _COMMITMENT.match(commitment):
            raise VaultMissing(f"not a rubric commitment: {commitment!r}")
        return self.root / f"{commitment}.json"

    def seal(self, rubric: Rubric) -> str:
        """Store ``rubric`` (idempotent, atomic tmp+replace); returns its commitment."""
        data = canonical_json(rubric.to_json()).encode("utf-8")
        commitment = hashlib.sha256(data).hexdigest()
        path = self._path(commitment)
        if path.is_file() and path.read_bytes() == data:
            return commitment
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".seal-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return commitment

    def open(self, commitment: str, *, role: str) -> Rubric:
        if role not in READER_ROLES:
            raise VaultDenied(f"role {role!r} may not open sealed rubrics")
        path = self._path(commitment)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise VaultMissing(f"no sealed rubric {commitment}") from None
        if hashlib.sha256(data).hexdigest() != commitment:
            raise VaultTampered(f"sealed rubric {commitment} does not match its hash")
        try:
            rubric = Rubric.from_json(json.loads(data.decode("utf-8")))
        except ValueError as exc:
            raise VaultTampered(f"sealed rubric {commitment} is malformed: {exc}") from exc
        if rubric.commitment() != commitment:
            raise VaultTampered(f"sealed rubric {commitment} is not in canonical form")
        return rubric

    def commitments(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json") if _COMMITMENT.match(p.stem))

    def versions(self, rubric_id: str, *, role: str = "orchestrator") -> list[Rubric]:
        """Every sealed version of ``rubric_id``, ascending by version (fails closed on tamper)."""
        found = [r for r in (self.open(c, role=role) for c in self.commitments()) if r.id == rubric_id]
        return sorted(found, key=lambda r: (r.version, r.commitment()))
