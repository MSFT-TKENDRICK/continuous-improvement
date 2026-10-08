"""Ledger layout under ``experiments/`` (design v1 §2-3, v2 C10/C14/C15).

::

    experiments/
      campaigns/<cid>/campaign.json
      campaigns/<cid>/frontier.json
      campaigns/<cid>/history.jsonl
      campaigns/<cid>/rounds/<eid>/experiment.json
      campaigns/<cid>/rounds/<eid>/decisions.json
      campaigns/<cid>/rounds/<eid>/eval/<name>.json
      sleep/state.json
      sleep/nights/<date>/experiment.json
      holdout-looks.jsonl

Run checkpoints are NOT ledger state (C14): they live in a per-run ephemeral directory
under ``CI_RUN_DIR`` (:func:`run_dir`), never committed, cached or uploaded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from ci_lab.contracts import CAMPAIGN_RE

LEDGER_ROOT = "experiments"
EXPERIMENT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,44}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EVAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


def _check(value: str, regex: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not regex.match(value) or ".." in value:
        raise ValueError(f"bad {what} {value!r}")
    return value


def night_date(value: str | _date) -> str:
    """Normalize a sleep night date to ``YYYY-MM-DD`` (accepts ``YYYYMMDD`` too)."""
    if isinstance(value, _date):
        return value.isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{8}", value):
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    _check(value, DATE_RE, "night date")
    _date.fromisoformat(value)
    return value


@dataclass(frozen=True)
class Layout:
    """Absolute ledger paths for a repository root (a working tree)."""

    repo: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo", Path(self.repo))

    @property
    def root(self) -> Path:
        return self.repo / LEDGER_ROOT

    def rel(self, path: str | os.PathLike[str]) -> str:
        """Repo-relative POSIX path of a ledger path."""
        return Path(path).absolute().relative_to(self.repo.absolute()).as_posix()

    # ------------------------------------------------------------ campaigns

    def campaigns_dir(self) -> Path:
        return self.root / "campaigns"

    def campaign_dir(self, cid: str) -> Path:
        return self.campaigns_dir() / _check(cid, CAMPAIGN_RE, "campaign id")

    def campaign_json(self, cid: str) -> Path:
        return self.campaign_dir(cid) / "campaign.json"

    def frontier_json(self, cid: str) -> Path:
        return self.campaign_dir(cid) / "frontier.json"

    def history_jsonl(self, cid: str) -> Path:
        return self.campaign_dir(cid) / "history.jsonl"

    def rounds_dir(self, cid: str) -> Path:
        return self.campaign_dir(cid) / "rounds"

    def round_dir(self, cid: str, eid: str) -> Path:
        return self.rounds_dir(cid) / _check(eid, EXPERIMENT_RE, "experiment id")

    def experiment_json(self, cid: str, eid: str) -> Path:
        return self.round_dir(cid, eid) / "experiment.json"

    def decisions_json(self, cid: str, eid: str) -> Path:
        return self.round_dir(cid, eid) / "decisions.json"

    def eval_dir(self, cid: str, eid: str) -> Path:
        return self.round_dir(cid, eid) / "eval"

    def eval_json(self, cid: str, eid: str, name: str) -> Path:
        name = name[:-5] if name.endswith(".json") else name
        return self.eval_dir(cid, eid) / f"{_check(name, EVAL_NAME_RE, 'eval name')}.json"

    def campaign_ids(self) -> list[str]:
        d = self.campaigns_dir()
        return sorted(p.name for p in d.iterdir() if p.is_dir() and CAMPAIGN_RE.match(p.name)) if d.is_dir() else []

    def round_ids(self, cid: str) -> list[str]:
        d = self.rounds_dir(cid)
        return sorted(p.name for p in d.iterdir() if p.is_dir() and EXPERIMENT_RE.match(p.name)) if d.is_dir() else []

    # ------------------------------------------------------------ sleep

    def sleep_dir(self) -> Path:
        return self.root / "sleep"

    def sleep_state(self) -> Path:
        return self.sleep_dir() / "state.json"

    def night_dir(self, date: str | _date) -> Path:
        return self.sleep_dir() / "nights" / night_date(date)

    def night_experiment(self, date: str | _date) -> Path:
        return self.night_dir(date) / "experiment.json"

    # ------------------------------------------------------------ global

    def holdout_looks(self) -> Path:
        return self.root / "holdout-looks.jsonl"


def run_root() -> Path:
    """``CI_RUN_DIR`` (default ``<cwd>/artifacts/runs``; ``artifacts/`` is git-ignored)."""
    env = os.environ.get("CI_RUN_DIR")
    return Path(env) if env else Path.cwd() / "artifacts" / "runs"


def run_dir(run_id: str, *, root: str | os.PathLike[str] | None = None, create: bool = True) -> Path:
    """Per-run ephemeral checkpoint directory (C14): resume only on the same machine/run."""
    base = Path(root) if root is not None else run_root()
    path = base / _check(run_id, RUN_ID_RE, "run id")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
