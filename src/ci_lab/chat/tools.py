"""Experiment-designer tools: pure functions over a :class:`ChatConfig`, plus their MAF wrappers.

Every public method of :class:`ChatTools` returns a JSON-able dict and never raises for bad
input; validation problems come back as ``{"ok": false, "errors": [...]}`` so the model can
fix them. :func:`maf_tools` wraps the methods as MAF ``FunctionTool``s; ``launch_campaign``
is ``approval_mode="always_require"`` so the AG-UI server refuses to run it until a human
approves the call (docs/chat.md).
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from ci_lab.campaign.defaults import DEFAULT_HYPER
from ci_lab.contracts import CAMPAIGN_RE, STRATEGIES, TEXT_COMPONENTS
from ci_lab.domain import DEFAULT_DOMAIN

__all__ = ["MAX_ARMS", "WORKFLOW_FILE", "ChatConfig", "ChatTools", "case_counts", "hyper_descriptions",
           "maf_tools"]

MAX_ARMS = 8
WORKFLOW_FILE = "campaign-scheduled.yml"
WORKFLOW_MAX_ROUNDS = 9  # campaign-scheduled.yml rejects rounds outside 1..9
TARGETS = ("local", "workflow")
_RUN_URL_RE = re.compile(r"https://github\.com/\S+/actions/runs/\d+")

_INT_KEYS = ("arms", "k", "max_rounds", "aa_repeats", "max_parallel_arms", "max_arm_attempts", "holdout_looks",
             "seed")
_OPT_POS_INT_KEYS = ("budget", "b_min", "b_max", "budget_tokens", "arm_budget_tokens", "guard_trials")
_EXTRA_DESCRIPTIONS = {
    "draft_prs": "open each shipped round as a stacked draft PR",
}
_CHAT_LIMITS = {
    "arms": f"1..{MAX_ARMS} (chat limit)",
    "aa_repeats": ">= 2 (A/A calibration needs a spread)",
    "k": ">= 1",
    "max_rounds": "1..99; rounds requested in a draft must not exceed it",
    "strategies": f"non-empty subset of {list(STRATEGIES)}",
    "heartbeat_s": "(0, 60]",
}


@dataclass
class ChatConfig:
    """Where the chat server reads and writes, and how it launches campaigns."""

    run_root: Path
    chat_dir: Path
    campaign_profile: str = "fake"
    repo: str | None = None
    ledger_dir: Path | None = None
    repo_root: Path = field(default_factory=Path.cwd)
    dry_run_launch: bool = False
    dry_run_publish: bool = True
    python: str = sys.executable
    gh: str = "gh"

    @property
    def ledger_root(self) -> Path:
        if self.ledger_dir is not None:
            return Path(self.ledger_dir)
        if self.campaign_profile == "fake":
            return Path(self.run_root) / "_fake" / "experiments"
        return Path(self.repo_root) / "experiments"

    @property
    def drafts_dir(self) -> Path:
        return Path(self.chat_dir) / "drafts"

    @property
    def launches_dir(self) -> Path:
        return Path(self.chat_dir) / "launches"


@functools.lru_cache(maxsize=4)
def case_counts(repo_root: str) -> dict[str, dict[str, int]]:
    """``{suite: {"evolve": n, "heldout": n, "ood": n}}`` of the frozen ASSERT sets (same splits as campaigns)."""
    from ci_lab.domain.harness import HarnessDomain, load_cases

    root = Path(repo_root)
    dataset = root / "evals" / "datasets" / "harness.yaml"
    cases = load_cases(root, dataset)
    if not cases:
        return {}
    domain = HarnessDomain(repo_root=root, dataset_path=dataset, cases=cases)
    suite_of = {c.case_id: c.suite.removeprefix("harness_") for c in cases}
    out: dict[str, dict[str, int]] = {}
    for split, ids in domain.splits().items():
        if split not in ("evolve", "heldout", "ood"):
            continue
        for cid in ids:
            row = out.setdefault(suite_of[cid], {"evolve": 0, "heldout": 0, "ood": 0})
            row[split] = row.get(split, 0) + 1
    return dict(sorted(out.items()))


@functools.lru_cache(maxsize=1)
def hyper_descriptions() -> dict[str, str]:
    """Descriptions of ``DEFAULT_HYPER`` keys, taken from the inline comments in ``campaign/defaults.py``."""
    from ci_lab.campaign import defaults

    text = Path(defaults.__file__).read_text(encoding="utf-8")
    found = dict(re.findall(r'^\s+"(\w+)":.*?#\s*(.+?)\s*$', text, flags=re.MULTILINE))
    return {k: found.get(k) or _EXTRA_DESCRIPTIONS.get(k, "") for k in DEFAULT_HYPER}


def _now() -> float:
    return round(time.time(), 3)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class ChatTools:
    """The designer's tool surface. Stateless apart from the files under ``config.chat_dir``."""

    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ read-only tools
    def list_strategies(self) -> dict[str, Any]:
        return {"strategies": list(STRATEGIES), "text_components": list(TEXT_COMPONENTS),
                "note": "RRSI Thompson-allocates strategies to arms each round; 'guard' arms run a paired "
                        "guard-off/on experiment instead of editing text components."}

    def list_suites(self) -> dict[str, Any]:
        counts = case_counts(str(self.config.repo_root))
        totals = {"evolve": 0, "heldout": 0, "ood": 0}
        suites = []
        for suite, splits in counts.items():
            for k in totals:
                totals[k] += splits.get(k, 0)
            suites.append({"suite": suite, "cases": sum(splits.values()), "splits": dict(splits)})
        return {"suites": suites, "totals": {**totals, "all": sum(totals.values())},
                "note": "Rounds and A/A calibration evaluate the evolve split; held-out looks use heldout; "
                        "ood is reported only."}

    def get_default_hyperparameters(self) -> dict[str, Any]:
        desc = hyper_descriptions()
        return {"hyperparameters": {k: {"default": v, "description": desc.get(k, ""),
                                        **({"limits": _CHAT_LIMITS[k]} if k in _CHAT_LIMITS else {})}
                                    for k, v in DEFAULT_HYPER.items()}}

    def list_campaigns(self) -> dict[str, Any]:
        campaigns: dict[str, dict[str, Any]] = {}
        root = self.config.ledger_root / "campaigns"
        if root.is_dir():
            for meta_path in sorted(root.glob("*/campaign.json")):
                cid = meta_path.parent.name
                summary = self._ledger_summary(cid)
                if summary is not None:
                    campaigns[cid] = {"cid": cid, "source": "ledger", **summary}
        drafts = self.config.drafts_dir
        if drafts.is_dir():
            for path in sorted(drafts.glob("*.json")):
                draft = _read_json(path)
                if not isinstance(draft, dict) or not CAMPAIGN_RE.match(path.stem):
                    continue
                entry = campaigns.setdefault(path.stem, {"cid": path.stem, "source": "draft"})
                entry["draft"] = {"target": draft.get("target"), "rounds": draft.get("rounds"),
                                  "created": draft.get("created")}
                launch = self._launch_record(path.stem)
                if launch:
                    entry["launch"] = launch
        return {"campaigns": list(campaigns.values()), "ledger_root": str(self.config.ledger_root),
                "run_root": str(self.config.run_root)}

    def campaign_status(self, cid: str) -> dict[str, Any]:
        if not isinstance(cid, str) or not CAMPAIGN_RE.match(cid):
            return {"ok": False, "errors": [f"bad campaign id {cid!r}: must match {CAMPAIGN_RE.pattern}"]}
        out: dict[str, Any] = {"ok": True, "cid": cid}
        draft = _read_json(self.config.drafts_dir / f"{cid}.json")
        if draft is not None:
            out["draft"] = draft
        launch = self._launch_record(cid)
        if launch:
            out["launch"] = launch
            if launch.get("log") and Path(launch["log"]).is_file():
                out["log_tail"] = Path(launch["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        summary = self._ledger_summary(cid)
        if summary is not None:
            out["campaign"] = summary
        if len(out) == 2:
            return {"ok": False, "errors": [f"no draft, launch or campaign named {cid!r}"]}
        return out

    # ------------------------------------------------------------------ draft
    def draft_campaign(self, cid: str, hyper: Mapping[str, Any] | None, rounds: int, target: str,
                       rationale: str) -> dict[str, Any]:
        """Validate and persist a draft to ``<chat_dir>/drafts/<cid>.json``."""
        result = self.validate_draft(cid, hyper, rounds, target, rationale)
        if result["ok"]:
            path = self.config.drafts_dir / f"{cid}.json"
            _write_json(path, result["draft"])
            result["path"] = str(path)
        return result

    def validate_draft(self, cid: str, hyper: Mapping[str, Any] | None, rounds: int, target: str,
                       rationale: str) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if not isinstance(cid, str) or not CAMPAIGN_RE.match(cid):
            errors.append(f"bad campaign id {cid!r}: use 3-41 chars of a-z, 0-9 and '-', starting "
                          f"with a letter or digit ({CAMPAIGN_RE.pattern})")
        else:
            if (self.config.ledger_root / "campaigns" / cid).exists():
                errors.append(f"campaign {cid!r} already exists in {self.config.ledger_root}; pick a new id")
            launch = self._launch_record(cid)
            if launch and launch.get("status") not in ("failed", "dry_run"):
                errors.append(f"campaign {cid!r} was already launched ({launch.get('status')}); pick a new id")
        if target not in TARGETS:
            errors.append(f"target must be one of {list(TARGETS)}, got {target!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("rationale must be a non-empty string")
        if hyper is None:
            hyper = {}
        if not isinstance(hyper, Mapping):
            errors.append("hyper must be an object of hyperparameter overrides")
            hyper = {}
        normalized, hyper_errors = self.normalize_hyper(hyper)
        errors += hyper_errors
        if not _is_int(rounds):
            errors.append(f"rounds must be an integer, got {rounds!r}")
        elif normalized is not None and not 1 <= rounds <= int(normalized["max_rounds"]):
            errors.append(f"rounds must be in 1..max_rounds ({normalized['max_rounds']}), got {rounds}")
        overrides = {k: v for k, v in (normalized or {}).items() if v != DEFAULT_HYPER[k]}
        if target == "workflow":
            if _is_int(rounds) and not 1 <= rounds <= WORKFLOW_MAX_ROUNDS:
                errors.append(f"the workflow target accepts rounds 1..{WORKFLOW_MAX_ROUNDS}")
            if overrides:
                warnings.append(f"{WORKFLOW_FILE} takes only cid and rounds; it runs the copilot profile with "
                                f"default hyperparameters, so these overrides are IGNORED: {sorted(overrides)}. "
                                "Use target 'local' to apply them.")
        if errors or normalized is None:
            return {"ok": False, "errors": errors, "warnings": warnings}

        workflow = target == "workflow"
        effective = dict(DEFAULT_HYPER) if workflow else normalized
        draft: dict[str, Any] = {
            "cid": cid, "target": target, "rounds": rounds, "rationale": rationale.strip(),
            "domain": DEFAULT_DOMAIN,
            "hyper": effective, "overrides": {} if workflow else overrides,
            "ignored_overrides": overrides if workflow else {},
            "estimate": self.estimate(effective, rounds),
            "profile": "copilot" if workflow else self.config.campaign_profile,
            "created": _now(), "warnings": warnings,
        }
        draft["commands"] = self.launch_commands(draft)
        return {"ok": True, "draft": draft, "warnings": warnings}

    def normalize_hyper(self, requested: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        """Merge ``requested`` over ``DEFAULT_HYPER`` with type and range checks (stricter than ``merge_hyper``)."""
        from ci_lab.campaign.driver import merge_hyper

        errors: list[str] = []
        unknown = sorted(set(requested) - set(DEFAULT_HYPER))
        if unknown:
            errors.append(f"unknown hyperparameters: {unknown} (known: {sorted(DEFAULT_HYPER)})")
        h = {k: v for k, v in requested.items() if k in DEFAULT_HYPER}
        for key in _INT_KEYS:
            if key in h and not _is_int(h[key]):
                errors.append(f"{key} must be an integer, got {h[key]!r}")
        for key in _OPT_POS_INT_KEYS:
            if key in h and h[key] is not None and not (_is_int(h[key]) and h[key] >= 1):
                errors.append(f"{key} must be null or an integer >= 1, got {h[key]!r}")
        for key in ("heartbeat_s", "guard_margin"):
            if key in h and not _is_num(h[key]):
                errors.append(f"{key} must be a number, got {h[key]!r}")
        if _is_num(h.get("guard_margin")) and h["guard_margin"] < 0:
            errors.append("guard_margin must be >= 0")
        if "draft_prs" in h and not isinstance(h["draft_prs"], bool):
            errors.append("draft_prs must be a boolean")
        if h.get("guard_stochastic") is not None and not isinstance(h["guard_stochastic"], bool):
            errors.append("guard_stochastic must be null or a boolean")
        if "rrsi_profile" in h and not isinstance(h["rrsi_profile"], str):
            errors.append("rrsi_profile must be a string")
        if "rrsi" in h and not isinstance(h["rrsi"], Mapping):
            errors.append("rrsi must be an object")
        if "strategies" in h:
            s = h["strategies"]
            if isinstance(s, str):
                s = [x.strip() for x in s.split(",") if x.strip()]
            if not isinstance(s, list) or not s or not all(isinstance(x, str) for x in s) \
                    or not set(s) <= set(STRATEGIES):
                errors.append(f"strategies must be a non-empty subset of {list(STRATEGIES)}, got {h['strategies']!r}")
            else:
                h["strategies"] = list(dict.fromkeys(s))
        if errors:
            return None, errors
        try:
            merged = merge_hyper(h)
        except (ValueError, TypeError) as exc:
            return None, [str(exc)]
        if not 1 <= merged["arms"] <= MAX_ARMS:
            errors.append(f"arms must be in 1..{MAX_ARMS}, got {merged['arms']}")
        if merged["aa_repeats"] < 2:
            errors.append(f"aa_repeats must be >= 2 for A/A calibration, got {merged['aa_repeats']}")
        if merged["b_min"] is not None and merged["b_max"] is not None and merged["b_min"] > merged["b_max"]:
            errors.append("b_min must be <= b_max")
        return (None, errors) if errors else (merged, [])

    def estimate(self, hyper: Mapping[str, Any], rounds: int) -> dict[str, Any]:
        """Case evaluations: ``(aa_repeats + rounds*arms) * cases * k`` over the evolve split, plus the
        per-round incumbent re-evaluation and the (separate, not launched here) held-out confirm look."""
        counts = case_counts(str(self.config.repo_root))
        evolve = sum(s.get("evolve", 0) for s in counts.values())
        heldout = sum(s.get("heldout", 0) for s in counts.values())
        k, arms, aa = int(hyper["k"]), int(hyper["arms"]), int(hyper["aa_repeats"])
        calibration = aa * evolve * k
        arm_evals = rounds * arms * evolve * k
        incumbent = rounds * evolve * k
        return {
            "formula": "(aa_repeats + rounds * arms) * cases * k",
            "cases": evolve, "k": k, "arms": arms, "rounds": rounds, "aa_repeats": aa,
            "evaluations": calibration + arm_evals,
            "breakdown": {"aa_calibration": calibration, "arms": arm_evals, "incumbent_reevaluation": incumbent},
            "evaluations_with_incumbent": calibration + arm_evals + incumbent,
            "heldout_confirm_per_look": 2 * heldout * k,
            "notes": ["cases = evolve-split cases of the frozen ASSERT sets",
                      ("each evaluation is one agent rollout plus judging; guard arms run paired guard-off/on "
                       "trials and may cost more"),
                      "the held-out confirm look (`campaign confirm`) is not part of the launch chain"],
        }

    # ------------------------------------------------------------------ launch
    def launch_commands(self, draft: Mapping[str, Any]) -> list[list[str]]:
        """The argv lists a launch runs: the local ``campaign new -> calibrate -> run`` chain, or ``gh workflow run``."""
        cfg = self.config
        cid = str(draft["cid"])
        if draft["target"] == "workflow":
            argv = [cfg.gh, "workflow", "run", WORKFLOW_FILE, "-f", f"cid={cid}", "-f", f"rounds={draft['rounds']}"]
            if cfg.repo:
                argv += ["--repo", cfg.repo]
            return [argv]
        common = ["--profile", str(draft.get("profile") or cfg.campaign_profile),
                  "--domain", DEFAULT_DOMAIN, "--run-dir", str(cfg.run_root)]
        if cfg.ledger_dir is not None:
            common += ["--ledger-dir", str(cfg.ledger_dir)]
        if cfg.repo:
            common += ["--repo", cfg.repo]
        if cfg.dry_run_publish:
            common += ["--dry-run-publish"]
        base = [cfg.python, "-m", "ci_lab.cli", "campaign"]
        hyper_args = [a for k, v in sorted(dict(draft.get("overrides") or {}).items())
                      for a in ("--hyper", f"{k}={json.dumps(v, separators=(',', ':'))}")]
        return [[*base, "new", cid, *common, *hyper_args],
                [*base, "calibrate", cid, *common],
                [*base, "run", cid, *common, "--rounds", str(draft["rounds"])]]

    def launch_campaign(self, cid: str) -> dict[str, Any]:
        """Launch the stored draft ``cid`` (re-validated). Gated by human approval in :func:`maf_tools`."""
        cfg = self.config
        if not isinstance(cid, str) or not CAMPAIGN_RE.match(cid):
            return {"ok": False, "launched": False, "errors": [f"bad campaign id {cid!r}"]}
        path = cfg.drafts_dir / f"{cid}.json"
        stored = _read_json(path)
        if not isinstance(stored, dict):
            return {"ok": False, "launched": False, "errors": [f"no draft for {cid!r}; call draft_campaign first"]}
        # Re-validate: the ledger, the launch registry and the limits may have changed since drafting.
        check = self.validate_draft(cid, stored.get("overrides") or stored.get("ignored_overrides") or {},
                                    stored.get("rounds"), stored.get("target"), stored.get("rationale") or "")
        if not check["ok"]:
            return {"ok": False, "launched": False, "errors": check["errors"]}
        draft = check["draft"]
        if draft["target"] == "workflow":
            argv = draft["commands"][0]
        else:
            argv = [cfg.python, "-m", "ci_lab.chat.launch", str(path)]
        record: dict[str, Any] = {"cid": cid, "target": draft["target"], "created": _now(), "draft": str(path),
                                  "argv": argv}
        result: dict[str, Any] = {"ok": True, "cid": cid, "target": draft["target"], "argv": argv,
                                  "commands": draft["commands"], "warnings": draft["warnings"]}
        if draft["target"] == "workflow":
            result["note"] = (f"{WORKFLOW_FILE} accepts only cid and rounds; it runs the copilot profile with "
                              "default hyperparameters")
            if draft["ignored_overrides"]:
                result["ignored_overrides"] = draft["ignored_overrides"]

        def finish(status: str, launched: bool, **extra: Any) -> dict[str, Any]:
            record.update(status=status, **{k: v for k, v in extra.items() if k in ("pid", "log", "run_url",
                                                                                   "error")})
            _write_json(cfg.launches_dir / f"{cid}.json", record)
            out = {**result, **{k: v for k, v in extra.items() if k != "error"}, "launched": launched,
                   "status": status, "launches": self.launches()}
            if "error" in extra:
                out.update(ok=False, errors=[extra["error"]])
            return out

        if cfg.dry_run_launch:
            return finish("dry_run", False, dry_run=True,
                          message="dry-run: recorded the command; nothing was executed")
        if draft["target"] == "workflow":
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=cfg.repo_root,
                                      env=_child_env(), check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return finish("failed", False, error=str(exc))
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                return finish("failed", False, error=f"gh exited {proc.returncode}: {out.strip()[-500:]}")
            m = _RUN_URL_RE.search(out)
            run_url = m.group(0) if m else (f"https://github.com/{cfg.repo}/actions/workflows/{WORKFLOW_FILE}"
                                            if cfg.repo else None)
            return finish("dispatched", True, run_url=run_url)
        log = cfg.launches_dir / f"{cid}.log"
        try:
            pid = _spawn_detached(argv, log, cwd=Path(cfg.repo_root))
        except OSError as exc:
            return finish("failed", False, error=str(exc))
        return finish("started", True, pid=pid, log=str(log))

    def launches(self) -> list[dict[str, Any]]:
        """Shared-state view of every launch: ``{cid, target, status, pid?, log?, run_url?}``."""
        out = []
        d = self.config.launches_dir
        if d.is_dir():
            for path in sorted(d.glob("*.json")):
                if path.name.endswith(".status.json"):
                    continue
                rec = self._launch_record(path.stem)
                if rec:
                    out.append({k: rec[k] for k in ("cid", "target", "status", "pid", "log", "run_url")
                                if rec.get(k) is not None})
        return out

    # ------------------------------------------------------------------ helpers
    def _launch_record(self, cid: str) -> dict[str, Any] | None:
        rec = _read_json(self.config.launches_dir / f"{cid}.json")
        if not isinstance(rec, dict):
            return None
        progress = _read_json(self.config.launches_dir / f"{cid}.status.json")
        if isinstance(progress, dict) and rec.get("status") == "started":
            rec["status"] = progress.get("status", rec["status"])
            rec["step"] = progress.get("step")
            if progress.get("returncode") is not None:
                rec["returncode"] = progress["returncode"]
        return rec

    def _ledger_summary(self, cid: str) -> dict[str, Any] | None:
        base = self.config.ledger_root / "campaigns" / cid
        meta = _read_json(base / "campaign.json")
        if not isinstance(meta, dict):
            return None
        cal = _read_json(base / "calibration.json")
        history = []
        hist_path = base / "history.jsonl"
        if hist_path.is_file():
            for line in hist_path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError):
                    history.append(json.loads(line))
        return {"profile": meta.get("profile"), "hyper": meta.get("hyper"),
                "calibrated": cal is not None, "delta": (cal or {}).get("delta"),
                "frontier": _read_json(base / "frontier.json"),
                "rounds": [{k: h.get(k) for k in ("eid", "decision", "winner")} for h in history],
                "confirmed": (base / "confirm.json").is_file(), "landed": (base / "land.json").is_file()}


def _child_env() -> dict[str, str]:
    """The server's environment minus the chat token (children never need it)."""
    return {k: v for k, v in os.environ.items() if k != "CI_CHAT_TOKEN"}


def _spawn_detached(argv: Sequence[str], log: Path, *, cwd: Path) -> int:
    """Start ``argv`` (no shell) detached from the server so it survives the server's exit."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=cwd, env=_child_env(), close_fds=True, **kwargs)
    return proc.pid


# ---------------------------------------------------------------------- MAF wrappers
def maf_tools(tools: ChatTools) -> list[Any]:
    """MAF ``FunctionTool``s over ``tools``. ``draft_campaign``/``launch_campaign`` push AG-UI shared state."""
    from agent_framework import tool
    from agent_framework_ag_ui import state_update

    def dump(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, default=str)

    def list_strategies() -> str:
        """List the arm strategies and text components a campaign can use."""
        return dump(tools.list_strategies())

    def list_suites() -> str:
        """List the frozen ASSERT suites with case counts per split (evolve, heldout, ood)."""
        return dump(tools.list_suites())

    def get_default_hyperparameters() -> str:
        """Return every campaign hyperparameter with its default value, description and limits."""
        return dump(tools.get_default_hyperparameters())

    def list_campaigns() -> str:
        """List existing campaigns, drafts and launches."""
        return dump(tools.list_campaigns())

    def campaign_status(cid: Annotated[str, Field(description="campaign id")]) -> str:
        """Report the draft, launch progress and ledger state of one campaign."""
        return dump(tools.campaign_status(cid))

    def draft_campaign(
        cid: Annotated[str, Field(description="new campaign id: 3-41 chars of a-z, 0-9, '-'")],
        hyper: Annotated[dict[str, Any], Field(description="hyperparameter overrides; {} keeps the defaults")],
        rounds: Annotated[int, Field(description="RRSI rounds to run after calibration (1..max_rounds)")],
        target: Annotated[Literal["local", "workflow"],
                          Field(description="'local' runs on this machine; 'workflow' dispatches "
                                            "campaign-scheduled.yml (cid and rounds only)")],
        rationale: Annotated[str, Field(description="why this design answers the user's question")],
    ) -> Any:
        """Validate and save a campaign draft with its cost estimate. Required before launch_campaign."""
        result = tools.draft_campaign(cid, hyper, rounds, target, rationale)
        if not result["ok"]:
            return dump(result)
        return state_update(dump(result), state={"draft": result["draft"]})

    def launch_campaign(cid: Annotated[str, Field(description="id of a campaign drafted with draft_campaign")]) -> Any:
        """Launch a drafted campaign. Requires explicit human approval; only launches a validated draft."""
        result = tools.launch_campaign(cid)
        launches = result.pop("launches", None)
        if launches is None:
            return dump(result)
        return state_update(dump(result), state={"launches": launches})

    plain = [list_strategies, list_suites, get_default_hyperparameters, list_campaigns, campaign_status,
             draft_campaign]
    return [*(tool(f) for f in plain), tool(launch_campaign, approval_mode="always_require")]
