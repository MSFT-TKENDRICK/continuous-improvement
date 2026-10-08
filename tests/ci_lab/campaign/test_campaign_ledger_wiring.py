"""21b: decision-time ``ledger.decisions.record_decisions``, C15 looks via ``ledger.looks``,
and the ``com.microsoft.ci.guard`` OES extension on round envelopes."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from ci_lab.campaign.defaults import build_envelope
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.fakes import fake_deps
from ci_lab.campaign.local import FileLedger
from ci_lab.contracts import Profile
from ci_lab.ledger.decisions import read_decisions
from ci_lab.ledger.layout import Layout
from ci_lab.ledger.looks import (
    LookBudgetExceeded,
    count_looks,
    looks,
    planned_looks,
    record_look,
)
from ci_lab.oes.models import GUARD_EXT
from ci_lab.oes.validate import EXTENSION_SCHEMAS, _schema_errors
from ci_lab.workflows.steps import (
    CampaignEnv,
    HoldoutExhausted,
    RoundContext,
    dataset_hash,
    guard_envelope_extension,
    reserve_holdout_look,
)

CID = "wire-camp"


def _round(tmp_path: Path, strategies: Sequence[str] = ("agent", "guard")):
    deps = fake_deps(tmp_path)

    def violations(edits: Sequence[str], case: str, trial: int) -> str | None:
        guarded = os.environ.get("CI_GUARDS") == "enforce" and any("tweak v2" in e for e in edits)
        return None if case != "c1" or guarded else "stub.unsafe"

    deps.domain.violation_fn = violations
    camp = Campaign.new(CID, "fake", {"arms": 2, "aa_repeats": 2, "strategies": list(strategies)},
                        deps=deps, run_root=tmp_path / "runs")
    asyncio.run(camp.calibrate())
    asyncio.run(camp.run(rounds=1))
    return camp, deps


def test_round_decisions_go_through_ledger_decisions(tmp_path: Path) -> None:
    _round(tmp_path, ("agent", "agent"))
    eid = f"{CID}-r01"
    data = read_decisions(tmp_path, CID, eid)
    assert data is not None and data["decision"] in ("ship", "do_not_ship")
    sel = json.loads((tmp_path / "runs" / eid / "selection.json").read_text())
    assert data == sel
    assert Layout(tmp_path).decisions_json(CID, eid).is_file()


def test_file_ledger_record_decisions_checks_verdict(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")  # root need not be named experiments
    rel = ledger.record_decisions(CID, f"{CID}-r02", {"decision": "rerun", "winner": None})
    assert rel == f"campaigns/{CID}/rounds/{CID}-r02/decisions.json"
    assert ledger.read_json(rel) == {"decision": "rerun", "winner": None}
    with pytest.raises(ValueError, match="decision must be one of"):
        ledger.record_decisions(CID, f"{CID}-r03", {"decision": "maybe"})


def test_file_ledger_record_look_is_c15(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "experiments")
    h = "a" * 64
    rec = ledger.record_look(h, experiment_id="c-one-confirm", planned=1, campaign_id="c-one")
    assert rec["look_no"] == 1 and rec["planned_looks"] == 1 and rec["campaign_id"] == "c-one"
    assert ledger.record_look(h, experiment_id="c-one-confirm", planned=1) == rec  # idempotent
    with pytest.raises(LookBudgetExceeded):
        ledger.record_look(h, experiment_id="c-two-confirm", planned=1)
    assert looks(tmp_path / "experiments" / "holdout-looks.jsonl", h) == [rec]


def test_confirm_look_is_recorded_by_ledger_looks(tmp_path: Path) -> None:
    camp, _ = _round(tmp_path, ("agent", "agent"))
    asyncio.run(camp.confirm())
    [line] = (tmp_path / "experiments" / "holdout-looks.jsonl").read_text().splitlines()
    rec = json.loads(line)
    assert rec["experiment_id"] == f"{CID}-confirm" and rec["campaign_id"] == CID
    assert rec["dataset_hash"].startswith("sha256:")
    assert rec["split"] == "heldout" and rec["look_no"] == 1 and rec["planned_looks"] == 1


def test_reserve_holdout_look_shared_per_experiment(tmp_path: Path) -> None:
    deps = fake_deps(tmp_path)
    env = CampaignEnv(CID, Profile.FAKE, {"holdout_looks": 2}, deps, tmp_path / "runs")
    first = reserve_holdout_look(env, f"{CID}-r01", "heldout")
    assert reserve_holdout_look(env, f"{CID}-r01", "heldout") == first  # guard arms of one round share it
    assert first["look_no"] == 1
    assert reserve_holdout_look(env, f"{CID}-confirm", "heldout")["look_no"] == 2
    with pytest.raises(HoldoutExhausted):
        reserve_holdout_look(env, f"{CID}-r02", "heldout")


def test_reserve_holdout_look_legacy_ledger(tmp_path: Path) -> None:
    deps = fake_deps(tmp_path)

    class Plain:  # LedgerStore without record_look
        def __init__(self, inner: FileLedger) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            if name in ("record_look", "record_decisions"):
                raise AttributeError(name)
            return getattr(self._inner, name)

    deps.ledger = Plain(deps.ledger)
    env = CampaignEnv(CID, Profile.FAKE, {"holdout_looks": 1}, deps, tmp_path / "runs")
    first = reserve_holdout_look(env, f"{CID}-confirm", "heldout")
    assert reserve_holdout_look(env, f"{CID}-confirm", "heldout") == first == {
        "dataset_hash": first["dataset_hash"], "look_no": 1}
    with pytest.raises(HoldoutExhausted):
        reserve_holdout_look(env, "other-camp-confirm", "heldout")


def _legacy_look(path: Path, cid: str, digest: str, **extra: str) -> dict[str, str]:
    """A pre-``ledger.looks`` record: bare hex hash, ``"<cid>|<hex>"`` key, no plan/look_id."""
    bare = digest.removeprefix("sha256:")
    rec = {"key": f"{cid}|{bare}", "campaign": cid, "dataset_hash": bare, "split": "heldout", **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return rec


def test_record_look_counts_legacy_bare_hash_entries(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "holdout-looks.jsonl"
    bare = "b" * 64
    _legacy_look(path, "old-camp", bare, eid="old-camp-confirm")
    for h in (bare, f"sha256:{bare}", f"SHA256:{bare.upper()}"):
        assert count_looks(path, h) == 1 and planned_looks(path, h) is None
    ledger = FileLedger(tmp_path / "experiments")
    with pytest.raises(LookBudgetExceeded):  # the old look still uses the 1-look budget
        ledger.record_look(f"sha256:{bare}", experiment_id="new-camp-confirm", planned=1, campaign_id="new-camp")
    again = ledger.record_look(f"sha256:{bare}", experiment_id="old-camp-confirm", planned=1, campaign_id="old-camp")
    assert again["look_no"] == 1 and again["eid"] == "old-camp-confirm"  # resumed confirm: idempotent
    assert ledger.record_look(f"sha256:{bare}", experiment_id="new-camp-confirm", planned=2)["look_no"] == 2
    assert count_looks(path, bare) == 2


def test_record_look_legacy_key_without_eid(tmp_path: Path) -> None:
    path = tmp_path / "holdout-looks.jsonl"
    bare = "c" * 64
    _legacy_look(path, "old-camp", bare)
    assert record_look(path, f"sha256:{bare}", experiment_id="old-camp-confirm", campaign_id="old-camp")["look_no"] == 1
    with pytest.raises(LookBudgetExceeded):
        record_look(path, f"sha256:{bare}", experiment_id="x-confirm", campaign_id="x")


@pytest.mark.parametrize("file_ledger", [True, False])
def test_reserve_holdout_look_honours_legacy_ledger_entries(tmp_path: Path, file_ledger: bool) -> None:
    deps = fake_deps(tmp_path)
    digest = dataset_hash(deps.domain.splits()["heldout"])
    assert digest.startswith("sha256:")
    _legacy_look(tmp_path / "experiments" / "holdout-looks.jsonl", CID, digest, eid=f"{CID}-confirm")
    if not file_ledger:
        class Plain:  # LedgerStore without record_look
            def __init__(self, inner: FileLedger) -> None:
                self._inner = inner

            def __getattr__(self, name: str):
                if name in ("record_look", "record_decisions"):
                    raise AttributeError(name)
                return getattr(self._inner, name)

        deps.ledger = Plain(deps.ledger)
    env = CampaignEnv(CID, Profile.FAKE, {"holdout_looks": 1}, deps, tmp_path / "runs")
    assert reserve_holdout_look(env, f"{CID}-confirm", "heldout") == {"dataset_hash": digest, "look_no": 1}
    with pytest.raises(HoldoutExhausted):  # same campaign, different experiment: a second look
        reserve_holdout_look(env, f"{CID}-r01", "heldout")
    with pytest.raises(HoldoutExhausted):
        reserve_holdout_look(env, "other-camp-confirm", "heldout")
    assert len((tmp_path / "experiments" / "holdout-looks.jsonl").read_text().splitlines()) == 1


def test_guard_extension_validates_and_lands_in_envelope(tmp_path: Path) -> None:
    _, deps = _round(tmp_path)
    ctx = RoundContext(CampaignEnv(CID, Profile.FAKE, {}, deps, tmp_path / "runs"), 1)
    ext = guard_envelope_extension(ctx, "v2")
    assert ext is not None and set(ext) == {GUARD_EXT}
    payload = ext[GUARD_EXT]
    assert payload["arm"] == "v2" and payload["split"] == "evolve" and payload["holdoutLook"] is False
    assert payload["ship"] == {"ok": True, "reasons": []}
    assert payload["incumbentBundleDigest"]
    assert _schema_errors(EXTENSION_SCHEMAS[GUARD_EXT], payload, f"extensions.{GUARD_EXT}", "guard") == []
    assert guard_envelope_extension(ctx, "v1") is None  # agent arm: no guard eval

    env = build_envelope("round", {"eid": "e-1", "decision": "ship", "extensions": ext})
    assert env["extensions"][GUARD_EXT] == payload
    assert "extensions" not in env["extensions"]["org.ci.rrsi"]


def test_guard_extension_on_heldout_carries_look(tmp_path: Path) -> None:
    _, deps = _round(tmp_path)
    run = tmp_path / "runs" / f"{CID}-r01" / "v2" / "arm.done"
    done = json.loads(run.read_text())
    done["guard"].update({"split": "heldout", "holdout_look": True, "dataset_hash": "sha256:" + "b" * 64, "look_no": 1,
                          "planned_looks": 1})
    run.write_text(json.dumps(done))
    ctx = RoundContext(CampaignEnv(CID, Profile.FAKE, {}, deps, tmp_path / "runs"), 1)
    payload = guard_envelope_extension(ctx, "v2")[GUARD_EXT]
    assert payload["holdoutLook"] is True
    assert payload["holdout"] == {"datasetHash": "sha256:" + "b" * 64, "plannedLooks": 1, "looksUsed": 1}
    assert _schema_errors(EXTENSION_SCHEMAS[GUARD_EXT], payload, f"extensions.{GUARD_EXT}", "guard") == []
