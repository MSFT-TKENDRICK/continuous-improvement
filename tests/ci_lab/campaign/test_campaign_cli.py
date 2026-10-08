from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci_lab.cli import main


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_cli_fake_profile_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--profile", "fake", "--run-dir", str(tmp_path), "--dry-run-publish"]
    code, out = _run(capsys, "campaign", "new", "cli-camp", *common, "--hyper", "aa_repeats=2",
                     "--hyper", "max_rounds=2")
    assert code == 0 and out["hyper"]["aa_repeats"] == 2 and not out["calibrated"]
    code, out = _run(capsys, "campaign", "calibrate", "cli-camp", *common)
    assert code == 0 and out["delta"] == pytest.approx(0.25)
    code, out = _run(capsys, "campaign", "run", "cli-camp", *common, "--rounds", "1")
    assert code == 0 and out["rounds"][0]["winner"] == "v1"
    code, out = _run(capsys, "campaign", "readjudicate", "cli-camp", "cli-camp-r01", *common)
    assert code == 0 and out["consistent"]
    code, out = _run(capsys, "campaign", "status", "cli-camp", *common)
    assert code == 0 and out["frontier"]["round"] == 1
    code, out = _run(capsys, "campaign", "confirm", "cli-camp", *common)
    assert code == 0 and out["decision"] == "ship"
    code, out = _run(capsys, "campaign", "land", "cli-camp", *common)
    assert code == 0 and len(out["layers"]) == 1
    assert (tmp_path / "_fake" / "experiments" / "campaigns" / "cli-camp" / "land.json").exists()


def test_cli_rejects_bad_hyper(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        main(["campaign", "new", "cli-camp", "--profile", "fake", "--run-dir", str(tmp_path), "--hyper", "bogus=1"])
