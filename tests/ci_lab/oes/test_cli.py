import copy
import json

import pytest

from ci_lab.cli import main
from ci_lab.oes.models import RRSI_EXT


@pytest.fixture
def tree(tmp_path, envelopes):
    root = tmp_path / "experiments"
    for kind, doc in envelopes.items():
        d = root / ("sleep" if kind == "sleep" else "campaigns/tone-a1")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kind}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def test_validate_directory_ok(tree, capsys):
    assert main(["oes", "validate", str(tree)]) == 0
    out = capsys.readouterr().out
    assert out.count("OK  ") == 4 and "4/4 valid" in out


def test_validate_glob_and_json(tree, capsys):
    assert main(["oes", "validate", str(tree / "**" / "*.json"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and len(report["files"]) == 4
    assert all(f["errors"] == [] for f in report["files"])


def test_validate_deduplicates_paths(tree, capsys):
    f = tree / "sleep" / "sleep.json"
    assert main(["oes", "validate", str(f), str(tree / "sleep"), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["files"]) == 1


def test_validate_reports_errors_and_exits_1(tree, envelopes, capsys):
    bad = copy.deepcopy(envelopes["round"])
    bad["decision"]["rationale"] = "edited"
    (tree / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert main(["oes", "validate", str(tree), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    errors = {f["path"].replace("\\", "/").rsplit("/", 1)[-1]: f["errors"] for f in report["files"]}
    assert errors["bad.json"][0].startswith("[content-hash]") and errors["round.json"] == []
    assert main(["oes", "validate", str(tree / "bad.json")]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_validate_no_match_exits_1(tmp_path, capsys):
    assert main(["oes", "validate", str(tmp_path / "nothing-*.json")]) == 1
    assert "no files matched" in capsys.readouterr().out


def test_validate_look_ledger(tree, envelopes, tmp_path, capsys):
    h = envelopes["confirm"]["extensions"][RRSI_EXT]["holdout"]["datasetHash"]
    ledger = tmp_path / "looks.jsonl"
    ledger.write_text(json.dumps({"datasetHash": h}) + "\n", encoding="utf-8")
    assert main(["oes", "validate", str(tree), "--look-ledger", str(ledger)]) == 0
    ledger.write_text((json.dumps({"datasetHash": h}) + "\n") * 2, encoding="utf-8")
    assert main(["oes", "validate", str(tree), "--look-ledger", str(ledger)]) == 1
    assert "[holdout-looks]" in capsys.readouterr().out
    ledger.write_text("not json\n", encoding="utf-8")
    assert main(["oes", "validate", str(tree), "--look-ledger", str(ledger)]) == 1


def test_validate_requires_paths():
    with pytest.raises(SystemExit):
        main(["oes", "validate"])
