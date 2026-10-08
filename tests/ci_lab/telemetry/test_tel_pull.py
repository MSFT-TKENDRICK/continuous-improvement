"""D5: ``telemetry pull`` — gh artifact download, digest + schema checks, cache."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from ci_lab.telemetry import pull
from ci_lab.telemetry.record import SchemaError

FIXTURES = Path(__file__).with_name("fixtures")


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


@pytest.fixture
def gh(monkeypatch, tmp_path):
    monkeypatch.setenv(pull.IMPORTS_ENV, str(tmp_path / "imports"))
    st = {"zip": _zip({"night/telemetry/spans-1.jsonl": (FIXTURES / "span_v1.jsonl").read_bytes()}),
          "digest": "auto", "calls": [], "id": 77}

    def fake(args):
        st["calls"].append(args)
        path = args[1]
        if path.endswith("/zip"):
            return st["zip"]
        digest = ("sha256:" + hashlib.sha256(st["zip"]).hexdigest()) if st["digest"] == "auto" else st["digest"]
        arts = [{"id": 1, "name": "spans", "expired": True},
                {"id": st["id"], "name": "spans", "expired": False, "digest": digest,
                 "created_at": "2026-01-01T00:00:00Z"},
                {"id": 2, "name": "other", "expired": False}]
        return json.dumps({"total_count": 3, "artifacts": arts}).encode()

    monkeypatch.setattr(pull, "_gh", fake)
    return st


def test_pull_verifies_and_caches(gh, tmp_path):
    m = pull.pull(12345, repo="o/r")
    assert m["cached"] is False and m["spans"] == 2 and m["traces"] == 1
    assert m["artifact_id"] == 77 and m["schemaVersion"] == 1 and m["services"] == ["ci-lab.night"]
    assert gh["calls"][0] == ["api", "repos/o/r/actions/runs/12345/artifacts?per_page=100"]
    assert gh["calls"][1] == ["api", "repos/o/r/actions/artifacts/77/zip"]
    d = tmp_path / "imports" / "12345"
    assert json.loads((d / "manifest.json").read_text("utf-8"))["sha256"] == m["sha256"]
    assert [p.name for p in pull.files_of(m)] == ["spans-1.jsonl"] and pull.files_of(m)[0].is_file()
    assert not (d / "artifact.zip").exists()
    assert [x["run_id"] for x in pull.list_imports()] == ["12345"]
    again = pull.pull("12345", repo="o/r")
    assert again["cached"] is True and len(gh["calls"]) == 3  # metadata only
    pull.pull("12345", repo="o/r", force=True)
    assert len(gh["calls"]) == 5


def test_pull_defaults_to_current_repo(gh):
    pull.pull(1)
    assert gh["calls"][0][1].startswith("repos/{owner}/{repo}/")


def test_digest_mismatch_rejected(gh, tmp_path):
    gh["digest"] = "sha256:" + "0" * 64
    with pytest.raises(pull.PullError, match="digest mismatch"):
        pull.pull(5, repo="o/r")
    assert not (tmp_path / "imports" / "5").exists()
    assert not (tmp_path / "imports").exists() or not list((tmp_path / "imports").iterdir())


def test_missing_digest_needs_opt_in(gh):
    gh["digest"] = None
    with pytest.raises(pull.PullError, match="--allow-no-digest"):
        pull.pull(5, repo="o/r")
    assert pull.pull(5, repo="o/r", allow_no_digest=True)["digest"] is None


def test_wrong_schema_version_rejected(gh, tmp_path):
    bad = (FIXTURES / "span_v1.jsonl").read_bytes().replace(b'"schemaVersion":1', b'"schemaVersion":2')
    gh["zip"] = _zip({"spans-9.jsonl": bad})
    with pytest.raises(SchemaError, match="schemaVersion"):
        pull.pull(6, repo="o/r")
    assert not list((tmp_path / "imports").iterdir())  # temp extraction dir cleaned up


def test_zip_slip_in_artifact_rejected(gh, tmp_path):
    gh["zip"] = _zip({"../escape.jsonl": b"{}"})
    with pytest.raises(Exception, match="unsafe"):
        pull.pull(7, repo="o/r")
    assert not (tmp_path / "imports" / "escape.jsonl").exists()


def test_missing_artifact_and_bad_run_id(gh):
    with pytest.raises(pull.PullError, match="no unexpired artifact"):
        pull.pull(8, repo="o/r", artifact="nope")
    with pytest.raises(ValueError):
        pull.pull("../../etc", repo="o/r")


def test_list_imports_empty(tmp_path):
    assert pull.list_imports(tmp_path / "missing") == []
