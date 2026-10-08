import hashlib

import pytest
from jsonschema import Draft202012Validator

from ci_lab.contracts import COMPONENTS
from ci_lab.oes import validate

VENDORED_SHA256 = "3c709822a2a29f7ce21c93aad31992cb65b219c8bdc106ea41aa4f3efe5eae15"
ALL = [validate.CORE_SCHEMA, *validate.EXTENSION_SCHEMAS.values()]


@pytest.mark.parametrize("name", ALL)
def test_schema_files_load_and_are_valid_draft_2020_12(name):
    schema = validate.load_schema(name)
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_vendored_core_schema_is_byte_exact():
    raw = (validate.schema_dir() / validate.CORE_SCHEMA).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VENDORED_SHA256
    source = (validate.schema_dir() / "SOURCE.md").read_text(encoding="utf-8")
    assert VENDORED_SHA256 in source and "openexperiment.org/schema/openexperiment-0.1.0.schema.json" in source


@pytest.mark.parametrize("name", list(validate.EXTENSION_SCHEMAS.values()))
def test_extension_schemas_are_closed_and_versioned(name):
    schema = validate.load_schema(name)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["version"]["const"] == "0.1.0"


def test_rrsi_prune_set_matches_contract_components():
    schema = validate.load_schema(validate.EXTENSION_SCHEMAS["com.microsoft.ci.rrsi"])
    assert schema["properties"]["pruneSet"]["items"] == {"$ref": "#/$defs/component"}
    assert tuple(schema["$defs"]["component"]["enum"]) == COMPONENTS


def test_schema_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_OES_SCHEMA_DIR", str(tmp_path))
    assert validate.schema_dir() == tmp_path
