"""Golden envelopes in ``fixtures/`` (also usable as examples: ``ci-lab oes validate tests/ci_lab/oes/fixtures``).

Regenerate after an intentional builder change: ``OES_UPDATE_GOLDEN=1 pytest tests/ci_lab/oes/test_golden.py``.
"""

import json
import os

import pytest

from ci_lab.oes import validate_envelope
from ci_lab.oes.validate import validate_file

KINDS = ["calibration", "round", "confirm", "sleep"]


@pytest.mark.parametrize("kind", KINDS)
def test_golden_envelope(fx, kind):
    path = fx.FIXTURES / f"{kind}.json"
    doc = fx.BUILDERS[kind]()
    if os.environ.get("OES_UPDATE_GOLDEN") == "1":
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    assert validate_file(path) == []
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert golden == doc, "builder output drifted from the golden fixture (see module docstring)"
    assert validate_envelope(golden) == []
