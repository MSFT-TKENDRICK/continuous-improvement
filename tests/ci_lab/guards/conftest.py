import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # `import guards_support` from guard tests


@pytest.fixture(autouse=True)
def _no_guard_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI_GUARDS", raising=False)
