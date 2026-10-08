import os

import pytest

# Importing order_support.agent calls assert_ai.auto_trace.enable(); keep it from
# instrumenting third-party libraries during unit tests.
os.environ.setdefault("PHOENIX_DISABLE_AUTO_INSTRUMENT", "1")

# Operator model selection (ci_lab.maf.models, CI_ASSERT_MODEL) must not leak from the developer's shell into tests.
os.environ.pop("CI_META_MODEL", None)
os.environ.pop("CI_ALLOWED_MODELS", None)
os.environ.pop("CI_ASSERT_MODEL", None)

@pytest.fixture(autouse=True, scope="session")
def _isolated_s1_admission_locks(tmp_path_factory):
    """Keep tests off the host-wide s1 judge queue (a live llama-server may be in use)."""
    mp = pytest.MonkeyPatch()
    mp.setenv("CI_S1_LOCK_DIR", str(tmp_path_factory.mktemp("s1-locks")))
    mp.delenv("CI_S1_MAX_INFLIGHT", raising=False)
    mp.delenv("CI_S1_ADMISSION_LOG", raising=False)
    yield
    mp.undo()


@pytest.fixture(autouse=True, scope="session")
def _isolated_governance(tmp_path_factory):
    """Governed agents (ci_lab.governance.maf) audit and queue approvals under tmp, never artifacts/."""
    root = tmp_path_factory.mktemp("governance")
    mp = pytest.MonkeyPatch()
    mp.setenv("CI_GOVERNANCE_AUDIT", str(root / "audit" / "decisions.jsonl"))
    mp.setenv("CI_GOVERNANCE_APPROVALS", str(root / "approvals"))
    for name in ("CI_GOVERNANCE_MODE", "CI_KILL_SWITCH"):
        mp.delenv(name, raising=False)
    yield
    mp.undo()
