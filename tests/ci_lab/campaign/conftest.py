from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _no_global_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``ci_lab.telemetry.setup`` installs a process-global tracer provider and MAF
    instrumentation, which would leak into later tests. Tests that exercise telemetry wiring
    override this stub themselves."""
    import ci_lab

    stub = types.ModuleType("ci_lab.telemetry")
    stub.setup = lambda component, **kw: None  # type: ignore[attr-defined]
    stub.shutdown = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ci_lab.telemetry", stub)
    monkeypatch.setattr(ci_lab, "telemetry", stub, raising=False)
