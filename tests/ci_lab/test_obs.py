import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import ATTR_EXPERIMENT, SPAN_ARM, SPAN_CAMPAIGN_ROUND


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


class Secret:
    def __str__(self):
        return "ssn=123-45-6789"


def test_noop_without_provider(tmp_path):
    with obs.span(SPAN_CAMPAIGN_ROUND, {ATTR_EXPERIMENT: "e1"}):
        obs.annotate({"x": 1})


def test_span_nesting_and_attr_cleaning(exporter):
    attrs = {ATTR_EXPERIMENT: "e1", "x": None, "p": ("a",), "obj": Secret(), "long": "z" * 5000,
             "mixed": [1, "a"]}
    with obs.span(SPAN_CAMPAIGN_ROUND, attrs), obs.span(SPAN_ARM, {"ci.arm": "a1"}):
        ids = obs.current_ids()
        obs.annotate({"agl.rollout_id": "r1"})
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans[SPAN_CAMPAIGN_ROUND].attributes
    assert spans[SPAN_ARM].parent.span_id == spans[SPAN_CAMPAIGN_ROUND].context.span_id
    assert "x" not in root
    assert tuple(root["p"]) == ("a",)
    assert root["obj"] == "<Secret>" and "123" not in json.dumps(dict(root), default=str)
    assert len(root["long"]) <= obs.MAX_ATTR_LEN + 1
    assert root["mixed"] == "<list>"
    assert spans[SPAN_ARM].attributes["agl.rollout_id"] == "r1"
    assert ids[0] == format(spans[SPAN_ARM].context.trace_id, "032x")


def test_new_trace_starts_fresh_root(exporter):
    with obs.span("outer"), obs.span(SPAN_CAMPAIGN_ROUND, new_trace=True):
        pass
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans[SPAN_CAMPAIGN_ROUND].parent is None
    assert spans[SPAN_CAMPAIGN_ROUND].context.trace_id != spans["outer"].context.trace_id


def test_exception_type_only(exporter):
    with pytest.raises(ValueError), obs.span("ci.step"):
        raise ValueError("customer email a@b.com")
    (s,) = exporter.get_finished_spans()
    assert s.status.status_code == trace.StatusCode.ERROR
    dumped = json.dumps([dict(e.attributes) for e in s.events]) + str(s.status.description)
    assert "a@b.com" not in dumped and "ValueError" in dumped


def test_link_to_and_validation(exporter):
    with obs.span("ci.round", links=[obs.link_to("AB" * 16, "cd" * 8)]):
        pass
    (s,) = exporter.get_finished_spans()
    assert format(s.links[0].context.trace_id, "032x") == "ab" * 16
    assert obs.link_to("0" * 32, "cd" * 8) is None
    assert obs.link_to("xyz", "cd" * 8) is None
    assert obs.link_to("ab" * 16, "cd" * 7) is None


def test_carrier_per_request_and_detach(exporter):
    with obs.span("client") as c:
        headers = obs.carrier()
        tid = c.get_span_context().trace_id
    assert trace.get_current_span().get_span_context().is_valid is False
    with obs.use_carrier(headers), obs.span("server.req"):
        pass
    assert trace.get_current_span().get_span_context().is_valid is False
    srv = next(s for s in exporter.get_finished_spans() if s.name == "server.req")
    assert srv.context.trace_id == tid


def test_wrap_ctx_thread_pool(exporter):
    with obs.span("parent") as p:
        tid = p.get_span_context().trace_id
        with ThreadPoolExecutor(1) as ex:
            got = ex.submit(obs.wrap_ctx(lambda: trace.get_current_span().get_span_context().trace_id)).result()
    assert got == tid


def test_child_env_propagates_traceparent(exporter, tmp_path):
    tp = TracerProvider()
    with tp.get_tracer("t").start_as_current_span("parent") as p:
        env = obs.child_env({"TRACEPARENT": "stale"})
        tid = format(p.get_span_context().trace_id, "032x")
    assert env[obs.TRACEPARENT_ENV].split("-")[1] == tid
    assert obs.child_env({"TRACEPARENT": "stale"}).get(obs.TRACEPARENT_ENV) is None
    code = ("from ci_lab import obs; from opentelemetry import trace; obs.attach_from_env();"
            "print(format(trace.get_current_span().get_span_context().trace_id,'032x'))")
    out = subprocess.run([sys.executable, "-c", code], env={**_base_env(), **env},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == tid


def _base_env():
    import os
    return {k: v for k, v in os.environ.items() if k != obs.TRACEPARENT_ENV}


def test_write_status_merges_per_writer(tmp_path):
    p = obs.write_status(tmp_path, "exp1", writer="w1", phase="propose",
                         arms={"a1": {"strategy": "gepa"}})
    obs.write_status(tmp_path, "exp1", writer="w1", phase="evaluate",
                     arms={"a1": {"state": "running"}, "a2": {"strategy": "agent"}})
    d = json.loads(p.read_text())
    assert p.parent.name == "status.d" and d["seq"] == 2 and d["writer"] == "w1"
    assert d["phase"] == "evaluate" and d["experiment_id"] == "exp1"
    assert d["arms"]["a1"]["strategy"] == "gepa" and d["arms"]["a1"]["state"] == "running"
    assert set(d["arms"]) == {"a1", "a2"}
    assert not list(p.parent.glob(".status-*"))
    with pytest.raises(ValueError):
        obs.write_status(tmp_path, "exp1", writer="../evil")


def test_write_status_records_trace_and_previous_link(exporter, tmp_path):
    with obs.span("ci.round") as s:
        obs.write_status(tmp_path, "exp1", writer="round", phase="plan")
        sid = format(s.get_span_context().span_id, "016x")
    (link,) = obs.previous_link(tmp_path, "exp1")
    assert format(link.context.span_id, "016x") == sid
    assert obs.previous_link(tmp_path, "nope") == []


_WRITER_CODE = """
import sys
from ci_lab import obs
run, arm = sys.argv[1], sys.argv[2]
for i in range(25):
    obs.write_status(run, "exp", writer=arm, arms={arm: {"i": i}})
"""


def test_concurrent_writers_lose_nothing(tmp_path):
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER_CODE, str(tmp_path), f"arm{n}"])
             for n in range(4)]
    assert all(p.wait(60) == 0 for p in procs)
    agg = obs.read_status(tmp_path, "exp")
    assert {a: v["i"] for a, v in agg["arms"].items()} == {f"arm{n}": 24 for n in range(4)}
    assert sorted(agg["writers"]) == [f"arm{n}" for n in range(4)]


def test_read_status_accepts_legacy_file(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "status.json").write_text(json.dumps(
        {"phase": "old", "updated": 1, "arms": {"a": {"updated": 1, "state": "x"}}}))
    obs.write_status(tmp_path, "exp", writer="new", phase="new", arms={"a": {"state": "y"}})
    agg = obs.read_status(tmp_path, "exp")
    assert agg["phase"] == "new" and agg["arms"]["a"]["state"] == "y"
