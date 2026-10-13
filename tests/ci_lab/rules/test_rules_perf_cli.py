"""Performance (100 rules x 200 steps) and CLI."""

import json
import time

from ci_lab.rules import build_bundle, evaluate_trajectory
from ci_lab.rules.cli import main
from ci_lab.rulespec import ExtractorSpec, RuleSpec

TOOLS = ["read_file", "search_kb", "write_file", "escalate_to_human", "verify_access"]


def _rules(n=100):
    out = []
    for k in range(n):
        tool = TOOLS[k % len(TOOLS)]
        kind = k % 5
        if kind == 0:
            req = {"kind": "prior", "tool": "read_file", "status": "ok",
                   "same": [["current.args.resource_id", "prior.args.resource_id"]],
                   "where": {"kind": "arg", "path": "prior.result.edit_allowed", "op": "eq", "value": True},
                   "cmp": [{"current": "current.args.amount", "op": "le", "prior": "prior.result.total"}]}
        elif kind == 1:
            req = {"kind": "state", "flag": "access_verified", "subject": "current.args.resource_id"}
        elif kind == 2:
            req = {"kind": "count", "tool": tool, "op": "le", "n": 1000 + k}
        elif kind == 3:
            req = {"kind": "any", "of": [{"kind": "arg", "path": "current.args.resource_id", "op": "matches",
                                          "value": rf"^A\d{{{1 + k % 4}}}"},
                                         {"kind": "not", "of": {"kind": "arg", "path": "current.args.amount",
                                                                "op": "gt", "value": k}}]}
        else:
            req = {"kind": "prior", "tool": "search_kb", "status": "any", "within": 1 + k % 7}
        out.append(RuleSpec.model_validate({
            "id": f"perf.r{k:03d}", "version": 1, "rung": "R2", "on": "tool_call", "target": tool,
            "require": req, "action": "warn", "template": "count.exceeded", "slots": {"tool": tool}}))
    return out


def _steps(n=200):
    from ci_lab.rulespec import TrajectoryStep

    steps = []
    k = 0
    while len(steps) < n:
        tool = TOOLS[k % len(TOOLS)]
        oid = f"A{k % 13}"
        steps.append(TrajectoryStep(i=len(steps), kind="tool_call", tool=tool, call_id=f"c{k}",
                                    args={"resource_id": oid, "amount": k % 90}))
        res = {"resource_id": oid, "total": 50.0, "edit_allowed": k % 3 != 0, "edit_limit_exceeded": False,
               "verified": True}
        steps.append(TrajectoryStep(i=len(steps), kind="tool_result", tool=tool, call_id=f"c{k}", result=res,
                                    status="ok" if k % 11 else "error"))
        k += 1
    return steps[:n]


def test_100_rules_x_200_steps_is_fast():
    ex = [ExtractorSpec(flag="access_verified", tool="verify_access", result_path="result.verified",
                        subject="args.resource_id", ttl_steps=20)]
    b = build_bundle(_rules(), ex)
    steps = _steps()
    evaluate_trajectory(b, steps)  # warm caches
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        got = evaluate_trajectory(b, steps)
        best = min(best, time.perf_counter() - t0)
    assert got  # the workload actually fires rules
    assert best < 0.5, f"{best * 1000:.1f} ms"  # target well under 100 ms; generous bound for CI noise


def test_cli_check_ok_and_errors(tmp_path, capsys, request):
    fx = request.path.parent / "fixtures"
    assert main(["rules", "check", str(fx / "seeds.yaml"), "--extractors", str(fx / "extractors.yaml")]) == 0
    assert "[RULES][OK] 7 rule(s)" in capsys.readouterr().out
    assert main(["rules", "check", str(fx / "seeds.yaml")]) == 1  # no extractor for access_verified
    out = capsys.readouterr().out
    assert "[RULES][ERROR]" in out and "  Violation: state flag 'access_verified' has no extractor" in out
    assert "  Fix: " in out


def test_cli_eval_prints_matches(tmp_path, capsys, request, tb):
    fx = request.path.parent / "fixtures"
    tb.inspect("A1")
    tb.call("write_file", resource_id="A1", amount=5)
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps({"steps": [s.model_dump(mode="json") for s in tb.steps]}), encoding="utf-8")
    rc = main(["rules", "eval", "--rules", str(fx / "seeds.yaml"), "--extractors", str(fx / "extractors.yaml"),
               "--trajectory", str(traj)])
    out = capsys.readouterr().out
    assert rc == 0
    got = json.loads(out)
    assert [m["rule"] for m in got] == ["change.requires_access"]
    assert got[0]["step_index"] == 2 and got[0]["action"] == "block" and got[0]["mode"] == "shadow"
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert main(["rules", "eval", "--rules", str(fx / "seeds.yaml"), "--extractors", str(fx / "extractors.yaml"),
                 "--trajectory", str(bad)]) == 1


def test_cli_register_plugs_into_ci_lab_parser():
    import argparse

    from ci_lab.rules.cli import register

    p = argparse.ArgumentParser()
    register(p.add_subparsers(dest="command"))
    args = p.parse_args(["rules", "check", "x.yaml"])
    assert args.func.__name__ == "cmd_check"
