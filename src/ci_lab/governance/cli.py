"""``ci-lab governance doctor|eval|audit-verify|approve|deny``."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ci_lab.governance.approvals import FileApprovalQueue
from ci_lab.governance.audit import AuditTrail
from ci_lab.governance.hypervisor import KillSwitchAdapter
from ci_lab.governance.policies import POLICIES, governance_mode

__all__ = ["AGT_MODULES", "doctor", "evaluate", "register"]

AGT_MODULES = ("agent_framework", "agentmesh", "agent_sre", "hypervisor")


def _policies() -> tuple[dict[str, Any], bool]:
    from ci_lab.governance.policies import _runtime

    out, ok = {}, True
    for name in POLICIES:
        try:
            manifest = _runtime(name).manifest
            missing = sorted(p for p, cfg in manifest.points.items() if cfg.policy_id not in manifest.policies)
            out[name] = {"points": sorted(manifest.points), "missing_policies": missing}
            ok = ok and bool(manifest.points) and not missing
        except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
            out[name], ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    return out, ok


def doctor() -> dict[str, Any]:
    """Health report; ``ok`` is false on any failed check (an engaged kill switch included)."""
    policies, policies_ok = _policies()
    try:
        mode: dict[str, Any] = {"value": governance_mode(), "ok": True}
    except RuntimeError as exc:
        mode = {"error": str(exc), "ok": False}
    modules = {m: importlib.util.find_spec(m) is not None for m in AGT_MODULES}
    native = importlib.util.find_spec("agent_control_specification") is not None
    trail = AuditTrail()
    res = trail.verify()
    audit = {"path": str(trail.path), "ok": res.ok, "entries": res.entries, "head": res.head,
             "denies": res.denies, "error": res.error}
    engaged = KillSwitchAdapter().engaged()
    report = {"policies": policies, "mode": mode, "agt_modules": modules, "acs_native": native,
              "audit": audit, "kill_switch": {"engaged": engaged},
              "pending_approvals": len(FileApprovalQueue().pending())}
    report["ok"] = policies_ok and mode["ok"] and all(modules.values()) and res.ok and not engaged
    return report


async def evaluate(policy: str, point: str, snapshot: dict[str, Any], *, mode: str = "enforce",
                   tool: str | None = None) -> dict[str, Any]:
    """Evaluate one point of a packaged policy (no approval resolution, no audit append)."""
    from ci_lab.governance.policies import _runtime

    res = await _runtime(policy).evaluate_intervention_point(point, snapshot, mode, tool)
    v = res.verdict
    return {"decision": str(getattr(v.decision, "value", v.decision)), "reason": v.reason,
            "message": v.message, "liftable": v.liftable, "transform": v.transform,
            "result_labels": list(v.result_labels), "input_identity": res.input_identity,
            "enforced_identity": res.enforced_identity}


def _read_json(src: str) -> Any:
    return json.loads(sys.stdin.read() if src == "-" else Path(src).read_text("utf-8"))


def _main(args: argparse.Namespace) -> int:
    cmd = args.governance_command
    if cmd == "doctor":
        report = doctor()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if cmd == "eval":
        out = asyncio.run(evaluate(args.policy, args.point, _read_json(args.snapshot),
                                   mode=args.mode or governance_mode(), tool=args.tool))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2 if out["decision"] == "deny" else 0
    if cmd == "audit-verify":
        res = AuditTrail(args.path).verify()
        print(json.dumps(asdict(res), indent=2))
        return 0 if res.ok else 1
    queue = FileApprovalQueue(args.dir)
    try:
        data = queue.decide(args.identity, cmd, reason=args.reason, by=args.by)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("governance", help="AGT/ACS governance: doctor, eval, audit, approvals")
    gsub = p.add_subparsers(dest="governance_command", required=True)
    gsub.add_parser("doctor", help="policies, mode, AGT imports, audit chain, kill switch")
    ev = gsub.add_parser("eval", help="evaluate a snapshot against a packaged policy point")
    ev.add_argument("--policy", choices=POLICIES, required=True)
    ev.add_argument("--point", required=True)
    ev.add_argument("--snapshot", required=True, metavar="PATH", help="snapshot JSON file or '-' (stdin)")
    ev.add_argument("--tool", default=None)
    ev.add_argument("--mode", choices=("enforce", "evaluate_only"), default=None)
    av = gsub.add_parser("audit-verify", help="recheck the decision hash chain")
    av.add_argument("--path", default=None, help="audit JSONL (default: $CI_GOVERNANCE_AUDIT)")
    for name in ("approve", "deny"):
        d = gsub.add_parser(name, help=f"{name} a held action by its enforced_identity")
        d.add_argument("identity", help="64-hex enforced_identity (optionally sha256:-prefixed)")
        d.add_argument("--reason", default=None)
        d.add_argument("--by", default=None)
        d.add_argument("--dir", default=None, help="approval queue (default: $CI_GOVERNANCE_APPROVALS)")
    p.set_defaults(func=_main)
