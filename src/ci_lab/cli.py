"""``ci-lab`` CLI. Subcommands live in their owning modules (``ci_lab.<module>.cli``)
and expose ``register(subparsers)``; modules not yet present are skipped so layers
can land independently."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import json
import sys

COMMAND_MODULES = ("oes", "providers", "campaign", "sleep", "telemetry", "judge", "rules", "lessons",
                   "lessons_arm", "lint", "template", "chat", "bus", "governance")
VERSIONED = ("agent-framework-core", "agent-framework-declarative", "agent-framework-github-copilot",
             "github-copilot-sdk", "agentlightning", "skillopt", "assert-ai")


def doctor() -> dict[str, object]:
    """Environment facts that experiment provenance relies on (and the no-.NET invariant)."""
    versions = {}
    for dist in VERSIONED:
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    dotnet = {name: importlib.util.find_spec(name) is not None for name in ("powerfx", "pythonnet", "clr")}
    return {"python": sys.version.split()[0], "versions": versions, "dotnet_bridges": dotnet,
            "no_dotnet": not any(dotnet.values())}


def doctor_models(*, env: dict[str, str] | None = None, list_models: object = None,
                  fetch: object = None) -> dict[str, object]:
    """Copilot-profile model check (``doctor --models``): every model a campaign (all strategies)
    or a sleep night would use, against the Copilot account's ``list_models()`` and the
    configured OpenAI-compatible endpoints. Needs a signed-in Copilot CLI (network)."""
    import asyncio

    from ci_lab.campaign.preflight import campaign_model_plan
    from ci_lab.contracts import STRATEGIES, Profile
    from ci_lab.providers.models import copilot_model_ids, model_ids, served_model_ids
    from ci_lab.sleep.wiring import model_uses

    try:
        plan = campaign_model_plan({"strategies": list(STRATEGIES)}, env=env)
        uses = list(dict.fromkeys([*plan.copilot, *model_uses(Profile.COPILOT)]))
        available = model_ids(asyncio.run((list_models or copilot_model_ids)()))  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    required = [{"model": u.model, "user": u.user, "override": u.override, "available": u.model in available}
                for u in uses]
    served = []
    for base, key, s_uses in plan.served:
        try:
            ids = model_ids((fetch or served_model_ids)(base, key))  # type: ignore[operator]
            err = None
        except Exception as exc:  # noqa: BLE001
            ids, err = (), f"{type(exc).__name__}: {exc}"
        served.append({"base_url": base, "served": list(ids), "error": err,
                       "required": [{"model": u.model, "user": u.user, "override": u.override,
                                     "available": u.model.removeprefix("openai/") in ids} for u in s_uses]})
    ok = all(r["available"] for r in required) and all(
        s["error"] is None and all(r["available"] for r in s["required"]) for s in served)
    return {"ok": ok, "copilot_available": list(available), "required": required, "served": served}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ci-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="print versions and verify the no-.NET invariant")
    doc.add_argument("--models", action="store_true",
                     help="also check every copilot-profile model id against the Copilot account (network)")
    doc.set_defaults(func=lambda a: _print_doctor(models=a.models))
    for name in COMMAND_MODULES:
        try:
            module = importlib.import_module(f"ci_lab.{name}.cli")
        except ModuleNotFoundError as exc:
            if exc.name == f"ci_lab.{name}.cli":
                continue
            raise
        module.register(sub)
    return parser


def _print_doctor(*, models: bool = False) -> int:
    report = doctor()
    ok = bool(report["no_dotnet"])
    if models:
        report["models"] = doctor_models()
        ok = ok and bool(report["models"]["ok"])  # type: ignore[index]
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
