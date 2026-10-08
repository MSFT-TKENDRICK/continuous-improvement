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

COMMAND_MODULES = ("oes", "providers", "campaign", "sleep")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ci-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="print versions and verify the no-.NET invariant")
    doc.set_defaults(func=lambda _a: _print_doctor())
    for name in COMMAND_MODULES:
        try:
            module = importlib.import_module(f"ci_lab.{name}.cli")
        except ModuleNotFoundError as exc:
            if exc.name == f"ci_lab.{name}.cli":
                continue
            raise
        module.register(sub)
    return parser


def _print_doctor() -> int:
    report = doctor()
    print(json.dumps(report, indent=2))
    return 0 if report["no_dotnet"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
