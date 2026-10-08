"""``ci-lab graph ...`` (validate / seal / run / show a task graph) and ``ci-lab bus ...`` (verify / tail /
heads of an agent-bus WAL). See ``examples/taskgraph/README.md``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ci_lab.bus.types import ROLES

POOL_DEFAULTS = {"s1": 1, "llm": 4, "cpu": os.cpu_count() or 1}
POOL_ENV = "CI_POOL_"  # CI_POOL_S1=2 -> pools["s1"] = 2


def _load_rubrics(paths: Sequence[str | Path]) -> list[Any]:
    import yaml

    from ci_lab.taskgraph.model import Rubric

    out = []
    for p in paths:
        doc = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        items = doc.get("rubrics", [doc]) if isinstance(doc, dict) else doc
        out += [Rubric.from_json(r) for r in items]
    return out


def pools_from_env(env: dict[str, str] | None = None) -> Any:
    from ci_lab.bus.pools import ResourcePools

    caps = dict(POOL_DEFAULTS)
    for k, v in (os.environ if env is None else env).items():
        if k.startswith(POOL_ENV) and k != POOL_ENV:
            caps[k[len(POOL_ENV):].lower()] = int(v)
    return ResourcePools(caps)


def make_voters_for(vault: Any, *, run_dir: Path, profile: str, s1_model: str | None) -> Any:
    """Deterministic oracles always; s1 / llm / assert voters only for deliverables whose rubric uses them."""
    from ci_lab.bus.voters.local import DeterministicCheckVoter
    from ci_lab.bus.voters.remote import AgentVoter, AssertVoter, S1RubricVoter

    lazy: dict[str, Any] = {}

    def voters_for(d: Any) -> list[Any]:
        measures = {c.measure for c in vault.open(d.rubric_commitment, role="orchestrator").criteria}
        out: list[Any] = [DeterministicCheckVoter()]
        if "s1" in measures and s1_model:
            out.append(S1RubricVoter(s1_model) if s1_model != "default" else S1RubricVoter())
        if "llm" in measures:
            if "critic" not in lazy:
                from ci_lab.meta.spec_loader import load_spec
                from ci_lab.providers.factory import make_chat_client

                lazy["critic"] = make_chat_client(profile=profile, model=load_spec("critic").model, purpose="critic")
            out.append(AgentVoter(lazy["critic"]))
        if "assert" in measures:
            if "domain" not in lazy:
                from ci_lab.domain.order_support import OrderSupportDomain

                lazy["domain"] = OrderSupportDomain(work_dir=run_dir / "assert")
            out.append(AssertVoter(lazy["domain"]))
        return out

    return voters_for


class _BothChallengers:
    def __init__(self, *challengers: Any) -> None:
        self.challengers = challengers

    async def propose(self, spec: Any, view: Any, attempt: str) -> list[Any]:
        return [p for c in self.challengers for p in await c.propose(spec, view, attempt)]


def make_challenger(kind: str, profile: str) -> Any:
    from ci_lab.adversary.challenger import (
        ADVERSARY_SPEC,
        DeterministicChallenger,
        LLMAdversary,
    )

    if kind == "off":
        return None
    made = []
    if kind in ("det", "both"):
        made.append(DeterministicChallenger())
    if kind in ("llm", "both"):
        from ci_lab.meta.spec_loader import load_spec
        from ci_lab.providers.factory import make_chat_client

        client = make_chat_client(profile=profile, model=load_spec(ADVERSARY_SPEC).model, purpose="proposer")

        async def complete(prompt: str) -> str:
            return str((await client.get_response(prompt)).text)

        made.append(LLMAdversary(complete))
    return made[0] if len(made) == 1 else _BothChallengers(*made)


def make_optimizers(kind: str, profile: str, *, seed: int = 0) -> list[Any]:
    """Hardener proposers tried after template patches. ``gepa`` evolves soft-criterion questions with the
    optimizer reflection LM of ``profile`` (``ci_lab.optim.lm.make_lm``, as the ``gepa`` arm strategy)."""
    if kind == "off":
        return []
    from ci_lab.adversary.optim_adapters import GepaSoftQuestionAdapter
    from ci_lab.optim.lm import make_lm

    return [GepaSoftQuestionAdapter(reflection_lm=make_lm(profile, "optimizer"), seed=seed).propose]


def cmd_validate(a: argparse.Namespace) -> int:
    from ci_lab.taskgraph.model import load_graph
    from ci_lab.taskgraph.validate import validate_graph
    from ci_lab.taskgraph.vault import RubricVault

    problems = validate_graph(load_graph(a.graph), vault=RubricVault(a.vault) if a.vault else None)
    for p in problems:
        print(f"{p.code}\t{p.where}\t{p.message}")
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_seal(a: argparse.Namespace) -> int:
    from ci_lab.taskgraph.validate import validate_rubric
    from ci_lab.taskgraph.vault import RubricVault

    vault, rc = RubricVault(a.vault), 0
    for r in _load_rubrics(a.rubric):
        if problems := validate_rubric(r):
            rc = 1
            print("\n".join(f"{p.code}\t{r.version_id}\t{p.message}" for p in problems))
            continue
        print(f"{r.version_id}\t{r.deliverable}\t{vault.seal(r)}")
    return rc


def cmd_run(a: argparse.Namespace) -> int:
    from ci_lab.bus import ids
    from ci_lab.bus.wal import AgentBus
    from ci_lab.taskgraph.model import load_graph
    from ci_lab.taskgraph.validate import validate_graph
    from ci_lab.taskgraph.vault import RubricVault

    run_dir = Path(a.run_dir)
    if a.optimizer != "off" and a.challenger == "off":
        print("error: --optimizer needs --challenger (hardening only runs on exploits)")
        return 2
    run_id = ids.run_id(a.run_id or time.strftime("r%Y%m%d-%H%M%S"))
    graph, vault, bus = load_graph(a.graph), RubricVault.for_run(run_dir), AgentBus(run_dir / "bus")
    for r in _load_rubrics(a.rubric):
        vault.seal(r)
    if problems := validate_graph(graph, vault=vault):
        print("\n".join(f"{p.code}\t{p.where}\t{p.message}" for p in problems))
        return 1
    if a.engine == "maf":
        try:
            from ci_lab.taskgraph.maf_engine import run_graph_maf as engine
        except ImportError as exc:
            print(f"error: --engine maf unavailable ({exc}); use --engine asyncio")
            return 2
    else:
        from ci_lab.taskgraph.scheduler import run_graph as engine
    if a.student == "fake":
        from ci_lab.taskgraph.students import fake_student as student
    else:
        from ci_lab.taskgraph.students import AgentStudentFactory

        student = AgentStudentFactory(context_root=Path(a.graph).parent, work_root=run_dir / "students",
                                      profile=a.profile)
    challenger = make_challenger(a.challenger, a.profile)
    hardener = None
    if challenger is not None:
        from ci_lab.adversary.harden import Hardener

        hardener = Hardener(vault, run_dir / "artifacts", optimizers=make_optimizers(a.optimizer, a.profile))
    voters_for = make_voters_for(vault, run_dir=run_dir, profile=a.profile, s1_model=a.s1_model)
    if not a.no_telemetry:  # spans + ci.bus.append events -> <run-dir>/telemetry/spans-*.jsonl
        from ci_lab import telemetry

        telemetry.setup("taskgraph", profile=a.profile, run_dir=run_dir)
    from opentelemetry import trace

    with trace.get_tracer("ci_lab.bus.cli").start_as_current_span(  # parents the manifest append
            "ci.taskgraph.cli", attributes={"ci.run": run_id, "ci.graph": graph.id, "ci.engine": a.engine}):
        result = asyncio.run(engine(graph, bus=bus, vault=vault, voters_for=voters_for, student_factory=student,
                                    run_id=run_id, pools=pools_from_env(), max_parallel=a.max_parallel,
                                    challenger=challenger, hardener=hardener, quorum=a.quorum,
                                    adopt_epoch_patches=not a.no_adopt_hardened))
    doc = result.to_json()
    (run_dir / "graph_result.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc, indent=2) if a.json else _summary(doc))
    return 0 if doc["ok"] else 1


def _summary(doc: dict[str, Any]) -> str:
    lines = [f"run {doc['run']}: ok={doc['ok']} wall_s={doc['wall_s']:.2f} critical_path={doc['critical_path']}"]
    for t, r in doc["tasks"].items():
        lines.append(f"  {t:<16} {r['status']:<10} score={r['score']} attempts={r['attempts']}"
                     f" exploits={r['exploits']}" + (f" reason={r['reason']}" if r.get("reason") else ""))
    return "\n".join(lines)


def cmd_show(a: argparse.Namespace) -> int:
    from ci_lab.bus.wal import AgentBus

    run_dir = Path(a.run_dir)
    doc = json.loads((run_dir / "graph_result.json").read_text(encoding="utf-8"))
    print(_summary(doc))
    for topic, h in sorted(AgentBus(run_dir / "bus").heads(doc["run"]).items()):
        print(f"  head {topic} {h}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from ci_lab.bus.wal import AgentBus, BusCorrupt

    bus, bad = AgentBus(a.bus_dir), 0
    for topic in bus.topics():
        try:
            n = len(bus.read(topic))
            print(f"ok\t{topic}\t{n}")
        except BusCorrupt as exc:
            bad += 1
            print(f"CORRUPT\t{topic}\t{exc}")
    print(f"{len(bus.topics())} topic(s), {bad} corrupt")
    return 1 if bad else 0


def cmd_tail(a: argparse.Namespace) -> int:
    from ci_lab.bus.types import visibility
    from ci_lab.bus.wal import AgentBus

    entries = AgentBus(a.bus_dir).read(a.topic)
    shown = [e for e in entries if a.role in visibility(e)]
    for e in shown[-a.n:] if a.n > 0 else []:
        ref = "" if e.ref is None else f" ref={e.ref}"
        print(f"{e.seq} {e.kind} {e.author.role}:{e.author.name}{ref} {json.dumps(e.body.to_json(), sort_keys=True)}")
    if hidden := len(entries) - len(shown):
        print(f"({hidden} entr{'y' if hidden == 1 else 'ies'} not visible to {a.role})")
    return 0


def cmd_heads(a: argparse.Namespace) -> int:
    from ci_lab.bus.wal import AgentBus

    for topic, h in sorted(AgentBus(a.bus_dir).heads(a.run).items()):
        print(f"{topic}\t{h}")
    return 0


def register(sub: Any) -> None:
    g = sub.add_parser("graph", help="validate / seal / run / show a task graph").add_subparsers(
        dest="graph_command", required=True)
    p = g.add_parser("validate", help="print graph problems (exit 1 on any)")
    p.add_argument("graph")
    p.add_argument("--vault", help="rubric vault dir: also check commitments, rubrics and leakage")
    p.set_defaults(func=cmd_validate)
    p = g.add_parser("seal", help="seal rubric(s) into a vault; prints version, deliverable, commitment")
    p.add_argument("rubric", nargs="+", help="rubric file(s): one rubric, a list, or {rubrics: [...]}")
    p.add_argument("--vault", required=True)
    p.set_defaults(func=cmd_seal)
    p = g.add_parser("run", help="run a task graph on the agent bus; writes <run-dir>/graph_result.json")
    p.add_argument("graph")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-id")
    p.add_argument("--rubric", action="append", default=[], help="seal these rubric file(s) into the run vault first")
    p.add_argument("--max-parallel", type=int)
    p.add_argument("--challenger", choices=("off", "det", "llm", "both"), default="off")
    p.add_argument("--optimizer", choices=("off", "gepa"), default="off",
                   help="hardener proposer after template patches (gepa: soft-question GEPA, Wilson-gated)")
    p.add_argument("--engine", choices=("asyncio", "maf"), default="asyncio")
    p.add_argument("--quorum", type=int, default=1)
    p.add_argument("--student", choices=("fake", "agent"), default="agent")
    p.add_argument("--profile", choices=("copilot", "offline"), default="copilot")
    p.add_argument("--s1-model", default="default", help="System-1 judge model for s1 criteria ('' = no s1 voter)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-telemetry", action="store_true", help="skip <run-dir>/telemetry span export")
    p.add_argument("--no-adopt-hardened", action="store_true",
                   help="start from the pinned rubrics, not earlier runs' accepted hardening in this --run-dir")
    p.set_defaults(func=cmd_run)
    p = g.add_parser("show", help="summarize <run-dir>/graph_result.json and the bus heads")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_show)

    b = sub.add_parser("bus", help="inspect an agent-bus WAL").add_subparsers(dest="bus_command", required=True)
    p = b.add_parser("verify", help="verify every topic hash chain (exit 1 on corrupt)")
    p.add_argument("bus_dir")
    p.set_defaults(func=cmd_verify)
    p = b.add_parser("tail", help="last entries of a topic visible to --role")
    p.add_argument("bus_dir")
    p.add_argument("topic")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--role", choices=sorted(ROLES), default="orchestrator")
    p.set_defaults(func=cmd_tail)
    p = b.add_parser("heads", help="head hash per topic of a run")
    p.add_argument("bus_dir")
    p.add_argument("run")
    p.set_defaults(func=cmd_heads)
