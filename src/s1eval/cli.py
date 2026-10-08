"""Command-line interface: ``s1eval {validate,conformance,serve,run,report,export-langsmith}``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .backends import LlamaCppLogprobBackend, make_backend
from .dataset import load_cases, validate_cases
from .metrics import compute_all
from .probes import (
    probe_batch_vs_single,
    probe_choice_order,
    probe_complement,
    probe_distractor,
    probe_variant_backend,
)
from .report import render
from .rubric import Rubric
from .runner import manifest, read_jsonl, run_cases, write_jsonl

ALL_PROBES = ("complement", "choice_order", "code_permutation", "distractor", "batch_vs_single")


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _backend_from_args(a: argparse.Namespace):
    return make_backend(
        a.backend,
        base_url=a.base_url,
        model=a.model,
        key_env=a.key_env,
        choice_permutations=a.choice_permutations,
        min_valid_mass=a.min_valid_mass,
    )


def cmd_validate(a: argparse.Namespace) -> int:
    rubric = Rubric.load(a.rubric)
    cases, sha = load_cases(a.dataset)
    validate_cases(cases, rubric)
    gold = [rubric.gold_pass(c.get("labels") or {}) for c in cases]
    human = [(c.get("labels") or {}).get("human_pass") for c in cases]
    disagree = [c["id"] for c, g, h in zip(cases, gold, human) if isinstance(h, bool) and g is not None and g != h]
    print(json.dumps({
        "cases": len(cases), "dataset_sha256": sha, "rubric_sha256": rubric.sha256,
        "gold_pass": {"pass": gold.count(True), "fail": gold.count(False), "ambiguous": gold.count(None)},
        "rule_vs_human_pass_disagreements": disagree,
    }, indent=2))
    return 0


def cmd_conformance(a: argparse.Namespace) -> int:
    be = LlamaCppLogprobBackend(a.llama_url)
    res = be.conformance()
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["passed"] else 1


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import make_server

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    be = LlamaCppLogprobBackend(a.llama_url, min_valid_mass=a.min_valid_mass, choice_permutations=a.choice_permutations)
    if not a.skip_conformance:
        res = be.conformance()
        if not res["passed"]:
            _p("conformance FAILED; refusing to serve:\n" + "\n".join(res["failures"]))
            return 1
        _p("conformance passed")
    token = os.environ.get(a.token_env) if a.token_env else None
    srv = make_server(be, a.host, a.port, token=token, model_name=a.model_name or be.model_label(),
                      insecure_allow_remote=a.insecure_allow_remote)
    _p(f"s1eval serving TypeSafe-compatible API on http://{a.host}:{srv.server_address[1]} (model {srv.model_name})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def _load_distractors(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [{"key": x["key"], "text": x["text"]} for x in d]


def cmd_run(a: argparse.Namespace) -> int:
    rubric = Rubric.load(a.rubric)
    cases, dsha = load_cases(a.dataset)
    validate_cases(cases, rubric)
    if a.limit:
        cases = cases[: a.limit]
    be = _backend_from_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conformance = None
    if isinstance(be, LlamaCppLogprobBackend) and not a.skip_conformance:
        conformance = be.conformance()
        (out / "conformance.json").write_text(json.dumps(conformance, indent=2, default=str), encoding="utf-8")
        if not conformance["passed"]:
            _p("conformance FAILED: " + "; ".join(conformance["failures"]))
            return 1
    _p(f"running {len(cases)} cases x {len(rubric.questions)} questions on {be.name}")
    records = run_cases(be, rubric, cases, repeats=a.repeats, progress=_p)
    write_jsonl(out / "records.jsonl", records)

    probes: dict[str, Any] = {}
    wanted = ALL_PROBES if a.probes == "all" else tuple(p for p in a.probes.split(",") if p and p != "none")
    pcases = cases[: a.probe_limit] if a.probe_limit else cases
    if "complement" in wanted and rubric.complements:
        probes["complement"] = probe_complement(be, rubric, pcases, records, _p)
    if "choice_order" in wanted:
        if isinstance(be, LlamaCppLogprobBackend) and be.choice_permutations > 1:
            probes["choice_order"] = {"skipped": "--choice-permutations > 1 already averages over option order, so "
                                                 "first-shown bias is not observable"}
        else:
            probes["choice_order"] = probe_choice_order(be, rubric, pcases, records, _p)
    if "code_permutation" in wanted and isinstance(be, LlamaCppLogprobBackend):
        variant = be.clone(code_seed=1 if be.code_seed != 1 else 2)
        probes["code_permutation"] = probe_variant_backend(variant, rubric, pcases, records, label="code_permutation", progress=_p)
    if "distractor" in wanted:
        probes["distractor"] = probe_distractor(be, rubric, pcases, records, _load_distractors(a.distractors), _p)
    if "batch_vs_single" in wanted:
        if isinstance(be, LlamaCppLogprobBackend):
            probes["batch_vs_single"] = {"skipped": "local backend already asks each question in its own request"}
        else:
            probes["batch_vs_single"] = probe_batch_vs_single(be, rubric, pcases, records, _p)

    man = manifest(be, rubric, dsha, {"s1eval_version": __version__, "repeats": a.repeats, "n_cases": len(cases),
                                      "probes": list(probes), "probe_cases": len(pcases)})
    metrics = compute_all(records, rubric)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out / "probes.json").write_text(json.dumps(probes, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(man, metrics, probes, conformance, title=f"s1eval: {rubric.name} on {be.name}"),
                                   encoding="utf-8")
    _p(f"wrote {out / 'report.md'}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    rubric = Rubric.load(a.rubric)
    d = Path(a.dir)
    records = read_jsonl(d / "records.jsonl")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    probes = json.loads((d / "probes.json").read_text(encoding="utf-8")) if (d / "probes.json").exists() else None
    conf = json.loads((d / "conformance.json").read_text(encoding="utf-8")) if (d / "conformance.json").exists() else None
    metrics = compute_all(records, rubric)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (d / "report.md").write_text(render(man, metrics, probes, conf, title=f"s1eval: {rubric.name}"), encoding="utf-8")
    _p(f"wrote {d / 'report.md'}")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from .langsmith_integration import ui_setup_text

    print(ui_setup_text(Rubric.load(a.rubric)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="s1eval", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_eval(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rubric", default="evals/rubrics/order_support.yaml")
        p.add_argument("--dataset", default="evals/datasets/order_support.yaml")

    p = sub.add_parser("validate", help="validate rubric + dataset (no model calls)")
    common_eval(p)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("conformance", help="verify llama-server logprob assumptions")
    p.add_argument("--llama-url", default=os.environ.get("S1EVAL_LLAMA_URL", "http://127.0.0.1:8081"))
    p.set_defaults(fn=cmd_conformance)

    p = sub.add_parser("serve", help="serve a TypeSafe-compatible /v1/systemone backed by llama-server")
    p.add_argument("--llama-url", default=os.environ.get("S1EVAL_LLAMA_URL", "http://127.0.0.1:8081"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token-env", default="S1EVAL_SERVER_TOKEN", help="env var holding the bearer token (optional on loopback)")
    p.add_argument("--model-name")
    p.add_argument("--min-valid-mass", type=float, default=0.5)
    p.add_argument("--choice-permutations", type=int, default=1)
    p.add_argument("--insecure-allow-remote", action="store_true")
    p.add_argument("--skip-conformance", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("run", help="run the eval (+ probes) and write a report")
    common_eval(p)
    p.add_argument("--backend", default="local",
                   help="local | systemone | openai | scripted | typesafe | langsmith-semif | langsmith-jev | openrouter")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--key-env", help="env var holding the API key")
    p.add_argument("--choice-permutations", type=int, default=1)
    p.add_argument("--min-valid-mass", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--probes", default="all", help=f"'all', 'none' or comma list of {ALL_PROBES}")
    p.add_argument("--probe-limit", type=int, default=0, help="run probes on the first N cases only")
    p.add_argument("--distractors", default="evals/datasets/distractors.yaml")
    p.add_argument("--skip-conformance", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="recompute metrics + report from a run directory")
    p.add_argument("--rubric", default="evals/rubrics/order_support.yaml")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("export-langsmith", help="print questions JSON + setup for the LangSmith UI evaluator")
    p.add_argument("--rubric", default="evals/rubrics/order_support.yaml")
    p.set_defaults(fn=cmd_export)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
