"""``ci-lab lessons`` — harvest | mine | validate | confirm.

Exit codes: 0 ok; 2 replay rejected / bad input; 3 sealed-split records were refused during harvest
(other records were still written); 4 confirm refused (non-human context).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ci_lab.lessons.common import (
    CANDIDATES_FILE,
    LESSONS_DIR,
    read_jsonl,
    write_jsonl_atomic,
)
from ci_lab.rulespec import canonical_json

NON_HUMAN_ENV = ("GITHUB_ACTIONS", "CI", "TF_BUILD", "CI_LAB_AGENT")


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("lessons", help="mine traces into lessons (harvest/mine/validate/confirm)")
    sub = p.add_subparsers(dest="lessons_cmd", required=True)

    h = sub.add_parser("harvest", help="adapt traces into rulespec.Trajectory JSONL")
    h.add_argument("--source", required=True, choices=["assert", "spans", "agl", "usage", "calibrate"])
    h.add_argument("--in", dest="in_path", required=True, type=Path)
    h.add_argument("--out", required=True, type=Path)
    h.add_argument("--split", default=None, choices=["evolve"],
                   help="assert records without a split are evolve (operator assertion)")
    h.add_argument("--pin", default=None, help="oracle+evaluator pin (fingerprint version)")
    h.add_argument("--slice", default=None, help="override slice (e.g. night id)")
    h.add_argument("--labels", type=Path, default=None, help="usage human labels JSONL {id, label}")
    h.set_defaults(func=_harvest)

    m = sub.add_parser("mine", help="family holdout split, fingerprint clustering, ladder routing")
    m.add_argument("--in", dest="in_path", required=True, type=Path)
    m.add_argument("--run", required=True, type=Path)
    m.add_argument("--registry", type=Path, default=None)
    m.add_argument("--min-support", type=int, default=3)
    m.add_argument("--min-slices", type=int, default=2)
    m.set_defaults(func=_mine)

    v = sub.add_parser("validate", help="replay rejection filter for candidate rules (never accepts)")
    v.add_argument("--rules", required=True, type=Path)
    v.add_argument("--in", dest="in_path", required=True, type=Path)
    v.add_argument("--run", type=Path, default=None, help="run dir holding lessons/candidates.jsonl")
    v.add_argument("--cluster", default=None)
    v.add_argument("--dataset", type=Path, action="append", default=[], help="eval dataset YAML (leak screen)")
    v.add_argument("--epsilon", type=float, default=None)
    v.add_argument("--out", type=Path, default=None)
    v.add_argument("--extractors", type=Path, action="append", default=[],
                   help="extractor YAML (required for rules using state flags)")
    v.set_defaults(func=_validate)

    c = sub.add_parser("confirm", help="human confirmation of a lesson cluster")
    c.add_argument("cluster_id")
    c.add_argument("--run", required=True, type=Path)
    c.add_argument("--by", required=True, help="reviewer name")
    c.add_argument("--reject", action="store_true")
    c.add_argument("--note", default="")
    c.add_argument("--yes", action="store_true", help="skip the interactive prompt (still human-only)")
    c.set_defaults(func=_confirm)


def _harvest(a: argparse.Namespace) -> int:
    from ci_lab.lessons.harvest import HarvestOptions, load_labels
    from ci_lab.lessons.pipeline import run_harvest

    opts = HarvestOptions(default_split=a.split, pin=a.pin, slice=a.slice, labels=load_labels(a.labels))
    _, stats = run_harvest(a.source, a.in_path, a.out, opts)
    print(json.dumps(stats.as_dict(), sort_keys=True))
    if stats.sealed:
        print(f"REFUSED {stats.sealed} sealed/unknown-split record(s) (B2/C15)", file=sys.stderr)
        return 3
    return 0


def _mine(a: argparse.Namespace) -> int:
    from ci_lab.lessons.cluster import MineConfig
    from ci_lab.lessons.pipeline import run_mine
    from ci_lab.lessons.registry import Registry

    reg = Registry.load(a.registry) if a.registry else None
    lines = run_mine(a.in_path, a.run, registry=reg,
                     config=MineConfig(min_support=a.min_support, min_slices=a.min_slices))
    print(json.dumps({"clusters": len(lines),
                      "routes": {x["cluster"]["id"]: x["cluster"]["route"] for x in lines}}, sort_keys=True))
    return 0


def _validate(a: argparse.Namespace) -> int:
    from ci_lab.lessons.common import load_clusters
    from ci_lab.lessons.pipeline import run_validate
    from ci_lab.lessons.replay import ReplayConfig, dataset_texts_from_yaml

    cluster = None
    if a.cluster:
        if a.run is None:
            print("--cluster needs --run", file=sys.stderr)
            return 2
        cluster = next((c for c in load_clusters(a.run) if c.id == a.cluster), None)
        if cluster is None:
            print(f"unknown cluster {a.cluster}", file=sys.stderr)
            return 2
    texts = [t for d in a.dataset for t in dataset_texts_from_yaml(d)]
    cfg = ReplayConfig(epsilon=a.epsilon) if a.epsilon is not None else None
    report = run_validate(a.rules, a.in_path, cluster=cluster, dataset_texts=texts, config=cfg,
                          extractors=a.extractors)
    body = report.model_dump_json(indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(body + "\n", encoding="utf-8")
    print(body)
    return 0 if report.ok else 2


def _non_human() -> str | None:
    for k in NON_HUMAN_ENV:
        if os.environ.get(k, "").strip().lower() in ("1", "true", "yes"):
            return k
    return None


def _confirm(a: argparse.Namespace) -> int:
    from ci_lab.lessons.cluster import (
        append_confirmation,
        apply_confirmations,
        load_confirmations,
    )

    if env := _non_human():
        print(f"refusing: lesson confirmation is a human decision ({env} is set)", file=sys.stderr)
        return 4
    path = a.run / LESSONS_DIR / CANDIDATES_FILE
    lines = list(read_jsonl(path)) if path.exists() else []
    if not any((x.get("cluster") or x).get("id") == a.cluster_id for x in lines):
        print(f"unknown cluster {a.cluster_id} in {path}", file=sys.stderr)
        return 2
    decision = "rejected" if a.reject else "confirmed"
    if not a.yes:
        if not sys.stdin.isatty():
            print("refusing: not an interactive terminal (pass --yes as a human operator)", file=sys.stderr)
            return 4
        ans = input(f"{decision} lesson cluster {a.cluster_id} as {a.by}? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            return 2
    append_confirmation(a.run, a.cluster_id, by=a.by, decision=decision, note=a.note)
    from ci_lab.rulespec import LessonCluster

    decisions = load_confirmations(a.run)
    out = []
    for x in lines:
        wrapped = "cluster" in x
        c = LessonCluster.model_validate(x["cluster"] if wrapped else x)
        c2 = apply_confirmations([c], decisions)[0].model_dump(mode="json")
        out.append({**x, "cluster": c2} if wrapped else c2)
    write_jsonl_atomic(path, (canonical_json(x) for x in out))
    print(json.dumps({"cluster_id": a.cluster_id, "decision": decision, "by": a.by}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci-lab")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
