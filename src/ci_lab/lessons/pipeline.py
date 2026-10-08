"""End-to-end phases used by the CLI and by sleep/campaign hooks: harvest → mine (cluster+route) → validate.

Each phase runs inside ``obs.span(SPAN_LESSONS, {ATTR_PHASE: ...})``. Outputs:

* ``<out>/trajectories.jsonl`` + ``<out>/harvest_report.json``;
* ``<run>/lessons/candidates.jsonl`` — one ``{"cluster": LessonCluster, "features": {...},
  "route_reasons": [...], "forced_structural": bool, "holdout_members": [ids]}`` per line (M17 input);
* ``<run>/lessons/mine_report.json``;
* ``<run>/lessons/confirmations.jsonl`` — human decisions (``ci-lab lessons confirm``), preserved across mines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ci_lab import contracts, obs
from ci_lab.lessons.cluster import MineConfig, load_confirmations, mine
from ci_lab.lessons.common import (
    CANDIDATES_FILE,
    HARVEST_REPORT,
    load_trajectories,
    save_trajectories,
    write_json_atomic,
    write_jsonl_atomic,
)
from ci_lab.lessons.common import lessons_dir as _lessons_dir
from ci_lab.lessons.harvest import HarvestOptions, HarvestStats, harvest
from ci_lab.lessons.registry import Registry
from ci_lab.lessons.replay import ReplayConfig, ReplayReport, validate
from ci_lab.lessons.route import route_all
from ci_lab.rulespec import LessonCluster, Trajectory, canonical_json

MINE_REPORT = "mine_report.json"


def _span(phase: str, **attrs: Any) -> Any:
    return obs.span(contracts.SPAN_LESSONS, {contracts.ATTR_PHASE: phase, **attrs})


def run_harvest(source: str, in_path: Path, out_dir: Path, opts: HarvestOptions | None = None
                ) -> tuple[list[Trajectory], HarvestStats]:
    stats = HarvestStats()
    with _span("harvest", **{"ci.lessons.source": source}) as s:
        trajs = harvest(source, in_path, opts, stats)
        total = save_trajectories(out_dir, trajs)
        report = {"source": source, **stats.as_dict(), "total_in_dir": total}
        write_json_atomic(out_dir / HARVEST_REPORT, report)
        s.set_attribute("ci.lessons.harvested", stats.harvested)
        s.set_attribute("ci.lessons.sealed_refused", stats.sealed)
    return trajs, stats


def run_mine(in_dir: Path, run_dir: Path, *, registry: Registry | None = None,
             config: MineConfig | None = None) -> list[dict[str, Any]]:
    """Cluster + route; writes ``<run>/lessons/candidates.jsonl``. Returns the written lines."""
    trajs = load_trajectories(in_dir)
    with _span("cluster") as s:
        res = mine(trajs, config, load_confirmations(run_dir))
        s.set_attribute("ci.lessons.clusters", len(res.clusters))
    with _span("route"):
        routed = route_all(res.clusters, res.members, trajs, registry)
    lines = [r.line(res.holdout_members.get(r.cluster.id, ())) for r in routed]
    ldir = _lessons_dir(run_dir)
    write_jsonl_atomic(ldir / CANDIDATES_FILE, (canonical_json(x) for x in lines))
    write_json_atomic(ldir / MINE_REPORT, {**res.report(),
                                           "routes": {r.cluster.id: r.cluster.route for r in routed}})
    return lines


def run_validate(rules_path: Path, in_dir: Path, *, cluster: LessonCluster | None = None,
                 dataset_texts: Sequence[str] = (), config: ReplayConfig | None = None,
                 engine: Any = None, work_dir: Path | None = None) -> ReplayReport:
    import yaml
    from pydantic import ValidationError

    from ci_lab.rulespec import RuleFile

    raw = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    trajs = load_trajectories(in_dir)
    with _span("replay", **({contracts.ATTR_LESSON: cluster.id} if cluster else {})) as s:
        try:
            rules: Any = RuleFile.model_validate(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            return ReplayReport(verdict="reject", ok=False, reasons=[f"invalid rule file: {str(exc)[:300]}"])
        report = validate(rules, trajs, cluster=cluster, dataset_texts=dataset_texts, config=config,
                          engine=engine, work_dir=work_dir)
        s.set_attribute("ci.lessons.verdict", report.verdict)
    return report
