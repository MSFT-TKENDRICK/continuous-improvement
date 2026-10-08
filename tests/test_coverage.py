"""Every live suite's test_set targets every taxonomy category, per ASSERT's own assignment builder."""

import asyncio
import json
import random
from pathlib import Path

import pytest
from assert_ai.runner import _load_context
from assert_ai.stages.stratification import normalize_stratification, run_stratification
from assert_ai.stages.test_set import build_generation_jobs, validate_sampling_config

from order_support import replay

LIVE_SUITES = [p for p in sorted((replay.REPO_ROOT / "evals" / "assert").glob("*/eval_config.yaml"))
               if p.parent.name != "judge_replay"]


def _assignments(path: Path, tmp_path: Path, seed: int) -> tuple[set[str], dict[str, list[str]]]:
    """Mirror assert_ai.stages.test_set.run/run_test_set/_generate_records without the LLM call."""
    ctx = _load_context(config=str(path), overrides=[f"artifacts_root={tmp_path}"])
    cfg = dict(ctx["stages"])["test_set"]
    taxonomy_path = path.parent / cfg["taxonomy_path"]
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    result = asyncio.run(run_stratification(taxonomy_path=str(taxonomy_path), out_dir=str(tmp_path),
                                            dimensions=cfg["stratify"]["dimensions"],
                                            context=ctx["context"]))
    raw = json.loads(Path(result["stratification_path"]).read_text(encoding="utf-8"))
    stratification = normalize_stratification(raw, taxonomy, inject_behavior=True)
    by_kind = {}
    for kind in ("prompt", "scenario"):
        kind_cfg = cfg[kind]
        sampling = validate_sampling_config(kind_cfg.get("sampling"), field_name=f"test_set.{kind}.sampling")
        _, rows = build_generation_jobs(taxonomy=taxonomy, stratification=stratification,
                                        sample_size=int(kind_cfg["sample_size"]),
                                        rng=random.Random(seed), sampling=sampling)
        by_kind[kind] = [row["behavior"] for row in rows or []]
    return {c["name"] for c in taxonomy["behavior_categories"]}, by_kind


@pytest.mark.parametrize("path", LIVE_SUITES, ids=lambda p: p.parent.name)
def test_every_category_is_targeted(path, tmp_path):
    # ASSERT's test_set.run() always uses seed 0; prompts alone must cover for any seed.
    for seed in range(5):
        categories, by_kind = _assignments(path, tmp_path / str(seed), seed)
        assert set(by_kind["prompt"]) == categories
        assert len(by_kind["prompt"]) == len(categories)
        assert len(by_kind["scenario"]) == len(set(by_kind["scenario"])) == 2
