"""One-process-per-case harness target runner.

The parent domain supplies only copied candidate/fixture paths and explicit execution pins.  This
module owns target execution, evaluator-side measurements, deterministic checks and frozen rubric
grading, then writes one bounded JSON result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.domain.harness_scoring import grade_case, load_rubric
from ci_lab.domain.harness_suites import RUNNERS, Execution
from ci_lab.harness_tree import HarnessTree
from ci_lab.metrics import RunMeter

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ENV = "CI_PROFILE"
TARGET_MODEL_ENV = "CI_LAB_TARGET_MODEL"
JUDGE_MODEL_ENV = "CI_LAB_JUDGE_MODEL"
S1_URL_ENV = "CI_S1_LLAMA_URL"


def _violation(rule_id: str, detail: str, severity: str = "major") -> dict[str, str]:
    return {"rule_id": rule_id, "severity": severity, "detail": detail[:500]}


def _preflight(profile: str, tier: str, target_model: str, judge_model: str) -> None:
    if profile == "fake":
        if tier != "ci":
            raise ValueError("profile=fake is only valid for tier=ci")
        return
    if not target_model:
        raise ValueError(f"{tier} tier requires a pinned target model")
    if tier in ("evolve", "confirm"):
        if not judge_model:
            raise ValueError(f"{tier} tier requires a pinned judge model")
        if not os.environ.get(S1_URL_ENV):
            raise ValueError(f"{tier} tier requires ${S1_URL_ENV}")


def _case(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("case JSON must be an object")
    suite = str(data.get("behavior") or "")
    if suite not in RUNNERS or not str(data.get("test_case_id") or "").startswith(suite + "_"):
        raise ValueError(f"unsupported harness case {data.get('test_case_id')!r}/{suite!r}")
    if not isinstance(data.get("expected"), Mapping):
        raise ValueError("case expected oracle is missing")
    return data


def _budgets(harness_dir: Path) -> dict[str, int]:
    return {str(k): int(v) for k, v in HarnessTree(harness_dir).manifest["caps"]["eval"].items()}


def _budget_violations(metrics: Mapping[str, float], caps: Mapping[str, int]) -> list[dict[str, str]]:
    values = {
        "max_llm_calls": metrics.get("llm_calls", 0),
        "max_tool_calls": metrics.get("tool_calls", 0),
        "max_tokens": metrics.get("tokens", 0),
        "timeout_s": metrics.get("wall_ms", 0) / 1000.0,
    }
    return [
        _violation("budget.exceeded", f"{name}: {values[name]:g} > {cap:g}")
        for name, cap in caps.items() if float(values[name]) > float(cap)
    ]


async def run_case(
    row: Mapping[str, Any],
    *,
    harness_dir: Path,
    profile: str,
    tier: str,
    target_model: str,
    judge_model: str,
) -> dict[str, Any]:
    _preflight(profile, tier, target_model, judge_model)
    errors = HarnessTree(harness_dir).validate()
    if errors:
        raise ValueError("invalid candidate harness: " + "; ".join(errors))
    suite = str(row["behavior"])
    meter = RunMeter()
    with meter:
        execution: Execution = await RUNNERS[suite](
            row, harness_dir, profile, target_model, meter)
    measurements = {
        **meter.as_dict(),
        "output_chars": float(len(execution.text)),
        "output_lines": float(len(execution.text.splitlines())),
        **execution.measurements,
    }
    grade = grade_case(
        load_rubric(REPO_ROOT, suite),
        deterministic_score=execution.deterministic_score,
        text=execution.text,
        measurements=measurements,
        profile=profile,
        judge_model=judge_model,
    )
    violations = [*execution.violations, *grade.violations]
    budget = _budget_violations(measurements, _budgets(harness_dir))
    violations.extend(budget)
    score: float | None = 0.0 if budget else grade.score
    if execution.served_model != target_model:
        score = None
        violations.append(_violation(
            "model.mismatch", f"served {execution.served_model!r}, pinned {target_model!r}"))
    injection = "injection" in suite
    return {
        "score": score,
        "violations": violations,
        "metrics": measurements,
        "rubric_scores": grade.rubric_scores,
        "subscores": grade.subscores,
        "served_model": execution.served_model,
        "judge_model": grade.judge_model,
        "excerpt": "" if injection else " ".join(execution.text.split())[:400],
    }


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


async def _main(args: argparse.Namespace) -> int:
    try:
        row = _case(Path(args.case_json))
        doc = await run_case(
            row,
            harness_dir=Path(args.harness_dir).resolve(),
            profile=args.profile,
            tier=args.tier,
            target_model=args.target_model,
            judge_model=args.judge_model,
        )
    except Exception as exc:  # noqa: BLE001 - child reports a typed missing trial, never a traceback
        doc = {
            "score": None,
            "violations": [_violation("eval.runner_error", f"{type(exc).__name__}: {exc}")],
            "metrics": {},
            "rubric_scores": {},
            "subscores": {},
            "served_model": "",
            "judge_model": args.judge_model,
            "excerpt": "",
        }
    _write(Path(args.out), doc)
    return 0 if doc["score"] is not None else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ci_lab.domain.harness_runner")
    p.add_argument("--harness-dir", required=True)
    p.add_argument("--case-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--profile", required=True, choices=("fake", "copilot", "offline"))
    p.add_argument("--tier", required=True, choices=("ci", "evolve", "confirm"))
    p.add_argument("--target-model", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--trial", required=True, type=int)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_main(parser().parse_args(argv)))


def _run_sync(make: Callable[[], Awaitable[str]]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = asyncio.run(make())
        except BaseException as exc:  # noqa: BLE001 - re-raised synchronously
            box["error"] = exc

    thread = threading.Thread(target=worker, name="harness-assert-chat", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return str(box["value"])


def _match_case(message: str) -> dict[str, Any]:
    from ci_lab.domain.harness import load_cases
    from ci_lab.domain.harness_assert_wrapper import current_case_id

    case_id = current_case_id() or os.environ.get("CI_CASE_ID")
    cases = load_cases()
    if case_id:
        for case in cases:
            if case.case_id == case_id:
                return dict(case.row)
    norm = " ".join(message.split())
    for case in cases:
        seed = case.row.get("seed") or {}
        if str(seed.get("description") or "") in norm or str(seed.get("title") or "") in norm:
            return dict(case.row)
    raise LookupError("could not identify frozen harness case")


async def _chat(message: str) -> str:
    from ci_lab.domain.harness import HarnessDomain

    source = Path(os.environ.get("CI_HARNESS_DIR") or REPO_ROOT / "harness").resolve()
    profile = os.environ.get(PROFILE_ENV, "fake")
    tier = "ci" if profile == "fake" else os.environ.get("CI_EVAL_TIER", "evolve")
    target = os.environ.get(TARGET_MODEL_ENV, "fake-model" if profile == "fake" else "")
    judge = os.environ.get(JUDGE_MODEL_ENV, "s1/scripted/default" if profile == "fake" else "")
    domain = HarnessDomain(profile=profile, tier=tier, target_model=target, judge_model=judge)
    with tempfile.TemporaryDirectory(prefix="ci-harness-assert-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        candidate = root / "candidate"
        shutil.copytree(source, candidate)
        row = domain._copy_case(_match_case(message), root)  # copied evaluator fixture, never the repo
        result = await run_case(row, harness_dir=candidate, profile=profile, tier=tier,
                                target_model=target, judge_model=judge)
        if result["score"] is None:
            raise RuntimeError(result["violations"][0]["detail"])
        return str(result.get("excerpt") or "")


def chat(message: str, history: list[dict[str, Any]] | None = None) -> str:
    """ASSERT callable target for ``harness_*`` configs."""
    del history
    return _run_sync(lambda: _chat(message))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
