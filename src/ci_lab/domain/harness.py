"""Self-hosted harness evaluation domain.

The candidate is always a validated :class:`ci_lab.harness_tree.HarnessTree`.  Each case/trial is
executed by :mod:`ci_lab.domain.harness_runner` in an isolated writable copy; the source candidate,
frozen cases, rubrics, registry and manifest are never made writable to target code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ci_lab.contracts import EvalResult, EvaluatorPin, FailureRecord, TaskScore, Violation
from ci_lab.harness_tree import FROZEN_MANIFEST, HarnessTree, HarnessTreeError, tree_digest
from ci_lab.metrics import surface_metrics
from ci_lab.mcp.codemode import ProcessTree

__all__ = [
    "DATASET_PATH",
    "DEFAULT_EVAL_CONCURRENCY",
    "FROZEN_GLOBS",
    "JUDGE_MODEL_ENV",
    "PROFILE_ENV",
    "SURFACE_GLOBS",
    "TARGET_MODEL_ENV",
    "HarnessCase",
    "HarnessDomain",
    "load_cases",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "evals" / "datasets" / "harness.yaml"
PROFILE_ENV = "CI_PROFILE"
TARGET_MODEL_ENV = "CI_LAB_TARGET_MODEL"
JUDGE_MODEL_ENV = "CI_LAB_JUDGE_MODEL"
S1_URL_ENV = "CI_S1_LLAMA_URL"
DEFAULT_EVAL_CONCURRENCY = 4
MAX_RESULT_BYTES = 1_000_000
INJECTION_MARKERS = ("injection",)
EXCERPT_CHARS = 400
_ENV_ALLOW = (
    "PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONPATH",
    "CI_COPILOT_SERVE_URL", "CI_COPILOT_SERVE_KEY", S1_URL_ENV,
    PROFILE_ENV, TARGET_MODEL_ENV, JUDGE_MODEL_ENV, "CI_META_MODEL",
)


def _manifest_globs() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], tuple[str, ...]]:
    tree = HarnessTree(REPO_ROOT / "harness")
    components = tree.component_globs()
    surface = tuple(dict.fromkeys(g for globs in components.values() for g in globs))
    frozen = tuple(dict.fromkeys((*tree.manifest["frozen"], "pyproject.toml", "uv.lock")))
    return components, surface, frozen


COMPONENT_GLOBS, SURFACE_GLOBS, FROZEN_GLOBS = _manifest_globs()


@dataclass(frozen=True)
class HarnessCase:
    case_id: str
    suite: str
    kind: str
    row: Mapping[str, Any]
    path: Path

    @property
    def category(self) -> str:
        dims = self.row.get("dimensions")
        return str(dims.get("behavior") if isinstance(dims, Mapping) else self.kind)

    @property
    def text(self) -> str:
        seed = self.row.get("seed")
        if not isinstance(seed, Mapping):
            return ""
        return "\n".join(str(seed.get(k) or "") for k in ("title", "description") if seed.get(k))


def _dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "ci_lab.harness.dataset.v1":
        raise ValueError(f"{path}: expected format ci_lab.harness.dataset.v1")
    return data


def load_cases(repo_root: Path = REPO_ROOT, dataset_path: Path = DATASET_PATH) -> list[HarnessCase]:
    """Load only frozen ``harness_*`` ASSERT cases named by the harness dataset."""
    doc = _dataset(dataset_path)
    suite_root = repo_root / str(doc["suite_root"])
    wanted = set()
    for value in doc["splits"].values():
        if isinstance(value, list):
            wanted.update(str(x) for x in value)
    cases: dict[str, HarnessCase] = {}
    for path in sorted(suite_root.glob("harness_*/test_set.jsonl")):
        suite = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("test_case_id") or "")
            if case_id in wanted:
                cases[case_id] = HarnessCase(case_id, suite, suite.removeprefix("harness_"), row, path)
    missing = sorted(wanted - set(cases))
    if missing:
        raise FileNotFoundError(f"frozen harness cases missing: {', '.join(missing)}")
    return [cases[k] for k in sorted(cases)]


def _hash_files(paths: Iterable[Path], *extra: str) -> str:
    h = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: p.as_posix()):
        h.update(path.relative_to(REPO_ROOT).as_posix().encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for value in extra:
        h.update(value.encode() + b"\0")
    return "sha256:" + h.hexdigest()


def _judge_provider(model: str) -> str:
    parts = model.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "s1" else (parts[0] if len(parts) > 1 else "unknown")


ProcessRunner = Callable[[Sequence[str], Path, Mapping[str, str], float], tuple[int, bytes, bytes, bool]]


def _run_process(command: Sequence[str], cwd: Path, env: Mapping[str, str],
                 timeout_s: float) -> tuple[int, bytes, bytes, bool]:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(list(command), cwd=cwd, env=dict(env), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=flags,
                            start_new_session=sys.platform != "win32")
    tree = ProcessTree(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return int(proc.returncode or 0), stdout, stderr, False
    except subprocess.TimeoutExpired:
        tree.kill()
        stdout, stderr = proc.communicate()
        return int(proc.returncode or 1), stdout, stderr, True
    finally:
        tree.close()


class HarnessDomain:
    """``contracts.Domain`` for the evolvable repo-root ``harness/`` tree."""

    name = "harness"
    surface_globs: Sequence[str] = SURFACE_GLOBS
    frozen_globs: Sequence[str] = FROZEN_GLOBS
    component_globs: Mapping[str, Sequence[str]] = COMPONENT_GLOBS

    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        dataset_path: Path | None = None,
        cases: Sequence[HarnessCase] | None = None,
        work_dir: Path | None = None,
        concurrency: int = DEFAULT_EVAL_CONCURRENCY,
        profile: str | None = None,
        tier: str | None = None,
        target_model: str | None = None,
        judge_model: str | None = None,
        python: str = sys.executable,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.dataset_path = Path(dataset_path or self.repo_root / "evals/datasets/harness.yaml")
        self._dataset = _dataset(self.dataset_path)
        self._cases = list(cases) if cases is not None else None
        self.work_dir = None if work_dir is None else Path(work_dir)
        self.concurrency = max(1, int(concurrency))
        self.profile = (profile or os.environ.get(PROFILE_ENV) or "fake").lower()
        self.tier = (tier or ("ci" if self.profile == "fake" else "evolve")).lower()
        self.target_model = target_model or os.environ.get(TARGET_MODEL_ENV) or (
            "fake-model" if self.profile == "fake" else "")
        self.judge_model = judge_model or os.environ.get(JUDGE_MODEL_ENV) or (
            "s1/scripted/default" if self.profile == "fake" else "")
        self.python = python
        self.process_runner = process_runner or _run_process
        self._details: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    def cases(self) -> list[HarnessCase]:
        if self._cases is None:
            self._cases = load_cases(self.repo_root, self.dataset_path)
        return self._cases

    def case(self, case_id: str) -> HarnessCase:
        try:
            return next(c for c in self.cases() if c.case_id == case_id)
        except StopIteration as exc:
            raise KeyError(case_id) from exc

    def splits(self) -> Mapping[str, Sequence[str]]:
        out: dict[str, Sequence[str]] = {}
        for name, value in self._dataset["splits"].items():
            if isinstance(value, Mapping) and "alias" in value:
                out[name] = tuple(self._dataset["splits"][str(value["alias"])])
            else:
                out[name] = tuple(value)
        return out

    def leak_corpus(self) -> Any:
        from ci_lab.tools.critic_checks import LeakCorpus, case_leak_material

        texts, titles = case_leak_material(c.row for c in self.cases())
        return LeakCorpus.build(texts, titles)

    def leak_texts(self) -> list[str]:
        return [c.text for c in self.cases() if c.text]

    @staticmethod
    def leak_literals() -> list[str]:
        return []

    @staticmethod
    def is_injection_suite(suite: str) -> bool:
        return any(marker in suite for marker in INJECTION_MARKERS)

    def _preflight(self) -> None:
        if self.tier == "ci" and self.profile != "fake":
            raise ValueError("harness ci tier requires profile=fake")
        if self.profile == "fake":
            return
        if not self.target_model:
            raise ValueError(f"harness {self.tier} tier requires pinned ${TARGET_MODEL_ENV}")
        if self.tier in ("evolve", "confirm"):
            if not self.judge_model:
                raise ValueError(f"harness {self.tier} tier requires pinned ${JUDGE_MODEL_ENV}")
            if not os.environ.get(S1_URL_ENV):
                raise ValueError(f"harness {self.tier} tier requires ${S1_URL_ENV}")

    def pin(self, served_judges: Iterable[str] = ()) -> EvaluatorPin:
        frozen = [
            self.dataset_path, FROZEN_MANIFEST,
            *sorted((self.repo_root / "evals/rubrics/harness").glob("*.yaml")),
            *sorted((self.repo_root / "evals/assert").glob("harness_*/test_set.jsonl")),
            *sorted((self.repo_root / "evals/assert").glob("harness_*/taxonomy.json")),
        ]
        tree = _hash_files(frozen, self.profile, self.tier, self.target_model, self.judge_model)
        return EvaluatorPin(tree, self.judge_model, _judge_provider(self.judge_model),
                            tuple(sorted(set(served_judges))))

    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult:
        self._preflight()
        split_name = "evolve" if split == "aa" else split
        if split_name not in self.splits():
            raise KeyError(split)
        source = Path(harness_dir).resolve()
        tree = HarnessTree(source)
        if errors := tree.validate():
            raise HarnessTreeError("invalid candidate harness: " + "; ".join(errors))
        before = tree_digest(source)
        ids = list(self.splits()[split_name])
        sem = asyncio.Semaphore(self.concurrency)
        jobs = [(case_id, trial) for case_id in ids for trial in range(max(1, int(k)))]

        async def one(case_id: str, trial: int) -> tuple[TaskScore, str | None]:
            async with sem:
                return await self._run_one(self.case(case_id), trial, source, before, split,
                                           experiment_id, variant)

        results = await asyncio.gather(*(one(case_id, trial) for case_id, trial in jobs))
        if tree_digest(source) != before:
            raise HarnessTreeError("source candidate changed during evaluation")
        return EvalResult(
            harness_tree=before,
            split=split,  # type: ignore[arg-type]
            pin=self.pin(judge for _, judge in results if judge),
            scores=[score for score, _ in results],
            surface=surface_metrics(source, HarnessTree(source).component_globs()),
        )

    async def _run_one(self, case: HarnessCase, trial: int, source: Path, harness_hash: str,
                       split: str, experiment_id: str, variant: str) -> tuple[TaskScore, str | None]:
        return await asyncio.to_thread(
            self._run_one_sync, case, trial, source, harness_hash, split, experiment_id, variant)

    def _run_one_sync(self, case: HarnessCase, trial: int, source: Path, harness_hash: str,
                      split: str, experiment_id: str, variant: str) -> tuple[TaskScore, str | None]:
        base = str(self.work_dir) if self.work_dir is not None else None
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"ci-harness-{case.case_id}-{trial}-", dir=base,
                                         ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            shutil.copytree(source, candidate)
            row = self._copy_case(case.row, root)
            case_json, out = root / "case.json", root / "outcome.json"
            case_json.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            env = self._child_env(root)
            command = [
                self.python, "-m", "ci_lab.domain.harness_runner",
                "--harness-dir", str(candidate), "--case-json", str(case_json), "--out", str(out),
                "--profile", self.profile, "--tier", self.tier,
                "--target-model", self.target_model, "--judge-model", self.judge_model,
                "--experiment-id", experiment_id, "--variant", variant, "--trial", str(trial),
            ]
            timeout = float(HarnessTree(source).manifest["caps"]["eval"]["timeout_s"])
            returncode, stdout, stderr, timed_out = self.process_runner(command, root, env, timeout)
            doc = self._read_outcome(out, returncode, stdout, stderr, timed_out)
        violations = tuple(
            Violation(str(v.get("rule_id") or "eval.invalid"), str(v.get("severity") or "major"),  # type: ignore[arg-type]
                      str(v.get("detail") or "")[:500])
            for v in doc.get("violations", [])
            if isinstance(v, Mapping)
        )
        metrics = doc.get("metrics") if isinstance(doc.get("metrics"), Mapping) else {}
        served_model = str(doc.get("served_model") or "") or None
        score_value = doc.get("score")
        score = float(score_value) if isinstance(score_value, int | float) and not isinstance(score_value, bool) else None
        if served_model and served_model != self.target_model:
            score = None
            violations = (*violations, Violation("model.mismatch", "major",
                                                  f"served {served_model!r}, pinned {self.target_model!r}"))
        task = TaskScore(
            case.case_id, trial, case.suite, score, violations,
            int(metrics.get("tokens_in") or 0), int(metrics.get("tokens_out") or 0), served_model,
            float(metrics.get("wall_ms") or 0), int(metrics.get("llm_calls") or 0),
            int(metrics.get("tool_calls") or 0),
            {str(k): float(v) for k, v in (doc.get("subscores") or {}).items()
             if isinstance(v, int | float) and not isinstance(v, bool)},
        )
        self._details[(harness_hash, case.case_id, split, trial)] = doc
        return task, str(doc.get("judge_model") or "") or None

    def _child_env(self, tmp: Path) -> dict[str, str]:
        env = {name: os.environ[name] for name in _ENV_ALLOW if os.environ.get(name)}
        env.update({
            "TEMP": str(tmp), "TMP": str(tmp), PROFILE_ENV: self.profile,
            TARGET_MODEL_ENV: self.target_model, JUDGE_MODEL_ENV: self.judge_model,
        })
        return env

    def _copy_case(self, row: Mapping[str, Any], root: Path) -> dict[str, Any]:
        fixtures = self.repo_root / str(self._dataset["fixture_root"])
        copied = root / "fixtures"
        shutil.copytree(fixtures, copied)

        def rewrite(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, str) and value.startswith(str(self._dataset["fixture_root"]).replace("\\", "/")):
                path, sep, fragment = value.partition("#")
                rel = Path(path).relative_to(Path(str(self._dataset["fixture_root"])))
                return str(copied / rel) + (sep + fragment if sep else "")
            return value

        return rewrite(dict(row))

    @staticmethod
    def _read_outcome(out: Path, returncode: int, stdout: bytes, stderr: bytes,
                      timed_out: bool) -> dict[str, Any]:
        if timed_out:
            return {"score": None, "violations": [{"rule_id": "eval.timeout", "severity": "major",
                                                    "detail": "runner process tree timed out"}], "metrics": {}}
        if not out.is_file():
            detail = f"runner exit {returncode}; stderr bytes={len(stderr)}; stdout bytes={len(stdout)}"
            return {"score": None, "violations": [{"rule_id": "eval.missing_result", "severity": "major",
                                                    "detail": detail}], "metrics": {}}
        data = out.read_bytes()
        if len(data) > MAX_RESULT_BYTES:
            return {"score": 0.0, "violations": [{"rule_id": "eval.result_too_large", "severity": "major",
                                                  "detail": f"{len(data)} > {MAX_RESULT_BYTES} bytes"}], "metrics": {}}
        try:
            doc = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"score": None, "violations": [{"rule_id": "eval.invalid_result", "severity": "major",
                                                   "detail": str(exc)[:500]}], "metrics": {}}
        if not isinstance(doc, dict) or not isinstance(doc.get("violations", []), list):
            return {"score": None, "violations": [{"rule_id": "eval.invalid_result", "severity": "major",
                                                   "detail": "runner result is not the expected object"}],
                    "metrics": {}}
        return doc

    def failures(self, result: EvalResult) -> list[FailureRecord]:
        grouped: dict[str, list[TaskScore]] = {}
        for score in result.scores:
            grouped.setdefault(score.case_id, []).append(score)
        failures: list[FailureRecord] = []
        for case_id, scores in sorted(grouped.items()):
            if all(s.score is not None and s.score >= 1.0 and not s.violations for s in scores):
                continue
            suite = scores[0].suite
            rules = {v.rule_id for s in scores for v in s.violations}
            rubric: dict[str, list[float]] = {}
            excerpt = ""
            for score in scores:
                if score.score is None:
                    rules.add("eval.missing_trial")
                detail = self._details.get((result.harness_tree, case_id, result.split, score.trial), {})
                for key, value in (detail.get("rubric_scores") or {}).items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        rubric.setdefault(str(key), []).append(float(value))
                if not self.is_injection_suite(suite) and not excerpt:
                    excerpt = str(detail.get("excerpt") or "")[:EXCERPT_CHARS]
            failures.append(FailureRecord(
                case_id, suite, self.case(case_id).category, tuple(sorted(rules)),
                {key: round(sum(values) / len(values), 4) for key, values in rubric.items()},
                "" if self.is_injection_suite(suite) else excerpt,
            ))
        return failures
