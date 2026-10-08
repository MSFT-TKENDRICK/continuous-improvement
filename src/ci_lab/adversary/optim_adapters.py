"""Optimizer adapters for the hardener (bus contract v2 §12, P12).

``GepaSoftQuestionAdapter`` evolves soft-criterion *questions* with the existing GEPA loop
(``optim/gepa.py``): questions are ``TextTarget`` files materialized in a temp dir, the
``EvolveScorer`` scores train-split corpus items with the hardener metric (exploit rejected / honest
accepted = 1) and the result is a non-template :class:`RubricCandidate` that must still pass
:func:`validate_candidate` on the held-out split. ``DspyAlignAdapter`` runs ``judge.align`` only with
≥4 human labels and only emits a new-epoch OES evaluator proposal; it never mutates a live rubric.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ci_lab.adversary.harden import (
    Corpus,
    CorpusItem,
    HardenDecision,
    RubricCandidate,
    Scorer,
    score_items,
    split_corpus,
    validate_candidate,
)
from ci_lab.judge import align, audit
from ci_lab.optim.gepa import GepaConfig, optimize_texts
from ci_lab.optim.scoring import CaseOutcome
from ci_lab.optim.targets import TextTarget
from ci_lab.taskgraph.model import Rubric
from ci_lab.taskgraph.vault import new_canary

__all__ = ["MIN_HUMAN_LABELS", "DspyAlignAdapter", "GepaSoftQuestionAdapter", "SoftQuestionScorer"]

MIN_HUMAN_LABELS = 4


def _with_questions(rubric: Rubric, questions: Mapping[str, str]) -> Rubric:
    crits = tuple(replace(c, check={**c.check, "question": questions[c.id].strip()}) if c.id in questions else c
                  for c in rubric.criteria)
    return replace(rubric, criteria=crits)


class SoftQuestionScorer:
    """``EvolveScorer``: writes candidate questions under ``root``, re-reads them into the rubric, scores items."""

    def __init__(self, rubric: Rubric, targets: Mapping[str, TextTarget], items: Mapping[str, CorpusItem],
                 scorer: Scorer, root: Path) -> None:
        self.rubric, self.targets, self.items, self.scorer, self.root = rubric, targets, items, scorer, root

    async def __call__(self, candidate: Mapping[str, str], case_ids: Sequence[str]) -> list[CaseOutcome]:
        for t in self.targets.values():
            t.write(self.root, candidate.get(t.id, t.read(self.root)))
        rubric = _with_questions(self.rubric, {cid: t.read(self.root) for cid, t in self.targets.items()})
        items = [self.items[c] for c in case_ids]
        accepted = await score_items(self.scorer, rubric, items)
        return [CaseOutcome(i.id, float(accepted[i.id] == (i.kind == "honest"))) for i in items]


class GepaSoftQuestionAdapter:
    """``propose`` is a :data:`~ci_lab.adversary.harden.CandidateProposer` for ``Hardener(optimizers=...)``;
    a ``scorer`` passed by the hardener (the one it gates with) overrides the constructor's."""

    def __init__(self, scorer: Scorer | None = None, *, reflection_lm: Any, config: GepaConfig | None = None,
                 seed: int = 0, rng: random.Random | None = None) -> None:
        self.scorer, self.reflection_lm, self.seed, self.rng = scorer, reflection_lm, seed, rng
        self.config = config or GepaConfig(seed=seed)

    async def propose(self, rubric: Rubric, corpus: Corpus, *, scorer: Scorer | None = None) -> RubricCandidate | None:
        """GEPA over the train split only; ``None`` when there is nothing to evolve or nothing changed."""
        scorer = scorer or self.scorer
        if scorer is None:
            raise ValueError("GepaSoftQuestionAdapter needs a scorer")
        train, _ = split_corpus(corpus, self.seed)
        items = {i.id: i for i in (*train.exploits, *train.honest)}
        soft = rubric.soft()
        if not soft or not items:
            return None
        targets = {c.id: TextTarget(f"soft/{c.id}.md") for c in soft}
        with tempfile.TemporaryDirectory(prefix="ci-soft-q-") as tmp:
            root = Path(tmp)
            for c in soft:
                targets[c.id].write(root, str(c.check["question"]))
            seed = {t.id: t.read(root) for t in targets.values()}
            evolve = SoftQuestionScorer(rubric, targets, items, scorer, root)
            result = await optimize_texts(seed, evolve, sorted(items), reflection_lm=self.reflection_lm,
                                          config=self.config)
        best = {cid: result.best.get(t.id, seed[t.id]) for cid, t in targets.items()}
        if all(best[cid].strip() == seed[targets[cid].id].strip() for cid in best):
            return None
        new = _with_questions(rubric, best)
        new = replace(new, version=rubric.version + 1, canary=new_canary(self.rng))
        return RubricCandidate(new, "gepa", False, ())

    async def harden(self, rubric: Rubric, corpus: Corpus) -> HardenDecision | None:
        """Propose, then gate with exactly the same :func:`validate_candidate` as template patches."""
        cand = await self.propose(rubric, corpus)
        if cand is None or self.scorer is None:  # propose raised already when there is no scorer
            return None
        return await validate_candidate(rubric, cand, corpus, self.scorer, seed=self.seed)


class DspyAlignAdapter:
    """``judge.align`` behind a ≥4 human-label gate; output is a new-epoch OES proposal file only."""

    def __init__(self, *, config: Path | str, labels: Path | str, transcripts: Path | str, lm: Any,
                 artifacts_dir: Path | str, **align_kwargs: Any) -> None:
        self.config, self.labels, self.transcripts, self.lm = Path(config), Path(labels), Path(transcripts), lm
        self.out_dir = Path(artifacts_dir) / "adversary" / "align"
        self.align_kwargs = align_kwargs

    def human_labels(self) -> int:
        """Distinct cases with at least one non-skipped human label."""
        if not self.labels.is_file():
            return 0
        return len({case for (case, _), v in audit.load_labels(self.labels).items() if v is not None})

    def run(self) -> Path | None:
        if self.human_labels() < MIN_HUMAN_LABELS:
            return None
        proposal = dict(align.run_align(config=self.config, labels=self.labels, transcripts=self.transcripts,
                                        out_dir=self.out_dir, lm=self.lm, **self.align_kwargs))
        proposal.update({"adopt": False, "requires_new_epoch": True, "x-ci-source": "adversary.dspy_align",
                         "x-ci-epoch": "new"})
        out = self.out_dir / "evaluator_proposal.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(proposal, indent=2, default=str), encoding="utf-8")
        return out
