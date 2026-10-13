# Harness suite rubrics

These rubrics use `ci_lab.taskgraph.model.Rubric`. Criteria with `role: quality`
alone contribute to the primary task score. Deterministic/ASSERT checks carry
most quality weight; System-1 checks only judge bounded explanatory quality and
never own correctness. Criteria with `role: resource` are reported as
`resource.<criterion_id>` and `resource_score`; they never enter primary quality.

All metric values are supplied by evaluator-side `RunMeter` or surface analysis.
Candidate text, tool arguments, and scripted model responses must not contain or
override measurements. Missing required metric values fail closed.

`wall_ms` is the runtime criterion for every suite. Proposal and taskgraph use
evaluator-measured `complexity_delta` for structural simplification, with an
optional edit-size budget. Triage, tool-use, and injection make no structural
simplification claim: their second resource criterion is `resource.efficiency`
using bounded calls or tokens.

Tier membership is defined only in `evals/datasets/harness.yaml`: `ci` runs every
case with profile `fake`; `evolve` runs at most three cases per suite with `k=1`;
`confirm` runs held-out finalists with `k=2`; `aa` aliases `evolve`. Hard runtime
caps come from `src/ci_lab/harness_tree/manifest.yaml` `caps.eval`, never cases.
