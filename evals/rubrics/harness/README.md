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

Each case's `fake_script` is a JSON-safe serialization of `FakeChatClient` steps:
`{"text": "..."}` becomes a string response, and
`{"tool_calls": [{"name": "...", "arguments": {...}}]}` becomes a list of
`ci_lab.testing.Call` objects. L7 converts these records directly; no Python
callables are stored in the dataset.

ASSERT's supported model-pin locations are used unchanged:
`default_model.name` for the target and `pipeline.judge.model.name` for System-1.
Live `evolve` and `confirm` preflight resolves the required environment pins
against those fields; no harness-only keys are added to ASSERT configs.
