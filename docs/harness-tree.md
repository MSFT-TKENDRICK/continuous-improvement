# Self-hosted harness tree (`harness/`)

`harness/` holds the agent-facing assets of the self-improvement harness itself, so the loop can evolve
them like any other harness. Library code takes an explicit `harness_dir`. `None` means the repo-root
`harness/` (`ci_lab.harness_tree.repo_harness_dir()`), and library code never reads the environment.

| Path | Component (owner) | Contents |
|---|---|---|
| `harness.yaml` | frozen | byte-identical copy of `src/ci_lab/harness_tree/manifest.yaml` |
| `agents/*.yaml` | agent (agl) | evolvable specs: analyst, proposer, reflector, failure_analyst, student |
| `prompts/**/*.md` | prompt (gepa) | their instruction files (`common.md` + one per agent) |
| `loops/loops.yaml` | loop (rrsi) | per-agent execution knobs, clamped to the frozen caps |
| `tools/tools.yaml` | client_tool (agl) | per-agent tool exposure subsets and descriptions |
| `mcp/exposure.yaml` | mcp (agl) | MCP exposure (see `ci_lab.mcp`) |
| `workflows/*.yaml` | workflow (agl) | target-agent MAF declarative workflows: `triage`, `propose` |
| `skills/<name>/SKILL.md` | skill (skillopt) | `harness-editing`, `trace-triage` (on the proposer's `skills_paths`) |
| `guards/**` | guard (guard) | guard rule files, written only by the guard strategy (none yet) |

## Frozen manifest

`src/ci_lab/harness_tree/manifest.yaml` (`format: ci_lab.harness.v1`) is the authority:

- component globs (relative to `harness/`);
- owners (mirroring `ci_lab.contracts.COMPONENT_OWNERS`);
- frozen repo globs;
- required agents;
- caps.

`HarnessTree(root).validate()` fails when `harness/harness.yaml` differs from it (newlines normalized), so a
candidate tree cannot widen its writable surface or relax a cap. `config`, `memory` and `context_mgmt` have
no file surface yet and map to no globs.

## Loops and tools overlays

`loops/loops.yaml` (`format: ci_lab.harness.loops.v1`) holds only target-agent execution knobs:

- `agents.<name>.max_nudges`: re-prompts when a run ends without the terminal `submit_*` call (default: the
  meta manifest's `max_nudges`, 2).
- `agents.<name>.max_tool_calls`: tool calls per run. Past it a call is not run and returns an `ERROR`
  asking for the submission. The terminal tool is never counted or blocked.
- `agents.<name>.max_turns`: model calls per run, nudges included. Past it the model is not called and
  the run ends without a submission (`MetaAgentError`).
- `code_mode.max_runs`: code-mode sandbox runs per agent run. It is validated and exposed by
  `HarnessTree.loops()` for the code-mode tool; nothing in this layer enforces it yet.

Values above `caps` in the frozen manifest fail `validate()` and are clamped at load. Absent knobs mean
no limit (`max_nudges`: the default). Campaign retries, critic attempts and taskgraph/bus voters and quorum
are frozen in `src/` and never read `harness/`.

`tools/tools.yaml` (`format: ci_lab.harness.tools.v1`) maps `agents.<name>` to `{tool: description | null}`.
A listed agent is exposed only the listed tools, each of which its spec must already bind, and the terminal
tool must stay (`SpecError` otherwise): the overlay can narrow, never add, capabilities. A description
replaces the bound function's own; `null` keeps it.

`load_spec(..., harness_dir=)` applies both to tree specs (`MetaAgentSpec.max_nudges`, `max_tool_calls`,
`max_turns`, `tools`, `tool_descriptions`); `harness_builder` enforces them with `loop_limit_middleware(spec)`
(`ToolBudgetMiddleware`, `TurnLimitMiddleware`). Frozen specs (critic, judge, adversary) are unaffected.

## Target workflows and skills

`harness/workflows/` holds MAF declarative workflows for the target agents (not the campaign arm
workflows, which stay frozen in `src/ci_lab/workflows/`):

- `triage.yaml`: `CiFailureAnalyst`, then `CiAnalyst`.
- `propose.yaml`: `CiProposer`, then the `self_check` function step.

They follow the same expression-free contract as the arm workflows (`ci_lab.workflows.assert_expression_free`)
and run with `ci_lab.workflows.runtime.build_workflow`/`run_or_resume`. `validate()` also requires every
agent to be a spec name of the tree and every function to be in the manifest's `workflow_functions`.

Each `skills/<name>/SKILL.md` needs YAML frontmatter with `name: <name>` and a `description`
(SkillOpt/MAF skill format).

## Snapshots

`snapshot(root)` returns `HarnessSnapshot(root, digest)`. The digest is a sha256 over the sorted relative
paths and bytes of every file in the tree. `__pycache__` is skipped, and a symlink fails.

Each campaign round pins its incumbent: the first meta agent of a round copies the tree (`MetaAgents.harness_dir`,
default the repo-root `harness/`) to `<round dir>/meta-harness` and records the snapshot in
`<round dir>/meta-harness.json`. The analyst, every proposal and repair, and the critic of that round (and of
a resumed round) load from that copy, which must still match the recorded digest.

## CLI

```
ci-lab harness validate [--dir DIR] [--json]   # exit 1 when HarnessTree(DIR).validate() reports errors
ci-lab harness metrics [--dir DIR] [--json]    # ci_lab.metrics.simplicity.surface_metrics per manifest component
```

`DIR` defaults to `$CI_HARNESS_DIR`, else the repo-root `harness/` (`default_root()`).

## Installed wheels

The wheel ships `ci_lab` only, not `harness/`. When running from an installed wheel, set `CI_HARNESS_DIR` or
pass `--harness-dir`/`--dir` to the CLI. Only CLI entry points consult `CI_HARNESS_DIR`
(`ci_lab.harness_tree.default_root()`).
