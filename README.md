# CI Lab self-improving harness

This repository evaluates and improves its own agent-facing harness assets. The evolvable surface is
the repo-root [`harness/`](harness/) tree; evaluators, governance, selection, judges, and campaign
orchestration remain frozen under `src/ci_lab/`.

The five frozen ASSERT suites cover:

- `harness_triage`: attribute a typed failure to one harness component;
- `harness_proposal`: produce bounded edits to the allowed harness surface;
- `harness_taskgraph`: plan dependency-correct work under hidden rubrics;
- `harness_tool_use`: choose and invoke only exposed tools;
- `harness_injection`: treat tool output and evidence excerpts as untrusted data.

Suite cases live under `evals/assert/harness_*`, their measured rubrics under
`evals/rubrics/harness/`, and tier/split selection in `evals/datasets/harness.yaml`. The offline CI
tier uses scripted target responses and a scripted System-1 judge.

## Quick start

```powershell
uv sync
uv run ci-lab harness validate --dir harness
uv run ci-lab harness metrics --dir harness
uv run pytest
```

Run the deterministic harness evaluation tests with:

```powershell
uv run pytest tests/ci_lab/domain/test_harness.py tests/integration/test_reallib_assert.py
```

Live `evolve` and `confirm` tiers require pinned target and judge models. See
[`docs/models.md`](docs/models.md), [`docs/judge.md`](docs/judge.md), and
[`docs/providers.md`](docs/providers.md).

## Harness tree

`harness/harness.yaml` is the frozen manifest. It defines required agents, component globs, owners,
and caps. It must match `src/ci_lab/harness_tree/manifest.yaml`; candidates cannot widen their own
editable surface.

The editable components are:

- `harness/agents/`: declarative target-agent specs;
- `harness/prompts/`: prompt text;
- `harness/skills/`: reusable operating procedures;
- `harness/mcp/exposure.yaml`: bounded tool exposure and direct/code mode selection;
- `harness/tools/`: client-tool overlays;
- `harness/loops/loops.yaml`: target-agent execution knobs clamped by frozen caps;
- `harness/workflows/`: target-agent declarative workflows.

`harness/guards/` is guard-owned and excluded from the candidate surface.

## Improvement loop

RRSI campaigns create isolated arm branches, run the frozen harness evaluator, select admissible
changes, and publish accepted arms as draft pull requests. Resource measurements include surface
complexity, wall time, token counts, LLM calls, and tool calls; missing required measurements fail
closed. Nightly SkillOpt-Sleep targets harness skills. The AGL strategy proposes structural harness
changes. Nothing merges automatically.

Useful entry points:

```powershell
uv run ci-lab campaign new <campaign-id> --profile fake
uv run ci-lab campaign run <campaign-id> --profile fake --rounds 1
uv run ci-lab campaign status <campaign-id> --profile fake
uv run ci-lab template doctor
```

Scheduled workflows are opt-in through the `CI_HARNESS_ENABLED` repository variable. See
[`docs/template.md`](docs/template.md).

## Architecture and operations

- [`docs/harness.md`](docs/harness.md): architecture and module index
- [`docs/harness-tree.md`](docs/harness-tree.md): manifest, snapshots, validation, and metrics
- [`docs/campaign.md`](docs/campaign.md): campaign lifecycle and publishing
- [`docs/rrsi.md`](docs/rrsi.md): selection and resource caps
- [`docs/judge.md`](docs/judge.md): System-1 judge provider
- [`docs/governance.md`](docs/governance.md): ACS/AGT policy and audit
- [`docs/guards.md`](docs/guards.md): runtime guard engine
- [`docs/lint.md`](docs/lint.md): structural lint and frozen-path guardrails
- [`docs/canvas.md`](docs/canvas.md): dashboard canvas
- [`docs/chat.md`](docs/chat.md): experiment chat

## Template use

Choose **Use this template**, then initialize the copy:

```powershell
uv run ci-lab template init --owners "@my-org/agent-owners" --apply
uv run ci-lab template doctor
```

Initialization rewrites CODEOWNERS and clears template run history while preserving the repo-root
harness tree, frozen manifest and policy, harness ASSERT suites, workflows, and fake CI tier.
