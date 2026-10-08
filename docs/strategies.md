# `ci_lab.strategies` — arm strategies

Design: §11.2. Each arm's `ArmDirective.strategy` selects how its edits are proposed:

```python
from ci_lab.strategies import get_strategy
strategy = get_strategy(directive.strategy, proposer=..., domain=..., lm=..., committer=...)
edits = await strategy.propose(ctx)   # list[contracts.Edit], len <= ctx.directive.edit_budget
```

`get_strategy(name, **deps)` accepts a shared superset of dependencies. Each strategy keeps
only the keyword arguments its constructor accepts.

- An unknown name raises `UnknownStrategy` (a `KeyError`).
- A missing required dependency raises `TypeError`.
- `guard` (contracts v2.4, §13) is provided by `ci_lab.lessons_arm.strategy`
  (`EXTERNAL["guard"]`). The first `get_strategy("guard", ...)` imports that module and
  calls its `register()`, which calls `register_strategy("guard", GuardStrategy)`. If the
  import fails, `get_strategy("guard")` raises `UnknownStrategy`. `available()` lists the
  names that resolve. `gepa` is the DSPy-based strategy (`ci_lab.optim`); there is no
  separate `dspy` name.
- Text strategies treat `**/harness/guards/**` as frozen (§13 B2), so only the `guard`
  strategy can write guard rule bundles.

Importing the package does not import dspy, gepa or skillopt_sleep (C26).

| name | class | deps | default focus |
|---|---|---|---|
| `agent` | `AgentStrategy` | `proposer: Callable[[ArmContext], Awaitable[list[Edit]]]` (M8a meta-agent) | — |
| `gepa` | `GepaStrategy` | `domain` or `scorer`; optional `lm`, `config: GepaConfig`, `committer`, `evolve_cases`, `k`, glob overrides | `prompt` |
| `skillopt` | `SkillOptStrategy` | as gepa, `config: SkillOptConfig` | `skill` (+ sibling `memory.md` if `memory` is in focus) |

## Common behaviour

- **Span:** all work runs inside `obs.span("ci.optimizer", {ci.strategy, ci.experiment,
  ci.variant=<arm>, ci.profile, ci.edit_budget})`. Cost and edit counts are added with
  `obs.annotate` (`ci.optimizer.*`, `ci.edits`).
- **Edit budget:** `edit_budget < 1` returns `[]` without doing any work. Returning more
  than `edit_budget` Edits raises `EditBudgetExceeded`, and returning non-`Edit` objects
  raises `TypeError`. Text strategies only optimize, and only commit, up to `edit_budget`
  components.
- **Plain Edits (C20):** optimizer text is written into the arm worktree. Each changed
  component is committed separately with `git_commit`, which stages exactly that file and
  adds the Co-authored-by trailer. That commit becomes one `Edit(component, hypothesis,
  files, commit)`. The critic, ASSERT, RRSI and OES then apply to it exactly as they do to
  agent edits.
- **Evolve only:** the default scorer is `DomainEvolveScorer`, which evaluates candidates
  in a scratch copy under `<run_dir>/optimizer/<arm>-<strategy>-scratch`. Evolve case ids
  are taken from the first available source:
  1. explicit `evolve_cases`;
  2. `domain.splits()["evolve"]`;
  3. `scorer.evolve_cases()`;
  4. the case ids of `ctx.failures`.
- **LM:** the injected `lm`, or else `make_lm(ctx.profile, "optimizer")`.
- **Budget:** `ctx.budget_tokens` caps the optimizer's metric calls (C18). When it is too
  small for one iteration, the strategy returns no edits.

## Cost report (ΔC accounting)

`ArmResult` has no cost field, so text strategies write
`<run_dir>/optimizer/<arm>-<strategy>.json` with these fields:

- `experiment_id`, `arm`, `strategy`;
- `cost`: summed metric calls and budget, refused calls, scorer, reflection and total
  tokens, candidates, wall seconds;
- `runs`: per-optimizer-run targets, changed targets, note, seed/best evolve-val score and
  diagnostics;
- `edits`;
- `acceptance: "diagnostic-only"`.

The arm worker or round should add `cost.total_tokens` to the arm's ΔC.

SkillOpt also saves `best_skill.md` / `best_memory.md` under
`<run_dir>/optimizer/<arm>-skillopt/<skill-dir>/`.

Strategies do not call `obs.write_status`. Status markers belong to the arm worker
(`writer=<arm>`).
