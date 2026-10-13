# Self-hosted harness tree (`harness/`)

`harness/` holds the agent-facing assets of the self-improvement harness itself, so the loop can evolve
them like any other harness. Library code takes an explicit `harness_dir`. `None` means the repo-root
`harness/` (`ci_lab.harness_tree.repo_harness_dir()`), and library code never reads the environment.

| Path | Component (owner) | Contents |
|---|---|---|
| `harness.yaml` | frozen | byte-identical copy of `src/ci_lab/harness_tree/manifest.yaml` |
| `agents/*.yaml` | agent (agl) | evolvable specs: analyst, proposer, reflector, failure_analyst, student |
| `prompts/**/*.md` | prompt (gepa) | their instruction files (`common.md` + one per agent) |

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

## Snapshots

`snapshot(root)` returns `HarnessSnapshot(root, digest)`. The digest is a sha256 over the sorted relative
paths and bytes of every file in the tree. `__pycache__` is skipped, and a symlink fails.

## Installed wheels

The wheel ships `ci_lab` only, not `harness/`. When running from an installed wheel, set `CI_HARNESS_DIR` or
pass `--harness-dir`/`--dir` to the CLI. Only CLI entry points consult `CI_HARNESS_DIR`
(`ci_lab.harness_tree.default_root()`).
