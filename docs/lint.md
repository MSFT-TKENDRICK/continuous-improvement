# `ci-lab lint` and `ci-lab reflect`: lessons as structure

Design: §13.1 rung R5, §13.4 (dev loop), §13.6 B2/B7/B8, §12.5 D1–D7.

Following poteto's "encode lessons in structure", every lesson we learn while building the harness becomes a rule that **fails CI**. Coding agents cannot talk their way past a rule the way they can ignore prose in AGENTS.md. Pick the strongest rung that fits. Once a rule exists, any prose saying the same thing is just a symptom.

| Layer | What enforces it | Can it be bypassed? |
|---|---|---|
| `lint/rules/*.yaml` + `ci-lab lint` | rule data interpreted by frozen checks (`src/ci_lab/lint/`) | no inline suppressions; allowlists live only in the rule's `exclude` |
| `.githooks/pre-commit` | `ci-lab lint --staged` on the index content | `git commit --no-verify` skips it locally (see the extension below) |
| `.github/workflows/lint.yml` | the same lint on every PR and every push to `main` | no; **authoritative** |
| `.github/extensions/ci-guardrails` | Copilot CLI PreToolUse hook | denies the agent's bypass attempts before they run |

## Running

```sh
uv run ci-lab lint                     # whole repo (git ls-files + untracked, not ignored)
uv run ci-lab lint --staged            # only staged files, reading the index (what will be committed)
uv run ci-lab lint --paths src/ci_lab  # files or dirs
uv run ci-lab lint --format json       # {errors, warnings, files, rules, elapsed_s, findings[]}
uv run python -m ci_lab.lint lint      # same thing; works before `lint` is in ci_lab.cli.COMMAND_MODULES
```

Exit codes: `0` means clean (warnings are allowed), `1` means at least one `error` finding, `2` means the rules failed to load or git failed. The whole repo lints in well under a second. Each run is wrapped in an `obs.span("ci.lint")` with `ci.lint.{mode,files,rules,errors,warnings}` attributes.

Output follows poteto's `lint-arch` format:

```
[LINT][ERROR] src/ci_lab/rules/x.py:2
  Violation: [b7.rules-re2-only] Stdlib `re` used in the rule engine (...) (call re.compile)
  Fix: Compile with RE2 (`import re2`) ...
  See: design §13.6 B7
[LINT] Failed with 1 error(s), 0 warning(s) (2 file(s), 14 rule(s), 0.19s).
```

### Enabling the pre-commit hook

Run this once per clone (the hook is never enabled for you):

```sh
git config core.hooksPath .githooks
```

The hook is POSIX `sh` and also runs under Git for Windows. It calls `uv run --native-tls -q ci-lab lint --staged`, falls back to `python -m ci_lab.lint`, and blocks the commit on errors.

## Rule schema (`lint/rules/*.yaml`)

The schema is pydantic with `extra=forbid` (`src/ci_lab/lint/spec.py`):

```yaml
schema_version: 1
rules:
  - id: obs.single-tracer-provider          # ^[a-z][a-z0-9_.-]{2,63}$, unique across files
    kind: banned_call
    names: [opentelemetry.trace.set_tracer_provider, "*.set_tracer_provider"]
    include: ["src/**/*.py"]                # repo-relative posix globs; ** spans dirs; {a,b}
    exclude: ["src/ci_lab/telemetry/**"]    # the allowlist
    message: ...
    fix: ...
    see: design §12.4 C28, §12.5 D4
    severity: error                         # error | warn
```

| kind | fields | checks |
|---|---|---|
| `banned_call` | `names` | AST call names resolved through import aliases (including relative imports); `*.name` matches any receiver |
| `banned_import` | `modules` | `import`, `from … import`, relative imports, `importlib.import_module("…")`, `__import__("…")`; prefix match |
| `banned_attr_arg` | `calls`, `banned` | attribute values built with `str()`, `repr()`, f-strings, `.format()` or `%` in `set_attribute`, `obs.span`, `annotate`, and similar calls |
| `banned_text` | `pattern` | RE2 only, at most 256 characters, compiled when the rule loads |
| `gha_banned_trigger` | `triggers` | workflow `on:` keys (handles the YAML `on`→`True` quirk; unparseable YAML fails closed) |
| `gha_pinned_sha` | none | every `uses:` must be `owner/repo[/path]@<40-hex>`; `./local` and `docker://…@sha256:` are allowed |
| `declarative_yaml_expression_free` | `banned_kinds`, `root_kinds` | in MAF declarative docs: no `=` PowerFx strings and no `If`/`ConditionGroup`/`Foreach`/`GotoAction` |
| `max_lines` | `max` | file length |

## Seed rules

| id | lesson | see |
|---|---|---|
| `obs.single-tracer-provider` | one TracerProvider, installed by `ci_lab.telemetry` | C28, D4 |
| `obs.no-stringified-span-attrs` | no `str()`/`repr()`/f-strings of objects in span attributes | D6 |
| `obs.no-traceparent-attach-in-services` | `obs.attach_from_env` only in one-shot entrypoints | D2 |
| `obs.no-raw-traceparent-env` | read `TRACEPARENT` only through `ci_lab.obs` | D2 |
| `obs.status-writes-via-obs` | `status.json` / `status.d` written only by `ci_lab.obs` | D1 |
| `gha.no-pull-request-target` | no `pull_request_target` | I4 |
| `gha.pin-actions-by-sha` | actions pinned by full commit SHA | C9 |
| `maf.declarative-expression-free` | expression-free declarative YAML (no .NET/PowerFx) | I3 |
| `dotnet.no-powerfx-imports` | no `powerfx`/`pythonnet`/`clr` imports | I3 |
| `b2.guards-rules-closed-view` | guards/rules never import the oracle, `assert_ai`, `ci_lab.judge` or `datasets` | B2 |
| `b2.guards-rules-no-eval-paths` | guards/rules never reference eval split paths | B2 |
| `c15.lessons-no-sealed-splits` | lessons never reference heldout/ood/confirm data | C15 |
| `b7.rules-re2-only` | no stdlib `re` in `src/ci_lab/rules/**` | B7 |
| `ext.no-console-log` | no `console.log` in Copilot extensions (stdout is JSON-RPC) | SDK docs |

Current allowlists, each pre-existing and legitimate:
- `src/order_support/agent.py`: the order-support CLI installs its own provider.
- `src/order_support/{assert_wrapper,cli}.py`: one-shot entrypoints that may attach `TRACEPARENT`.

## ci-guardrails (Copilot CLI extension)

The Copilot CLI discovers `.github/extensions/ci-guardrails/extension.mjs` automatically. This extension is the dev-loop equivalent of noodle's `block-sleep` PreToolUse hook. The pure policy lives in `policy.mjs` and is tested with `node --test tests/ci_lab/lint/policy.test.mjs`. It denies:

- `git commit --no-verify` / `-n`, including short-option clusters and abbreviations
- `git -c core.hooksPath=…` overrides
- force-push (`-f`, `--force[-with-lease]`, `+ref`) or delete targeting `main`/`master`
- `git config` that sets `core.hooksPath` to anything other than `.githooks`, or unsets it
- edits or creates of frozen contracts (`src/ci_lab/contracts.py`, `src/ci_lab/rulespec.py`, `src/ci_lab/rules/templates.yaml`, `harness/guards/BUNDLE.lock`), including `apply_patch` and shell writes, unless `CI_ALLOW_CONTRACT_EDIT=1`

Deny reasons use the same `Violation`/`Fix`/`See` format as the lint output.

## `ci-lab reflect`: mining dev transcripts (opt-in, B8)

```sh
uv run ci-lab reflect --source copilot-sessions --i-consent-local-mining \
    [--sessions-dir ~/.copilot/session-state] [--min-sessions 2] [--keep] [--draft-pr]
```

- **Local and opt-in.** Without `--i-consent-local-mining` it exits 2 before reading anything. Nothing leaves the machine.
- **Repo-origin provenance.** Only sessions whose `session.start` cwd/gitRoot (or `workspace.yaml` cwd) is under this repo are read.
- **Deterministic detectors** produce `rulespec.CorrectionRecord`s:
  - `lint_fail`: `Violation: [<known rule id>]` in tool output
  - `revert`: `git checkout -- / restore / reset --hard / revert` of files the agent edited
  - `repeated_fix`: at least 2 edits of the same `dir/*.ext` after failures
  - `user_correction`: a correction phrase (don't/never/stop/again/…) plus a trigger from the trusted catalog `src/ci_lab/lint/templates.yaml`, recorded as `rule_hint`
- **Privacy.** Raw text is inspected in memory and discarded. Events that look like secrets (tokens, keys, JWTs) are dropped whole and counted; user messages and commands are also dropped for PII (emails, phones, SSNs).
  - Outputs use a closed vocabulary only: template ids and their trusted rule text, existing rule ids, `dir/*.ext` globs for directories that exist in the repo, strict tool names, and counts. No excerpts, args, messages or hashes ever appear.
- **Convergence.** A lesson is proposed only when at least `--min-sessions` (≥2) distinct sessions agree. Proposed rules are `severity: warn` (shadow).
- **Output.** `artifacts/reflect/proposals.yaml` (gitignored); `--keep` adds `records.jsonl`.
- **`--draft-pr`** commits `lint/rules/lessons-<date>.yaml` to a new local branch `lessons/reflect-<date>` using git plumbing, so the working tree and index are untouched. It then prints the `git push … && gh pr create --draft …` command for you to run; it does not run it.
