# Rules engine (`ci_lab.rules`)

Lessons learned from traces are encoded **as structure** (design §13): declarative YAML rules
(`ci_lab.rulespec` models) interpreted by this frozen, pure engine. The model cannot bypass a
rule. It only sees the rendered remediation, which is template text from a trusted catalog.

The engine is pure. It has no OTel state, no network and no LLM. Its only dependencies are
`ci_lab.rulespec`, `pyyaml`, `google-re2` and `pydantic`. Python's `re` is never used.

## Public API

```python
from ci_lab.rules import (Bundle, RuleLoadError, load_bundle, load_with_lkg, default_templates,
                          compute_flags, Match, evaluate, evaluate_trajectory, redact,
                          write_lkg, build_bundle, load_templates, Problem, REDACTED, TRAJECTORY_END)

load_bundle(rule_paths, extractor_paths=(), *, templates=None) -> Bundle
load_with_lkg(guards_dir, lock, extractor_paths=(), *, templates=None) -> tuple[Bundle, bool]
write_lkg(guards_dir, bundle, lock=None) -> Path          # lock defaults to <guards_dir>/BUNDLE.lock
build_bundle(rules, extractors=(), *, templates=None) -> Bundle   # in-memory, same validation
default_templates() -> Mapping[str, TemplateSpec]         # trusted catalog (templates.yaml)
evaluate(bundle, view: GuardView, *, on: "tool_call"|"response"|"trajectory") -> list[Match]
evaluate_trajectory(bundle, steps) -> list[Match]
compute_flags(bundle, steps) -> dict[str, set[str]]       # flag -> normalized subjects valid at end
redact(text, matches, bundle) -> str
```

`Match(rule, message, fix, see, step_index)` holds the rendered template text.
`Bundle` exposes `rules` (sorted by `(rung, id)`), `extractors`, `templates`, `digest`
(`rulespec.bundle_digest(rules)`), `config_digest` (the rules digest plus extractors and the
templates in use), `sources`, `source_digests`, `rule(id)`, `rules_for(on)`, `render(rule)`,
`pattern(p)` and `redact_patterns(rule)`.

`RuleLoadError.details` is a `list[Problem(file, violation, fix)]` that contains **every**
problem found. Loading is transactional: either the whole bundle loads or nothing does.

## Firing

`fires = when AND NOT require`. A missing `when` counts as true. A rule therefore states the
invariant (`require`) that must hold whenever its context (`when`) applies. Every firing rule is
returned (all-match telemetry). Matches at one step are ordered by `(rung, id)`.

## Targets and `on`

| `on` | pending step | rules considered |
|------|--------------|------------------|
| `tool_call` | the `tool_call` about to run | `target == "*"` or `target == pending.tool` |
| `response` | the assistant `response` about to be sent | `target == "*"` or `target == pending.tool` |
| `trajectory` (R4) | a completed `tool_call`, or `None` | see below |

`evaluate(view, on=...)` requires `view.pending` to be of the matching kind (`tool_call` or
`response`). Otherwise it raises `ValueError`. For `on="trajectory"` with `pending=None`, only
`target: "*"` trajectory rules run. Context is the whole of `view.steps` and the step index is
`TRAJECTORY_END` (-1).

**R4 semantics** (`evaluate_trajectory`, used offline after a run completes):

1. For each `tool_call` step *i*, the engine evaluates the `tool_call` rules for that tool and
   `"*"`. It also evaluates the **trajectory rules whose `target` is exactly that tool name**.
   `prior` and `count` see only steps before *i*.
2. For each `response` step *i*, it evaluates the `response` rules.
3. Once at the end, it evaluates the **trajectory rules with `target: "*"`** against the whole
   trajectory, with `step_index = TRAJECTORY_END`.

Output order is step order, then `(rung, id)`, with end-of-trajectory matches last.

## Paths

The namespaces are `current.args.*` (the pending call), `prior.args.*` and `prior.result.*`
(the step bound by a `prior` predicate). Extractor paths use `args.*` and `result.*`.
Segments are `.name`, which applies only to dicts, and `[n]`, which applies only to lists
(`n >= 0`). Any missing segment or wrong container type makes the path **missing**, and a
predicate over a missing path is **false**. The exception is `op: exists`, which is true iff the
path is present (`value: false` inverts it, so true iff absent).

## Typing (no coercion)

- `bool` is not a number. Numbers (`int`/`float`) compare only with numbers, and strings only
  with strings. Any type mismatch makes the predicate **false**, including for `ne`.
- `eq/ne/lt/le/gt/ge` follow these rules. Ordering on other types is false.
- `in` is true iff some element is strictly equal to the value.
- `nin` is true iff the value is present and scalar, nothing in the list is strictly equal to
  it, and the list is empty or contains an element of the same type. A type mismatch is false,
  never vacuously true.
- `matches` is an RE2 **search** on a string value. The input is capped at `TEXT_MAX_BYTES`
  UTF-8 bytes and is never split mid-code-point.

## Tool status and pairing

A `tool_result` pairs with its `tool_call` by `call_id`. Without a usable `call_id`, it pairs
with the **earliest still-open call of the same tool**. A blocked result prefers a blocked call,
and a normal result takes a non-blocked call. A call's status is:

- `blocked` if the call step itself is blocked;
- otherwise the paired result's `status` (`ok`/`error`) if that result lies **before** the
  pending step;
- otherwise none (no result yet).

## Predicates

- **arg**: compares `path op value` using the typing rules above.
- **text**: an RE2 search over the pending response text, capped at `TEXT_MAX_BYTES`. It is
  false when there is no response text.
- **prior** `{tool, status, within, same, cmp, where}`: true iff some earlier `tool_call` step
  satisfies all of the following:
  - its tool is `tool` (`"*"` = any tool);
  - its status matches. `ok` and `error` must match exactly. `any` excludes `blocked` but
    accepts calls with no result yet;
  - with `within: n`, it is among the last *n* `tool_call` steps (all tools) before the pending
    step;
  - every `same: [a, b]` pair is equal. Strings are compared after
    `rulespec.normalize_subject` on both sides, numbers exactly, and missing values or
    mismatched types are unequal;
  - every `cmp {current, op, prior}` holds under typed comparison against **that same** step;
  - `where` holds with `prior.args` / `prior.result` bound to that step.
- **count** `{tool, status, op, n}`: counts the earlier `tool_call` steps of `tool` with a
  matching status (`any` counts every attempt, including blocked ones), then compares
  `count op n`.
- **state** `{flag, subject, value}`: true iff the flag is currently set for the normalized
  subject. With no `subject`, it is true if the flag is set for any subject. `value: false`
  negates the result. A missing or non-scalar subject value makes the predicate false.
- **all / any / not**: boolean composition.

### Flags (extractors)

Flags come **only** from trusted extractors in `compute_flags`. Model-supplied `step.flags` are
ignored. An extractor `{flag, tool, result_path, equals, subject, ttl_steps}` produces an event
for every paired, non-blocked call of `tool` whose result has `status: ok` and a dict `result`.
The event's subject is `normalize_subject(<subject path>)`, which must be a string or number.

- The **latest** event per `(flag, subject)` wins. If `result_path == equals` (strict
  equality), the flag is set. Otherwise it is **revoked**, so a later non-verifying result clears
  an earlier verification.
- **TTL**: the flag is valid while (`tool_call` steps before the pending step) − (`tool_call`
  steps before the setting call) `< ttl_steps`.
- A state flag verified for resource A does **not** satisfy
  `subject: current.args.resource_id` for resource B.

## Redaction

`redact(text, matches, bundle)` takes the `require` text patterns of the **matched** rules whose
`action` is `redact`. It computes every span against the **original full text**, so later rules
also see the original text and nothing past the evaluation cap leaks. Overlapping and adjacent
spans are merged, and each span is replaced by `REDACTED` (`"[redacted]"`). Existing tokens are
opaque, so `redact(redact(t)) == redact(t)`.

## RE2 safety

All `matches`/`text` patterns are compiled with `re2` at load time. A pattern longer than
`REGEX_MAX_LEN` (256), or one that RE2 rejects (backreferences, lookaround, invalid classes), is
a load error. Evaluation is linear-time, and the fuzz tests run every pattern against
adversarial inputs (long repeats, Unicode) under a hard time budget.

## Loader checks

YAML is parsed with YAML 1.2 booleans, so `on:` stays a string. Duplicate keys are rejected,
and unknown keys are rejected by the `RuleFile`/`ExtractorFile` models. The loader also reports:

- duplicate rule ids and duplicate extractors (same `flag, tool, result_path, subject`);
- unknown template ids, and `slots` that differ from the template's slot set (they must match
  exactly);
- RE2 errors, and `exists` values that are not bool;
- `state` flags with no extractor (pass extractor paths);
- `redact` rules that have no text predicate in `require`;
- **conflicts** within the same `(target, on)`:
  - identical `when` + `require` but a different `action`/`mode`;
  - contradictory `require` conjuncts (after flattening `all`) when the `when`s are identical or
    absent. Contradictions are `eq`/`eq` with different constants, `eq`/`ne` with the same
    constant, `eq` vs `in` without that value, `eq` vs `nin` with that value, disjoint
    `in`/`in`, `exists` vs not-`exists`, and `state` true vs false.

## LKG (N1)

`write_lkg(guards_dir, bundle)` re-hashes every source file, copies the files to
`<guards_dir>/.lkg/` (removing stale copies) and writes the lock:

```json
{"digest": "<bundle digest>", "files": {"rules.yaml": "sha256:<hex>"}}
```

`load_with_lkg(guards_dir, lock)` loads `<guards_dir>/*.yaml|*.yml`:

- If that succeeds, it returns `(bundle, False)`.
- Otherwise it loads the `.lkg` copies, which must have basename-only names, hashes matching the
  lock and a digest equal to `lock.digest`, and returns `(bundle, True)`.
- If both fail, it raises `RuleLoadError` with both sets of problems.
- A missing `guards_dir` gives an empty bundle.

## Template catalog

`src/ci_lab/rules/templates.yaml` is the trusted catalog. Its message and fix text is
imperative, contains no data, and slots are only `{name}` substitutions. It contains:

- preconditions: `precondition.missing`, `precondition.same_subject`,
  `precondition.condition_unmet`, `precondition.prior_call`, `precondition.state_flag`
- ordering: `sequence.required`, `verify.before_lookup`
- arguments: `arg.out_of_range`, `arg.not_allowed`, `arg.constraint`, `arg.exceeds_prior`,
  `amount.not_exceed_prior`, `count.exceeded`
- responses: `pii.redacted`, `response.redact_pattern`, `response.blocked`
- other: `trajectory.violation`, and `guard.terminal`, the terminal safe response after
  `MAX_GUARD_BLOCKS_PER_TURN` blocks

## CLI

```text
python -m ci_lab.rules.cli rules check <paths...> [--extractors ...] [--templates catalog.yaml]
python -m ci_lab.rules.cli rules eval --rules <paths...> [--extractors ...] --trajectory run.json
```

`check` prints `[RULES][ERROR] <file>` / `Violation:` / `Fix:` blocks and exits with 1, or
prints `[RULES][OK] N rule(s) ...` and exits with 0. `eval` prints the matches as JSON (rule,
rung, on, target, action, mode, step_index, message, fix, see). The trajectory may be a list of
`TrajectoryStep`, `{steps: [...]}` or a `Trajectory`. `register(subparsers)` is provided for
`ci-lab rules`; `"rules"` is listed in `ci_lab.cli.COMMAND_MODULES`.

## Seeds

The §13.7 seed lessons live as golden fixtures in `tests/ci_lab/rules/fixtures/seeds.yaml`
(with `extractors.yaml`) and are exercised by `tests/ci_lab/rules/test_rules_seeds.py`.
