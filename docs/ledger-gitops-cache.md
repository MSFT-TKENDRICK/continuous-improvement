# Ledger, gitops and cache (M6)

Durable state, git plumbing and shared caches for the self-improving harness
(design v1 §3–4; v2 C6, C8, C13, C14). Stdlib-only; Windows and POSIX.

## `ci_lab.ledger` — durable experiment state

| Module | API | Notes |
|---|---|---|
| `layout` | `Layout(repo)`, `run_dir(run_id)` | `experiments/campaigns/<cid>/{campaign.json, frontier.json, history.jsonl, rounds/<eid>/{experiment.json, decisions.json, eval/*.json}}`, `experiments/sleep/{state.json, nights/<date>/experiment.json}`, `experiments/holdout-looks.jsonl`. Ids are validated. Bulky run artifacts go under `CI_RUN_DIR` (default `./artifacts/runs`), not in the ledger (C14). |
| `atomic` | `atomic_write_{bytes,text,json}`, `read_json`, `append_jsonl`, `read_jsonl` | Same-dir temp file + fsync + `os.replace` (retried on Windows sharing violations) + POSIX dir fsync. JSONL readers skip a torn last line; appends repair it first. |
| `lock` | `FileLock`, `lock_for`, `ledger_lock`, `LockTimeout` | Cross-process exclusive lock (`msvcrt.locking` / `fcntl.flock`) with timeout; reentrant per thread / asyncio task; sync and `async with`. Lock files live in `<git-common-dir>/ci-lab/` so they never enter `experiments/`. |
| `commit` | `ledger_commit(repo, paths, message, *, ref="refs/heads/main", expected_old=None)` | Refuses any path (given or resulting from the tree diff) outside `experiments/**`. Builds the tree in a private `GIT_INDEX_FILE` seeded from the ref's parent, so the user's index is never touched, then `commit-tree` + `update-ref <ref> <new> <old>` (CAS). A moved ref raises `LedgerConflict`. Unchanged trees are a no-op. |
| `frontier` | `read_frontier(path)`, `cas_frontier(path, expected_incumbent, new, *, experiment_id=None)` | Compare-and-swap of the incumbent under a lock; `FrontierConflict` on mismatch; idempotent when already applied. |
| `decisions` | `record_decisions(repo, cid, eid, decisions)`, `read_decisions` | Atomic `rounds/<eid>/decisions.json`; `decision` must be `ship` / `do_not_ship` / `rerun`. |
| `outbox` | `FileOutbox(path)` (`contracts.Outbox`) | JSONL journal of `started` / `succeeded(result)` / `failed` per op id. `run_once`/`arun_once`: succeeded → stored result; otherwise `reconcile()` first and only run `fn` if it returns `None`. A per-op cross-process lock serializes concurrent runners. |
| `looks` | `record_look`, `count_looks`, `planned_looks`, `LookBudgetExceeded` | Global held-out look ledger keyed by dataset hash; idempotent per (dataset, experiment); refuses looks beyond the plan fixed by the first record. |

If `ledger_commit` targets the branch checked out in the user's worktree, the ref moves
but the worktree/index do not: `git status` will show the ledger change as a reverse
diff until the user updates. Prefer a dedicated ref such as `exp-ledger/<cid>`.

Observability (design §12.3): `record_decisions` and `cas_frontier` each emit a
`ci.step{ci.phase=record}` span via `ci_lab.obs` carrying `oes.experiment_id`
(+ `oes.decision`, or `rrsi.round`/`ci.score` for the frontier); `record_decisions` also
`obs.annotate`s the caller's current (round) span with `oes.experiment_id`/`oes.decision`.
Without a tracer
provider (installed only by `ci_lab.telemetry.setup`) these are no-ops.

## `ci_lab.gitops` — git plumbing

* `git` — subprocess wrapper: list args only, `--end-of-options`/`--` separators,
  `GIT_TERMINAL_PROMPT=0`, `GCM_INTERACTIVE=never`, `GIT_OPTIONAL_LOCKS=0` for reads,
  scrubbed `GIT_DIR`/`GIT_INDEX_FILE`; `tree_hash(repo, commit, path="")`; `write_lock(repo)`
  (single-writer mutex for ref-changing ops).
* `names` — contracts validators re-exported, plus `check_ref_format` (real
  `git check-ref-format`), `arm_branch`, `archive_tag(eid, arm)` = `exp-archive/<eid>/<arm>`,
  `sleep_branch`, `ledger_branch`.
* `slots` — `SlotPool(repo, cid, root=CI_WT_ROOT)` with worktrees at `<root>/<cid>/s<n>`
  (default root `C:\x` on Windows, `~/.ci-wt` elsewhere). `acquire(base, branch)` leases a
  warm slot or creates one; `reset` does `git switch --discard-changes -C` + `git clean -fdx -e .venv`;
  `release` detaches HEAD so the branch is free; `remove` deletes the worktree. Leases of
  dead local processes are reclaimed. `evaluator(commit)` gives a detached `<cid>/eval` tree.
* `publish_refs` — `push_with_lease(repo, remote, branch, expected_remote_sha)`
  (`--force-with-lease` pinned to the expected sha, `None` = must not exist; idempotent),
  `remote_sha`, `tag_archive(repo, eid, arm, commit, remote=None)` (create-only CAS),
  `is_ancestor`.
* `safe_path` (C13) — `safe_join(root, rel)` rejects absolute, UNC and drive-relative paths,
  `..`, `.git` (any case, 8.3 `GIT~1`), NUL/control/`<>:"|?*` (so ADS `a:s`), trailing dot/space
  and reserved device names; walks each existing component and rejects symlinks, junctions
  and other reparse points (`FILE_ATTRIBUTE_REPARSE_POINT`) and case aliases; the final real
  path must be inside the real root. `safe_write_*` re-checks after creating parents.
  `match_globs(rel, globs)` is posix-style (`*`, `?`, `[...]`, `**/`).

## `ci_lab.cache` — shared caches

* `cow.detect(path, source=None)` → `CowInfo(mode, filesystem, reason)`, mode `clone` |
  `hardlink` | `copy`. Windows: `GetVolumeInformationW` via ctypes (ReFS or
  `FILE_SUPPORTS_BLOCK_REFCOUNTING` → clone). Linux: `FICLONE` probe (+ fs name from
  mountinfo). macOS: `clonefile` probe. A `source` on another volume forces `copy`.
* `env.shared_env(worktree, no_sync=None)` / `with_shared_env` — shared `UV_CACHE_DIR`
  (`<CI_CACHE_DIR or CI_WT_ROOT/.cache>/uv`, an explicit `UV_CACHE_DIR` wins),
  `UV_LINK_MODE` (`clone` on CoW, else `hardlink`), shared `PYTHONPYCACHEPREFIX`,
  `UV_NO_SYNC=1` when `no_sync=True` (removed when `False`); drops `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV`.
* `venv.provision(worktree, golden=None)` — `uv sync` (`--native-tls` if `UV_NATIVE_TLS`,
  `--frozen` if `uv.lock` exists) with the shared env. With a golden `.venv` on a CoW volume the
  venv is cloned first and `uv sync` fixes the delta. Console-script launchers in a clone may
  still reference the golden path: run tools via `python -m` / `uv run`.
* `evalcache.EvalCache(root=None, enabled=True)` — `<root>/<pin_hash>/<harness_tree>/<split>/<case>/<trial>.json`
  (root `CI_EVAL_CACHE` or `<cache_root>/eval`). `pin_hash` covers evaluator tree, judge model
  and provider; served judge models are stored and checked on read. Unsafe case ids are hashed.
  `EvalCache.for_profile(Profile.COPILOT)` is disabled (C8). Missing trials are never cached.

## Environment variables

`CI_WT_ROOT`, `CI_CACHE_DIR`, `CI_EVAL_CACHE`, `CI_RUN_DIR`, `UV_CACHE_DIR`, `UV_NATIVE_TLS`.
