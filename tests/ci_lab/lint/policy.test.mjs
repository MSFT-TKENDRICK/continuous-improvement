// node --test tests/ci_lab/lint/policy.test.mjs — pure ci-guardrails policy (no SDK, no I/O).
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  evaluate,
  frozenTarget,
  splitCommands,
  toHookOutput,
} from "../../../.github/extensions/ci-guardrails/policy.mjs";

const sh = (command, ctx = {}, toolName = "powershell") =>
  evaluate({ toolName, toolArgs: { command }, workingDirectory: "C:\\x\\ci\\lint" }, ctx);
const rule = (d) => d?.rule ?? null;

test("splitCommands is quote-aware", () => {
  assert.deepEqual(splitCommands(`git commit -m "a; b" && git push`), [
    ["git", "commit", "-m", "a; b"],
    ["git", "push"],
  ]);
});

test("commit --no-verify / -n is denied in all spellings", () => {
  for (const c of [
    "git commit --no-verify -m x",
    "git commit -n -m x",
    "git commit -anm x",
    "git commit --no-veri -m x",
    "cd repo; git -C . commit -qn -m 'x'",
    "uv run pytest && git commit --no-verify -m 'msg'",
    "C:\\Program Files\\Git\\bin\\git.exe commit -n",
  ]) {
    assert.equal(rule(sh(c)), "no-verify", c);
  }
});

test("ordinary commits and look-alikes are allowed", () => {
  for (const c of [
    "git commit -m 'skip --no-verify next time'",
    "git commit -m -n",
    "git commit -am 'fix -n flag'",
    "git commit -F msg.txt",
    "git log -n 3",
    "git commit --no-edit --amend",
  ]) {
    assert.equal(rule(sh(c)), null, c);
  }
});

test("force-push to main is denied; feature branches allowed", () => {
  const onMain = { currentBranch: () => "main" };
  const onFeat = { currentBranch: () => "dev/lint" };
  for (const [c, ctx] of [
    ["git push --force origin main", onFeat],
    ["git push -f origin HEAD:main", onFeat],
    ["git push origin +main", onFeat],
    ["git push --force-with-lease origin refs/heads/main", onFeat],
    ["git push -f", onMain],
    ["git push origin --delete main", onFeat],
    ["git push --force --all", onFeat],
    ["git push -f origin HEAD", onMain],
  ]) {
    assert.equal(rule(sh(c, ctx)), "force-push-main", c);
  }
  for (const [c, ctx] of [
    ["git push origin main", onFeat],
    ["git push --force origin dev/lint", onFeat],
    ["git push -f", onFeat],
    ["git push -u origin dev/lint", onMain],
  ]) {
    assert.equal(rule(sh(c, ctx)), null, c);
  }
});

test("core.hooksPath may only point at .githooks", () => {
  for (const c of [
    "git config core.hooksPath /dev/null",
    "git config --local core.hooksPath ''",
    "git config --unset core.hooksPath",
    "git config unset core.hooksPath",
    "git config set core.hooksPath x",
    "git config --global core.HooksPath .git/hooks",
    "git -c core.hooksPath=/dev/null commit -m x",
  ]) {
    assert.equal(rule(sh(c)), "hooks-path", c);
  }
  for (const c of [
    "git config core.hooksPath .githooks",
    "git config core.hooksPath ./.githooks/",
    "git config core.hooksPath",
    "git config --get core.hooksPath",
    "git config user.name x",
  ]) {
    assert.equal(rule(sh(c)), null, c);
  }
});

test("frozen contract edits are denied unless CI_ALLOW_CONTRACT_EDIT=1", () => {
  const cwd = "C:\\x\\ci\\lint";
  const edit = (toolName, toolArgs, env = {}) => evaluate({ toolName, toolArgs, workingDirectory: cwd }, { env });
  assert.equal(rule(edit("edit", { path: "C:\\x\\ci\\lint\\src\\ci_lab\\contracts.py", old_str: "a" })),
    "frozen-contract");
  assert.equal(rule(edit("create", { path: "harness/guards/BUNDLE.lock", file_text: "" })), "frozen-contract");
  assert.equal(rule(edit("edit", { path: "src/ci_lab/rules/../rulespec.py" })), "frozen-contract");
  assert.equal(rule(edit("apply_patch", { input: "*** Begin Patch\n*** Update File: src/ci_lab/rules/templates.yaml\n" })),
    "frozen-contract");
  assert.equal(rule(edit("edit", { path: "src/ci_lab/contracts.py" }, { CI_ALLOW_CONTRACT_EDIT: "1" })), null);
  assert.equal(rule(edit("edit", { path: "src/ci_lab/obs.py" })), null);
  assert.equal(rule(edit("view", { path: "src/ci_lab/contracts.py" })), null);
  assert.equal(rule(sh("Set-Content src/ci_lab/contracts.py 'x'")), "frozen-contract");
  assert.equal(rule(sh("echo x > src\\ci_lab\\rulespec.py", {}, "bash")), "frozen-contract");
  assert.equal(rule(sh("Get-Content src/ci_lab/contracts.py 2>&1 | Select -First 5")), null);
  assert.equal(rule(sh("git diff src/ci_lab/contracts.py")), null);
  assert.equal(rule(sh("Set-Content src/ci_lab/contracts.py 'x'", { env: { CI_ALLOW_CONTRACT_EDIT: "1" } })), null);
  assert.equal(frozenTarget("/home/u/repo/SRC/ci_lab/Contracts.py"), "src/ci_lab/contracts.py");
});

test("deny output uses lint-arch reason and SDK shape", () => {
  const out = toHookOutput(sh("git commit --no-verify"));
  assert.equal(out.permissionDecision, "deny");
  assert.match(out.permissionDecisionReason, /^\[LINT\]\[ERROR\] guardrail:no-verify\n {2}Violation: .+\n {2}Fix: .+\n {2}See: .+$/);
  assert.equal(toHookOutput(null), undefined);
});
