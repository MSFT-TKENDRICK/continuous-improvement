// ci-guardrails policy (design §13.1 R5, §13.4): pure deny logic for the Copilot CLI PreToolUse hook.
// Dev-loop analogue of poteto/noodle's block-sleep hook: structure that coding agents cannot talk past.
// No I/O here; extension.mjs supplies env and the current branch.

import path from "node:path";

export const FROZEN_PATHS = Object.freeze([
  "src/ci_lab/contracts.py",
  "src/ci_lab/rulespec.py",
  "src/ci_lab/rules/templates.yaml",
  "harness/guards/BUNDLE.lock",
  "src/order_support/oracle.py",
]);
// Frozen directories (any file below them). `harness/guards/` matches at any depth, e.g.
// src/order_support/harness/guards/**; the others are repo-relative. Mirrored in .github/CODEOWNERS.
export const FROZEN_DIRS = Object.freeze([
  "lint/rules/",
  "harness/guards/",
  "src/ci_lab/rules/",
]);
export const HOOKS_PATH = ".githooks";
const PROTECTED_BRANCHES = new Set(["main", "master"]);
const SHELL_TOOLS = new Set(["powershell", "bash", "shell", "sh", "pwsh", "cmd", "run_in_terminal", "terminal"]);
const EDIT_TOOL_RE = /edit|create|write|patch|replace|insert/i;
const PATH_KEYS = ["path", "file_path", "filePath", "filename", "target", "destination"];
const PATCH_FILE_RE = /^\*\*\* (?:Update|Add|Delete) File: (.+)$|^\*\*\* Move to: (.+)$/gm;
const SHELL_WRITE_RE = new RegExp(
  [
    String.raw`(?<![\d=\-])>(?!&)`,
    String.raw`\b(?:set-content|add-content|out-file|new-item|copy-item|move-item|remove-item|rename-item)\b`,
    String.raw`\bsed\s+-i`,
    String.raw`\b(?:tee|cp|mv|rm|del|truncate|touch)\s`,
    String.raw`writealltext|writeallbytes|write_text|write_bytes`,
    String.raw`\bgit\s+(?:checkout|restore|rm|mv|apply|stash\s+pop)\b`,
  ].join("|"),
  "i",
);

function reason(rule, violation, fix, see) {
  return { rule, reason: `[LINT][ERROR] guardrail:${rule}\n  Violation: ${violation}\n  Fix: ${fix}\n  See: ${see}` };
}

// ------------------------------------------------------------------ shell lexing

/** Split a shell command line into simple commands (token arrays). Quote-aware; POSIX sh and PowerShell. */
export function splitCommands(command) {
  const commands = [];
  let tokens = [];
  let cur = "";
  let has = false;
  let quote = null;
  const pushTok = () => {
    if (has) tokens.push(cur);
    cur = "";
    has = false;
  };
  const pushCmd = () => {
    pushTok();
    if (tokens.length) commands.push(tokens);
    tokens = [];
  };
  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (quote) {
      if (c === quote) quote = null;
      else cur += c;
      continue;
    }
    if (c === '"' || c === "'") {
      quote = c;
      has = true;
    } else if (c === ";" || c === "\n" || c === "\r" || c === "|" || c === "&" || c === "(" || c === ")"
      || c === "{" || c === "}") {
      pushCmd();
    } else if (c === " " || c === "\t") {
      pushTok();
    } else {
      cur += c;
      has = true;
    }
  }
  pushCmd();
  return commands;
}

const GIT_GLOBAL_WITH_VALUE = new Set(["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env",
  "--exec-path"]);

/** Parse a token array as a git invocation: {configOverrides, sub, args} or null. */
export function parseGit(tokens) {
  let i = tokens.findIndex((t) => /^(?:.*[\\/])?git(?:\.exe)?$/i.test(t));
  if (i < 0) return null;
  const configOverrides = [];
  i += 1;
  while (i < tokens.length && tokens[i].startsWith("-")) {
    const t = tokens[i];
    if (t === "-c") {
      configOverrides.push(tokens[i + 1] ?? "");
      i += 2;
    } else if (t.startsWith("-c") && t.length > 2 && !t.startsWith("--")) {
      configOverrides.push(t.slice(2));
      i += 1;
    } else if (GIT_GLOBAL_WITH_VALUE.has(t)) {
      i += 2;
    } else {
      i += 1;
    }
  }
  if (i >= tokens.length) return { configOverrides, sub: null, args: [] };
  return { configOverrides, sub: tokens[i], args: tokens.slice(i + 1) };
}

function isLongAbbrev(tok, full, minLen) {
  const name = tok.split("=")[0];
  return name.length >= minLen && full.startsWith(name);
}

function normHooksPath(v) {
  return v.replace(/\\/g, "/").replace(/^\.\//, "").replace(/\/+$/, "");
}

// ------------------------------------------------------------------ git rules

const COMMIT_VALUE_LONG = new Set(["--message", "--file", "--author", "--date", "--reuse-message",
  "--reedit-message", "--fixup", "--squash", "--template", "--trailer", "--cleanup", "--pathspec-from-file"]);

function commitSkipsHooks(args) {
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t === "--") break;
    if (t.startsWith("--")) {
      if (isLongAbbrev(t, "--no-verify", "--no-veri".length)) return true;
      if (COMMIT_VALUE_LONG.has(t)) i += 1;
      continue;
    }
    if (/^-[A-Za-z]+/.test(t)) {
      for (let j = 1; j < t.length; j++) {
        const c = t[j];
        if (c === "n") return true;
        if ("mFCctS".includes(c)) {
          if (j === t.length - 1 && c !== "S") i += 1;
          break;
        }
      }
    }
  }
  return false;
}

function pushForceToProtected(args, currentBranch) {
  let force = false;
  let wide = false;
  let del = false;
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t === "--") {
      positional.push(...args.slice(i + 1));
      break;
    }
    if (t.startsWith("--")) {
      if (isLongAbbrev(t, "--force", "--forc".length) || isLongAbbrev(t, "--force-with-lease", "--force-".length)) {
        force = true;
      } else if (t === "--all" || t === "--mirror" || t === "--branches") {
        wide = true;
      } else if (t === "--delete") {
        del = true;
      } else if (["--repo", "--receive-pack", "--exec", "--push-option"].includes(t)) {
        i += 1;
      }
      continue;
    }
    if (/^-[A-Za-z]+$/.test(t)) {
      if (t.includes("f")) force = true;
      if (t.includes("d")) del = true;
      if (t.endsWith("o")) i += 1;
      continue;
    }
    positional.push(t);
  }
  const refspecs = positional.slice(1);
  if (refspecs.some((r) => r.startsWith("+"))) force = true;
  if (!force && !del) return false;
  if (wide) return true;
  const dst = (r) => {
    const s = r.replace(/^\+/, "");
    const d = s.includes(":") ? s.slice(s.lastIndexOf(":") + 1) : s;
    return d.replace(/^refs\/heads\//, "");
  };
  if (refspecs.length) return refspecs.some((r) => PROTECTED_BRANCHES.has(dst(r) === "HEAD" ? currentBranch?.() : dst(r)));
  return force && PROTECTED_BRANCHES.has(currentBranch?.() ?? "");
}

const CONFIG_VALUE_FLAGS = new Set(["--file", "-f", "--blob", "--type", "--default", "--comment", "--value"]);
const CONFIG_READ_FLAGS = new Set(["--get", "--get-all", "--get-regexp", "--get-urlmatch", "-l", "--list",
  "--show-origin", "--show-scope", "--name-only"]);

function configChangesHooksPath(args) {
  let unset = false;
  let read = false;
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("-")) {
      if (t === "--unset" || t === "--unset-all" || t === "--remove-section" || t === "--rename-section") unset = true;
      else if (CONFIG_READ_FLAGS.has(t)) read = true;
      else if (CONFIG_VALUE_FLAGS.has(t)) i += 1;
      continue;
    }
    positional.push(t);
  }
  if (["set", "unset", "get", "list", "rename-section", "remove-section", "edit"].includes(positional[0])) {
    const sub = positional.shift();
    if (sub === "unset" || sub.endsWith("-section")) unset = true;
    if (sub === "get" || sub === "list") read = true;
  }
  const [key, value] = positional;
  if (!key) return false;
  const lk = key.toLowerCase();
  if (lk !== "core.hookspath" && !(unset && lk === "core")) return false;
  if (unset) return true;
  if (read || value === undefined) return false;
  return normHooksPath(value) !== HOOKS_PATH;
}

function overrideChangesHooksPath(overrides) {
  return overrides.some((kv) => {
    const eq = kv.indexOf("=");
    const k = (eq < 0 ? kv : kv.slice(0, eq)).toLowerCase();
    return k === "core.hookspath" && normHooksPath(eq < 0 ? "" : kv.slice(eq + 1)) !== HOOKS_PATH;
  });
}

// ------------------------------------------------------------------ paths

export function normalizePath(p, workingDirectory) {
  let s = String(p).replace(/\\/g, "/");
  const abs = /^(?:[A-Za-z]:)?\//.test(s);
  if (!abs && workingDirectory) s = `${String(workingDirectory).replace(/\\/g, "/")}/${s}`;
  return path.posix.normalize(s).toLowerCase();
}

function dirHit(n, dir) {
  const d = dir.toLowerCase();
  const base = d.replace(/\/$/, "");
  return n === base || n.startsWith(d) || n.includes(`/${d}`) || n.endsWith(`/${base}`);
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const DIR_SHELL_RES = FROZEN_DIRS.map((d) => [
  d,
  new RegExp(`(?:^|[^a-z0-9_.-])${escapeRe(d.toLowerCase().replace(/\/$/, ""))}(?=$|/|[^a-z0-9_.-])`),
]);

/** The frozen file, or `<dir>**` for a path under a frozen directory, else null. */
export function frozenTarget(p, workingDirectory) {
  const n = normalizePath(p, workingDirectory);
  const file = FROZEN_PATHS.find((f) => n === f.toLowerCase() || n.endsWith(`/${f.toLowerCase()}`));
  if (file) return file;
  const dir = FROZEN_DIRS.find((d) => dirHit(n, d));
  return dir ? `${dir}**` : null;
}

function frozenInShell(lower) {
  const file = FROZEN_PATHS.find((f) => lower.includes(f.toLowerCase()));
  if (file) return file;
  const dir = DIR_SHELL_RES.find(([, re]) => re.test(lower));
  return dir ? `${dir[0]}**` : null;
}

function editTargets(toolArgs) {
  const out = [];
  if (!toolArgs || typeof toolArgs !== "object") return out;
  for (const k of PATH_KEYS) if (typeof toolArgs[k] === "string") out.push(toolArgs[k]);
  if (Array.isArray(toolArgs.edits)) {
    for (const e of toolArgs.edits) for (const k of PATH_KEYS) if (typeof e?.[k] === "string") out.push(e[k]);
  }
  for (const v of Object.values(toolArgs)) {
    if (typeof v !== "string" || !v.includes("*** ")) continue;
    for (const m of v.matchAll(PATCH_FILE_RE)) out.push((m[1] ?? m[2]).trim());
  }
  return out;
}

// ------------------------------------------------------------------ entry point

const SEE_HOOKS = "docs/lint.md; design §13.1 R5, §13.4";
const SEE_CONTRACTS = "FLEET.md (frozen contracts); design §13.6";

function shellDecision(command, ctx) {
  const lower = command.replace(/\\/g, "/").toLowerCase();
  for (const tokens of splitCommands(command)) {
    const git = parseGit(tokens);
    if (!git) continue;
    if (overrideChangesHooksPath(git.configOverrides)) {
      return reason("hooks-path", "`git -c core.hooksPath=...` bypasses the repo's pre-commit lint gate.",
        "Drop the -c override; hooks live in .githooks (git config core.hooksPath .githooks).", SEE_HOOKS);
    }
    if (git.sub === "commit" && commitSkipsHooks(git.args)) {
      return reason("no-verify", "`git commit --no-verify`/`-n` skips the pre-commit lint gate.",
        "Commit without --no-verify and fix what `ci-lab lint --staged` reports; CI runs the same rules.",
        SEE_HOOKS);
    }
    if (git.sub === "push" && pushForceToProtected(git.args, ctx.currentBranch)) {
      return reason("force-push-main", "Force-push (or delete) targeting main rewrites shared history.",
        "Push a feature branch and open a PR; never force-push main.", SEE_HOOKS);
    }
    if (git.sub === "config" && configChangesHooksPath(git.args)) {
      return reason("hooks-path", "Changing core.hooksPath away from .githooks disables the pre-commit lint gate.",
        "Leave core.hooksPath at .githooks (git config core.hooksPath .githooks).", SEE_HOOKS);
    }
  }
  if (ctx.env?.CI_ALLOW_CONTRACT_EDIT !== "1" && SHELL_WRITE_RE.test(lower)) {
    const hit = frozenInShell(lower);
    if (hit) return contractDenial(hit);
  }
  return null;
}

function contractDenial(target) {
  return reason("frozen-contract", `${target} is a frozen contract; agents may not modify it.`,
    "Propose the change to the contracts owner (or set CI_ALLOW_CONTRACT_EDIT=1 for an approved contract PR).",
    SEE_CONTRACTS);
}

/**
 * Decide a PreToolUse event. Returns null (no opinion) or {rule, reason}.
 * @param {{toolName: string, toolArgs: unknown, workingDirectory?: string}} input
 * @param {{env?: Record<string, string|undefined>, currentBranch?: () => string|undefined}} ctx
 */
export function evaluate(input, ctx = {}) {
  const toolName = String(input?.toolName ?? "");
  const args = input?.toolArgs;
  const cwd = input?.workingDirectory;
  if (args && typeof args === "object" && typeof args.command === "string"
    && (SHELL_TOOLS.has(toolName.toLowerCase()) || !EDIT_TOOL_RE.test(toolName))) {
    return shellDecision(args.command, ctx);
  }
  if (EDIT_TOOL_RE.test(toolName) && ctx.env?.CI_ALLOW_CONTRACT_EDIT !== "1") {
    for (const p of editTargets(args)) {
      const hit = frozenTarget(p, cwd);
      if (hit) return contractDenial(hit);
    }
  }
  return null;
}

/** Map a policy decision to the SDK's onPreToolUse output. */
export function toHookOutput(decision) {
  return decision ? { permissionDecision: "deny", permissionDecisionReason: decision.reason } : undefined;
}
