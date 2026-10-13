// Builds the experiment chat bundle into the canvas extension (ui/chat/).
// Deterministic: running it twice yields identical files (CI diffs the committed output).
import { build } from "esbuild";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
export const outdir = join(repo, ".github", "extensions", "ci-harness-dashboard", "ui", "chat");
const stubs = join(here, "stubs");
const MAX_FILE = 1024 * 1024;

// Modules the chat never needs that would pull in eval, <style> injection, remote assets or
// unwanted vendors. Each is replaced by a small local stub in ./stubs.
const STUB = {
  "streamdown": "streamdown.jsx",
  "@copilotkit/web-inspector": "empty.js",
  "@copilotkit/a2ui-renderer": "a2ui.jsx",
  "@copilotkit/mcp-apps-renderer": "empty.js",
  "@copilotkit/web-components/threads-drawer": "threads-drawer.js",
  "@jetbrains/websandbox": "empty.js",
  "@ag-ui/proto": "ag-ui-proto.js",
  "react-style-singleton": "style-singleton.js",
};

// Text patches applied to dependency files: vendor names the repo bans (in a warning string and a
// docs URL) and the hosted-cloud API origin. Each pattern must match, so a version bump that
// changes the text fails the build instead of silently shipping it.
const CLOUD = [/https:\/\/api\.cloud\.copilotkit\.ai/g, "https://copilot-cloud.invalid"];
// zod's JIT probe calls `new Function("")`; under `script-src 'self'` that fires a CSP violation
// even though zod catches it. Report "no eval" so zod uses its interpreter path directly.
const ZOD_NO_EVAL = [/const F = Function;\s*new F\(""\);\s*return true;/g, "return false;"];
const ZOD_NO_COMPILE = [/const F = Function;/g, 'const F = function () { throw new Error("zod JIT disabled (CSP)"); };'];
const SCRUB = [
  { file: /[\\/]@ag-ui[\\/]client[\\/]dist[\\/]index\.mjs$/, replace: [[/\(e\.g\. @ag-ui\/l[a-z]+\)/g, "(the framework adapter)"]] },
  { file: /[\\/]@copilotkit[\\/]shared[\\/]dist[\\/]utils[\\/]errors\.mjs$/, replace: [[/\/l[a-z]+-python\/coagent-troubleshooting\/[^`"']*/g, "/troubleshooting/common-issues"]] },
  { file: /[\\/]@copilotkit[\\/]shared[\\/]dist[\\/]constants[\\/]index\.mjs$/, replace: [CLOUD] },
  { file: /[\\/]@copilotkit[\\/]react-core[\\/]dist[\\/]copilotkit-[A-Za-z0-9_-]+\.mjs$/, replace: [CLOUD] },
  { file: /[\\/]zod[\\/]v4[\\/]core[\\/]util\.js$/, replace: [ZOD_NO_EVAL] },
  { file: /[\\/]zod[\\/]v4[\\/]core[\\/]doc\.js$/, replace: [ZOD_NO_COMPILE] },
];
const BANNED = /lang(chain|smith|graph)/i;
// Mirrors the repo lint rule ext.no-console-log (debug output is dropped via `pure` below).
const FORBIDDEN_IN_JS = [/\bconsole\.(?:log|info|debug)\s*\(/, /api\.cloud\.copilotkit\.ai/, /\beval\(/, /new Function\(/, /=\s*Function\s*[;,)]/, /segment\.(io|com)/i, /scarf/i];

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");

const stubPlugin = {
  name: "stub-modules",
  setup(b) {
    const filter = new RegExp(`^(${Object.keys(STUB).map(escape).join("|")})$`);
    b.onResolve({ filter }, (args) => ({ path: join(stubs, STUB[args.path]) }));
    // zod v4 re-exports ~40 locales; only English is used.
    b.onResolve({ filter: /^\.\.\/locales\/index\.js$/ }, (args) =>
      /[\\/]zod[\\/]v4[\\/]/.test(args.importer) ? { path: join(stubs, "zod-locales.js") } : undefined);
    // Stylesheets are built separately into chat.css; JS-side CSS imports become no-ops.
    b.onResolve({ filter: /\.css$/ }, (args) => (args.kind === "entry-point" ? undefined : { path: args.path, namespace: "no-css" }));
    b.onLoad({ filter: /.*/, namespace: "no-css" }, () => ({ contents: "", loader: "js" }));
  },
};

const scrubPlugin = {
  name: "scrub-strings",
  setup(b) {
    for (const rule of SCRUB) {
      b.onLoad({ filter: rule.file }, (args) => {
        let text = readFileSync(args.path, "utf8");
        for (const [from, to] of rule.replace) {
          if (!from.test(text)) throw new Error(`scrub: ${from} not found in ${relative(here, args.path)}`);
          from.lastIndex = 0;
          text = text.replace(from, to);
        }
        return { contents: text, loader: "js" };
      });
    }
  },
};

const common = {
  absWorkingDir: here,
  outdir,
  bundle: true,
  minify: true,
  treeShaking: true,
  sourcemap: false,
  legalComments: "external",
  charset: "utf8",
  logLevel: "warning",
  metafile: true,
};

export async function bundle() {
  rmSync(outdir, { recursive: true, force: true });
  mkdirSync(outdir, { recursive: true });
  const js = await build({
    ...common,
    entryPoints: { chat: "src/main.js" },
    format: "esm",
    splitting: true,
    platform: "browser",
    target: ["es2022", "chrome120", "edge120"],
    jsx: "automatic",
    chunkNames: "chunk-[hash]",
    loader: { ".svg": "dataurl" },
    pure: ["console.log", "console.info", "console.debug"],
    define: { "process.env.NODE_ENV": '"production"', "process.env": "{}", "global": "globalThis" },
    plugins: [stubPlugin, scrubPlugin],
  });
  const css = await build({
    ...common,
    entryPoints: { chat: "src/styles.css" },
    loader: { ".svg": "dataurl", ".woff2": "empty", ".woff": "empty", ".ttf": "empty" },
  });
  const inputs = [...Object.keys(js.metafile.inputs), ...Object.keys(css.metafile.inputs)];
  writeFileSync(join(outdir, "THIRD_PARTY_LICENSES.txt"), licenses(inputs));
  return check();
}

function packageOf(input) {
  const parts = input.split(/[\\/]/);
  const i = parts.lastIndexOf("node_modules");
  if (i < 0) return null;
  const name = parts[i + 1].startsWith("@") ? `${parts[i + 1]}/${parts[i + 2]}` : parts[i + 1];
  return { name, dir: join(here, ...parts.slice(0, i + 1 + name.split("/").length)) };
}

function licenses(inputs) {
  const pkgs = new Map();
  for (const input of inputs) {
    const p = packageOf(input);
    if (p && !pkgs.has(p.name)) pkgs.set(p.name, p.dir);
  }
  const out = [
    "Third-party software bundled in ui/chat (built from web/experiment-chat by build.mjs; do not edit).",
    "Some modules are replaced by local stubs; see web/experiment-chat/stubs.",
    "",
  ];
  for (const name of [...pkgs.keys()].sort()) {
    const dir = pkgs.get(name);
    const pj = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
    const license = typeof pj.license === "string" ? pj.license : pj.license?.type ?? "UNKNOWN";
    out.push("=".repeat(72), `${name}@${pj.version} (${license})`, "=".repeat(72));
    const file = readdirSync(dir).sort().find((f) => /^(licen[cs]e|copying)(\.|$)/i.test(f));
    out.push(file ? readFileSync(join(dir, file), "utf8").replace(/\r\n/g, "\n").trim() : `License: ${license} (no license file shipped in the package)`, "");
  }
  return out.join("\n") + "\n";
}

/** Fail the build on oversized files, banned vendor names, eval or cloud endpoints. */
export function check() {
  const problems = [];
  const files = readdirSync(outdir).sort();
  const rows = [];
  let total = 0;
  for (const f of files) {
    const path = join(outdir, f);
    const buf = readFileSync(path);
    total += buf.length;
    rows.push(`${relative(repo, path).split(sep).join("/")}  ${buf.length} B  ${createHash("sha256").update(buf).digest("hex").slice(0, 12)}`);
    if (buf.length > MAX_FILE) problems.push(`${f} is ${buf.length} B (> 1 MiB)`);
    const text = buf.toString("utf8");
    if (BANNED.test(text)) problems.push(`${f} contains a banned vendor name`);
    if (f.endsWith(".js")) for (const re of FORBIDDEN_IN_JS) if (re.test(text)) problems.push(`${f} matches ${re}`);
  }
  return { rows, total, problems };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!existsSync(join(here, "node_modules"))) {
    process.stderr.write("node_modules missing: run `npm ci --ignore-scripts` in web/experiment-chat first\n");
    process.exit(1);
  }
  const { rows, total, problems } = await bundle();
  process.stderr.write(`${rows.join("\n")}\ntotal ${total} B\n`);
  if (problems.length) {
    process.stderr.write(`bundle check failed:\n  ${problems.join("\n  ")}\n`);
    process.exit(1);
  }
}
