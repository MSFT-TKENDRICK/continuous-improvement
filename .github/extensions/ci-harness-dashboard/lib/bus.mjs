// Agent-bus WAL reader for the Bus tab (bus contract v2 §2-§3, src/ci_lab/bus/wal.py).
// The canvas is an ORCHESTRATOR view: rows carry only allowlisted, orchestrator-visible body
// fields (ids, enums, numbers, counts). Free text that may quote rubric material (vote reasons,
// verdict corrections, proposal summaries, note text, intent/outcome detail) is dropped, and
// artifacts and sealed/vault dirs are never opened; artifacts appear as sha256/bytes refs only.
import fsp from "node:fs/promises";
import path from "node:path";

export const BUS_LIMITS = Object.freeze({ searchDepth: 3, searchDirs: 400, roots: 16, runs: 50, topics: 64, walBytes: 2 * 1024 * 1024, rows: 500 });
const SUFFIX = ".wal.jsonl";
const GENESIS = "0".repeat(64);
const HEX64 = /^[0-9a-f]{64}$/;
const RUN_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const TASK_RE = /^(?:_run|[a-z0-9][a-z0-9_-]{0,63})$/;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._@/:-]{0,159}$/;
const SKIP_DIRS = new Set(["sealed", "vault", "artifacts", "students", "challenger", "telemetry", "node_modules"]);
const ROLES = new Set(["orchestrator", "planner", "examiner", "student", "voter", "judge", "adversary", "hardener"]);
const NOTE_TEXTS = new Set(["torn_tail_repaired"]);
const TERMINAL = { commit: "committed", reject: "rejected", abort: "aborted" };
const FLAGS = { exploit: "exploit", rubric_patch: "patch" };

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const N = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const B = (v) => (typeof v === "boolean" ? v : null);
const ID = (v) => (typeof v === "string" && ID_RE.test(v) ? v : null);
const len = (v) => (Array.isArray(v) ? v.length : null);
const oneOf = (v, ...ok) => (ok.includes(v) ? v : null);
const clip = (v, n) => (typeof v === "string" ? (v.length > n ? `${v.slice(0, n)}…` : v) : null);
const short = (v) => (typeof v === "string" && HEX64.test(v) ? v.slice(0, 12) : null);

/** Per-kind allowlist of orchestrator-visible body fields (mirrors `types.visibility`: the
 * orchestrator may see every kind). Anything not listed here never leaves this module. */
const PROJECT = Object.freeze({
    manifest: (b) => ({ run: ID(b.run), code_rev: ID(b.code_rev), voters: len(b.voters), quorum: N(b.quorum), max_parallel: N(b.max_parallel) }),
    intent: (b) => ({ action: ID(b.action), key: ID(b.key), attempt: ID(b.attempt) }),
    outcome: (b) => ({ intent_seq: N(b.intent_seq), ok: B(b.ok), decision: oneOf(b.detail?.decision, "commit", "revise", "reject") }),
    proposal: (b) => ({ proposal: ID(b.proposal), rubric_version: ID(b.rubric_version) }),
    vote: (b) => ({
        proposal: ID(b.proposal), voter: ID(b.voter), measure: oneOf(b.measure, "deterministic", "assert", "s1", "llm"), criterion: ID(b.criterion),
        passed: B(b.passed), score: N(b.score), confidence: N(b.confidence), reasons: len(b.reasons),
    }),
    verdict: (b) => {
        const crit = isObj(b.criteria) ? Object.values(b.criteria).filter(isObj) : [];
        return {
            proposal: ID(b.proposal), decision: oneOf(b.decision, "commit", "revise", "reject"), score: N(b.score), criteria: crit.length,
            failed_required: crit.filter((c) => c.required === true && c.passed === false).length, votes: len(b.votes), escalated: B(b.escalated),
            correction: b.correction != null,
        };
    },
    commit: (b) => ({ proposal: ID(b.proposal), verdict_seq: N(b.verdict_seq) }),
    reject: (b) => ({ attempt: ID(b.attempt), reason: clip(b.reason, 120) }),
    abort: (b) => ({ attempt: ID(b.attempt), reason: clip(b.reason, 120) }),
    exploit: (b) => ({
        attempt: ID(b.attempt), gamer: ID(b.gamer), adversary_proposal: ID(b.adversary_proposal), soft_pref: oneOf(b.soft_pref, "student", "adversary", "tie"),
        soft_pass_adversary: B(b.soft_pass_adversary), oracle_invalid: len(b.oracle_invalid),
    }),
    rubric_patch: (b) => ({
        rubric_id: ID(b.rubric_id), from_version: ID(b.from_version), to_version: ID(b.to_version), accepted: B(b.accepted),
        applies_from_attempt: N(b.applies_from_attempt), diff_sha256: short(b.diff_sha256),
    }),
    note: (b) => ({ text: NOTE_TEXTS.has(b.text) ? b.text : null, data: isObj(b.data) ? Object.keys(b.data).length : null }),
});
export const ORCHESTRATOR_KINDS = Object.freeze(Object.keys(PROJECT));

/** One WAL line -> table row, or null when the kind is not orchestrator-visible. */
export function projectEntry(e) {
    const project = Object.hasOwn(PROJECT, e.kind) ? PROJECT[e.kind] : null;
    if (!project) return null;
    const body = isObj(e.body) ? e.body : {};
    const fields = Object.fromEntries(Object.entries(project(body)).filter(([, v]) => v !== null));
    const a = isObj(body.artifact) && HEX64.test(body.artifact.sha256) ? { sha256: body.artifact.sha256, bytes: N(body.artifact.bytes) } : null;
    return {
        seq: e.seq,
        kind: e.kind,
        role: ROLES.has(e.author?.role) ? e.author.role : null,
        name: typeof e.author?.name === "string" && NAME_RE.test(e.author.name) ? e.author.name : null,
        ref: Number.isInteger(e.ref) ? e.ref : null,
        ts: clip(e.ts, 40),
        summary: Object.entries(fields).map(([k, v]) => `${k}=${v}`).join(" "),
        fields,
        artifact: a,
        flag: FLAGS[e.kind] ?? null,
    };
}

/**
 * Parse one topic WAL like `wal._parse`: dense seq from 0, `prev` = previous `hash`, matching
 * topic. A torn final line is tolerated (`torn`); any other damage stops at the verified prefix
 * and sets `corrupt`. With `truncated` (tail read) the chain is checked from the first kept line.
 * Hashes are linked, not recomputed (Python float canonicalization differs from JS).
 */
export function parseWal(text, topic, { truncated = false, maxRows = BUS_LIMITS.rows } = {}) {
    const lines = text.replace(/^\uFEFF/, "").split("\n");
    const tail = lines.pop();
    const t = { topic, task: topic.split("/").pop(), entries: 0, rows: [], counts: {}, exploits: 0, patches: 0, state: "open", torn: !!tail, corrupt: null, truncated, hidden: 0, rowsTruncated: false };
    let prevHash = truncated ? null : GENESIS;
    let next = truncated ? null : 0;
    for (let i = 0; i < lines.length; i++) {
        let e;
        try {
            e = JSON.parse(lines[i]);
        } catch {
            if (i === lines.length - 1 && !tail) t.torn = true;
            else t.corrupt = `unparseable line ${i + 1}`;
            break;
        }
        const shapeOk = isObj(e) && Number.isInteger(e.seq) && typeof e.kind === "string" && isObj(e.author) && HEX64.test(e.hash) && HEX64.test(e.prev);
        if (!shapeOk || e.topic !== topic || (next !== null && (e.seq !== next || e.prev !== prevHash))) {
            t.corrupt = `broken seq/hash chain at line ${i + 1}`;
            break;
        }
        prevHash = e.hash;
        next = e.seq + 1;
        t.entries++;
        t.counts[e.kind] = (t.counts[e.kind] ?? 0) + 1;
        if (TERMINAL[e.kind]) t.state = TERMINAL[e.kind];
        const row = projectEntry(e);
        if (!row) {
            t.hidden++;
            continue;
        }
        if (row.flag === "exploit") t.exploits++;
        if (row.flag === "patch") t.patches++;
        t.rows.push(row);
    }
    if (t.rows.length > maxRows) {
        t.rows = t.rows.slice(-maxRows);
        t.rowsTruncated = true;
    }
    return t;
}

/** Bounded BFS for `bus/` dirs under `artifacts/` and the run roots (never into sealed/vault/artifacts). */
export async function findBusRoots(src, limits = BUS_LIMITS) {
    const seen = new Set();
    const queue = [];
    for (const base of [path.join(src.roots.repo, "artifacts"), ...src.roots.runs]) {
        const real = await src.realDir(base);
        if (real && src.allowed(real) && !seen.has(real)) {
            seen.add(real);
            queue.push([real, 0]);
        }
    }
    const found = [];
    for (let visited = 0; queue.length && visited < limits.searchDirs && found.length < limits.roots; visited++) {
        const [dir, depth] = queue.shift();
        for (const d of await src.list(dir, { dirs: true })) {
            if (seen.has(d.path)) continue;
            seen.add(d.path);
            if (d.name === "bus") found.push(d.path);
            else if (depth + 1 < limits.searchDepth && !SKIP_DIRS.has(d.name.toLowerCase())) queue.push([d.path, depth + 1]);
        }
    }
    return found.slice(0, limits.roots);
}

const walCache = new Map();
async function readWal(src, p, topic, limits) {
    const f = await src.safeFile(p);
    if (!f) return null;
    const key = `${f.real}\0${limits.walBytes}\0${limits.rows}`;
    const hit = walCache.get(key);
    if (hit && hit.mtimeMs === f.st.mtimeMs && hit.size === f.st.size) return hit.value;
    let value = null;
    let fh;
    try {
        fh = await fsp.open(f.real, "r");
        const start = Math.max(0, f.st.size - limits.walBytes);
        const buf = Buffer.alloc(f.st.size - start);
        const { bytesRead } = await fh.read(buf, 0, buf.length, start);
        let text = buf.subarray(0, bytesRead).toString("utf8");
        if (start > 0) text = text.slice(text.indexOf("\n") + 1);
        value = parseWal(text, topic, { truncated: start > 0, maxRows: limits.rows });
    } catch {
        src.warn(`unreadable bus WAL: ${src.rel(p)}`);
    } finally {
        await fh?.close().catch(() => {});
    }
    if (walCache.size >= 2000) walCache.delete(walCache.keys().next().value);
    walCache.set(key, { mtimeMs: f.st.mtimeMs, size: f.st.size, value });
    return value;
}

/** Every discovered bus root -> runs -> topics (orchestrator projection, see `parseWal`). */
export async function readBus(src, limits = BUS_LIMITS) {
    const out = [];
    for (const root of await findBusRoots(src, limits)) {
        let runs = (await src.list(root, { dirs: true })).filter((d) => RUN_RE.test(d.name));
        runs = (await src.mtimes(runs)).sort((a, b) => b.mtimeMs - a.mtimeMs);
        for (const r of runs) {
            if (out.length >= limits.runs) return out;
            const files = (await src.list(r.path, { dirs: false }))
                .filter((f) => f.name.endsWith(SUFFIX) && TASK_RE.test(f.name.slice(0, -SUFFIX.length)))
                .sort((a, b) => a.name.localeCompare(b.name))
                .slice(0, limits.topics);
            const topics = [];
            for (const f of files) {
                const t = await readWal(src, f.path, `${r.name}/${f.name.slice(0, -SUFFIX.length)}`, limits);
                if (!t) continue;
                if (t.corrupt) src.warn(`bus topic ${t.topic}: ${t.corrupt}`);
                topics.push(t);
            }
            if (topics.length) out.push({ root: src.rel(root), run: r.name, mtimeMs: r.mtimeMs, topics });
        }
    }
    return out;
}
