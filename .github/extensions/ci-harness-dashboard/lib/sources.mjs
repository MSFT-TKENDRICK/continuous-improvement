// All on-disk format knowledge for the dashboard lives here (design §12.2, C35).
// Every reader is tolerant: missing/oversized/corrupt files yield null (plus a warning), never a throw.
// Integrators adjusting to final module layouts should only need to touch this file.
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { isWithin, isSafeId } from "./security.mjs";
import { readBus } from "./bus.mjs";

export const LIMITS = Object.freeze({
    jsonBytes: 4 * 1024 * 1024,
    jsonlTailBytes: 8 * 1024 * 1024,
    jsonlLines: 5000,
    dirEntries: 2000,
    runDirs: 200,
    roundsPerCampaign: 200,
    campaigns: 100,
    spanFiles: 32,
    spanFileBytes: 4 * 1024 * 1024,
    spanLines: 20000,
    aglFiles: 400,
    aglFileBytes: 512 * 1024,
    evalRuns: 60,
    scoreLines: 5000,
    sleepNights: 120,
    warnings: 50,
    cacheEntries: 4000,
});

// ---------------------------------------------------------------- layout constants
export const LAYOUT = Object.freeze({
    ledgerDir: "experiments",
    runDirDefaults: ["artifacts/ci-runs", "artifacts/runs"],
    resultsDirs: ["artifacts/results"],
    extraTelemetryDirs: ["artifacts/telemetry"],
    fakeDir: "_fake",
    statusDir: "status.d", // contracts.RUN_STATUS_DIR (one <writer>.json per writer)
    statusFile: "status.json", // contracts.RUN_STATUS_FILE (legacy single-file marker)
    envelopeNames: ["envelope.json", "experiment.json"],
    aglDirNames: ["agl", "journal", "rollouts"],
    skipRunDirs: new Set(["telemetry", "_fake", "inc-cache", "agl", "journal", "rollouts", "results", "cache"]),
    spanFileRe: /^spans.*\.jsonl$/i,
    rolloutFileRe: /^ro-[0-9a-f]{6,64}\.jsonl$/i,
});

const RRSI_EXT = "com.microsoft.ci.rrsi";
const SLEEP_EXT = "com.microsoft.ci.sleep";

// ---------------------------------------------------------------- small helpers
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);
const str = (v) => (typeof v === "string" ? v : typeof v === "number" ? String(v) : null);
const num = (v) => {
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
    return null;
};
const pick = (o, ...keys) => {
    if (!isObj(o)) return undefined;
    for (const k of keys) if (o[k] !== undefined && o[k] !== null) return o[k];
    return undefined;
};
const firstNum = (o, ...keys) => {
    for (const k of keys) {
        const n = num(pick(o, k));
        if (n !== null) return n;
    }
    return null;
};
/** Epoch seconds from epoch seconds/ms or an ISO string. */
export function toEpochSec(v) {
    if (typeof v === "number" && Number.isFinite(v)) return v > 1e12 ? v / 1000 : v;
    if (typeof v === "string" && v) {
        const n = Number(v);
        if (Number.isFinite(n)) return n > 1e12 ? n / 1000 : n;
        const t = Date.parse(v);
        if (Number.isFinite(t)) return t / 1000;
    }
    return null;
}

// ---------------------------------------------------------------- read cache
const cache = new Map();
function cacheGet(key, st) {
    const hit = cache.get(key);
    if (hit && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit;
    return null;
}
function cacheSet(key, st, value) {
    if (cache.size >= LIMITS.cacheEntries) cache.delete(cache.keys().next().value);
    cache.set(key, { mtimeMs: st.mtimeMs, size: st.size, value });
}
export function clearReadCache() {
    cache.clear();
}

// ---------------------------------------------------------------- Sources
export class Sources {
    /**
     * @param {string} repoRoot absolute repo root
     * @param {{env?: Record<string,string|undefined>, limits?: Partial<typeof LIMITS>}} [opts]
     */
    constructor(repoRoot, opts = {}) {
        this.repoRoot = path.resolve(repoRoot);
        this.env = opts.env ?? process.env;
        this.homeDir = opts.homeDir ?? os.homedir();
        this.limits = { ...LIMITS, ...(opts.limits ?? {}) };
        this.allowedRoots = [];
        this.warnings = [];
        this.roots = { repo: this.repoRoot, ledger: [], runs: [], results: [], imports: [], telemetry: [], agl: [] };
    }

    warn(msg) {
        if (this.warnings.length < this.limits.warnings) this.warnings.push(msg);
    }

    rel(p) {
        const r = path.relative(this.repoRoot, p);
        if (r && !r.startsWith("..") && !path.isAbsolute(r)) return r.split(path.sep).join("/");
        for (const imp of this.roots.imports ?? []) {
            const ri = path.relative(imp, p);
            if (!ri.startsWith("..") && !path.isAbsolute(ri)) return `imports:/${ri.split(path.sep).join("/")}`;
        }
        return p;
    }

    /** True for a path whose realpath lies inside an allowed root. */
    allowed(real) {
        return this.allowedRoots.some((root) => isWithin(root, real));
    }

    async realDir(p) {
        try {
            const lst = await fsp.lstat(p);
            if (lst.isSymbolicLink() || !lst.isDirectory()) return null;
            const real = await fsp.realpath(p);
            return real;
        } catch {
            return null;
        }
    }

    /** Resolve the candidate root directories (called on every scan; cheap). */
    async resolveRoots() {
        const repoReal = (await this.realDir(this.repoRoot)) ?? this.repoRoot;
        this.repoRoot = repoReal;
        const allowed = [repoReal];
        const runCandidates = [];
        const envRun = this.env.CI_RUN_DIR;
        if (envRun) runCandidates.push(path.resolve(this.repoRoot, envRun));
        for (const d of LAYOUT.runDirDefaults) runCandidates.push(path.join(this.repoRoot, d));
        const runs = [];
        for (const c of runCandidates) {
            const real = await this.realDir(c);
            if (real && !runs.includes(real)) {
                runs.push(real);
                if (!allowed.some((r) => isWithin(r, real))) allowed.push(real);
            }
        }
        this.allowedRoots = allowed;
        // Same env var as `ci_lab.telemetry.pull.IMPORTS_ENV`.
        const importsCandidate = this.env.CI_IMPORTS_DIR
            ? path.resolve(this.env.CI_IMPORTS_DIR)
            : path.join(this.homeDir, ".ci-lab", "imports");
        const imports = await this.realDir(importsCandidate);
        if (imports && !allowed.some((r) => isWithin(r, imports))) allowed.push(imports);
        const ledger = [];
        for (const c of [path.join(repoReal, LAYOUT.ledgerDir), ...runs.map((r) => path.join(r, LAYOUT.fakeDir, LAYOUT.ledgerDir))]) {
            const real = await this.realDir(c);
            if (real && this.allowed(real) && !ledger.includes(real)) ledger.push(real);
        }
        const results = [];
        for (const d of LAYOUT.resultsDirs) {
            const real = await this.realDir(path.join(repoReal, d));
            if (real) results.push(real);
        }
        for (const r of runs) {
            const real = await this.realDir(path.join(r, "results"));
            if (real && !results.includes(real)) results.push(real);
        }
        const fakeRuns = [];
        for (const r of runs) {
            const real = await this.realDir(path.join(r, LAYOUT.fakeDir));
            if (real) fakeRuns.push(real);
        }
        this.roots = { repo: repoReal, ledger, runs: [...runs, ...fakeRuns], results, imports: imports ? [imports] : [], telemetry: [], agl: [] };
        return this.roots;
    }

    /** Directories worth fs.watch-ing (existing roots; recursion handled by the hub). */
    async watchDirs() {
        await this.resolveRoots();
        const dirs = [...this.roots.ledger, ...this.roots.runs, ...this.roots.results, ...this.roots.imports];
        for (const d of LAYOUT.extraTelemetryDirs) {
            const real = await this.realDir(path.join(this.roots.repo, d));
            if (real) dirs.push(real);
        }
        return [...new Set(dirs)];
    }

    // ------------------------------------------------------------ primitive safe reads
    async safeFile(p) {
        try {
            const lst = await fsp.lstat(p);
            if (lst.isSymbolicLink() || !lst.isFile()) return null;
            const real = await fsp.realpath(p);
            if (!this.allowed(real)) {
                this.warn(`outside allowed roots: ${this.rel(p)}`);
                return null;
            }
            return { real, st: lst };
        } catch {
            return null;
        }
    }

    async exists(p) {
        return (await this.safeFile(p)) !== null;
    }

    async readJson(p, maxBytes = this.limits.jsonBytes) {
        const f = await this.safeFile(p);
        if (!f) return null;
        if (f.st.size > maxBytes) {
            this.warn(`too large (${f.st.size} B): ${this.rel(p)}`);
            return null;
        }
        const key = `json:${f.real}`;
        const hit = cacheGet(key, f.st);
        if (hit) return hit.value;
        let value = null;
        try {
            const text = await fsp.readFile(f.real, "utf8");
            value = text.trim() ? JSON.parse(text.replace(/^\uFEFF/, "")) : {};
        } catch {
            this.warn(`unparseable JSON: ${this.rel(p)}`);
            value = null;
        }
        cacheSet(key, f.st, value);
        return value;
    }

    /** Tail of a JSONL file: last `maxBytes` bytes, last `maxLines` parseable object lines. */
    async readJsonlTail(p, { maxBytes = this.limits.jsonlTailBytes, maxLines = this.limits.jsonlLines } = {}) {
        const empty = { rows: [], truncated: false, bad: 0, size: 0 };
        const f = await this.safeFile(p);
        if (!f) return null;
        const key = `jsonl:${maxBytes}:${maxLines}:${f.real}`;
        const hit = cacheGet(key, f.st);
        if (hit) return hit.value;
        let out = { ...empty, size: f.st.size };
        let fh;
        try {
            fh = await fsp.open(f.real, "r");
            const start = Math.max(0, f.st.size - maxBytes);
            const len = f.st.size - start;
            const buf = Buffer.alloc(len);
            let off = 0;
            while (off < len) {
                const { bytesRead } = await fh.read(buf, off, len - off, start + off);
                if (!bytesRead) break;
                off += bytesRead;
            }
            let text = buf.subarray(0, off).toString("utf8");
            if (start > 0) text = text.slice(text.indexOf("\n") + 1);
            else text = text.replace(/^\uFEFF/, "");
            const lines = text.split(/\r?\n/).filter((l) => l.trim());
            const keep = lines.slice(-maxLines);
            const rows = [];
            let bad = 0;
            for (const line of keep) {
                try {
                    const v = JSON.parse(line);
                    if (isObj(v)) rows.push(v);
                    else bad++;
                } catch {
                    bad++;
                }
            }
            out = { rows, truncated: start > 0 || lines.length > keep.length, bad, size: f.st.size };
        } catch {
            this.warn(`unreadable JSONL: ${this.rel(p)}`);
        } finally {
            await fh?.close().catch(() => {});
        }
        cacheSet(key, f.st, out);
        return out;
    }

    async countLines(p, maxBytes = this.limits.jsonlTailBytes) {
        const t = await this.readJsonlTail(p, { maxBytes, maxLines: 1e6 });
        return t ? t.rows.length : null;
    }

    /** Non-symlink child entries of a directory inside the allowed roots. */
    async list(dir, { dirs = null, max = this.limits.dirEntries } = {}) {
        const real = await this.realDir(dir);
        if (!real || !this.allowed(real)) return [];
        let ents;
        try {
            ents = await fsp.readdir(real, { withFileTypes: true });
        } catch {
            return [];
        }
        const out = [];
        for (const e of ents) {
            if (e.isSymbolicLink() || e.name.startsWith(".")) continue;
            if (dirs === true && !e.isDirectory()) continue;
            if (dirs === false && !e.isFile()) continue;
            out.push({ name: e.name, path: path.join(real, e.name), dir: e.isDirectory() });
            if (out.length >= max) {
                this.warn(`directory listing capped at ${max}: ${this.rel(real)}`);
                break;
            }
        }
        return out;
    }

    async mtimes(entries) {
        return Promise.all(
            entries.map(async (e) => {
                try {
                    return { ...e, mtimeMs: (await fsp.lstat(e.path)).mtimeMs };
                } catch {
                    return { ...e, mtimeMs: 0 };
                }
            }),
        );
    }

    async readEnvelope(dir) {
        for (const n of LAYOUT.envelopeNames) {
            const p = path.join(dir, n);
            const raw = await this.readJson(p);
            if (raw) return normalizeEnvelope(raw, this.rel(p));
        }
        return null;
    }

    // ------------------------------------------------------------ ledger: campaigns
    async readCampaigns() {
        const out = [];
        for (const ledger of this.roots.ledger) {
            const dirs = await this.list(path.join(ledger, "campaigns"), { dirs: true });
            for (const d of dirs.slice(0, this.limits.campaigns)) {
                if (!isSafeId(d.name)) continue;
                out.push(await this.readCampaign(d.path, d.name));
            }
        }
        return out;
    }

    async readCampaign(dir, cid) {
        const campaign = (await this.readJson(path.join(dir, "campaign.json"))) ?? {};
        const frontier = await this.readJson(path.join(dir, "frontier.json"));
        const calibration = await this.readJson(path.join(dir, "calibration.json"));
        const history = await this.readJsonlTail(path.join(dir, "history.jsonl"), { maxLines: 1000 });
        const confirm = await this.readJson(path.join(dir, "confirm.json"));
        const land = await this.readJson(path.join(dir, "land.json"));
        const stack = await this.readJson(path.join(dir, "stack.json"));
        const hyper = isObj(campaign.hyper) ? campaign.hyper : {};
        const rounds = [];
        const roundDirs = [
            ...(await this.list(path.join(dir, "rounds"), { dirs: true })),
            ...(await this.list(dir, { dirs: true })).filter((e) => !["rounds", "calibration", "confirm"].includes(e.name)),
        ];
        roundDirs.sort((a, b) => a.name.localeCompare(b.name));
        for (const rd of roundDirs.slice(-this.limits.roundsPerCampaign)) {
            if (!isSafeId(rd.name)) continue;
            const envelope = await this.readEnvelope(rd.path);
            const decisions = await this.readJson(path.join(rd.path, "decisions.json"));
            if (!envelope && !decisions) continue;
            rounds.push({ eid: rd.name, source: this.rel(rd.path), envelope, decisions: isObj(decisions) ? decisions : null });
        }
        const calEnvelope = await this.readEnvelope(path.join(dir, "calibration"));
        const confirmEnvelope = await this.readEnvelope(path.join(dir, "confirm"));
        return {
            campaignId: str(pick(campaign, "campaignId", "campaign_id", "id")) ?? cid,
            source: this.rel(dir),
            profile: str(pick(campaign, "profile")),
            domain: str(pick(campaign, "domain")),
            baseCommit: str(pick(campaign, "base_commit", "baseCommit")),
            budget: {
                rounds: firstNum(hyper, "max_rounds", "rounds", "T"),
                tokens: firstNum(hyper, "token_budget", "budget_tokens", "max_tokens"),
            },
            frontier: frontier
                ? {
                      commit: str(pick(frontier, "incumbent_commit", "commit")),
                      tree: str(pick(frontier, "incumbent_tree", "tree")),
                      score: num(pick(frontier, "score", "incumbent_score")),
                      round: num(pick(frontier, "round")),
                      eid: str(pick(frontier, "eid", "experiment_id")),
                  }
                : null,
            calibration: calibration
                ? { delta: num(pick(calibration, "delta")), tokens: num(pick(calibration, "tokens")), envelope: calEnvelope }
                : calEnvelope
                  ? { delta: calEnvelope.delta, tokens: null, envelope: calEnvelope }
                  : null,
            history: (history?.rows ?? []).map(normalizeHistoryRow),
            rounds,
            confirm: confirm || confirmEnvelope ? { ...smallObj(confirm), envelope: confirmEnvelope } : null,
            land: land ? smallObj(land) : null,
            stackSize: Array.isArray(stack) ? stack.length : arr(pick(stack, "stack", "items")).length || null,
        };
    }

    /** Standalone envelopes directly under a ledger root (e.g. evaluator experiments). */
    async readStandaloneExperiments() {
        const out = [];
        for (const ledger of this.roots.ledger) {
            for (const d of await this.list(ledger, { dirs: true })) {
                if (["campaigns", "sleep"].includes(d.name) || !isSafeId(d.name)) continue;
                const env = await this.readEnvelope(d.path);
                if (env) out.push(env);
            }
        }
        return out;
    }

    // ------------------------------------------------------------ ledger: sleep
    async readSleep() {
        const res = { state: null, nights: [], tasks: { reviewed: null, pending: null }, source: null };
        for (const ledger of this.roots.ledger) {
            const dir = path.join(ledger, "sleep");
            if (!(await this.realDir(dir))) continue;
            res.source = this.rel(dir);
            const st = await this.readJson(path.join(dir, "state.json"));
            if (st) {
                res.state = {
                    night: num(pick(st, "night")),
                    lastNightId: str(pick(st, "last_night_id", "lastNightId")),
                    lastDate: str(pick(st, "last_date", "lastDate")),
                    lastStatus: str(pick(st, "last_status", "lastStatus")),
                    lastBaseSha: str(pick(st, "last_base_sha")),
                    acceptedTotal: num(pick(st, "accepted_total", "acceptedTotal")),
                    history: arr(st.history).slice(-this.limits.sleepNights).map((h) => ({
                        nightId: str(pick(h, "night_id", "nightId", "id")),
                        night: num(pick(h, "night")),
                        status: str(pick(h, "status")),
                        deltaLcb: num(pick(h, "delta_lcb", "deltaLcb")),
                        nTasks: num(pick(h, "n_tasks", "nTasks")),
                    })),
                };
            }
            const envFiles = (await this.list(path.join(dir, "envelopes"), { dirs: false })).filter((e) => e.name.endsWith(".json"));
            for (const f of envFiles.slice(-this.limits.sleepNights)) {
                const raw = await this.readJson(f.path);
                if (raw) res.nights.push(normalizeEnvelope(raw, this.rel(f.path)));
            }
            const nightDirs = await this.list(path.join(dir, "nights"), { dirs: true });
            for (const nd of nightDirs.slice(-this.limits.sleepNights)) {
                const env = await this.readEnvelope(nd.path);
                if (env && !res.nights.some((n) => n.id && n.id === env.id)) res.nights.push(env);
            }
            res.tasks.reviewed = await this.countLines(path.join(dir, "tasks.jsonl"));
            res.tasks.pending = await this.countLines(path.join(dir, "tasks.pending.jsonl"));
        }
        res.nights.sort((a, b) => (a.sleep?.nightIndex ?? 0) - (b.sleep?.nightIndex ?? 0) || String(a.id).localeCompare(String(b.id)));
        return res;
    }

    async readHoldoutLooks() {
        let n = 0;
        for (const ledger of this.roots.ledger) n += (await this.countLines(path.join(ledger, "holdout-looks.jsonl"))) ?? 0;
        return n;
    }

    // ------------------------------------------------------------ run dirs (CI_RUN_DIR)
    async readRuns() {
        const out = [];
        for (const root of this.roots.runs) {
            let dirs = (await this.list(root, { dirs: true })).filter((d) => !LAYOUT.skipRunDirs.has(d.name) && isSafeId(d.name));
            dirs = (await this.mtimes(dirs)).sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, this.limits.runDirs);
            for (const d of dirs) {
                const run = await this.readRun(d.path, d.name, d.mtimeMs);
                if (run) out.push(run);
            }
        }
        return out;
    }

    /** JS port of `ci_lab.obs.read_status`: status.d/<writer>.json markers + legacy status.json. */
    async readStatus(dir) {
        const docs = [];
        const files = (await this.list(path.join(dir, LAYOUT.statusDir), { dirs: false, max: 256 })).filter((e) => e.name.endsWith(".json"));
        files.sort((a, b) => a.name.localeCompare(b.name));
        for (const f of files) docs.push(await this.readJson(f.path, 256 * 1024));
        docs.push(await this.readJson(path.join(dir, LAYOUT.statusFile), 256 * 1024));
        return aggregateStatus(docs);
    }

    async readRun(dir, name, mtimeMs) {
        const status = await this.readStatus(dir);
        const begin = await this.readJson(path.join(dir, "begin.json"));
        const selection = await this.readJson(path.join(dir, "selection.json"));
        const done = await this.readJson(path.join(dir, "round.done"));
        const armsJson = await this.readJson(path.join(dir, "arms.json"));
        const stopped = await this.exists(path.join(dir, "STOP"));
        const lastInc = await this.readJson(path.join(dir, "last_incumbent_eval.json"));
        if (!status && !begin && !selection && !done && !armsJson && !stopped && !lastInc) return null;
        const directives = arr(pick(begin, "directives")).filter(isObj);
        const armNames = new Set();
        for (const d of directives) if (isSafeId(str(d.arm))) armNames.add(d.arm);
        for (const a of arr(pick(armsJson, "order"))) if (isSafeId(str(a))) armNames.add(a);
        if (isObj(status?.arms)) for (const a of Object.keys(status.arms)) if (isSafeId(a)) armNames.add(a);
        const arms = [];
        for (const arm of [...armNames].slice(0, 32)) {
            const ad = await this.readJson(path.join(dir, arm, "arm.done"));
            const result = isObj(ad?.result) ? ad.result : null;
            const directive = directives.find((d) => d.arm === arm) ?? (isObj(ad?.directive) ? ad.directive : {});
            arms.push({
                arm,
                strategy: str(pick(directive, "strategy")),
                component: str(pick(directive, "component")),
                done: !!ad,
                status: str(pick(result, "status")) ?? (ad ? "done" : null),
                score: result ? (firstNum(result, "score") ?? firstNum(result.eval, "score", "mean", "mean_score", "evolve_score")) : null,
                reason: ad && typeof ad.reason === "string" ? ad.reason.slice(0, 300) : null,
            });
        }
        let st = null;
        if (isObj(status)) {
            const sarms = {};
            if (isObj(status.arms)) {
                for (const [a, v] of Object.entries(status.arms)) {
                    if (!isObj(v)) continue;
                    sarms[a] = { strategy: str(v.strategy), state: str(v.state), phase: str(v.phase), updated: toEpochSec(v.updated) };
                }
            }
            st = {
                phase: str(status.phase),
                state: str(pick(status, "state", "status")),
                updated: toEpochSec(status.updated),
                heartbeat: num(pick(status, "heartbeat", "heartbeat_s", "heartbeat_interval")),
                traceId: str(pick(status.trace, "trace_id", "traceId")) ?? str(pick(status, "trace_id", "traceId")),
                spanId: str(pick(status.trace, "span_id", "spanId")),
                writers: arr(status.writers).filter((w) => typeof w === "string").slice(0, 64),
                pid: num(pick(status, "pid")),
                campaignId: str(pick(status, "campaign_id", "campaignId")),
                round: num(pick(status, "round")),
                arms: sarms,
            };
        }
        return {
            experimentId: str(pick(status, "experiment_id")) ?? name,
            source: this.rel(dir),
            mtimeMs,
            status: st,
            begin: begin
                ? { round: num(pick(begin, "round")), baseCommit: str(pick(begin, "base_commit")), eid: str(pick(begin, "eid")) }
                : null,
            selection: selection ? { decision: str(pick(selection, "decision")), winner: str(pick(selection, "winner")) } : null,
            done: done ? { decision: str(pick(done, "decision")), winner: str(pick(done, "winner")), pr: num(pick(done, "pr")) ?? str(pick(done, "pr")) } : null,
            stopped,
            arms,
        };
    }

    // ------------------------------------------------------------ telemetry spans JSONL
    /**
     * Imported CI runs (`ci-lab telemetry pull --run <id>`, design §12.5 D5):
     * `<imports>/<run_id>/manifest.json` (`files` relative to the run dir) + `*.jsonl`.
     */
    async readImports() {
        const out = [];
        const isJsonl = (n) => n.toLowerCase().endsWith(".jsonl");
        for (const root of this.roots.imports) {
            let dirs = (await this.list(root, { dirs: true })).filter((d) => isSafeId(d.name));
            dirs = (await this.mtimes(dirs)).sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, this.limits.runDirs);
            for (const d of dirs) {
                const meta = await this.readJson(path.join(d.path, "manifest.json"), 256 * 1024);
                const files = [];
                const listed = arr(pick(meta, "files")).filter((f) => typeof f === "string" && isJsonl(f));
                if (listed.length) {
                    for (const f of listed.slice(0, 200)) {
                        const p = path.resolve(d.path, f);
                        if (isWithin(d.path, p)) files.push(p);
                    }
                } else {
                    for (const e of await this.list(d.path, { max: 200 })) {
                        if (!e.dir && isJsonl(e.name)) files.push(e.path);
                        else if (e.dir) files.push(...(await this.list(e.path, { dirs: false, max: 200 })).filter((x) => isJsonl(x.name)).map((x) => x.path));
                    }
                }
                const digest = str(pick(meta, "digest"));
                out.push({
                    runId: str(pick(meta, "run_id", "runId")) ?? d.name,
                    dir: d.name,
                    mtimeMs: d.mtimeMs,
                    importedAt: toEpochSec(pick(meta, "pulled", "imported_at")) ?? d.mtimeMs / 1000,
                    artifactCreated: str(pick(meta, "artifact_created")),
                    artifact: str(pick(meta, "artifact")),
                    schemaVersion: num(pick(meta, "schemaVersion", "schema_version")),
                    digest: digest ? digest.slice(0, 80) : null,
                    digestVerified: meta ? !!digest : null,
                    repo: str(pick(meta, "repo")),
                    spans: num(pick(meta, "spans")),
                    traces: num(pick(meta, "traces")),
                    skippedLines: num(pick(meta, "skipped_lines")),
                    services: arr(pick(meta, "services")).filter((s) => typeof s === "string").slice(0, 16),
                    files,
                });
            }
        }
        return out;
    }

    async telemetryFiles(runs, imports = []) {
        const dirs = [];
        for (const r of this.roots.runs) dirs.push(path.join(r, "telemetry"));
        for (const run of runs ?? []) dirs.push(path.resolve(this.repoRoot, run.source, "telemetry"));
        for (const d of LAYOUT.extraTelemetryDirs) dirs.push(path.join(this.roots.repo, d));
        let files = [];
        const seen = new Set();
        for (const d of dirs) {
            const real = await this.realDir(d);
            if (!real || seen.has(real)) continue;
            seen.add(real);
            this.roots.telemetry.push(real);
            files.push(...(await this.list(real, { dirs: false })).filter((e) => LAYOUT.spanFileRe.test(e.name)));
        }
        for (const imp of imports) for (const p of imp.files) files.push({ name: path.basename(p), path: p, importRun: imp.runId });
        files = (await this.mtimes(files)).sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, this.limits.spanFiles);
        return files;
    }

    /** Raw OTLP-JSON-ish objects (one per line); the model flattens them. */
    async readSpanLines(runs, imports = []) {
        const lines = [];
        const files = await this.telemetryFiles(runs, imports);
        let budget = this.limits.spanLines;
        for (const f of files) {
            if (budget <= 0) break;
            const t = await this.readJsonlTail(f.path, { maxBytes: this.limits.spanFileBytes, maxLines: budget });
            if (!t) continue;
            if (f.importRun) for (const row of t.rows) lines.push({ ...row, __import: f.importRun });
            else lines.push(...t.rows);
            budget -= t.rows.length;
        }
        return { lines, files: files.map((f) => this.rel(f.path)) };
    }

    // ------------------------------------------------------------ AGL rollout journal
    async readRollouts(runs) {
        const dirs = [];
        const add = (base) => {
            for (const n of LAYOUT.aglDirNames) dirs.push(path.join(base, n));
        };
        for (const r of this.roots.runs) add(r);
        for (const run of runs ?? []) {
            const base = path.resolve(this.repoRoot, run.source);
            add(base);
            for (const a of run.arms ?? []) add(path.join(base, a.arm));
        }
        let files = [];
        const seen = new Set();
        for (const d of dirs) {
            const real = await this.realDir(d);
            if (!real || seen.has(real)) continue;
            seen.add(real);
            this.roots.agl.push(real);
            files.push(...(await this.list(real, { dirs: false })).filter((e) => LAYOUT.rolloutFileRe.test(e.name)));
        }
        files = (await this.mtimes(files)).sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, this.limits.aglFiles);
        const out = [];
        for (const f of files) {
            const t = await this.readJsonlTail(f.path, { maxBytes: this.limits.aglFileBytes, maxLines: 2000 });
            if (t) out.push(summarizeRollout(t.rows, f.name.replace(/\.jsonl$/i, ""), this.rel(f.path)));
        }
        return out;
    }

    // ------------------------------------------------------------ ASSERT results
    async readEvals() {
        const out = [];
        for (const root of this.roots.results) {
            for (const suite of await this.list(root, { dirs: true })) {
                if (!isSafeId(suite.name)) continue;
                const suiteJson = await this.readJson(path.join(suite.path, "suite.json"));
                let runs = (await this.list(suite.path, { dirs: true })).filter((r) => isSafeId(r.name));
                runs = (await this.mtimes(runs)).sort((a, b) => b.mtimeMs - a.mtimeMs).slice(0, this.limits.evalRuns);
                for (const r of runs) {
                    const ev = await this.readEvalRun(r.path, suite.name, r.name, suiteJson);
                    if (ev) out.push({ ...ev, mtimeMs: r.mtimeMs });
                }
            }
        }
        return out;
    }

    async readEvalRun(dir, suiteId, runId, suiteJson) {
        const manifest = await this.readJson(path.join(dir, "manifest.json"));
        const metrics = await this.readJson(path.join(dir, "metrics.json"));
        const scores = await this.readJsonlTail(path.join(dir, "scores.jsonl"), { maxLines: this.limits.scoreLines });
        if (!manifest && !metrics && !scores) return null;
        const scales = {};
        const rows = [];
        for (const row of scores?.rows ?? []) {
            if (row.type && row.type !== "prompt" && row.type !== "conversation" && row.type !== "case") continue;
            const verdict = isObj(row.verdict) ? row.verdict : {};
            const dims = {};
            if (isObj(verdict.dimensions)) {
                for (const [k, v] of Object.entries(verdict.dimensions)) {
                    if (typeof v === "boolean" || typeof v === "number" || typeof v === "string") dims[k] = typeof v === "string" ? v.slice(0, 80) : v;
                }
            }
            if (isObj(row.dimension_scales)) {
                for (const [k, s] of Object.entries(row.dimension_scales)) {
                    if (scales[k] || !isObj(s)) continue;
                    scales[k] = { type: str(s.type), values: arr(s.values).map((v) => (isObj(v) ? v.value : v)).filter((v) => v !== undefined) };
                }
            }
            rows.push({
                caseId: str(pick(row, "test_case_id", "case_id", "id")),
                behavior: str(row.behavior),
                judgeModel: str(row.judge_model),
                target: str(row.target),
                testerModel: str(row.tester_model),
                status: str(row.judge_status) ?? (row.judge_error ? "error" : "ok"),
                error: !!row.judge_error,
                dims,
                notApplicable: arr(row.not_applicable_score_keys).filter((k) => typeof k === "string"),
                scenario: str(pick(row.dimensions, "scenario")),
            });
        }
        const totals = isObj(metrics?.totals) ? metrics.totals : {};
        return {
            suiteId,
            runId,
            source: this.rel(dir),
            createdAt: str(pick(suiteJson, "created_at")),
            manifest: manifest
                ? {
                      status: str(manifest.status),
                      startedAt: str(manifest.started_at),
                      endedAt: str(manifest.ended_at),
                      heartbeatAt: str(manifest.heartbeat_at),
                      pid: num(manifest.pid),
                      stages: isObj(manifest.stages) ? Object.fromEntries(Object.entries(manifest.stages).map(([k, v]) => [k, str(v)])) : {},
                  }
                : null,
            metrics: metrics
                ? {
                      elapsedS: num(metrics.elapsed_s),
                      calls: num(totals.calls),
                      inputTokens: num(totals.input_tokens),
                      outputTokens: num(totals.output_tokens),
                      cachedInputTokens: num(totals.cached_input_tokens),
                      cacheHitRate: num(totals.cache_hit_rate),
                      models: isObj(metrics.per_model) ? Object.keys(metrics.per_model).slice(0, 20) : [],
                  }
                : null,
            scores: rows,
            scales,
            truncated: !!scores?.truncated,
            badLines: scores?.bad ?? 0,
        };
    }

    // ------------------------------------------------------------ full scan
    async scan() {
        this.warnings = [];
        await this.resolveRoots();
        const campaigns = await this.readCampaigns();
        const standalone = await this.readStandaloneExperiments();
        const sleep = await this.readSleep();
        const holdoutLooks = await this.readHoldoutLooks();
        const runs = await this.readRuns();
        const imports = await this.readImports();
        const spans = await this.readSpanLines(runs, imports);
        const rollouts = await this.readRollouts(runs);
        const evals = await this.readEvals();
        const bus = await readBus(this);
        return {
            scannedAt: Date.now() / 1000,
            roots: { ...this.roots, allowed: [...this.allowedRoots] },
            campaigns,
            standalone,
            sleep,
            holdoutLooks,
            runs,
            imports: imports.map(({ files, ...rest }) => ({ ...rest, files: files.length })),
            spanLines: spans.lines,
            spanFiles: spans.files,
            rollouts,
            evals,
            bus,
            warnings: [...this.warnings],
        };
    }
}

// ---------------------------------------------------------------- normalizers (pure)
function smallObj(o) {
    if (!isObj(o)) return {};
    const out = {};
    for (const [k, v] of Object.entries(o)) {
        if (v === null || ["string", "number", "boolean"].includes(typeof v)) out[k] = typeof v === "string" ? v.slice(0, 300) : v;
    }
    return out;
}

/**
 * Mirror of `ci_lab.obs.read_status`: docs sorted by `updated`; top-level fields from the most
 * recently updated writer win; arms merge per arm by their own `updated`; `writers` lists ids.
 */
export function aggregateStatus(docs) {
    const sorted = docs.filter(isObj).sort((a, b) => (toEpochSec(a.updated) ?? 0) - (toEpochSec(b.updated) ?? 0));
    if (!sorted.length) return null;
    const out = {};
    const arms = {};
    for (const doc of sorted) {
        if (isObj(doc.arms)) {
            for (const [arm, val] of Object.entries(doc.arms)) {
                if (!isObj(val)) continue;
                if ((toEpochSec(val.updated) ?? 0) >= (toEpochSec(arms[arm]?.updated) ?? 0)) arms[arm] = val;
            }
        }
        for (const [k, v] of Object.entries(doc)) if (!["arms", "writer", "seq"].includes(k)) out[k] = v;
    }
    if (Object.keys(arms).length) out.arms = arms;
    out.writers = sorted.map((d) => d.writer).filter(Boolean);
    return out;
}

export function normalizeHistoryRow(h) {
    return {
        eid: str(pick(h, "eid", "experiment_id")),
        round: num(pick(h, "round")),
        decision: str(pick(h, "decision")),
        winner: str(pick(h, "winner")),
        tokens: num(pick(h, "tokens")),
        score: num(pick(h, "score", "incumbent_score", "new_incumbent_score")),
        deltaS: num(pick(h, "delta_s", "deltaS")),
        deltaC: num(pick(h, "delta_c", "deltaC")),
        arms: arr(h?.arms)
            .filter(isObj)
            .map((a) => ({
                arm: str(a.arm),
                component: str(a.component),
                strategy: str(a.strategy),
                hypotheses: Array.isArray(a.hypotheses) ? a.hypotheses.length : num(a.hypotheses),
                accepted: typeof a.accepted === "boolean" ? a.accepted : null,
                score: num(a.score),
                status: str(a.status),
            })),
    };
}

/** OES 0.1.0 envelope (+ RRSI / sleep extensions) → compact, render-safe record. */
export function normalizeEnvelope(env, source = null) {
    if (!isObj(env)) return null;
    const ext = isObj(env.extensions) ? env.extensions : {};
    const x = isObj(ext[RRSI_EXT]) ? ext[RRSI_EXT] : {};
    const s = isObj(ext[SLEEP_EXT]) ? ext[SLEEP_EXT] : null;
    const e = isObj(env.experiment) ? env.experiment : {};
    const xv = isObj(x.variants) ? x.variants : {};
    const sel = isObj(x.selection) ? x.selection : null;
    const dec = isObj(env.decision) ? env.decision : {};
    const sc = isObj(env.scorecard) ? env.scorecard : {};
    return {
        id: str(e.id),
        title: str(e.title),
        hypothesis: str(e.hypothesis),
        status: str(e.status),
        tags: arr(e.tags).filter((t) => typeof t === "string"),
        kind: str(x.kind) ?? (s ? "sleep" : "experiment"),
        designType: str(env.design?.type),
        campaignId: str(x.campaignId),
        round: num(x.round),
        split: str(x.split),
        delta: num(x.delta),
        deltaMethod: str(x.deltaMethod),
        budget: num(x.budget),
        stall: typeof x.stall === "boolean" ? x.stall : null,
        ciLowerBound: num(x.ciLowerBound),
        parentExperimentId: str(x.lineage?.parentExperimentId),
        incumbentCommit: str(x.lineage?.incumbentCommit),
        judgeModel: str(x.evaluatorPin?.judgeModel),
        variants: arr(env.variants)
            .filter(isObj)
            .map((v) => {
                const xe = isObj(xv[v.id]) ? xv[v.id] : {};
                return {
                    id: str(v.id),
                    name: str(v.name),
                    role: str(v.role),
                    description: str(v.description),
                    status: str(xe.status) ?? str(v.config?.status),
                    harnessTree: str(v.config?.harnessTree) ?? str(xe.harnessTree),
                    refs: arr(v.codeReferences)
                        .filter(isObj)
                        .map((r) => ({ type: str(r.type), sha: str(r.sha), ref: str(r.ref), component: str(r.component) })),
                    edits: arr(xe.edits)
                        .filter(isObj)
                        .map((ed) => ({ component: str(ed.component), hypothesis: str(ed.hypothesis), commit: str(ed.commit) })),
                    critic: isObj(xe.critic)
                        ? { passed: xe.critic.passed === true, reasons: arr(xe.critic.reasons).filter((r) => typeof r === "string").slice(0, 10) }
                        : null,
                    headCommit: str(xe.headCommit),
                };
            }),
        metrics: arr(env.metrics)
            .filter(isObj)
            .map((m) => ({ id: str(m.id), name: str(m.name), role: str(m.role), direction: str(m.direction) })),
        sampleSizes: isObj(env.results?.sampleSizes) ? env.results.sampleSizes : {},
        results: arr(env.results?.metricResults)
            .filter(isObj)
            .map((r) => ({
                metricId: str(r.metricId),
                variantId: str(r.comparison?.variantId),
                baselineVariantId: str(r.comparison?.baselineVariantId),
                role: str(r.role),
                baselineValue: num(r.baselineValue),
                variantValue: num(r.variantValue),
                diff: num(r.absoluteDifference),
                ci: isObj(r.confidenceInterval)
                    ? { level: num(r.confidenceInterval.level), lower: num(r.confidenceInterval.lower), upper: num(r.confidenceInterval.upper) }
                    : null,
                status: str(r.resultStatus),
                impact: str(r.decisionImpact),
            })),
        scorecard: { summary: str(sc.summary), overallResult: str(sc.overallResult), recommendedAction: str(sc.recommendedAction), qualityStatus: str(sc.qualityStatus) },
        decision: {
            status: str(dec.status),
            outcome: str(dec.outcome),
            rationale: str(dec.rationale),
            decidedBy: str(dec.decidedBy?.name) ?? str(dec.decidedBy?.type) ?? str(dec.decidedBy),
            decidedAt: str(dec.decidedAt),
        },
        selection: sel
            ? {
                  winner: str(sel.winner),
                  incumbentScore: num(sel.incumbentScore),
                  newIncumbentScore: num(sel.newIncumbentScore),
                  candidates: arr(sel.candidates)
                      .filter(isObj)
                      .map((c) => ({
                          variantId: str(c.variantId),
                          admissible: c.admissible === true,
                          deltaS: num(c.deltaS),
                          deltaC: num(c.deltaC),
                          novelty: num(c.novelty),
                          ciLowerBound: num(c.ciLowerBound),
                          rule: str(c.rule),
                          reasons: arr(c.reasons).filter((r) => typeof r === "string").slice(0, 10),
                      })),
              }
            : null,
        quality: arr(env.qualityChecks)
            .filter(isObj)
            .map((q) => ({ checkType: str(q.checkType), status: str(q.status), severity: str(q.severity), message: str(q.message) })),
        sleep: s
            ? {
                  night: str(s.night),
                  nightIndex: num(s.nightIndex),
                  tasks: {
                      total: num(s.tasks?.total),
                      reviewed: num(s.tasks?.byOrigin?.reviewed),
                      harvested: num(s.tasks?.byOrigin?.harvested),
                  },
                  gate: {
                      skilloptPassed: typeof s.gate?.skillopt?.passed === "boolean" ? s.gate.skillopt.passed : null,
                      skilloptScore: num(s.gate?.skillopt?.score),
                      skilloptBaseline: num(s.gate?.skillopt?.baselineScore),
                      assertPassed: typeof s.gate?.assert?.passed === "boolean" ? s.gate.assert.passed : null,
                      ciLowerBound: num(s.gate?.assert?.ciLowerBound),
                      deltaS: num(s.gate?.assert?.deltaS),
                      safetyBaseline: num(s.gate?.assert?.safetyViolations?.baseline),
                      safetyCandidate: num(s.gate?.assert?.safetyViolations?.candidate),
                  },
                  used: isObj(s.budget?.used) ? smallObj(s.budget.used) : {},
                  limits: isObj(s.budget?.limits) ? smallObj(s.budget.limits) : {},
                  adoptionPr: str(s.adoptionPr) ?? num(s.adoptionPr),
                  skillPath: str(s.skillPath),
              }
            : null,
        source,
    };
}

/** Summarize one AGL journal file (`{"v":1,"kind":"start|event|finish",...}` lines). */
export function summarizeRollout(rows, fallbackId, source) {
    let start = null;
    let finish = null;
    let events = 0;
    let score = null;
    let reward = null;
    let lastTs = null;
    let errors = 0;
    const types = {};
    for (const r of rows) {
        const ts = toEpochSec(r.ts);
        if (ts !== null) lastTs = Math.max(lastTs ?? 0, ts);
        if (r.kind === "start") start = start ?? r;
        else if (r.kind === "finish") finish = finish ?? r;
        else if (r.kind === "event") {
            events++;
            const t = str(r.event_type) ?? "event";
            types[t] = (types[t] ?? 0) + 1;
            if (t === "ci.score") score = firstNum(r.data, "value", "score") ?? score;
            if (t === "reward") reward = firstNum(r.data, "value", "reward") ?? (num(r.data) ?? reward);
            if (t === "ci.error") errors++;
        }
    }
    const key = start?.key;
    return {
        rolloutId: str(pick(start ?? rows[0], "rollout_id")) ?? fallbackId,
        attemptId: str(pick(finish ?? start, "attempt_id")),
        key: typeof key === "string" ? key.slice(0, 200) : isObj(key) ? smallObj(key) : null,
        status: str(finish?.status) ?? (start ? "running" : "unknown"),
        startTs: toEpochSec(start?.ts),
        endTs: toEpochSec(finish?.ts),
        lastTs,
        events,
        eventTypes: types,
        score,
        reward,
        errors,
        source,
    };
}
