// Pure aggregation over the normalized scan produced by sources.mjs (no I/O, no clock reads).
import { isSensitiveAttr, sensitiveEnabled, isTraceId } from "./security.mjs";

export const DEFAULT_HEARTBEAT_SEC = 30;
export const SPAN_SCHEMA_VERSION = 1; // contracts.SPAN_SCHEMA_VERSION
const ASSERT_STALE_SEC = 300;
const MAX_ATTR_LEN = 500;
const MAX_TRACES = 500;

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);
const round4 = (x) => (typeof x === "number" && Number.isFinite(x) ? Math.round(x * 1e4) / 1e4 : null);
const CAMPAIGN_SUFFIX_RE = /^(.+?)-(r\d+|cal|confirm)$/;

export function campaignOf(experimentId) {
    const m = typeof experimentId === "string" ? CAMPAIGN_SUFFIX_RE.exec(experimentId) : null;
    return m ? m[1] : null;
}

// ================================================================ campaigns / experiments
function winnerCandidate(env) {
    const sel = env?.selection;
    if (!sel) return null;
    const cands = arr(sel.candidates);
    return cands.find((c) => c.variantId && c.variantId === sel.winner) ?? null;
}

export function experimentSummary(env) {
    const w = winnerCandidate(env);
    return {
        id: env.id,
        title: env.title,
        kind: env.kind,
        campaignId: env.campaignId ?? (env.kind === "sleep" ? null : campaignOf(env.id)),
        round: env.round,
        status: env.status,
        outcome: env.decision?.outcome ?? null,
        recommended: env.scorecard?.recommendedAction ?? null,
        winner: env.selection?.winner ?? null,
        deltaS: w?.deltaS ?? null,
        deltaC: w?.deltaC ?? null,
        ciLowerBound: env.ciLowerBound ?? w?.ciLowerBound ?? env.sleep?.gate?.ciLowerBound ?? null,
        decidedAt: env.decision?.decidedAt ?? null,
        night: env.sleep?.night ?? null,
        source: env.source,
    };
}

export function collectExperiments(scan) {
    const out = new Map();
    const add = (env) => {
        if (!env || !env.id) return;
        if (!out.has(env.id)) out.set(env.id, env);
    };
    for (const c of arr(scan.campaigns)) {
        for (const r of arr(c.rounds)) add(r.envelope ? { ...r.envelope, campaignId: r.envelope.campaignId ?? c.campaignId } : null);
        add(c.calibration?.envelope ? { ...c.calibration.envelope, campaignId: c.calibration.envelope.campaignId ?? c.campaignId } : null);
        add(c.confirm?.envelope ? { ...c.confirm.envelope, campaignId: c.confirm.envelope.campaignId ?? c.campaignId } : null);
    }
    for (const e of arr(scan.standalone)) add(e);
    for (const e of arr(scan.sleep?.nights)) add(e);
    return out;
}

export function campaignView(c) {
    const byEid = new Map();
    for (const h of arr(c.history)) {
        if (!h.eid) continue;
        byEid.set(h.eid, { eid: h.eid, round: h.round, decision: h.decision, winner: h.winner, tokens: h.tokens, score: h.score, deltaS: h.deltaS, deltaC: h.deltaC, arms: h.arms.length, accepted: h.arms.filter((a) => a.accepted).length });
    }
    for (const r of arr(c.rounds)) {
        const env = r.envelope;
        const eid = env?.id ?? r.eid;
        const cur = byEid.get(eid) ?? byEid.get(r.eid) ?? { eid, round: null, decision: null, winner: null, tokens: null, score: null, deltaS: null, deltaC: null, arms: 0, accepted: 0 };
        if (env) {
            if (env.kind && !["round", "experiment"].includes(env.kind)) continue;
            const w = winnerCandidate(env);
            cur.round ??= env.round;
            cur.decision ??= env.decision?.outcome ?? null;
            cur.winner ??= env.selection?.winner ?? null;
            cur.score ??= env.selection?.newIncumbentScore ?? null;
            cur.deltaS ??= w?.deltaS ?? null;
            cur.deltaC ??= w?.deltaC ?? null;
            cur.arms ||= env.variants.filter((v) => v.role !== "baseline").length;
            cur.admissible = arr(env.selection?.candidates).filter((x) => x.admissible).length;
            cur.ciLowerBound = env.ciLowerBound ?? w?.ciLowerBound ?? null;
        }
        byEid.set(cur.eid, cur);
    }
    const rounds = [...byEid.values()].sort((a, b) => (a.round ?? 1e9) - (b.round ?? 1e9) || String(a.eid).localeCompare(String(b.eid)));
    const tokensUsed = rounds.reduce((s, r) => s + (r.tokens ?? 0), 0) + (c.calibration?.tokens ?? 0);
    const decisions = {};
    for (const r of rounds) if (r.decision) decisions[r.decision] = (decisions[r.decision] ?? 0) + 1;
    const last = rounds.at(-1) ?? null;
    return {
        campaignId: c.campaignId,
        profile: c.profile,
        domain: c.domain,
        source: c.source,
        incumbent: c.frontier ?? null,
        delta: c.calibration?.delta ?? c.calibration?.envelope?.delta ?? null,
        rounds,
        latestRound: last,
        trend: rounds.map((r) => ({ round: r.round, eid: r.eid, score: r.score, deltaS: r.deltaS, deltaC: r.deltaC })),
        budget: {
            tokensUsed: tokensUsed || null,
            tokensLimit: c.budget?.tokens ?? null,
            tokenBurn: c.budget?.tokens && tokensUsed ? round4(tokensUsed / c.budget.tokens) : null,
            roundsDone: rounds.length,
            roundsLimit: c.budget?.rounds ?? null,
        },
        decisions,
        confirm: c.confirm ? { outcome: c.confirm.envelope?.decision?.outcome ?? c.confirm.outcome ?? c.confirm.decision ?? null, id: c.confirm.envelope?.id ?? null } : null,
        land: c.land ?? null,
        stackSize: c.stackSize ?? null,
        stopped: false,
    };
}

// ================================================================ live (status.d / status.json)
const DONE_STATES = new Set(["done", "completed", "complete", "finished", "succeeded", "failed", "error", "cancelled", "canceled", "stopped", "skipped"]);

export function liveView(runs, { now, heartbeatSec = DEFAULT_HEARTBEAT_SEC } = {}) {
    const out = [];
    for (const r of arr(runs)) {
        const st = r.status;
        const updated = st?.updated ?? (r.mtimeMs ? r.mtimeMs / 1000 : null);
        const hb = st?.heartbeat && st.heartbeat > 0 ? st.heartbeat : heartbeatSec;
        const age = updated !== null && now !== undefined ? Math.max(0, now - updated) : null;
        const finished = !!r.done || DONE_STATES.has(String(st?.state ?? "").toLowerCase()) || DONE_STATES.has(String(st?.phase ?? "").toLowerCase());
        const stale = !finished && age !== null && age > 2 * hb;
        const armMap = new Map();
        for (const a of arr(r.arms)) armMap.set(a.arm, { arm: a.arm, strategy: a.strategy, component: a.component, state: a.done ? (a.status ?? "done") : null, phase: null, ageSec: null, stale: false, done: a.done, score: a.score });
        for (const [arm, v] of Object.entries(st?.arms ?? {})) {
            const cur = armMap.get(arm) ?? { arm, strategy: null, component: null, state: null, phase: null, ageSec: null, stale: false, done: false, score: null };
            const aAge = v.updated !== null && v.updated !== undefined && now !== undefined ? Math.max(0, now - v.updated) : null;
            const aDone = cur.done || DONE_STATES.has(String(v.state ?? "").toLowerCase());
            Object.assign(cur, {
                strategy: v.strategy ?? cur.strategy,
                state: cur.done ? cur.state : (v.state ?? cur.state),
                phase: v.phase ?? cur.phase,
                ageSec: aAge === null ? null : Math.round(aAge),
                stale: !aDone && !finished && aAge !== null && aAge > 2 * hb,
                done: aDone,
            });
            armMap.set(arm, cur);
        }
        out.push({
            experimentId: r.experimentId,
            campaignId: st?.campaignId ?? campaignOf(r.experimentId),
            source: r.source,
            phase: st?.phase ?? (r.done ? "done" : r.selection ? "select" : r.begin ? "running" : null),
            state: finished ? "done" : stale ? "stale" : st || r.begin ? "running" : "idle",
            stale,
            ageSec: age === null ? null : Math.round(age),
            heartbeatSec: hb,
            updated,
            traceId: st?.traceId ?? null,
            writers: st?.writers ?? [],
            round: st?.round ?? r.begin?.round ?? null,
            decision: r.done?.decision ?? r.selection?.decision ?? null,
            winner: r.done?.winner ?? r.selection?.winner ?? null,
            pr: r.done?.pr ?? null,
            stopped: !!r.stopped,
            arms: [...armMap.values()],
        });
    }
    const rank = { running: 0, stale: 1, idle: 2, done: 3 };
    out.sort((a, b) => rank[a.state] - rank[b.state] || (b.updated ?? 0) - (a.updated ?? 0));
    return out;
}

// ================================================================ spans (OTLP-JSON from JSONL and Aspire)
function decodeAnyValue(v) {
    if (!isObj(v)) return v ?? null;
    if ("stringValue" in v) return v.stringValue;
    if ("boolValue" in v) return v.boolValue;
    if ("intValue" in v) {
        const n = Number(v.intValue);
        return Number.isSafeInteger(n) ? n : String(v.intValue);
    }
    if ("doubleValue" in v) return Number(v.doubleValue);
    if ("arrayValue" in v) return arr(v.arrayValue?.values).map(decodeAnyValue);
    if ("kvlistValue" in v) return attrMap(v.kvlistValue?.values);
    if ("bytesValue" in v) return "<bytes>";
    return null;
}

/** OTLP KeyValue[] or a plain map → plain map. */
export function attrMap(attrs) {
    if (Array.isArray(attrs)) {
        const out = {};
        for (const kv of attrs) if (isObj(kv) && typeof kv.key === "string") out[kv.key] = decodeAnyValue(kv.value);
        return out;
    }
    if (isObj(attrs)) {
        if (Array.isArray(attrs.attributes)) return attrMap(attrs.attributes);
        const out = {};
        for (const [k, v] of Object.entries(attrs)) out[k] = isObj(v) && ("stringValue" in v || "intValue" in v || "boolValue" in v || "doubleValue" in v || "arrayValue" in v) ? decodeAnyValue(v) : v;
        return out;
    }
    return {};
}

function normId(id, hexLen) {
    if (typeof id !== "string" || !id) return null;
    if (new RegExp(`^[0-9a-fA-F]{${hexLen}}$`).test(id)) return id.toLowerCase();
    if (/^[A-Za-z0-9+/]+={0,2}$/.test(id)) {
        const hex = Buffer.from(id, "base64").toString("hex");
        if (hex.length === hexLen) return hex;
    }
    return null;
}

export function nanosToMs(v) {
    if (typeof v === "number" && Number.isFinite(v)) return v / 1e6;
    if (typeof v === "string" && /^\d{1,24}$/.test(v)) return Number(BigInt(v) / 1000n) / 1000;
    if (typeof v === "string" && v) {
        const t = Date.parse(v);
        if (Number.isFinite(t)) return t;
    }
    return null;
}

function normStatus(s) {
    if (!isObj(s)) return { code: 0, message: null };
    let code = s.code ?? s.status_code ?? 0;
    if (typeof code === "string") code = /ERROR/i.test(code) ? 2 : /OK/i.test(code) ? 1 : Number(code) || 0;
    return { code, message: typeof s.message === "string" ? s.message.slice(0, MAX_ATTR_LEN) : null };
}

function normSpan(s, resource, scope, origin) {
    const traceId = normId(s.traceId ?? s.trace_id, 32);
    const spanId = normId(s.spanId ?? s.span_id, 16);
    if (!traceId || !spanId) return null;
    const startMs = nanosToMs(s.startTimeUnixNano ?? s.start_time_unix_nano ?? s.start_time);
    const endMs = nanosToMs(s.endTimeUnixNano ?? s.end_time_unix_nano ?? s.end_time);
    const status = normStatus(s.status);
    return {
        traceId,
        spanId,
        parentSpanId: normId(s.parentSpanId ?? s.parent_span_id, 16),
        name: typeof s.name === "string" ? s.name.slice(0, 200) : "(unnamed)",
        kind: s.kind ?? null,
        startMs,
        endMs,
        durationMs: startMs !== null && endMs !== null ? Math.max(0, endMs - startMs) : null,
        status,
        error: status.code === 2,
        attributes: attrMap(s.attributes),
        resource,
        scope: typeof scope?.name === "string" ? scope.name : null,
        events: arr(s.events).length,
        exceptionTypes: arr(s.events)
            .filter((e) => isObj(e) && e.name === "exception")
            .map((e) => attrMap(e.attributes)["exception.type"])
            .filter((t) => typeof t === "string")
            .slice(0, 5),
        links: arr(s.links)
            .filter(isObj)
            .map((l) => ({ traceId: normId(l.traceId ?? l.trace_id, 32), spanId: normId(l.spanId ?? l.span_id, 16) }))
            .filter((l) => l.traceId),
        origin,
    };
}

// Span-per-line records follow `ci_lab.telemetry.record` (key `schemaVersion`); like its
// `validate()`, records with a missing or unknown version are rejected. OTLP batches
// (`resourceSpans`, Aspire responses) are unversioned and accepted.
function schemaOk(o) {
    if (Array.isArray(o.resourceSpans ?? o.data?.resourceSpans)) return true;
    return Number(o.schemaVersion) === SPAN_SCHEMA_VERSION;
}

/**
 * Flatten any mix of: span-per-line records (JSONL exporter), `{resourceSpans:[...]}` batches,
 * and Aspire `/api/telemetry/*` responses `{data:{resourceSpans:[...]}}` → normalized spans.
 * Unknown span schema versions are rejected (design §12.5 D7).
 */
export function flattenOtlp(input, origin = "jsonl") {
    const out = [];
    let rejected = 0;
    const items = Array.isArray(input) ? input : [input];
    for (const item of items) {
        if (!isObj(item)) continue;
        if (!schemaOk(item)) {
            rejected++;
            continue;
        }
        const o = item.__import ? `import:${item.__import}` : origin;
        const rs = item.resourceSpans ?? item.data?.resourceSpans;
        if (Array.isArray(rs)) {
            for (const r of rs) {
                if (!isObj(r)) continue;
                const resource = attrMap(r.resource);
                for (const ss of arr(r.scopeSpans ?? r.instrumentationLibrarySpans)) {
                    for (const s of arr(ss?.spans)) {
                        const n = isObj(s) ? normSpan(s, resource, ss.scope ?? ss.instrumentationLibrary, o) : null;
                        if (n) out.push(n);
                    }
                }
            }
            continue;
        }
        const n = normSpan(item, attrMap(item.resource), item.scope ?? item.instrumentationScope, o);
        if (n) out.push(n);
    }
    out.rejected = rejected;
    return out;
}

/** Dedupe by traceId/spanId, preferring ended spans and local JSONL over remote copies. */
export function mergeSpans(...lists) {
    const map = new Map();
    for (const list of lists) {
        for (const s of arr(list)) {
            const k = `${s.traceId}/${s.spanId}`;
            const cur = map.get(k);
            if (!cur || (cur.endMs === null && s.endMs !== null)) map.set(k, s);
        }
    }
    return [...map.values()];
}

export function spanCategory(s) {
    if (s.name.startsWith("ci.")) return "harness";
    if (s.attributes["gen_ai.operation.name"] !== undefined || /^(invoke_agent|chat|execute_tool|workflow|executor|create_agent|embeddings|text_completion|generate_content)\b/.test(s.name)) return "genai";
    return "other";
}

const service = (s) => (typeof s.resource?.["service.name"] === "string" ? s.resource["service.name"] : null);

export function traceList(spans, { limit = MAX_TRACES } = {}) {
    const groups = new Map();
    for (const s of arr(spans)) {
        let g = groups.get(s.traceId);
        if (!g) groups.set(s.traceId, (g = []));
        g.push(s);
    }
    const out = [];
    for (const [traceId, list] of groups) {
        const ids = new Set(list.map((s) => s.spanId));
        const roots = list.filter((s) => !s.parentSpanId || !ids.has(s.parentSpanId));
        roots.sort((a, b) => (a.startMs ?? Infinity) - (b.startMs ?? Infinity));
        const root = roots[0] ?? list[0];
        const start = Math.min(...list.map((s) => s.startMs ?? Infinity));
        const end = Math.max(...list.map((s) => s.endMs ?? -Infinity));
        const attr = (k) => list.map((s) => s.attributes[k]).find((v) => v !== undefined && v !== null) ?? null;
        out.push({
            traceId,
            name: root.name,
            startMs: Number.isFinite(start) ? start : null,
            durationMs: Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, end - start) : null,
            spans: list.length,
            errors: list.filter((s) => s.error).length,
            genai: list.filter((s) => spanCategory(s) === "genai").length,
            campaignId: attr("ci.campaign_id"),
            experimentId: attr("oes.experiment_id"),
            round: attr("rrsi.round"),
            night: attr("sleep.night"),
            services: [...new Set(list.map(service).filter(Boolean))].slice(0, 8),
            origin: [...new Set(list.map((s) => s.origin))].join(","),
            incomplete: roots.length !== 1 || list.some((s) => s.endMs === null || (s.parentSpanId && !ids.has(s.parentSpanId))),
        });
    }
    out.sort((a, b) => (b.startMs ?? 0) - (a.startMs ?? 0));
    return out.slice(0, limit);
}

function fmtAttr(v) {
    if (v === null || v === undefined) return "";
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return s.length > MAX_ATTR_LEN ? `${s.slice(0, MAX_ATTR_LEN)}…` : s;
}

/** C29: drop GenAI content attributes unless the span/resource declares sensitive mode. */
export function visibleAttributes(span) {
    const allow = sensitiveEnabled(span.attributes, span.resource);
    const attrs = {};
    let redacted = 0;
    for (const [k, v] of Object.entries(span.attributes ?? {})) {
        if (!allow && isSensitiveAttr(k)) {
            redacted++;
            continue;
        }
        attrs[k] = fmtAttr(v);
    }
    return { attrs, redacted, sensitive: allow };
}

