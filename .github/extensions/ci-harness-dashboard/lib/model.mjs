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

/** Collapsible tree for one trace: round → arm → step → MAF workflow → executor → agent → chat/tool. */
export function buildSpanTree(spans, traceId) {
    if (!isTraceId(traceId)) return null;
    const list = arr(spans).filter((s) => s.traceId === traceId.toLowerCase());
    if (!list.length) return null;
    const byId = new Map(list.map((s) => [s.spanId, s]));
    const kids = new Map();
    const roots = [];
    for (const s of list) {
        if (s.parentSpanId && byId.has(s.parentSpanId) && s.parentSpanId !== s.spanId) {
            if (!kids.has(s.parentSpanId)) kids.set(s.parentSpanId, []);
            kids.get(s.parentSpanId).push(s);
        } else roots.push(s);
    }
    const t0 = Math.min(...list.map((s) => s.startMs ?? Infinity));
    const byStart = (a, b) => (a.startMs ?? Infinity) - (b.startMs ?? Infinity);
    const visited = new Set();
    const node = (s, depth) => {
        visited.add(s.spanId);
        const { attrs, redacted, sensitive } = visibleAttributes(s);
        const children = depth >= 64 ? [] : (kids.get(s.spanId) ?? []).filter((c) => !visited.has(c.spanId)).sort(byStart).map((c) => node(c, depth + 1));
        return {
            spanId: s.spanId,
            parentSpanId: s.parentSpanId,
            name: s.name,
            category: spanCategory(s),
            service: service(s),
            offsetMs: s.startMs !== null && Number.isFinite(t0) ? s.startMs - t0 : null,
            durationMs: s.durationMs,
            error: s.error,
            statusMessage: s.status.message,
            exceptionTypes: s.exceptionTypes,
            attributes: attrs,
            redacted,
            sensitive,
            links: s.links,
            origin: s.origin,
            children,
        };
    };
    const tree = roots.sort(byStart).map((r) => node(r, 0));
    // Cycles leave spans unreachable from any root: surface them as extra roots.
    for (const s of list.sort(byStart)) if (!visited.has(s.spanId)) tree.push(node(s, 0));
    const summary = traceList(list)[0];
    return { ...summary, roots: tree };
}

// ================================================================ evals (ASSERT)
function dimStats(rows, key, scale) {
    const vals = rows.map((r) => r.dims[key]).filter((v) => v !== undefined && v !== null);
    if (!vals.length) return { key, type: scale?.type ?? null, n: 0 };
    if (vals.every((v) => typeof v === "boolean")) {
        const t = vals.filter(Boolean).length;
        return { key, type: "boolean", n: vals.length, trueCount: t, rate: round4(t / vals.length) };
    }
    if (vals.every((v) => typeof v === "number")) {
        const dist = {};
        for (const v of vals) dist[v] = (dist[v] ?? 0) + 1;
        return { key, type: "numeric", n: vals.length, mean: round4(vals.reduce((a, b) => a + b, 0) / vals.length), min: Math.min(...vals), max: Math.max(...vals), distribution: dist };
    }
    const dist = {};
    for (const v of vals) dist[String(v)] = (dist[String(v)] ?? 0) + 1;
    const order = arr(scale?.values).map(String);
    return { key, type: "ordinal", n: vals.length, distribution: dist, scale: order, mode: Object.entries(dist).sort((a, b) => b[1] - a[1])[0][0] };
}

export function evalView(ev, { now } = {}) {
    const rows = arr(ev.scores);
    const keys = new Set();
    for (const r of rows) for (const k of Object.keys(r.dims)) keys.add(k);
    const dimensions = [...keys].map((k) => dimStats(rows, k, ev.scales?.[k]));
    const judges = [...new Set(rows.map((r) => r.judgeModel).filter(Boolean))];
    let agreement = null;
    if (judges.length > 1) {
        const byCase = new Map();
        for (const r of rows) {
            const k = `${r.caseId}|${r.target ?? ""}`;
            if (!byCase.has(k)) byCase.set(k, []);
            byCase.get(k).push(r);
        }
        const multi = [...byCase.values()].filter((g) => new Set(g.map((r) => r.judgeModel)).size > 1);
        const byDim = {};
        for (const k of keys) {
            const comparable = multi.filter((g) => g.filter((r) => r.dims[k] !== undefined).length > 1);
            if (!comparable.length) continue;
            const agree = comparable.filter((g) => new Set(g.filter((r) => r.dims[k] !== undefined).map((r) => String(r.dims[k]))).size === 1).length;
            byDim[k] = round4(agree / comparable.length);
        }
        agreement = { judges: judges.length, cases: multi.length, byDim };
    }
    const errors = rows.filter((r) => r.error || (r.status && r.status !== "ok")).length;
    const notApplicable = rows.reduce((s, r) => s + r.notApplicable.length, 0);
    const flags = [];
    if (errors) flags.push(`${errors} judge error(s)`);
    if (ev.truncated) flags.push("scores.jsonl truncated to tail");
    if (ev.badLines) flags.push(`${ev.badLines} unparseable line(s)`);
    if (rows.length && notApplicable / (rows.length * Math.max(1, keys.size)) > 0.25) flags.push("many not-applicable scores");
    const hb = Date.parse(ev.manifest?.heartbeatAt ?? "");
    const running = ev.manifest?.status === "running";
    if (running && Number.isFinite(hb) && now !== undefined && now - hb / 1000 > ASSERT_STALE_SEC) flags.push("stale: heartbeat older than 5 min");
    if (agreement) for (const [k, v] of Object.entries(agreement.byDim)) if (v < 0.8) flags.push(`low judge agreement on ${k} (${Math.round(v * 100)}%)`);
    return {
        suiteId: ev.suiteId,
        runId: ev.runId,
        source: ev.source,
        status: ev.manifest?.status ?? null,
        startedAt: ev.manifest?.startedAt ?? ev.createdAt ?? null,
        endedAt: ev.manifest?.endedAt ?? null,
        elapsedS: ev.metrics?.elapsedS ?? null,
        cases: new Set(rows.map((r) => r.caseId)).size,
        rows: rows.length,
        judgeModels: judges,
        targets: [...new Set(rows.map((r) => r.target).filter(Boolean))].slice(0, 10),
        scenarios: Object.entries(rows.reduce((m, r) => (r.scenario ? ((m[r.scenario] = (m[r.scenario] ?? 0) + 1), m) : m), {})).map(([name, n]) => ({ name, n })),
        errors,
        notApplicable,
        dimensions,
        agreement,
        flags,
        metrics: ev.metrics,
        mtimeMs: ev.mtimeMs ?? null,
    };
}

// ================================================================ sleep / rollouts
export function sleepView(sleep) {
    const nights = arr(sleep?.nights).map((n) => ({
        id: n.id,
        night: n.sleep?.night ?? null,
        nightIndex: n.sleep?.nightIndex ?? null,
        status: n.status,
        outcome: n.decision?.outcome ?? null,
        rationale: n.decision?.rationale ?? null,
        tasks: n.sleep?.tasks ?? null,
        gate: n.sleep?.gate ?? null,
        used: n.sleep?.used ?? {},
        adoptionPr: n.sleep?.adoptionPr ?? null,
        skillPath: n.sleep?.skillPath ?? null,
        source: n.source,
    }));
    return {
        source: sleep?.source ?? null,
        state: sleep?.state ?? null,
        tasks: sleep?.tasks ?? { reviewed: null, pending: null },
        nights,
        harvested: nights.reduce((s, n) => s + (n.tasks?.harvested ?? 0), 0),
        skillsUpdated: [...new Set(nights.filter((n) => n.outcome === "ship" && n.skillPath).map((n) => n.skillPath))],
        pendingPrs: nights.filter((n) => n.adoptionPr && n.outcome === "ship").map((n) => ({ night: n.night ?? n.id, pr: n.adoptionPr })),
    };
}

export function rolloutView(rollouts) {
    const list = arr(rollouts);
    const byStatus = {};
    for (const r of list) byStatus[r.status] = (byStatus[r.status] ?? 0) + 1;
    const scores = list.map((r) => r.score).filter((x) => typeof x === "number");
    return {
        total: list.length,
        byStatus,
        meanScore: scores.length ? round4(scores.reduce((a, b) => a + b, 0) / scores.length) : null,
        recent: [...list].sort((a, b) => (b.lastTs ?? 0) - (a.lastTs ?? 0)).slice(0, 50),
    };
}

/** Bus tab: per-run totals over the already orchestrator-projected topics from `lib/bus.mjs`. */
export function busView(runs) {
    const KEYS = ["topics", "entries", "exploits", "patches", "committed", "aborted", "rejected", "corrupt", "torn"];
    const zero = () => Object.fromEntries(KEYS.map((k) => [k, 0]));
    const totals = { runs: 0, ...zero() };
    const list = arr(runs).map((r) => {
        const c = zero();
        for (const t of arr(r.topics)) {
            c.topics++;
            c.entries += t.entries;
            c.exploits += t.exploits;
            c.patches += t.patches;
            if (c[t.state] !== undefined) c[t.state]++;
            if (t.corrupt) c.corrupt++;
            if (t.torn) c.torn++;
        }
        totals.runs++;
        for (const k of KEYS) totals[k] += c[k];
        return { ...r, counts: c };
    });
    return { runs: list.sort((a, b) => (b.mtimeMs ?? 0) - (a.mtimeMs ?? 0)), totals };
}

// ================================================================ whole model
export function buildModel(scan, { now, heartbeatSec = DEFAULT_HEARTBEAT_SEC, extraSpans = [] } = {}) {
    const t = now ?? scan.scannedAt;
    const experiments = collectExperiments(scan);
    const live = liveView(scan.runs, { now: t, heartbeatSec });
    const stoppedCampaigns = new Set(arr(scan.runs).filter((r) => r.stopped).map((r) => r.experimentId));
    const campaigns = arr(scan.campaigns).map((c) => ({ ...campaignView(c), stopped: stoppedCampaigns.has(c.campaignId) }));
    const fileSpans = flattenOtlp(arr(scan.spanLines), "jsonl");
    const spans = mergeSpans(fileSpans, extraSpans);
    const evals = arr(scan.evals).map((e) => evalView(e, { now: t })).sort((a, b) => (b.mtimeMs ?? 0) - (a.mtimeMs ?? 0));
    return {
        generatedAt: t,
        heartbeatSec,
        roots: {
            ledger: arr(scan.roots?.ledger).length,
            runs: arr(scan.roots?.runs).length,
            results: arr(scan.roots?.results).length,
            imports: arr(scan.roots?.imports).length,
        },
        campaigns,
        experiments: [...experiments.values()].map(experimentSummary),
        experimentsById: experiments,
        live,
        spans,
        rejectedSpans: fileSpans.rejected ?? 0,
        spanFiles: arr(scan.spanFiles),
        traces: traceList(spans),
        evals,
        sleep: sleepView(scan.sleep),
        holdoutLooks: scan.holdoutLooks ?? 0,
        rollouts: rolloutView(scan.rollouts),
        bus: busView(scan.bus),
        imports: arr(scan.imports),
        warnings: arr(scan.warnings),
    };
}

export function experimentDetail(model, id) {
    const env = model.experimentsById.get(id);
    const run = model.live.find((r) => r.experimentId === id) ?? null;
    if (!env && !run) return null;
    const traceIds = new Set(model.traces.filter((t) => t.experimentId === id).map((t) => t.traceId));
    if (run?.traceId) traceIds.add(run.traceId);
    return { envelope: env ?? null, summary: env ? experimentSummary(env) : null, run, traceIds: [...traceIds] };
}

/** Compact JSON for the agent (`get_summary`): counts and headlines only, no secrets, no content. */
export function summarize(model) {
    const lastDecisions = [...model.experiments]
        .filter((e) => e.outcome)
        .sort((a, b) => String(b.decidedAt ?? "").localeCompare(String(a.decidedAt ?? "")))
        .slice(0, 5)
        .map((e) => ({ id: e.id, kind: e.kind, outcome: e.outcome, winner: e.winner, deltaS: e.deltaS, deltaC: e.deltaC }));
    const count = (s) => model.live.filter((r) => r.state === s).length;
    return {
        generatedAt: model.generatedAt,
        campaigns: model.campaigns.map((c) => ({
            id: c.campaignId,
            incumbentScore: c.incumbent?.score ?? null,
            incumbentRound: c.incumbent?.round ?? null,
            rounds: c.rounds.length,
            lastDecision: c.latestRound?.decision ?? null,
            stopped: c.stopped,
            tokenBurn: c.budget.tokenBurn,
        })),
        live: {
            running: count("running"),
            stale: count("stale"),
            done: count("done"),
            active: model.live.filter((r) => r.state === "running" || r.state === "stale").slice(0, 5).map((r) => ({ id: r.experimentId, phase: r.phase, state: r.state, ageSec: r.ageSec, arms: r.arms.length })),
        },
        experiments: model.experiments.length,
        lastDecisions,
        evals: model.evals.slice(0, 5).map((e) => ({ suite: e.suiteId, run: e.runId, status: e.status, cases: e.cases, flags: e.flags.length })),
        sleep: {
            lastNight: model.sleep.state?.lastNightId ?? model.sleep.nights.at(-1)?.id ?? null,
            lastStatus: model.sleep.state?.lastStatus ?? null,
            acceptedTotal: model.sleep.state?.acceptedTotal ?? null,
            pendingTasks: model.sleep.tasks.pending,
        },
        traces: model.traces.length,
        traceErrors: model.traces.reduce((s, t) => s + t.errors, 0),
        rollouts: { total: model.rollouts.total, byStatus: model.rollouts.byStatus },
        bus: model.bus.totals,
        imports: model.imports.length,
        warnings: model.warnings.length,
    };
}
