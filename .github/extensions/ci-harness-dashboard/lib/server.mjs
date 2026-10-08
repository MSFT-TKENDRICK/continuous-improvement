// Per-canvas-instance loopback server (design §12.2/§12.4 C31): static UI, JSON API, SSE.
// The iframe has no privileged bridge; it talks to this server only (fetch + EventSource).
import http from "node:http";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { flattenOtlp, mergeSpans, buildSpanTree, traceList, experimentDetail, summarize } from "./model.mjs";
import { hostAllowed, originAllowed, securityHeaders, newToken, tokenEquals, isSafeId, isTraceId, stripSecrets } from "./security.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_UI_DIR = path.resolve(here, "..", "ui");
export const VIEWS = ["overview", "live", "experiment", "traces", "evals", "sleep", "aspire"];
const STATIC = {
    "/": ["index.html", "text/html; charset=utf-8"],
    "/index.html": ["index.html", "text/html; charset=utf-8"],
    "/app.js": ["app.js", "text/javascript; charset=utf-8"],
    "/app.css": ["app.css", "text/css; charset=utf-8"],
};
const BODY_MAX = 16 * 1024;
const TOKEN_HEADER = "x-ci-token";

/** Validate/normalize a partial UI state ({view, campaignId, experimentId, traceId}); throws on bad input. */
export function normalizeUiState(input) {
    const out = {};
    if (!input || typeof input !== "object") return out;
    if (input.view !== undefined && input.view !== null) {
        if (!VIEWS.includes(input.view)) throw new RangeError(`unknown view: ${String(input.view).slice(0, 40)}`);
        out.view = input.view;
    }
    for (const k of ["campaignId", "experimentId"]) {
        if (input[k] === undefined) continue;
        if (input[k] === null || input[k] === "") out[k] = null;
        else if (isSafeId(input[k])) out[k] = input[k];
        else throw new RangeError(`invalid ${k}`);
    }
    if (input.traceId !== undefined) {
        if (input.traceId === null || input.traceId === "") out.traceId = null;
        else if (isTraceId(input.traceId)) out.traceId = input.traceId.toLowerCase();
        else throw new RangeError("invalid traceId (expected 32 hex chars)");
    }
    return out;
}

/**
 * @param {{hub: import('./hub.mjs').Hub, aspire?: import('./aspire.mjs').AspireClient, instanceId?: string,
 *          initialState?: object, log?: Function, uiDir?: string, pingMs?: number, aspireTimeoutMs?: number}} opts
 */
export async function startInstanceServer(opts) {
    const { hub, aspire = null, log = () => {}, uiDir = DEFAULT_UI_DIR, pingMs = 15000 } = opts;
    const token = newToken();
    let ui = { view: "overview", campaignId: null, experimentId: null, traceId: null, ...normalizeUiState(opts.initialState ?? {}), seq: 0 };
    const clients = new Set();
    const staticCache = new Map();
    let port = 0;

    const send = (res, status, body, headers = {}) => {
        const isStr = typeof body === "string" || Buffer.isBuffer(body);
        const payload = isStr ? body : JSON.stringify(body);
        res.writeHead(status, securityHeaders({ "Content-Type": isStr ? "text/plain; charset=utf-8" : "application/json; charset=utf-8", ...headers }));
        res.end(payload);
    };
    const fail = (res, status, code, message) => send(res, status, { error: { code, message } });

    const broadcast = (event, data) => {
        const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
        for (const c of clients) c.write(frame);
    };
    const onChange = ({ version, reason }) => broadcast("changed", { version, reason });
    hub.on("change", onChange);
    const pinger = setInterval(() => broadcast("ping", { t: Date.now() }), pingMs);
    pinger.unref?.();

    async function staticFile(name) {
        if (!staticCache.has(name)) staticCache.set(name, await fsp.readFile(path.join(uiDir, name), "utf8"));
        return staticCache.get(name);
    }

    async function allSpans({ withAspire }) {
        const local = hub.model?.spans ?? [];
        if (!withAspire || !aspire) return { spans: local, aspire: { included: false } };
        const data = await aspire.traces().catch(() => null);
        if (!data) return { spans: local, aspire: { included: false, error: aspire.lastError ?? "unavailable" } };
        const remote = flattenOtlp(data, "aspire");
        return { spans: mergeSpans(local, remote), aspire: { included: true, spans: remote.length } };
    }

    async function readBody(req) {
        const ct = String(req.headers["content-type"] ?? "");
        if (!ct.toLowerCase().startsWith("application/json")) throw Object.assign(new Error("expected application/json"), { status: 415 });
        let size = 0;
        const chunks = [];
        for await (const chunk of req) {
            size += chunk.length;
            if (size > BODY_MAX) throw Object.assign(new Error("body too large"), { status: 413 });
            chunks.push(chunk);
        }
        const text = Buffer.concat(chunks).toString("utf8");
        if (!text.trim()) return {};
        try {
            const v = JSON.parse(text);
            return v && typeof v === "object" && !Array.isArray(v) ? v : {};
        } catch {
            throw Object.assign(new Error("invalid JSON"), { status: 400 });
        }
    }

    function setUi(partial, { broadcastUi = true } = {}) {
        const next = normalizeUiState(partial);
        if (next.traceId && !next.view) next.view = "traces";
        if (next.experimentId && !next.view) next.view = "experiment";
        ui = { ...ui, ...next, seq: ui.seq + 1 };
        if (broadcastUi) broadcast("ui", ui);
        return ui;
    }

    async function handleGet(req, res, url) {
        const p = url.pathname;
        if (STATIC[p]) {
            const [file, type] = STATIC[p];
            let body = await staticFile(file);
            if (file === "index.html") body = body.replace('content="__CI_TOKEN__"', `content="${token}"`);
            res.writeHead(200, securityHeaders({ "Content-Type": type }));
            return res.end(body);
        }
        if (p === "/events") {
            res.writeHead(200, securityHeaders({ "Content-Type": "text/event-stream; charset=utf-8", Connection: "keep-alive", "X-Accel-Buffering": "no" }));
            res.write(`retry: 2000\nevent: hello\ndata: ${JSON.stringify({ ui, version: hub.version })}\n\n`);
            clients.add(res);
            req.on("close", () => clients.delete(res));
            return;
        }
        if (p === "/favicon.ico") return send(res, 204, "");
        if (!p.startsWith("/api/")) return fail(res, 404, "not_found", "not found");
        const m = hub.model;
        if (!m) return fail(res, 503, "not_ready", "data hub is still loading");
        const meta = { version: hub.version, generatedAt: m.generatedAt, heartbeatSec: m.heartbeatSec };
        const route = p.slice(5);
        if (route === "summary") {
            return send(res, 200, {
                ...meta,
                ui,
                summary: summarize(m),
                campaigns: m.campaigns,
                roots: m.roots,
                imports: m.imports,
                rollouts: { total: m.rollouts.total, byStatus: m.rollouts.byStatus, meanScore: m.rollouts.meanScore },
                warnings: m.warnings.slice(0, 20),
            });
        }
        if (route === "experiments") {
            const cid = url.searchParams.get("campaign");
            const list = cid ? m.experiments.filter((e) => e.campaignId === cid) : m.experiments;
            return send(res, 200, { ...meta, campaigns: m.campaigns, experiments: list });
        }
        if (route.startsWith("experiment/")) {
            const id = decodeURIComponent(route.slice(11));
            if (!isSafeId(id)) return fail(res, 400, "bad_id", "invalid experiment id");
            const d = experimentDetail(m, id);
            return d ? send(res, 200, { ...meta, ...d }) : fail(res, 404, "not_found", "experiment not found");
        }
        if (route === "live") return send(res, 200, { ...meta, live: m.live, rollouts: m.rollouts });
        if (route === "traces") {
            const withAspire = url.searchParams.get("aspire") !== "0";
            const { spans, aspire: a } = await allSpans({ withAspire });
            return send(res, 200, { ...meta, traces: traceList(spans), aspire: a, rejectedSpans: m.rejectedSpans, spanFiles: m.spanFiles, imports: m.imports });
        }
        if (route.startsWith("trace/")) {
            const id = decodeURIComponent(route.slice(6));
            if (!isTraceId(id)) return fail(res, 400, "bad_id", "invalid trace id (expected 32 hex chars)");
            let tree = buildSpanTree(m.spans, id);
            let origin = "jsonl";
            if (aspire) {
                const data = await aspire.trace(id).catch(() => null);
                if (data) {
                    const merged = mergeSpans(m.spans.filter((s) => s.traceId === id.toLowerCase()), flattenOtlp(data, "aspire"));
                    tree = buildSpanTree(merged, id) ?? tree;
                    origin = tree ? "merged" : origin;
                }
            }
            if (!tree) return fail(res, 404, "not_found", "trace not found");
            const link = aspire ? await aspire.deepLink(id) : null;
            return send(res, 200, { ...meta, trace: tree, origin, aspireUrl: link });
        }
        if (route === "evals") return send(res, 200, { ...meta, evals: m.evals });
        if (route === "sleep") return send(res, 200, { ...meta, sleep: m.sleep, holdoutLooks: m.holdoutLooks });
        if (route === "aspire") {
            const status = aspire ? await aspire.status() : { configured: false, reachable: false };
            return send(res, 200, { ...meta, aspire: stripSecrets(status) });
        }
        return fail(res, 404, "not_found", "unknown endpoint");
    }

    async function handlePost(req, res, url) {
        // CSRF / cross-origin guards: same-origin fetch only, plus the per-instance token.
        if (!originAllowed(req.headers.origin, port)) return fail(res, 403, "bad_origin", "cross-origin request rejected");
        if (req.headers["sec-fetch-site"] && !["same-origin", "none"].includes(req.headers["sec-fetch-site"])) {
            return fail(res, 403, "bad_origin", "cross-site request rejected");
        }
        if (!tokenEquals(String(req.headers[TOKEN_HEADER] ?? ""), token)) return fail(res, 403, "bad_token", "missing or invalid token");
        const body = await readBody(req);
        const p = url.pathname;
        if (p === "/api/ui") {
            try {
                return send(res, 200, { ui: setUi(body, { broadcastUi: false }) });
            } catch (e) {
                return fail(res, 400, "bad_request", e.message);
            }
        }
        if (p === "/api/refresh") {
            await hub.refresh("manual");
            return send(res, 200, { version: hub.version });
        }
        if (p === "/api/aspire/login") {
            const traceId = body.traceId === undefined || body.traceId === null ? null : String(body.traceId);
            if (traceId !== null && !isTraceId(traceId)) return fail(res, 400, "bad_id", "invalid trace id");
            const u = aspire ? await aspire.loginUrl(traceId) : null;
            if (!u) return fail(res, 404, "not_configured", "Aspire Dashboard is not running; run `ci-lab dashboard up`");
            log("Aspire login URL handed to the canvas on user request", "info");
            return send(res, 200, { url: u });
        }
        return fail(res, 404, "not_found", "unknown endpoint");
    }

    const server = http.createServer(async (req, res) => {
        try {
            if (!hostAllowed(req.headers.host, port)) return fail(res, 421, "bad_host", "unexpected Host header");
            const url = new URL(req.url ?? "/", `http://127.0.0.1:${port}`);
            if (req.method === "GET") return await handleGet(req, res, url);
            if (req.method === "POST") return await handlePost(req, res, url);
            return send(res, 405, { error: { code: "method_not_allowed", message: "method not allowed" } }, { Allow: "GET, POST" });
        } catch (e) {
            if (e?.status) return fail(res, e.status, "bad_request", e.message);
            log(`canvas server error: ${e?.message ?? e}`, "error");
            if (!res.headersSent) fail(res, 500, "internal", "internal error");
            else res.end();
        }
    });
    server.keepAliveTimeout = 5000;
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
    });
    port = server.address().port;

    return {
        port,
        url: `http://127.0.0.1:${port}/`,
        instanceId: opts.instanceId ?? null,
        token,
        get ui() {
            return ui;
        },
        /** Merge + push UI state to every connected iframe (used by canvas actions). */
        pushUi(partial) {
            return setUi(partial, { broadcastUi: true });
        },
        clientCount: () => clients.size,
        async close() {
            hub.off("change", onChange);
            clearInterval(pinger);
            for (const c of clients) c.end();
            clients.clear();
            await new Promise((r) => {
                server.close(() => r());
                server.closeAllConnections?.();
            });
        },
    };
}
