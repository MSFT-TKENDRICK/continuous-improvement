// Aspire Dashboard client (design §12.3/§12.4 C30). Reads the `ci-lab dashboard up` state file and
// queries the dashboard's `/api/telemetry/*` endpoints with the `x-api-key` header.
// Secrets (api_key, browser_token, otlp_key) never leave this module except `loginUrl()`, which the
// server hands to the iframe only in response to an explicit user click.
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { isLoopbackUrl, isTraceId } from "./security.mjs";

const STATE_MAX_BYTES = 64 * 1024;
const RESPONSE_MAX_BYTES = 16 * 1024 * 1024;

/** Same env var / default as `ci_lab.contracts.DASHBOARD_STATE_ENV`. */
export function statePath(env = process.env, homeDir = os.homedir()) {
    return env.CI_DASHBOARD_STATE ? path.resolve(env.CI_DASHBOARD_STATE) : path.join(homeDir, ".ci-lab", "dashboard.json");
}

function pidAlive(pid) {
    if (!Number.isInteger(pid) || pid <= 0) return null;
    try {
        process.kill(pid, 0);
        return true;
    } catch (e) {
        return e?.code === "EPERM";
    }
}

function baseUrl(u) {
    try {
        const x = new URL(u);
        return `${x.protocol}//${x.host}`;
    } catch {
        return null;
    }
}

export class AspireClient {
    /**
     * @param {{env?: object, homeDir?: string, fetchImpl?: typeof fetch, timeoutMs?: number, cacheMs?: number, now?: () => number}} [opts]
     */
    constructor(opts = {}) {
        this.env = opts.env ?? process.env;
        this.homeDir = opts.homeDir ?? os.homedir();
        this.fetch = opts.fetchImpl ?? globalThis.fetch;
        this.timeoutMs = opts.timeoutMs ?? 3000;
        this.cacheMs = opts.cacheMs ?? 5000;
        this.now = opts.now ?? Date.now;
        this.cache = new Map();
        this.lastError = null;
    }

    get statePath() {
        return statePath(this.env, this.homeDir);
    }

    /** Raw state (contains secrets — internal use only) or null. */
    async readState() {
        try {
            const p = this.statePath;
            const st = await fsp.lstat(p);
            if (!st.isFile() || st.isSymbolicLink() || st.size > STATE_MAX_BYTES) return null;
            const data = JSON.parse((await fsp.readFile(p, "utf8")).replace(/^\uFEFF/, ""));
            if (!data || typeof data !== "object" || Array.isArray(data)) return null;
            const apiUrl = typeof data.api_url === "string" ? data.api_url : typeof data.ui_url === "string" ? data.ui_url : null;
            return {
                pid: Number.isInteger(data.pid) ? data.pid : null,
                version: typeof data.version === "string" ? data.version : null,
                started: typeof data.started === "string" ? data.started : null,
                uiUrl: typeof data.ui_url === "string" && isLoopbackUrl(data.ui_url) ? baseUrl(data.ui_url) : null,
                apiUrl: apiUrl && isLoopbackUrl(apiUrl) ? baseUrl(apiUrl) : null,
                otlpUrl: typeof data.otlp_url === "string" && isLoopbackUrl(data.otlp_url) ? baseUrl(data.otlp_url) : null,
                nonLoopback: [data.ui_url, data.api_url].some((u) => typeof u === "string" && !isLoopbackUrl(u)),
                apiKey: typeof data.api_key === "string" ? data.api_key : null,
                browserToken: typeof data.browser_token === "string" ? data.browser_token : null,
            };
        } catch {
            return null;
        }
    }

    async get(pathname, { cache = true } = {}) {
        const st = await this.readState();
        if (!st?.apiUrl || !st.apiKey) return { ok: false, status: 0, error: st ? "dashboard API not configured" : "no dashboard state" };
        const key = `${st.apiUrl}${pathname}`;
        const hit = cache ? this.cache.get(key) : null;
        if (hit && this.now() - hit.at < this.cacheMs) return hit.value;
        let value;
        try {
            const res = await this.fetch(key, {
                headers: { "x-api-key": st.apiKey, accept: "application/json" },
                redirect: "manual",
                signal: AbortSignal.timeout(this.timeoutMs),
            });
            const len = Number(res.headers.get("content-length") ?? 0);
            if (len > RESPONSE_MAX_BYTES) value = { ok: false, status: res.status, error: "response too large" };
            else if (res.status === 404) value = { ok: false, status: 404, error: "not found" };
            else if (res.status === 401 || res.status === 403) value = { ok: false, status: res.status, error: "API key rejected" };
            else if (!res.ok) value = { ok: false, status: res.status, error: `HTTP ${res.status}` };
            else {
                const text = await res.text();
                if (text.length > RESPONSE_MAX_BYTES) value = { ok: false, status: res.status, error: "response too large" };
                else value = { ok: true, status: res.status, data: JSON.parse(text) };
            }
        } catch (e) {
            // Our own wording only: never echo request details (headers carry the key).
            const name = e?.name === "TimeoutError" || e?.name === "AbortError" ? "timeout" : e instanceof SyntaxError ? "invalid JSON" : "unreachable";
            value = { ok: false, status: 0, error: name };
        }
        this.lastError = value.ok ? null : value.error;
        this.cache.set(key, { at: this.now(), value });
        if (this.cache.size > 64) this.cache.delete(this.cache.keys().next().value);
        return value;
    }

    /** Sanitized status for the iframe / `dashboard_status` (no secrets, no token-bearing URLs). */
    async status() {
        const st = await this.readState();
        if (!st) return { configured: false, reachable: false, hint: "run `ci-lab dashboard up` to start the Aspire Dashboard" };
        const probe = st.apiUrl && st.apiKey ? await this.get("/api/telemetry/resources") : { ok: false, error: "dashboard API not configured" };
        const resources = !probe.ok ? [] : Array.isArray(probe.data) ? probe.data : Array.isArray(probe.data?.data) ? probe.data.data : [];
        return {
            configured: true,
            reachable: probe.ok,
            version: st.version,
            started: st.started,
            uiUrl: st.uiUrl,
            otlpUrl: st.otlpUrl,
            pidAlive: pidAlive(st.pid),
            loginAvailable: !!(st.uiUrl && st.browserToken),
            nonLoopbackIgnored: st.nonLoopback || undefined,
            resources: resources
                .filter((r) => r && typeof r === "object")
                .slice(0, 50)
                .map((r) => ({ name: String(r.name ?? ""), displayName: String(r.displayName ?? r.name ?? ""), hasTraces: r.hasTraces === true })),
            error: probe.ok ? null : probe.error,
        };
    }

    /** OTLP-JSON `{data:{resourceSpans}}` of recent traces, or null. */
    async traces() {
        const r = await this.get("/api/telemetry/traces");
        return r.ok ? r.data : null;
    }

    async trace(traceId) {
        if (!isTraceId(traceId)) return null;
        const r = await this.get(`/api/telemetry/traces/${traceId.toLowerCase()}`);
        return r.ok ? r.data : null;
    }

    /** Token-free deep link (opening it without a session cookie redirects to the login page). */
    async deepLink(traceId) {
        const st = await this.readState();
        if (!st?.uiUrl) return null;
        return isTraceId(traceId) ? `${st.uiUrl}/traces/detail/${traceId.toLowerCase()}` : `${st.uiUrl}/`;
    }

    /** Token-bearing login URL; only for a user-initiated click (server POST /api/aspire/login). */
    async loginUrl(traceId) {
        const st = await this.readState();
        if (!st?.uiUrl || !st.browserToken) return null;
        const u = new URL("/login", st.uiUrl);
        u.searchParams.set("t", st.browserToken);
        if (isTraceId(traceId)) u.searchParams.set("returnUrl", `/traces/detail/${traceId.toLowerCase()}`);
        return u.toString();
    }
}
