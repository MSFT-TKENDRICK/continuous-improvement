// Security primitives shared by the server, sources and model (design §12.4 C29–C31, C35).
import { randomBytes, timingSafeEqual } from "node:crypto";
import path from "node:path";

export const CSP =
    "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; " +
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors *";

export function securityHeaders(extra = {}) {
    return {
        "Content-Security-Policy": CSP,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "Cross-Origin-Opener-Policy": "same-origin",
        ...extra,
    };
}

/** C31: only exact loopback Host headers for our own port are accepted. */
export function hostAllowed(host, port) {
    if (typeof host !== "string" || !port) return false;
    const h = host.trim().toLowerCase();
    return h === `127.0.0.1:${port}` || h === `localhost:${port}`;
}

export function originAllowed(origin, port) {
    if (origin === undefined) return true;
    if (typeof origin !== "string") return false;
    const o = origin.trim().toLowerCase();
    return o === `http://127.0.0.1:${port}` || o === `http://localhost:${port}`;
}

export function newToken(bytes = 24) {
    return randomBytes(bytes).toString("base64url");
}

export function tokenEquals(a, b) {
    if (typeof a !== "string" || typeof b !== "string" || !a || !b) return false;
    const ba = Buffer.from(a);
    const bb = Buffer.from(b);
    return ba.length === bb.length && timingSafeEqual(ba, bb);
}

/** True when `child` is `root` or lies beneath it (both should already be realpaths). */
export function isWithin(root, child) {
    const rel = path.relative(root, child);
    if (rel === "") return true;
    return !rel.startsWith("..") && !path.isAbsolute(rel);
}

export function isLoopbackUrl(value) {
    try {
        const u = new URL(value);
        if (u.protocol !== "http:" && u.protocol !== "https:") return false;
        return ["127.0.0.1", "localhost", "[::1]"].includes(u.hostname.toLowerCase());
    } catch {
        return false;
    }
}

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/;
const TRACE_RE = /^[0-9a-fA-F]{32}$/;
export const isSafeId = (s) => typeof s === "string" && ID_RE.test(s);
export const isTraceId = (s) => typeof s === "string" && TRACE_RE.test(s);

// C29: GenAI content attributes (prompts, completions, tool payloads) are hidden unless the
// span or its resource says sensitive mode was explicitly enabled.
const SENSITIVE_KEYS = new Set([
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.prompt",
    "gen_ai.completion",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.tool.definitions",
]);
export function isSensitiveAttr(key) {
    if (typeof key !== "string") return false;
    if (SENSITIVE_KEYS.has(key)) return true;
    if (key.startsWith("gen_ai.")) {
        return /(^|\.)(content|messages|system_instructions|arguments|result|prompt|completion)(\.|$)/.test(key.slice(7));
    }
    return /\.content$/.test(key) || /(^|\.)(input|output)\.messages$/.test(key);
}

const TRUE = new Set([true, "true", "1", 1, "on", "yes"]);
export const SENSITIVE_FLAGS = ["ci.sensitive", "ci.telemetry.sensitive", "gen_ai.sensitive_data"];
export function sensitiveEnabled(...attrMaps) {
    for (const m of attrMaps) {
        if (!m || typeof m !== "object") continue;
        for (const k of SENSITIVE_FLAGS) if (TRUE.has(m[k])) return true;
    }
    return false;
}

const SECRET_KEY_RE = /(token|api[_-]?key|otlp[_-]?key|secret|password|authorization|cookie)/i;
/** Deep-copies JSON-ish data dropping any key that looks secret (defence in depth for C30). */
export function stripSecrets(value, depth = 0) {
    if (depth > 12 || value === null || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map((v) => stripSecrets(v, depth + 1));
    const out = {};
    for (const [k, v] of Object.entries(value)) {
        if (SECRET_KEY_RE.test(k)) continue;
        out[k] = stripSecrets(v, depth + 1);
    }
    return out;
}
