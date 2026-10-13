// Experiment-chat backend manager: lazily runs `ci-lab chat serve` (docs/chat.md, docs/canvas.md
// "Experiment chat") and hands the canvas proxy its loopback endpoint + bearer token.
// One process per repo root, shared by every open canvas instance (ref-counted), restarted with
// backoff on crash, killed when the last instance closes or the extension exits. Stdin stays a
// pipe so the child also exits on EOF if this process dies without running exit hooks.
import { spawn as nodeSpawn } from "node:child_process";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { newToken } from "./security.mjs";

export const CHAT_PROFILES = ["copilot", "fake"];
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1"]);
const DEFAULTS = Object.freeze({
    listenTimeoutMs: 120_000,
    stopGraceMs: 3_000,
    backoffBaseMs: 1_000,
    backoffMaxMs: 30_000,
    maxRestarts: 5,
    stableMs: 60_000,
    stdoutMax: 64 * 1024,
    logBurst: 20,
    logWindowMs: 10_000,
    logLineMax: 500,
});

export class ChatUnavailable extends Error {
    constructor(code, message, retryAfterMs = null) {
        super(message);
        this.code = code;
        this.status = 503;
        this.retryAfterMs = retryAfterMs;
    }
}

/** Resolve argv: CI_CHAT_COMMAND (JSON array) or the default `uv run ... ci-lab chat serve`. */
export function chatCommand(env, runDir) {
    if (env.CI_CHAT_COMMAND !== undefined && env.CI_CHAT_COMMAND !== "") {
        let argv;
        try {
            argv = JSON.parse(env.CI_CHAT_COMMAND);
        } catch {
            argv = null;
        }
        if (!Array.isArray(argv) || !argv.length || !argv.every((a) => typeof a === "string" && a.length > 0 && !a.includes("\0"))) {
            return { error: "CI_CHAT_COMMAND must be a JSON array of non-empty strings" };
        }
        return { argv, custom: true };
    }
    const profile = env.CI_CHAT_PROFILE || "copilot";
    if (!CHAT_PROFILES.includes(profile)) return { error: `CI_CHAT_PROFILE must be one of: ${CHAT_PROFILES.join(", ")}` };
    return { argv: ["uv", "run", "--no-sync", "ci-lab", "chat", "serve", "--profile", profile, "--run-dir", runDir], custom: false, profile };
}

/** Run root the dashboard hub watches first (sources.mjs: $CI_RUN_DIR, else artifacts/ci-runs). */
export function chatRunDir(repoRoot, env) {
    return env.CI_RUN_DIR ? path.resolve(repoRoot, env.CI_RUN_DIR) : path.join(repoRoot, "artifacts", "ci-runs");
}

/**
 * Windows: `shell:false` cannot run .cmd/.bat shims, so bare names resolve to a `.exe` on PATH.
 * Elsewhere spawn() searches PATH itself. Returns null when nothing executable is found.
 */
export function resolveExecutable(name, { env = process.env, platform = process.platform, exists = fs.existsSync } = {}) {
    if (platform !== "win32") return name;
    if (/[\\/]/.test(name)) return exists(name) ? name : null;
    const exe = /\.exe$/i.test(name) ? name : `${name}.exe`;
    const pathVar = env.PATH ?? env.Path ?? "";
    for (const dir of pathVar.split(";")) {
        if (!dir) continue;
        const candidate = path.join(dir.replace(/^"|"$/g, ""), exe);
        if (exists(candidate)) return candidate;
    }
    return null;
}

/** Parse the contract's single stdout line; throws on anything else. */
export function parseListening(line) {
    let v;
    try {
        v = JSON.parse(line);
    } catch {
        throw new Error("chat server wrote a non-JSON stdout line");
    }
    if (!v || v.event !== "listening") throw new Error("chat server stdout line is not a listening event");
    const host = String(v.host ?? "");
    const port = v.port;
    if (!LOOPBACK.has(host)) throw new Error("chat server is not listening on loopback");
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("chat server reported an invalid port");
    const p = typeof v.path === "string" && /^\/[A-Za-z0-9/_-]{0,64}$/.test(v.path) ? v.path : "/agui";
    return { host: host === "localhost" ? "127.0.0.1" : host, port, path: p };
}

export class ChatBackend extends EventEmitter {
    /**
     * @param {{repoRoot: string, env?: object, log?: Function, spawn?: Function, platform?: string,
     *          exists?: Function, options?: Partial<typeof DEFAULTS>}} opts
     */
    constructor(opts) {
        super();
        this.repoRoot = path.resolve(opts.repoRoot);
        this.env = opts.env ?? process.env;
        this.log = opts.log ?? (() => {});
        this.spawnImpl = opts.spawn ?? nodeSpawn;
        this.platform = opts.platform ?? process.platform;
        this.exists = opts.exists ?? fs.existsSync;
        this.o = { ...DEFAULTS, ...(opts.options ?? {}) };
        this.refs = 0;
        this.state = this.env.CI_CHAT_DISABLED === "1" ? "disabled" : "stopped";
        this.child = null;
        this.endpoint = null;
        this.token = null;
        this.starting = null;
        this.failures = 0;
        this.restarts = 0;
        this.lastError = this.state === "disabled" ? "disabled by CI_CHAT_DISABLED=1" : null;
        this.nextAttemptAt = 0;
        this.retryTimer = null;
        this.readySince = null;
        this.logTokens = this.o.logBurst;
        this.logWindowStart = 0;
        this.suppressed = 0;
    }

    /** Public, secret-free status for the UI and the `chat_status` action. */
    status() {
        const cmd = this.state === "disabled" ? null : chatCommand(this.env, chatRunDir(this.repoRoot, this.env));
        const retryInMs = this.state === "crashed" && this.nextAttemptAt > Date.now() ? this.nextAttemptAt - Date.now() : null;
        return {
            state: this.state,
            pid: this.child?.pid ?? null,
            port: this.state === "ready" ? (this.endpoint?.port ?? null) : null,
            profile: cmd?.custom ? "custom" : (cmd?.profile ?? null),
            program: cmd?.argv ? path.basename(cmd.argv[0]) : null,
            runDir: chatRunDir(this.repoRoot, this.env),
            restarts: this.restarts,
            lastError: this.lastError,
            retryInMs,
            readySince: this.readySince,
            viewers: this.refs,
        };
    }

    /** Start (if needed) and resolve to `{host, port, path, token}` for the proxy; throws ChatUnavailable. */
    async ensure() {
        if (this.state === "disabled") throw new ChatUnavailable("chat_disabled", "experiment chat is disabled (CI_CHAT_DISABLED=1)");
        if (this.state === "ready" && this.child && this.endpoint) return { ...this.endpoint, token: this.token };
        if (this.starting) return this.starting;
        const wait = this.nextAttemptAt - Date.now();
        if (this.state === "crashed" && wait > 0) {
            throw new ChatUnavailable("chat_restarting", `experiment chat backend failed (${this.lastError ?? "unknown error"}); retrying in ${Math.ceil(wait / 1000)}s`, wait);
        }
        return this.start();
    }

    start() {
        clearTimeout(this.retryTimer);
        this.retryTimer = null;
        const runDir = chatRunDir(this.repoRoot, this.env);
        const cmd = chatCommand(this.env, runDir);
        if (cmd.error) return Promise.reject(this.failStart("chat_config", cmd.error, { permanent: true }));
        const program = resolveExecutable(cmd.argv[0], { env: this.env, platform: this.platform, exists: this.exists });
        if (!program) {
            return Promise.reject(this.failStart("chat_command_not_found", `cannot find ${path.basename(cmd.argv[0])} on PATH (install uv, or set CI_CHAT_COMMAND)`));
        }
        const token = newToken(24);
        const childEnv = { ...this.env, CI_CHAT_TOKEN: token, CI_RUN_DIR: runDir, PYTHONUNBUFFERED: "1" };
        delete childEnv.CI_CHAT_COMMAND;
        this.state = "starting";
        this.token = token;
        this.endpoint = null;
        let child;
        try {
            child = this.spawnImpl(program, cmd.argv.slice(1), { cwd: this.repoRoot, env: childEnv, stdio: ["pipe", "pipe", "pipe"], shell: false, windowsHide: true });
        } catch (e) {
            return Promise.reject(this.failStart("chat_spawn_failed", `could not start the chat backend: ${this.scrub(e?.message ?? e)}`));
        }
        this.child = child;
        this.emit("status", this.status());
        this.starting = new Promise((resolve, reject) => {
            let settled = false;
            let buf = "";
            let lastStderr = "";
            const settle = (err, value) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                this.starting = null;
                if (err) reject(err);
                else resolve(value);
            };
            const timer = setTimeout(() => {
                this.lastError = `no listening line within ${Math.round(this.o.listenTimeoutMs / 1000)}s`;
                settle(new ChatUnavailable("chat_start_timeout", `experiment chat backend did not start: ${this.lastError}`));
                this.kill(child);
            }, this.o.listenTimeoutMs);
            timer.unref?.();
            child.stdout?.setEncoding?.("utf8");
            child.stdout?.on("data", (chunk) => {
                if (this.endpoint && this.child === child) {
                    this.logLine("chat stdout", String(chunk));
                    return;
                }
                buf += String(chunk);
                const nl = buf.indexOf("\n");
                if (nl < 0) {
                    if (buf.length > this.o.stdoutMax) {
                        this.lastError = "oversized stdout before the listening line";
                        settle(new ChatUnavailable("chat_bad_listening", `experiment chat backend did not start: ${this.lastError}`));
                        this.kill(child);
                    }
                    return;
                }
                const line = buf.slice(0, nl).trim();
                const rest = buf.slice(nl + 1);
                buf = "";
                try {
                    this.endpoint = parseListening(line);
                } catch (e) {
                    this.lastError = e.message;
                    settle(new ChatUnavailable("chat_bad_listening", `experiment chat backend did not start: ${e.message}`));
                    this.kill(child);
                    return;
                }
                if (rest.trim()) this.logLine("chat stdout", rest);
                this.state = "ready";
                this.readySince = Date.now();
                this.lastError = null;
                this.log(`ci-harness-dashboard: experiment chat ready on 127.0.0.1:${this.endpoint.port} (pid ${child.pid ?? "?"})`, "info");
                this.emit("status", this.status());
                settle(null, { ...this.endpoint, token });
            });
            child.stderr?.setEncoding?.("utf8");
            child.stderr?.on("data", (chunk) => {
                const text = String(chunk);
                const lines = text.split(/\r?\n/).filter((l) => l.trim());
                if (lines.length) lastStderr = lines[lines.length - 1];
                this.logLine("chat", text);
            });
            child.stdin?.on?.("error", () => {});
            child.on("error", (e) => {
                const code = e?.code === "ENOENT" ? "chat_command_not_found" : "chat_spawn_failed";
                const msg = e?.code === "ENOENT" ? `cannot find ${path.basename(cmd.argv[0])} (install uv, or set CI_CHAT_COMMAND)` : `chat backend error: ${this.scrub(e?.message ?? e)}`;
                this.lastError = msg;
                settle(new ChatUnavailable(code, msg));
                if (this.child === child) this.onExit(child, null, null, msg);
            });
            child.on("exit", (code, signal) => {
                const why = `exited with ${signal ? `signal ${signal}` : `code ${code}`}${lastStderr ? `: ${this.scrub(errorText(lastStderr)).slice(0, 200)}` : ""}`;
                settle(new ChatUnavailable("chat_exited", `experiment chat backend ${why}`));
                this.onExit(child, code, signal, why);
            });
        });
        this.starting.catch(() => {});
        return this.starting;
    }

    failStart(code, message, { permanent = false } = {}) {
        this.state = "crashed";
        this.lastError = message;
        this.failures++;
        this.nextAttemptAt = permanent ? Date.now() + this.o.backoffMaxMs : Date.now() + this.backoffDelay();
        this.emit("status", this.status());
        return new ChatUnavailable(code, message);
    }

    backoffDelay() {
        return Math.min(this.o.backoffMaxMs, this.o.backoffBaseMs * 2 ** Math.max(0, this.failures - 1));
    }

    onExit(child, code, signal, why) {
        if (this.child !== child) return;
        const wasStopping = child.__ciStopping === true;
        const stable = this.readySince !== null && Date.now() - this.readySince >= this.o.stableMs;
        this.child = null;
        this.endpoint = null;
        this.token = null;
        this.readySince = null;
        if (wasStopping) {
            this.state = this.state === "disabled" ? "disabled" : "stopped";
            this.emit("status", this.status());
            return;
        }
        if (stable) this.failures = 0;
        this.failures++;
        this.state = "crashed";
        this.lastError = why;
        // Exit code 2 is the server's "bad startup input" contract: retrying will not help soon.
        const permanent = code === 2;
        const delay = permanent ? this.o.backoffMaxMs : this.backoffDelay();
        this.nextAttemptAt = Date.now() + delay;
        this.log(`ci-harness-dashboard: experiment chat backend ${why}`, "warning");
        if (this.refs > 0 && !permanent && this.failures <= this.o.maxRestarts) {
            this.retryTimer = setTimeout(() => {
                this.retryTimer = null;
                if (this.refs <= 0 || this.state !== "crashed") return;
                this.restarts++;
                this.start().catch(() => {});
            }, delay);
            this.retryTimer.unref?.();
        }
        this.emit("status", this.status());
    }

    kill(child, { force = false } = {}) {
        if (!child) return;
        child.__ciStopping = child.__ciStopping || force;
        try {
            child.stdin?.end?.();
        } catch {
            /* already closed */
        }
        try {
            child.kill?.();
        } catch {
            /* already gone */
        }
    }

    /** Graceful stop: close stdin (contract: exit on EOF), then terminate after a grace period. */
    async stop() {
        clearTimeout(this.retryTimer);
        this.retryTimer = null;
        const child = this.child;
        if (!child) {
            if (this.state !== "disabled") this.state = "stopped";
            return;
        }
        child.__ciStopping = true;
        const exited = new Promise((r) => {
            if (child.exitCode !== null && child.exitCode !== undefined) return r();
            child.once("exit", () => r());
        });
        try {
            child.stdin?.end?.();
        } catch {
            /* ignore */
        }
        const t = setTimeout(() => this.kill(child, { force: true }), this.o.stopGraceMs);
        t.unref?.();
        await Promise.race([exited, new Promise((r) => setTimeout(r, this.o.stopGraceMs + 2000).unref?.())]);
        clearTimeout(t);
        if (this.child === child) this.onExit(child, null, null, "stopped");
    }

    /** Synchronous last-resort kill (process exit hook). */
    killSync() {
        clearTimeout(this.retryTimer);
        if (this.child) this.kill(this.child, { force: true });
    }

    scrub(text) {
        let s = String(text);
        if (this.token) s = s.split(this.token).join("***");
        return s.replace(/(x-ci-chat-token|CI_CHAT_TOKEN)(["'\s:=]+)[A-Za-z0-9_-]{8,}/gi, "$1$2***");
    }

    /** Rate-limited, token-scrubbed logging of child output (never stdout: that is the RPC channel). */
    logLine(prefix, text) {
        const now = Date.now();
        if (now - this.logWindowStart >= this.o.logWindowMs) {
            if (this.suppressed) this.log(`${prefix}: … ${this.suppressed} line(s) suppressed`, "info");
            this.logWindowStart = now;
            this.logTokens = this.o.logBurst;
            this.suppressed = 0;
        }
        for (const raw of String(text).split(/\r?\n/)) {
            const line = raw.trim();
            if (!line) continue;
            if (this.logTokens <= 0) {
                this.suppressed++;
                continue;
            }
            this.logTokens--;
            const msg = this.scrub(line);
            this.log(`${prefix}: ${msg.length > this.o.logLineMax ? `${msg.slice(0, this.o.logLineMax)}…` : msg}`, /\b(ERROR|CRITICAL|Traceback)\b/.test(line) ? "warning" : "info");
        }
    }
}

function errorText(line) {
    try {
        const v = JSON.parse(line);
        if (v && typeof v.error === "string") return v.error;
    } catch {
        /* plain text */
    }
    return line;
}

// ---------------------------------------------------------------- shared pool (one per repo root)
const pool = new Map();
let exitHooked = false;

const keyOf = (root) => {
    const k = path.resolve(root);
    return process.platform === "win32" ? k.toLowerCase() : k;
};

/** Shared backend for `repoRoot` (no process is spawned until `ensure()`); pair with `releaseChat`. */
export function acquireChat(repoRoot, opts = {}) {
    const key = keyOf(repoRoot);
    let backend = pool.get(key);
    if (!backend) {
        backend = new ChatBackend({ ...opts, repoRoot });
        pool.set(key, backend);
    }
    backend.refs++;
    if (!exitHooked) {
        exitHooked = true;
        process.once("exit", () => {
            for (const b of pool.values()) b.killSync();
        });
    }
    return backend;
}

/** Drop one reference; the last one stops the child process. */
export async function releaseChat(backend) {
    if (!backend) return;
    backend.refs = Math.max(0, backend.refs - 1);
    if (backend.refs > 0) return;
    for (const [k, b] of pool) if (b === backend) pool.delete(k);
    await backend.stop();
}

export function activeChats() {
    return pool.size;
}
