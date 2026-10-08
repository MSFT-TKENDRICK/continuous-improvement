// One data hub per repo root, shared by every canvas instance (design §12.2): watches the known
// directories (fs.watch, 250 ms debounce) with a 5 s poll fallback, rebuilds the model and fans
// out change notifications. Rebuilds are cheap thanks to the (path, mtime, size) read cache.
import { EventEmitter } from "node:events";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { Sources } from "./sources.mjs";
import { buildModel, DEFAULT_HEARTBEAT_SEC, summarize } from "./model.mjs";

const hubs = new Map();

function fingerprint(model) {
    const live = model.live.map(({ ageSec, ...rest }) => rest);
    const h = createHash("sha1");
    h.update(
        JSON.stringify([model.campaigns, model.experiments, live, model.traces, model.evals, model.sleep, model.rollouts, model.imports, model.warnings, model.rejectedSpans]),
    );
    return h.digest("hex");
}

export class Hub extends EventEmitter {
    /**
     * @param {string} repoRoot
     * @param {{env?: object, homeDir?: string, heartbeatSec?: number, debounceMs?: number, pollMs?: number,
     *          log?: (msg: string, level?: string) => void, now?: () => number, watch?: boolean}} [opts]
     */
    constructor(repoRoot, opts = {}) {
        super();
        this.setMaxListeners(100);
        this.repoRoot = repoRoot;
        this.env = opts.env ?? process.env;
        const envHb = Number(this.env.CI_DASHBOARD_HEARTBEAT_SEC);
        this.heartbeatSec = opts.heartbeatSec ?? (Number.isFinite(envHb) && envHb > 0 ? envHb : DEFAULT_HEARTBEAT_SEC);
        this.debounceMs = opts.debounceMs ?? 250;
        this.pollMs = opts.pollMs ?? 5000;
        this.log = opts.log ?? (() => {});
        this.now = opts.now ?? (() => Date.now() / 1000);
        this.watchEnabled = opts.watch ?? true;
        this.sources = new Sources(repoRoot, { env: this.env, homeDir: opts.homeDir });
        this.watchers = new Map();
        this.model = null;
        this.version = 0;
        this.fp = null;
        this.refreshing = null;
        this.pending = false;
        this.debounce = null;
        this.poller = null;
        this.closed = false;
        this.refs = 0;
    }

    async start() {
        await this.refresh("start");
        await this.syncWatchers();
        this.poller = setInterval(() => {
            this.syncWatchers()
                .then(() => this.refresh("poll"))
                .catch((e) => this.log(`hub poll failed: ${e?.message ?? e}`, "warning"));
        }, this.pollMs);
        this.poller.unref?.();
        return this;
    }

    /** Debounced refresh triggered by watcher events. */
    poke(reason = "watch") {
        if (this.closed) return;
        clearTimeout(this.debounce);
        this.debounce = setTimeout(() => {
            this.refresh(reason).catch((e) => this.log(`hub refresh failed: ${e?.message ?? e}`, "warning"));
        }, this.debounceMs);
        this.debounce.unref?.();
    }

    /** Rebuild the model; concurrent calls coalesce into one follow-up rebuild. */
    async refresh(reason = "manual") {
        if (this.closed) return this.model;
        if (this.refreshing) {
            // Remember the strongest pending reason: a coalesced manual refresh must still bump the version.
            if (this.pending !== "manual") this.pending = reason;
            return this.refreshing;
        }
        this.refreshing = (async () => {
            try {
                let why = reason;
                for (;;) {
                    this.pending = false;
                    const scan = await this.sources.scan();
                    const model = buildModel(scan, { now: this.now(), heartbeatSec: this.heartbeatSec });
                    const fp = fingerprint(model);
                    this.model = model;
                    if (fp !== this.fp || why === "manual") {
                        this.fp = fp;
                        this.version++;
                        this.emit("change", { version: this.version, reason: why });
                    }
                    if (!this.pending || this.closed) break;
                    why = this.pending;
                }
                return this.model;
            } finally {
                this.refreshing = null;
            }
        })();
        return this.refreshing;
    }

    async syncWatchers() {
        if (!this.watchEnabled || this.closed) return;
        let dirs = [];
        try {
            dirs = await this.sources.watchDirs();
        } catch {
            return;
        }
        const want = new Set(dirs);
        for (const [d, w] of this.watchers) {
            if (!want.has(d)) {
                w.close();
                this.watchers.delete(d);
            }
        }
        for (const d of want) {
            if (this.watchers.has(d)) continue;
            try {
                await fsp.access(d);
                const w = fs.watch(d, { recursive: true, persistent: false }, () => this.poke("watch"));
                w.on("error", () => {
                    // Fall back to polling for this directory; the next poll may re-add it.
                    w.close();
                    this.watchers.delete(d);
                });
                this.watchers.set(d, w);
            } catch {
                /* poll fallback covers it */
            }
        }
    }

    summary() {
        return this.model ? summarize(this.model) : null;
    }

    status() {
        return {
            repo: path.basename(this.repoRoot),
            version: this.version,
            generatedAt: this.model?.generatedAt ?? null,
            watching: this.watchers.size,
            pollMs: this.pollMs,
            heartbeatSec: this.heartbeatSec,
            roots: this.model?.roots ?? null,
            warnings: this.model?.warnings?.length ?? 0,
        };
    }

    close() {
        this.closed = true;
        clearInterval(this.poller);
        clearTimeout(this.debounce);
        for (const w of this.watchers.values()) w.close();
        this.watchers.clear();
        this.removeAllListeners();
    }
}

/** Shared, ref-counted hub for a repo root; call `releaseHub(hub)` when done. */
export async function acquireHub(repoRoot, opts = {}) {
    let key = path.resolve(repoRoot);
    try {
        key = await fsp.realpath(key);
    } catch {
        /* keep resolved path */
    }
    if (process.platform === "win32") key = key.toLowerCase();
    let entry = hubs.get(key);
    if (!entry) {
        const hub = new Hub(repoRoot, opts);
        entry = { hub, ready: hub.start() };
        hubs.set(key, entry);
        entry.ready.catch(() => hubs.delete(key));
    }
    entry.hub.refs++;
    await entry.ready;
    return entry.hub;
}

export function releaseHub(hub) {
    if (!hub) return;
    hub.refs = Math.max(0, hub.refs - 1);
    if (hub.refs > 0) return;
    for (const [k, e] of hubs) if (e.hub === hub) hubs.delete(k);
    hub.close();
}

export function activeHubs() {
    return hubs.size;
}
