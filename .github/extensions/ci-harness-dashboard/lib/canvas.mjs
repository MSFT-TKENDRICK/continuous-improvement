// Canvas controller: instance lifecycle + agent actions, independent of the SDK so it can be tested
// with plain `node --test`. extension.mjs only wires these handlers into `createCanvas`.
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { acquireHub, releaseHub } from "./hub.mjs";
import { AspireClient } from "./aspire.mjs";
import { startInstanceServer, VIEWS, normalizeUiState } from "./server.mjs";
import { summarize, experimentDetail, buildSpanTree } from "./model.mjs";
import { stripSecrets, isSafeId, isTraceId } from "./security.mjs";
import { acquireChat, releaseChat, chatCommand, chatRunDir } from "./chat.mjs";

export const CANVAS_ID = "ci-harness-dashboard";
export const DISPLAY_NAME = "CI Harness Dashboard";
export const DESCRIPTION =
    "Live dashboard for the self-improving harness: campaigns, rounds and arms, OES experiments, ASSERT evals, sleep nights, OpenTelemetry traces (local JSONL and Aspire) and an experiment chat for drafting and approving RRSI campaign launches.";

const idSchema = { type: "string", minLength: 1, maxLength: 160, pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]*$" };
const traceSchema = { type: "string", pattern: "^[0-9a-fA-F]{32}$" };
const viewSchema = { type: "string", enum: VIEWS };

export const OPEN_INPUT_SCHEMA = {
    type: "object",
    properties: {
        repoRoot: { type: "string", description: "Repository root to monitor (defaults to the session working directory)." },
        view: { ...viewSchema, description: "Initial view." },
        campaignId: { ...idSchema, description: "Campaign to select." },
        experimentId: { ...idSchema, description: "Experiment (OES envelope / round) to focus." },
        traceId: { ...traceSchema, description: "Trace to open (32 hex chars)." },
    },
    additionalProperties: false,
};

const obj = (properties, required = []) => ({ type: "object", properties, required, additionalProperties: false });

const ACTION_DEFS = [
    { name: "refresh", description: "Re-scan the repository now and refresh every open view.", inputSchema: obj({}) },
    { name: "show_view", description: `Switch the dashboard to a view (${VIEWS.join(", ")}).`, inputSchema: obj({ view: viewSchema }, ["view"]) },
    { name: "select_campaign", description: "Show a campaign's frontier, rounds and budget in the Experiment view.", inputSchema: obj({ campaignId: idSchema }, ["campaignId"]) },
    { name: "focus_experiment", description: "Open one experiment (round, calibration, confirm or sleep night) in the Experiment view.", inputSchema: obj({ experimentId: idSchema }, ["experimentId"]) },
    { name: "focus_trace", description: "Open one trace's span tree in the Traces view.", inputSchema: obj({ traceId: traceSchema }, ["traceId"]) },
    { name: "get_summary", description: "Return a compact JSON summary of campaigns, live rounds, decisions, evals, sleep and traces.", inputSchema: obj({}) },
    { name: "dashboard_status", description: "Return dashboard health: data roots, watcher state, connected views and Aspire reachability (no secrets).", inputSchema: obj({}) },
    { name: "chat_status", description: "Return the experiment-chat backend state (starting, ready, crashed, disabled or stopped), profile, run root and last error (no secrets).", inputSchema: obj({}) },
];

/** Walk up from `start` to the nearest directory containing `.git`; falls back to `start`. */
export async function findRepoRoot(start) {
    let dir = path.resolve(start);
    for (;;) {
        try {
            await fsp.lstat(path.join(dir, ".git"));
            return dir;
        } catch {
            /* keep walking */
        }
        const up = path.dirname(dir);
        if (up === dir) return path.resolve(start);
        dir = up;
    }
}

/**
 * @param {{CanvasError: new (code: string, message: string) => Error, log?: (msg: string, level?: string) => void,
 *          env?: object, homeDir?: string, hubOptions?: object, aspire?: AspireClient, cwd?: () => string}} deps
 */
export function createDashboard(deps) {
    const { CanvasError } = deps;
    const log = deps.log ?? (() => {});
    const env = deps.env ?? process.env;
    const cwd = deps.cwd ?? (() => process.cwd());
    const aspire = deps.aspire ?? new AspireClient({ env, homeDir: deps.homeDir });
    const hubOptions = { env, homeDir: deps.homeDir, log, ...(deps.hubOptions ?? {}) };
    const chatPool = deps.chatPool ?? { acquire: acquireChat, release: releaseChat };
    /** instanceId → Promise<{server, hub, repoRoot, lastInput}> */
    const instances = new Map();

    const fail = (code, message) => {
        throw new CanvasError(code, message);
    };

    function validateOpenInput(input) {
        if (input === undefined || input === null) return {};
        if (typeof input !== "object" || Array.isArray(input)) fail("invalid_input", "input must be an object");
        const extra = Object.keys(input).filter((k) => !(k in OPEN_INPUT_SCHEMA.properties));
        if (extra.length) fail("invalid_input", `unknown input field(s): ${extra.join(", ").slice(0, 100)}`);
        if (input.repoRoot !== undefined && (typeof input.repoRoot !== "string" || !path.isAbsolute(input.repoRoot))) fail("invalid_input", "repoRoot must be an absolute path");
        try {
            normalizeUiState(input);
        } catch (e) {
            fail("invalid_input", e.message);
        }
        return input;
    }

    async function resolveRoot(input, ctx) {
        const start = input.repoRoot ?? ctx?.session?.workingDirectory ?? cwd();
        let st;
        try {
            st = await fsp.stat(start);
        } catch {
            fail("repo_not_found", "repository root does not exist");
        }
        if (!st.isDirectory()) fail("repo_not_found", "repository root is not a directory");
        // An explicit repoRoot is honoured as given; the working-directory fallback walks up to `.git`.
        return input.repoRoot ? path.resolve(start) : findRepoRoot(start);
    }

    const uiFrom = (input) => {
        const { repoRoot, ...ui } = input;
        return ui;
    };
    const sameRoot = (a, b) => (process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b);

    async function disposeEntry(entry) {
        try {
            await entry.server.close();
        } finally {
            releaseHub(entry.hub);
            await chatPool.release(entry.chat).catch((e) => log(`experiment chat stop error: ${e?.message ?? e}`, "warning"));
        }
    }

    async function createEntry(instanceId, root, input) {
        const hub = await acquireHub(root, hubOptions);
        // Shared per repo root; nothing is spawned until the chat view first talks to /agui.
        const chat = chatPool.acquire(root, { env, log });
        try {
            const server = await startInstanceServer({ hub, aspire, chat, instanceId, initialState: uiFrom(input), log, port: deps.port ?? 0 });
            log(`ci-harness-dashboard: instance ${instanceId} serving ${path.basename(root)} on ${server.url}`, "info");
            return { server, hub, chat, repoRoot: root, lastInput: JSON.stringify(input) };
        } catch (e) {
            releaseHub(hub);
            await chatPool.release(chat).catch(() => {});
            throw e;
        }
    }

    function statusLine(hub) {
        const s = hub.summary();
        if (!s) return "loading";
        const parts = [];
        if (s.live.running) parts.push(`${s.live.running} running`);
        if (s.live.stale) parts.push(`${s.live.stale} stale`);
        parts.push(`${s.experiments} experiments`);
        if (s.traces) parts.push(`${s.traces} traces`);
        return parts.join(" · ");
    }

    /** Idempotent: re-opening an instance focuses/rehydrates it and applies only changed input. */
    async function open(ctx) {
        const input = validateOpenInput(ctx.input);
        const root = await resolveRoot(input, ctx);
        const id = ctx.instanceId;
        let pending = instances.get(id);
        let entry = pending ? await pending.catch(() => null) : null;
        if (entry && !sameRoot(entry.repoRoot, root)) {
            instances.delete(id);
            await disposeEntry(entry);
            entry = null;
        }
        if (!entry) {
            pending = createEntry(id, root, input);
            instances.set(id, pending);
            try {
                entry = await pending;
            } catch (e) {
                instances.delete(id);
                log(`ci-harness-dashboard: open failed: ${e?.message ?? e}`, "error");
                fail("open_failed", "could not start the dashboard server");
            }
        } else {
            const key = JSON.stringify(input);
            if (key !== entry.lastInput) {
                entry.lastInput = key;
                const ui = uiFrom(input);
                if (Object.keys(ui).length) entry.server.pushUi(ui);
            }
        }
        return { url: entry.server.url, title: `${DISPLAY_NAME} — ${path.basename(entry.repoRoot)}`, status: statusLine(entry.hub) };
    }

    async function onClose(ctx) {
        const pending = instances.get(ctx.instanceId);
        if (!pending) return;
        instances.delete(ctx.instanceId);
        const entry = await pending.catch(() => null);
        if (entry) await disposeEntry(entry);
    }

    async function closeAll() {
        const all = [...instances.values()];
        instances.clear();
        await Promise.all(all.map((p) => p.then(disposeEntry, () => {})));
    }

    async function requireInstance(ctx) {
        const pending = instances.get(ctx.instanceId);
        const entry = pending ? await pending.catch(() => null) : null;
        if (!entry) fail("canvas_not_open", "this dashboard instance is not open; open the canvas first");
        return entry;
    }

    /** Run `fn(hub)` against the instance's hub, or a temporary hub for the session's repo. */
    async function withHub(ctx, fn) {
        const pending = instances.get(ctx.instanceId);
        const entry = pending ? await pending.catch(() => null) : null;
        if (entry) return fn(entry.hub, entry);
        const root = await resolveRoot({}, ctx);
        const hub = await acquireHub(root, hubOptions);
        try {
            return await fn(hub, null);
        } finally {
            releaseHub(hub);
        }
    }

    const input = (ctx) => (ctx.input && typeof ctx.input === "object" && !Array.isArray(ctx.input) ? ctx.input : {});
    const push = (entry, ui) => {
        try {
            return entry.server.pushUi(ui);
        } catch (e) {
            fail("invalid_input", e.message);
        }
    };
    const shown = (entry, ui) => ({ ok: true, ui: { view: ui.view, campaignId: ui.campaignId, experimentId: ui.experimentId, traceId: ui.traceId }, viewers: entry.server.clientCount() });

    const handlers = {
        async refresh(ctx) {
            const entry = await requireInstance(ctx);
            await entry.hub.refresh("manual");
            return { ok: true, version: entry.hub.version, status: statusLine(entry.hub) };
        },
        async show_view(ctx) {
            const { view } = input(ctx);
            if (!VIEWS.includes(view)) fail("invalid_input", `view must be one of: ${VIEWS.join(", ")}`);
            const entry = await requireInstance(ctx);
            return shown(entry, push(entry, { view }));
        },
        async select_campaign(ctx) {
            const { campaignId } = input(ctx);
            if (!isSafeId(campaignId)) fail("invalid_input", "invalid campaignId");
            const entry = await requireInstance(ctx);
            const known = !!entry.hub.model?.campaigns.some((c) => c.campaignId === campaignId);
            return { ...shown(entry, push(entry, { view: "experiment", campaignId, experimentId: null })), known };
        },
        async focus_experiment(ctx) {
            const { experimentId } = input(ctx);
            if (!isSafeId(experimentId)) fail("invalid_input", "invalid experimentId");
            const entry = await requireInstance(ctx);
            const d = entry.hub.model ? experimentDetail(entry.hub.model, experimentId) : null;
            return { ...shown(entry, push(entry, { view: "experiment", experimentId })), found: !!d, summary: d?.summary ?? null, runState: d?.run?.state ?? null };
        },
        async focus_trace(ctx) {
            const { traceId } = input(ctx);
            if (!isTraceId(traceId)) fail("invalid_input", "traceId must be 32 hex characters");
            const entry = await requireInstance(ctx);
            const t = entry.hub.model ? buildSpanTree(entry.hub.model.spans, traceId) : null;
            const ui = push(entry, { view: "traces", traceId });
            return { ...shown(entry, ui), foundLocally: !!t, name: t?.name ?? null, spans: t?.spans ?? null, errors: t?.errors ?? null };
        },
        async get_summary(ctx) {
            return withHub(ctx, async (hub) => {
                if (!hub.model) await hub.refresh("summary");
                return summarize(hub.model);
            });
        },
        async dashboard_status(ctx) {
            const asp = stripSecrets(await aspire.status());
            return withHub(ctx, async (hub, entry) =>
                stripSecrets({
                    open: !!entry,
                    instanceId: ctx.instanceId ?? null,
                    url: entry?.server.url ?? null,
                    viewers: entry?.server.clientCount() ?? 0,
                    ui: entry ? { ...entry.server.ui } : null,
                    hub: hub.status(),
                    aspire: { configured: asp.configured, reachable: asp.reachable, version: asp.version ?? null, uiUrl: asp.uiUrl ?? null, error: asp.error ?? null, hint: asp.hint ?? null },
                    instances: instances.size,
                }),
            );
        },
        async chat_status(ctx) {
            const pending = instances.get(ctx.instanceId);
            const entry = pending ? await pending.catch(() => null) : null;
            if (entry?.chat) return stripSecrets({ open: true, ...entry.chat.status() });
            // No open canvas: report what would run, without spawning anything.
            const root = await resolveRoot({}, ctx);
            const disabled = env.CI_CHAT_DISABLED === "1";
            const cmd = disabled ? null : chatCommand(env, chatRunDir(root, env));
            const lastError = disabled ? "disabled by CI_CHAT_DISABLED=1" : (cmd?.error ?? null);
            return stripSecrets({
                open: false,
                state: disabled ? "disabled" : "stopped",
                profile: cmd?.custom ? "custom" : (cmd?.profile ?? null),
                program: cmd?.argv ? path.basename(cmd.argv[0]) : null,
                runDir: chatRunDir(root, env),
                lastError,
                hint: "open the canvas and pick the Chat tab to start the backend",
            });
        },
    };

    const actions = ACTION_DEFS.map((a) => ({ ...a, handler: (ctx) => handlers[a.name](ctx) }));
    return { open, onClose, actions, closeAll, instances, aspire };
}

/** True when `p` exists synchronously (used by dev-preview argument checks). */
export function existsDir(p) {
    try {
        return fs.statSync(p).isDirectory();
    } catch {
        return false;
    }
}
