// CI Harness Dashboard UI. No framework, no innerHTML: every node is built with createElement and
// textContent (C32). Talks only to its own loopback server (fetch + EventSource).
const TOKEN = document.querySelector('meta[name="ci-token"]')?.content ?? "";
const VIEWS = [
    ["overview", "Overview"],
    ["live", "Live"],
    ["experiment", "Experiment"],
    ["traces", "Traces"],
    ["evals", "Evals"],
    ["sleep", "Sleep"],
    ["bus", "Bus"],
    ["chat", "Chat"],
    ["aspire", "Aspire"],
];

const state = {
    ui: { view: "overview", campaignId: null, experimentId: null, traceId: null, seq: -1 },
    version: 0,
    data: {},
    clock: { server: null, client: null },
    traceFilter: "",
    openSpans: new Set(),
    loginUrl: null,
    busTopic: null,
};

// ------------------------------------------------------------------ DOM helpers
function h(tag, props = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props ?? {})) {
        if (v === undefined || v === null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "text") el.textContent = String(v);
        else if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
        else if (k === "data") for (const [dk, dv] of Object.entries(v)) el.dataset[dk] = String(dv);
        else if (k in el && typeof v !== "string") el[k] = v;
        else el.setAttribute(k, String(v));
    }
    append(el, children);
    return el;
}
function append(el, children) {
    for (const c of children.flat(Infinity)) {
        if (c === null || c === undefined || c === false) continue;
        el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
}
const svgNs = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}) {
    const el = document.createElementNS(svgNs, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
}
const code = (t) => h("code", { text: t });

// ------------------------------------------------------------------ formatting
const fmtNum = (v, d = 3) => (typeof v === "number" && Number.isFinite(v) ? String(Math.round(v * 10 ** d) / 10 ** d) : "—");
const fmtSigned = (v, d = 3) => (typeof v === "number" && Number.isFinite(v) ? (v > 0 ? "+" : "") + fmtNum(v, d) : "—");
const fmtPct = (v) => (typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—");
const fmtInt = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toLocaleString() : "—");
function fmtDur(ms) {
    if (typeof ms !== "number" || !Number.isFinite(ms)) return "—";
    if (ms < 1) return "<1 ms";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
    const m = Math.floor(ms / 60000);
    return `${m}m ${Math.round((ms % 60000) / 1000)}s`;
}
function fmtAge(sec) {
    if (typeof sec !== "number" || !Number.isFinite(sec)) return "—";
    sec = Math.max(0, Math.round(sec));
    if (sec < 60) return `${sec}s ago`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
    if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
    return `${Math.floor(sec / 86400)}d ago`;
}
function fmtTime(epochMs) {
    if (typeof epochMs !== "number" || !Number.isFinite(epochMs)) return "—";
    return new Date(epochMs).toLocaleString();
}
const shortSha = (s) => (typeof s === "string" ? s.slice(0, 10) : "—");
const serverNow = () => (state.clock.server === null ? Date.now() / 1000 : state.clock.server + (Date.now() - state.clock.client) / 1000);
/** Live-updating relative time label for an epoch-seconds timestamp. */
function ageEl(epochSec) {
    if (typeof epochSec !== "number") return h("span", { class: "muted", text: "—" });
    const el = h("span", { class: "age", data: { epoch: epochSec }, text: fmtAge(serverNow() - epochSec) });
    el.title = new Date(epochSec * 1000).toLocaleString();
    return el;
}
function badge(text, kind = "") {
    return h("span", { class: `badge ${kind}`.trim(), text });
}
const stateKind = (s) =>
    ({ running: "info", stale: "warn", done: "ok", failed: "bad", error: "bad", ship: "ok", accepted: "ok", reject: "bad", rejected: "bad", stop: "warn", completed: "ok", succeeded: "ok" })[
        String(s ?? "").toLowerCase()
    ] ?? "";

function table(headers, rows, { caption, cls = "" } = {}) {
    return h(
        "div",
        { class: "table-wrap" },
        h(
            "table",
            { class: cls },
            caption ? h("caption", { text: caption }) : null,
            h("thead", {}, h("tr", {}, headers.map((t) => h("th", { scope: "col", text: t })))),
            h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c ?? "—"))))),
        ),
    );
}
function kv(pairs) {
    return h(
        "dl",
        { class: "kv" },
        pairs.filter(([, v]) => v !== undefined).map(([k, v]) => [h("dt", { text: k }), h("dd", {}, v === null || v === "" ? "—" : v)]),
    );
}
function section(title, ...children) {
    return h("section", { class: "card" }, h("h2", { text: title }), ...children);
}
function empty(title, ...hint) {
    return h("div", { class: "empty" }, h("p", { class: "empty-title", text: title }), h("p", { class: "muted" }, ...hint));
}
function linkBtn(text, onClick, title) {
    return h("button", { type: "button", class: "link", title, on: { click: onClick } }, text);
}

// ------------------------------------------------------------------ server I/O
async function api(path) {
    const res = await fetch(path, { headers: { accept: "application/json" }, cache: "no-store" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(body?.error?.message ?? `HTTP ${res.status}`), { status: res.status });
    if (typeof body.generatedAt === "number") state.clock = { server: body.generatedAt, client: Date.now() };
    return body;
}
async function post(path, body = {}) {
    const res = await fetch(path, {
        method: "POST",
        headers: { "content-type": "application/json", "x-ci-token": TOKEN },
        body: JSON.stringify(body),
        cache: "no-store",
    });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(out?.error?.message ?? `HTTP ${res.status}`);
    return out;
}

function navigate(partial) {
    const next = { ...state.ui, ...partial };
    state.ui = next;
    post("/api/ui", { view: next.view, campaignId: next.campaignId, experimentId: next.experimentId, traceId: next.traceId })
        .then((r) => {
            if (r?.ui?.seq !== undefined) state.ui.seq = r.ui.seq;
        })
        .catch(() => {});
    render();
}

const ENDPOINTS = {
    overview: () => [["summary", "/api/summary"]],
    live: () => [["live", "/api/live"]],
    experiment: () => [
        ["experiments", "/api/experiments"],
        ...(state.ui.experimentId ? [["experiment", `/api/experiment/${encodeURIComponent(state.ui.experimentId)}`]] : []),
    ],
    traces: () => [["traces", "/api/traces"], ...(state.ui.traceId ? [["trace", `/api/trace/${state.ui.traceId}`]] : [])],
    evals: () => [["evals", "/api/evals"]],
    sleep: () => [["sleep", "/api/sleep"]],
    bus: () => [["bus", "/api/bus"]],
    aspire: () => [["aspire", "/api/aspire"]],
    chat: () => [["chatStatus", "/api/chat/status"]],
};

let loadSeq = 0;
async function load() {
    const my = ++loadSeq;
    const view = state.ui.view;
    const wanted = ENDPOINTS[view]?.() ?? [];
    const results = await Promise.all(
        wanted.map(([key, url]) =>
            api(url)
                .then((v) => [key, url, v, null])
                .catch((e) => [key, url, null, e]),
        ),
    );
    if (my !== loadSeq) return;
    for (const [key, url, v, e] of results) state.data[key] = { url, value: v, error: e ? e.message : null, status: e?.status ?? 200 };
    render();
}
let loadTimer = null;
function scheduleLoad(delay = 150) {
    clearTimeout(loadTimer);
    loadTimer = setTimeout(load, delay);
}
const got = (key, url) => {
    const d = state.data[key];
    return d && (!url || d.url === url) ? d : null;
};

// ------------------------------------------------------------------ views
function viewOverview() {
    const d = got("summary");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const v = d.value;
    const s = v.summary;
    const out = [];
    const nothing = !v.campaigns.length && !s.experiments && !s.live.running && !s.live.done && !s.evals.length && !s.traces && !s.sleep.lastNight;
    if (nothing) {
        out.push(
            empty(
                "No harness data found in this repository yet.",
                "Start a campaign with ",
                code("ci-lab campaign run --profile fake"),
                ", run an ASSERT suite, or start telemetry with ",
                code("ci-lab dashboard up"),
                ". Data appears here automatically as files are written.",
            ),
        );
    }
    out.push(
        h(
            "div",
            { class: "stats" },
            stat("Running", s.live.running, s.live.stale ? `${s.live.stale} stale` : null, s.live.stale ? "warn" : ""),
            stat("Experiments", s.experiments),
            stat("Traces", s.traces, s.traceErrors ? `${s.traceErrors} errors` : null, s.traceErrors ? "bad" : ""),
            stat("Rollouts", s.rollouts.total),
            stat("Eval runs", s.evals.length),
            stat("Imported CI runs", s.imports),
        ),
    );
    for (const c of v.campaigns) out.push(campaignCard(c));
    if (s.live.active.length) {
        out.push(
            section(
                "Active rounds",
                table(
                    ["Experiment", "Phase", "State", "Age", "Arms"],
                    s.live.active.map((r) => [expLink(r.id), r.phase ?? "—", badge(r.state, stateKind(r.state)), fmtAge(r.ageSec), String(r.arms)]),
                ),
            ),
        );
    }
    if (s.lastDecisions.length) {
        out.push(
            section(
                "Latest decisions",
                table(
                    ["Experiment", "Kind", "Outcome", "Winner", "ΔS", "ΔC"],
                    s.lastDecisions.map((e) => [expLink(e.id), e.kind ?? "—", badge(e.outcome, stateKind(e.outcome)), e.winner ?? "—", fmtSigned(e.deltaS), fmtSigned(e.deltaC)]),
                ),
            ),
        );
    }
    if (s.evals.length) {
        out.push(
            section(
                "Recent ASSERT runs",
                table(
                    ["Suite", "Run", "Status", "Cases", "Flags"],
                    s.evals.map((e) => [e.suite, e.run, badge(e.status ?? "?", stateKind(e.status)), String(e.cases), e.flags ? badge(String(e.flags), "warn") : "0"]),
                ),
            ),
        );
    }
    if (s.sleep.lastNight) {
        out.push(
            section(
                "Sleep",
                kv([
                    ["Last night", s.sleep.lastNight],
                    ["Status", s.sleep.lastStatus ? badge(s.sleep.lastStatus, stateKind(s.sleep.lastStatus)) : null],
                    ["Accepted total", fmtInt(s.sleep.acceptedTotal)],
                    ["Pending tasks", fmtInt(s.sleep.pendingTasks)],
                ]),
            ),
        );
    }
    if (v.imports.length) out.push(importsCard(v.imports));
    if (v.warnings.length) {
        out.push(h("details", { class: "card warnings" }, h("summary", { text: `${v.warnings.length} reader warning(s)` }), h("ul", {}, v.warnings.map((w) => h("li", { text: w })))));
    }
    return out;
}
function stat(label, value, sub, kind = "") {
    return h("div", { class: `stat ${kind}`.trim() }, h("div", { class: "stat-value", text: fmtInt(value ?? 0) }), h("div", { class: "stat-label", text: label }), sub ? h("div", { class: "stat-sub", text: sub }) : null);
}
function campaignCard(c) {
    const inc = c.incumbent;
    const burn = c.budget.tokenBurn;
    return h(
        "section",
        { class: "card" },
        h("h2", {}, linkBtn(c.campaignId, () => navigate({ view: "experiment", campaignId: c.campaignId, experimentId: null }), "Open campaign"), c.stopped ? badge("STOP", "warn") : null),
        kv([
            ["Profile", c.profile],
            ["Domain", c.domain],
            ["Incumbent", inc ? h("span", {}, code(shortSha(inc.commit)), ` score ${fmtNum(inc.score)} (round ${inc.round ?? "—"})`) : null],
            ["Rounds", `${c.budget.roundsDone}${c.budget.roundsLimit ? ` / ${c.budget.roundsLimit}` : ""}`],
            ["Decisions", Object.keys(c.decisions).length ? Object.entries(c.decisions).map(([k, n]) => badge(`${k} ${n}`, stateKind(k))) : null],
            ["δ (calibrated)", fmtNum(c.delta, 4)],
            [
                "Token burn",
                burn !== null ? h("span", {}, h("meter", { min: 0, max: 1, low: 0.7, high: 0.9, optimum: 0, value: Math.min(1, burn), title: fmtPct(burn) }), ` ${fmtPct(burn)}`) : fmtInt(c.budget.tokensUsed),
            ],
            ["Confirm", c.confirm?.outcome ? badge(c.confirm.outcome, stateKind(c.confirm.outcome)) : null],
        ]),
        c.trend.length > 1 ? sparkline(c.trend.map((t) => t.score)) : null,
    );
}
function sparkline(values) {
    const pts = values.map((v, i) => [i, v]).filter(([, v]) => typeof v === "number");
    if (pts.length < 2) return null;
    const W = 240;
    const H = 40;
    const lo = Math.min(...pts.map((p) => p[1]));
    const hi = Math.max(...pts.map((p) => p[1]));
    const x = (i) => (i / Math.max(1, values.length - 1)) * (W - 8) + 4;
    const y = (v) => (hi === lo ? H / 2 : H - 4 - ((v - lo) / (hi - lo)) * (H - 8));
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "spark", role: "img", "aria-label": `Score trend ${fmtNum(pts[0][1])} → ${fmtNum(pts.at(-1)[1])}` });
    s.append(svg("polyline", { points: pts.map(([i, v]) => `${x(i)},${y(v)}`).join(" "), fill: "none", class: "spark-line" }));
    for (const [i, v] of pts) s.append(svg("circle", { cx: x(i), cy: y(v), r: 2.5, class: "spark-dot" }));
    return s;
}
function importsCard(imports) {
    return section(
        "Imported CI runs",
        h("p", { class: "muted" }, "Pulled with ", code("ci-lab telemetry pull --run <id>"), "; their spans appear under Traces."),
        table(
            ["Run", "Repo", "Spans", "Traces", "Digest", "Pulled"],
            imports.map((i) => [i.runId, i.repo ?? "—", fmtInt(i.spans), fmtInt(i.traces), i.digestVerified ? badge("verified", "ok") : badge("unverified", "warn"), ageEl(i.importedAt)]),
        ),
    );
}
const expLink = (id) => (id ? linkBtn(id, () => navigate({ view: "experiment", experimentId: id }), "Open experiment") : "—");
const traceLink = (id, label) => (id ? linkBtn(label ?? `${id.slice(0, 8)}…`, () => navigate({ view: "traces", traceId: id }), "Open trace") : "—");

