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
    aspire: () => [["aspire", "/api/aspire"]],
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

function viewLive() {
    const d = got("live");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const { live, heartbeatSec, rollouts } = d.value;
    if (!live.length) {
        return empty(
            "No run directories with live status.",
            "Rounds write ",
            code("<run_dir>/<experiment>/status.d/<writer>.json"),
            " while they run (",
            code("CI_RUN_DIR"),
            ", default ",
            code("artifacts/ci-runs"),
            "). Start one with ",
            code("ci-lab campaign run --profile fake"),
            ".",
        );
    }
    const out = [h("p", { class: "muted small", text: `Stale = no update for more than 2 × heartbeat (${heartbeatSec}s default).` })];
    for (const r of live) {
        out.push(
            h(
                "section",
                { class: `card run ${r.state}` },
                h("h2", {}, expLink(r.experimentId), " ", badge(r.state, stateKind(r.state)), r.stopped ? badge("STOP", "warn") : null),
                kv([
                    ["Phase", r.phase],
                    ["Round", r.round === null ? null : String(r.round)],
                    ["Updated", ageEl(r.updated)],
                    ["Writers", r.writers.length ? r.writers.join(", ") : null],
                    ["Decision", r.decision ? h("span", {}, badge(r.decision, stateKind(r.decision)), r.winner ? ` → ${r.winner}` : "") : null],
                    ["PR", r.pr === null ? null : `#${r.pr}`],
                    ["Trace", r.traceId ? traceLink(r.traceId) : null],
                ]),
                r.arms.length
                    ? table(
                          ["Arm", "Strategy", "State", "Phase", "Updated", "Score"],
                          r.arms.map((a) => [
                              a.arm,
                              a.strategy ?? "—",
                              h("span", {}, badge(a.state ?? "?", stateKind(a.state)), a.stale ? badge("stale", "warn") : null),
                              a.phase ?? "—",
                              a.ageSec === null ? "—" : fmtAge(a.ageSec),
                              fmtNum(a.score),
                          ]),
                      )
                    : null,
            ),
        );
    }
    if (rollouts?.total) {
        out.push(
            section(
                "AGL rollouts",
                kv([
                    ["Total", fmtInt(rollouts.total)],
                    ["By status", Object.entries(rollouts.byStatus).map(([k, n]) => badge(`${k} ${n}`, stateKind(k)))],
                    ["Mean score", fmtNum(rollouts.meanScore)],
                ]),
                table(
                    ["Rollout", "Status", "Events", "Score", "Last"],
                    rollouts.recent.slice(0, 15).map((r) => [h("span", { class: "mono", text: r.rolloutId }), badge(r.status, stateKind(r.status)), String(r.events), fmtNum(r.score), ageEl(r.lastTs)]),
                ),
            ),
        );
    }
    return out;
}

function viewExperiment() {
    const d = got("experiments");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const { campaigns, experiments } = d.value;
    const out = [];
    const cid = state.ui.campaignId ?? (state.ui.experimentId ? null : campaigns[0]?.campaignId ?? null);
    if (state.ui.experimentId) {
        out.push(h("p", {}, linkBtn("← all experiments", () => navigate({ experimentId: null }))));
        out.push(experimentDetailView());
        return out;
    }
    if (!campaigns.length && !experiments.length) {
        return empty("No experiments recorded yet.", "The ledger lives under ", code("experiments/"), " (OES envelopes). Run ", code("ci-lab campaign run --profile fake"), " to produce one.");
    }
    if (campaigns.length) {
        const sel = h(
            "select",
            { "aria-label": "Campaign", on: { change: (e) => navigate({ campaignId: e.target.value || null }) } },
            campaigns.map((c) => {
                const o = h("option", { value: c.campaignId, text: c.campaignId });
                o.selected = c.campaignId === cid;
                return o;
            }),
        );
        out.push(h("div", { class: "toolbar" }, h("label", {}, "Campaign ", sel)));
        const c = campaigns.find((x) => x.campaignId === cid);
        if (c) {
            out.push(campaignCard(c));
            out.push(
                section(
                    "Rounds",
                    c.rounds.length
                        ? table(
                              ["#", "Experiment", "Decision", "Winner", "Score", "ΔS", "ΔC", "CI lb", "Arms"],
                              c.rounds.map((r) => [
                                  r.round === null ? "—" : String(r.round),
                                  expLink(r.eid),
                                  r.decision ? badge(r.decision, stateKind(r.decision)) : "—",
                                  r.winner ?? "—",
                                  fmtNum(r.score),
                                  fmtSigned(r.deltaS),
                                  fmtSigned(r.deltaC),
                                  fmtNum(r.ciLowerBound),
                                  `${r.accepted ?? 0}/${r.arms ?? 0}`,
                              ]),
                          )
                        : empty("No rounds yet."),
                ),
            );
        }
    }
    const list = cid ? experiments.filter((e) => e.campaignId === cid || !e.campaignId) : experiments;
    if (list.length) {
        out.push(
            section(
                "Experiments",
                table(
                    ["Experiment", "Kind", "Outcome", "Winner", "ΔS", "ΔC", "CI lb"],
                    list.map((e) => [expLink(e.id), e.kind ?? "—", e.outcome ? badge(e.outcome, stateKind(e.outcome)) : "—", e.winner ?? "—", fmtSigned(e.deltaS), fmtSigned(e.deltaC), fmtNum(e.ciLowerBound)]),
                ),
            ),
        );
    }
    return out;
}

function experimentDetailView() {
    const d = got("experiment", `/api/experiment/${encodeURIComponent(state.ui.experimentId)}`);
    if (!d) return loading();
    if (d.error) return d.status === 404 ? empty(`Experiment ${state.ui.experimentId} not found.`, "It may not have been written yet; this view updates automatically.") : errorBox(d.error);
    const { envelope: e, run, traceIds } = d.value;
    const out = [];
    if (e) {
        out.push(
            h(
                "section",
                { class: "card" },
                h("h2", {}, e.id, " ", badge(e.kind ?? "experiment"), e.decision.outcome ? badge(e.decision.outcome, stateKind(e.decision.outcome)) : null),
                e.title ? h("p", { text: e.title }) : null,
                e.hypothesis ? h("p", { class: "muted", text: e.hypothesis }) : null,
                kv([
                    ["Campaign", e.campaignId],
                    ["Round", e.round === null ? null : String(e.round)],
                    ["Split", e.split],
                    ["Status", e.status],
                    ["δ", fmtNum(e.delta, 4)],
                    ["CI lower bound", fmtNum(e.ciLowerBound)],
                    ["Judge", e.judgeModel],
                    ["Incumbent", e.incumbentCommit ? code(shortSha(e.incumbentCommit)) : null],
                    ["Recommended", e.scorecard.recommendedAction],
                    ["Decided", e.decision.decidedAt],
                ]),
                e.decision.rationale ? h("p", { class: "rationale", text: e.decision.rationale }) : null,
            ),
        );
        if (e.selection) {
            out.push(
                section(
                    "Selection",
                    table(
                        ["Variant", "Admissible", "ΔS", "ΔC", "CI lb", "Rule"],
                        e.selection.candidates.map((c) => [
                            h("span", {}, c.variantId, c.variantId === e.selection.winner ? badge("winner", "ok") : null),
                            c.admissible ? "yes" : h("span", { title: c.reasons.join("; ") }, "no"),
                            fmtSigned(c.deltaS),
                            fmtSigned(c.deltaC),
                            fmtNum(c.ciLowerBound),
                            c.rule ?? "—",
                        ]),
                    ),
                ),
            );
        }
        if (e.variants.length) {
            out.push(
                section(
                    "Variants",
                    ...e.variants.map((v) =>
                        h(
                            "details",
                            { class: "variant" },
                            h("summary", {}, `${v.id} `, v.role ? badge(v.role) : null, v.status ? badge(v.status, stateKind(v.status)) : null),
                            v.description ? h("p", { text: v.description }) : null,
                            kv([
                                ["Head", v.headCommit ? code(shortSha(v.headCommit)) : null],
                                ["Tree", v.harnessTree ? code(shortSha(v.harnessTree)) : null],
                                ["Critic", v.critic ? (v.critic.passed ? badge("passed", "ok") : h("span", {}, badge("failed", "bad"), ` ${v.critic.reasons.join("; ")}`)) : null],
                            ]),
                            v.edits.length ? table(["Component", "Hypothesis", "Commit"], v.edits.map((x) => [x.component ?? "—", x.hypothesis ?? "—", x.commit ? code(shortSha(x.commit)) : "—"])) : null,
                        ),
                    ),
                ),
            );
        }
        if (e.results.length) {
            out.push(
                section(
                    "Metric results",
                    table(
                        ["Metric", "Variant", "Baseline", "Value", "Diff", "CI", "Status"],
                        e.results.map((r) => [
                            r.metricId ?? "—",
                            r.variantId ?? "—",
                            fmtNum(r.baselineValue),
                            fmtNum(r.variantValue),
                            fmtSigned(r.diff),
                            r.ci ? `[${fmtNum(r.ci.lower)}, ${fmtNum(r.ci.upper)}]` : "—",
                            r.status ?? "—",
                        ]),
                    ),
                ),
            );
        }
        if (e.quality.length) {
            out.push(section("Quality checks", table(["Check", "Status", "Severity", "Message"], e.quality.map((q) => [q.checkType ?? "—", badge(q.status ?? "?", q.status === "passed" ? "ok" : q.status === "failed" ? "bad" : ""), q.severity ?? "—", q.message ?? "—"]))));
        }
        if (e.sleep) out.push(sleepNightCard(e));
        if (e.source) out.push(h("p", { class: "muted small" }, "Source: ", code(e.source)));
    }
    if (run) {
        out.push(
            section(
                "Run status",
                kv([
                    ["State", badge(run.state, stateKind(run.state))],
                    ["Phase", run.phase],
                    ["Updated", ageEl(run.updated)],
                    ["Writers", run.writers.join(", ") || null],
                ]),
                run.arms.length ? table(["Arm", "Strategy", "State", "Phase"], run.arms.map((a) => [a.arm, a.strategy ?? "—", badge(a.state ?? "?", stateKind(a.state)), a.phase ?? "—"])) : null,
            ),
        );
    }
    if (traceIds.length) out.push(section("Traces", h("ul", {}, traceIds.map((t) => h("li", {}, traceLink(t, t))))));
    return out;
}

function viewTraces() {
    const d = got("traces");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const v = d.value;
    const out = [];
    if (state.ui.traceId) {
        out.push(h("p", {}, linkBtn("← all traces", () => navigate({ traceId: null }))));
        out.push(traceDetailView());
        return out;
    }
    const notes = [];
    if (v.aspire.included) notes.push(`includes ${v.aspire.spans} span(s) from the Aspire Dashboard`);
    else if (v.aspire.error && v.aspire.error !== "no dashboard state") notes.push(`Aspire: ${v.aspire.error}`);
    if (v.rejectedSpans) notes.push(`${v.rejectedSpans} span record(s) rejected (unknown schemaVersion)`);
    if (!v.traces.length) {
        return empty(
            "No traces yet.",
            "Spans are written to ",
            code("<run_dir>/telemetry/spans-<pid>.jsonl"),
            " and, when running, to the Aspire Dashboard (",
            code("ci-lab dashboard up"),
            "). CI runs can be imported with ",
            code("ci-lab telemetry pull --run <id>"),
            ".",
        );
    }
    const filter = h("input", {
        type: "search",
        placeholder: "Filter by name, campaign, experiment…",
        "aria-label": "Filter traces",
        value: state.traceFilter,
        on: {
            input: (e) => {
                state.traceFilter = e.target.value;
                render();
                const el = document.querySelector('input[type="search"]');
                el?.focus();
                el?.setSelectionRange(el.value.length, el.value.length);
            },
        },
    });
    const q = state.traceFilter.trim().toLowerCase();
    const list = q ? v.traces.filter((t) => [t.name, t.campaignId, t.experimentId, t.traceId, t.night, ...(t.services ?? [])].some((x) => String(x ?? "").toLowerCase().includes(q))) : v.traces;
    out.push(h("div", { class: "toolbar" }, filter, h("span", { class: "muted small", text: `${list.length} of ${v.traces.length}` })));
    if (notes.length) out.push(h("p", { class: "muted small", text: notes.join(" · ") }));
    out.push(
        table(
            ["Trace", "Started", "Duration", "Spans", "Errors", "Context"],
            list.slice(0, 200).map((t) => [
                h("span", {}, traceLink(t.traceId, t.name), t.incomplete ? badge("partial", "warn") : null, t.origin.includes("import") ? badge("CI", "info") : null),
                fmtTime(t.startMs),
                fmtDur(t.durationMs),
                `${t.spans}${t.genai ? ` (${t.genai} GenAI)` : ""}`,
                t.errors ? badge(String(t.errors), "bad") : "0",
                [t.campaignId, t.experimentId, t.round !== null && t.round !== undefined ? `round ${t.round}` : null, t.night].filter(Boolean).join(" · ") || "—",
            ]),
            { cls: "traces" },
        ),
    );
    return out;
}

function traceDetailView() {
    const d = got("trace", `/api/trace/${state.ui.traceId}`);
    if (!d) return loading();
    if (d.error) return d.status === 404 ? empty(`Trace ${state.ui.traceId} not found.`, "Spans are exported when they end; long-running rounds appear after their first step finishes.") : errorBox(d.error);
    const { trace: t, aspireUrl } = d.value;
    const total = t.durationMs || 1;
    const header = h(
        "section",
        { class: "card" },
        h("h2", {}, t.name, " ", t.errors ? badge(`${t.errors} error(s)`, "bad") : badge("ok", "ok")),
        kv([
            ["Trace id", code(t.traceId)],
            ["Started", fmtTime(t.startMs)],
            ["Duration", fmtDur(t.durationMs)],
            ["Spans", `${t.spans} (${t.genai} GenAI)`],
            ["Context", [t.campaignId, t.experimentId, t.night].filter(Boolean).join(" · ") || null],
            ["Services", t.services.join(", ") || null],
        ]),
        h(
            "div",
            { class: "toolbar" },
            h("button", { type: "button", class: "btn", on: { click: () => expandAll(t.roots, true) } }, "Expand all"),
            h("button", { type: "button", class: "btn", on: { click: () => expandAll(t.roots, false) } }, "Collapse all"),
            aspireUrl ? h("button", { type: "button", class: "btn primary", title: "Sign in to the Aspire Dashboard and open this trace", on: { click: () => openAspire(t.traceId) } }, "Open in Aspire") : null,
        ),
        loginFallback(),
    );
    const tree = h("ul", { class: "tree", role: "tree" }, t.roots.map((n) => spanNode(n, total)));
    return [header, h("section", { class: "card" }, tree)];
}
function expandAll(nodes, open) {
    const walk = (n) => {
        if (open) state.openSpans.add(n.spanId);
        else state.openSpans.delete(n.spanId);
        n.children.forEach(walk);
    };
    nodes.forEach(walk);
    render();
}
function spanNode(n, total) {
    const isOpen = state.openSpans.has(n.spanId) || (n.children.length && !state.openSpans.has(`-${n.spanId}`) && n.category === "harness");
    const bar = h("span", { class: "bar-track", "aria-hidden": "true" }, h("span", { class: `bar-fill ${n.category} ${n.error ? "err" : ""}` }));
    const fill = bar.firstChild;
    fill.style.marginLeft = `${Math.max(0, Math.min(100, ((n.offsetMs ?? 0) / total) * 100))}%`;
    fill.style.width = `${Math.max(0.5, Math.min(100, ((n.durationMs ?? 0) / total) * 100))}%`;
    const toggle = n.children.length
        ? h("button", {
              type: "button",
              class: "twisty",
              "aria-label": isOpen ? "Collapse" : "Expand",
              "aria-expanded": isOpen ? "true" : "false",
              text: isOpen ? "▾" : "▸",
              on: {
                  click: () => {
                      if (isOpen) {
                          state.openSpans.delete(n.spanId);
                          state.openSpans.add(`-${n.spanId}`);
                      } else {
                          state.openSpans.add(n.spanId);
                          state.openSpans.delete(`-${n.spanId}`);
                      }
                      render();
                  },
              },
          })
        : h("span", { class: "twisty-pad", "aria-hidden": "true" });
    const attrs = Object.entries(n.attributes);
    const details = h(
        "details",
        { class: "span-attrs" },
        h("summary", { text: `${attrs.length} attribute(s)${n.redacted ? ` · ${n.redacted} GenAI content attribute(s) hidden` : ""}` }),
        attrs.length ? h("dl", { class: "kv mono" }, attrs.map(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })])) : null,
        n.redacted ? h("p", { class: "muted small", text: "Content capture was off for this span (C29). Enable sensitive telemetry only on fake/local profiles." }) : null,
    );
    return h(
        "li",
        { role: "treeitem", "aria-expanded": n.children.length ? String(!!isOpen) : undefined, class: `span ${n.category}` },
        h(
            "div",
            { class: "span-row" },
            toggle,
            h("span", { class: "span-name", title: n.name, text: n.name }),
            n.category === "genai" ? badge("GenAI", "info") : null,
            n.error ? badge(n.exceptionTypes[0] ?? n.statusMessage ?? "error", "bad") : null,
            n.origin && n.origin !== "jsonl" ? badge(n.origin.startsWith("import") ? "CI" : n.origin, "") : null,
            h("span", { class: "span-dur", text: fmtDur(n.durationMs) }),
        ),
        bar,
        details,
        isOpen && n.children.length ? h("ul", { role: "group" }, n.children.map((c) => spanNode(c, total))) : null,
    );
}

async function openAspire(traceId) {
    state.loginUrl = null;
    try {
        const { url } = await post("/api/aspire/login", { traceId: traceId ?? null });
        // `noopener` in the features string makes window.open return null, so sever the opener manually.
        const w = window.open(url, "_blank");
        if (w) {
            try {
                w.opener = null;
            } catch {
                /* cross-origin proxy */
            }
        } else {
            state.loginUrl = url;
            render();
        }
    } catch (e) {
        state.loginUrl = { error: e.message };
        render();
    }
}
function loginFallback() {
    const u = state.loginUrl;
    if (!u) return null;
    if (typeof u === "object") return h("p", { class: "error", text: u.error });
    const input = h("input", { type: "text", readOnly: true, value: u, "aria-label": "Aspire sign-in link", class: "mono" });
    return h(
        "div",
        { class: "fallback" },
        h("p", { class: "muted small", text: "The host blocked the pop-up. Copy this one-time sign-in link into your browser (it contains a login token; don't share it):" }),
        h(
            "div",
            { class: "toolbar" },
            input,
            h(
                "button",
                {
                    type: "button",
                    class: "btn",
                    on: {
                        click: async () => {
                            try {
                                await navigator.clipboard.writeText(u);
                            } catch {
                                input.select();
                                document.execCommand?.("copy");
                            }
                        },
                    },
                },
                "Copy",
            ),
            h("button", { type: "button", class: "btn", on: { click: () => ((state.loginUrl = null), render()) } }, "Dismiss"),
        ),
    );
}

function viewEvals() {
    const d = got("evals");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const { evals } = d.value;
    if (!evals.length) {
        return empty("No ASSERT results found.", "Results are read from ", code("artifacts/results/<suite>/<run>/"), " (", code("manifest.json"), ", ", code("metrics.json"), ", ", code("scores.jsonl"), "). Run an ASSERT suite to populate them.");
    }
    return evals.map((e) =>
        h(
            "section",
            { class: "card" },
            h("h2", {}, `${e.suiteId} / ${e.runId} `, badge(e.status ?? "?", stateKind(e.status))),
            e.flags.length ? h("ul", { class: "flags" }, e.flags.map((f) => h("li", {}, badge("!", "warn"), ` ${f}`))) : null,
            kv([
                ["Cases", `${e.cases} (${e.rows} rows)`],
                ["Judges", e.judgeModels.join(", ") || null],
                ["Targets", e.targets.join(", ") || null],
                ["Elapsed", e.elapsedS === null ? null : fmtDur(e.elapsedS * 1000)],
                ["Calls / tokens", e.metrics ? `${fmtInt(e.metrics.calls)} calls · ${fmtInt(e.metrics.inputTokens)} in · ${fmtInt(e.metrics.outputTokens)} out` : null],
                ["Cache hit", e.metrics ? fmtPct(e.metrics.cacheHitRate) : null],
                ["Errors", e.errors ? badge(String(e.errors), "bad") : "0"],
                ["Scenarios", e.scenarios.map((s) => `${s.name} (${s.n})`).join(", ") || null],
            ]),
            e.dimensions.length
                ? table(
                      ["Dimension", "Type", "n", "Result", e.agreement ? "Judge agreement" : null].filter(Boolean),
                      e.dimensions.map((dm) => [dm.key, dm.type ?? "—", String(dm.n), dimResult(dm), e.agreement ? fmtPct(e.agreement.byDim[dm.key]) : null].filter((x) => x !== null)),
                  )
                : null,
            h("p", { class: "muted small" }, "Source: ", code(e.source)),
        ),
    );
}
function dimResult(d) {
    if (d.type === "boolean") return h("span", {}, h("meter", { min: 0, max: 1, value: d.rate ?? 0, title: fmtPct(d.rate) }), ` ${fmtPct(d.rate)} true`);
    if (d.type === "numeric") return `mean ${fmtNum(d.mean)} (${d.min}–${d.max})`;
    if (d.type === "ordinal") {
        const keys = d.scale?.length ? d.scale.filter((k) => d.distribution[k]) : Object.keys(d.distribution);
        return keys.map((k) => `${k}: ${d.distribution[k]}`).join(", ");
    }
    return "—";
}

function viewSleep() {
    const d = got("sleep");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const { sleep: s, holdoutLooks } = d.value;
    if (!s.state && !s.nights.length) {
        return empty("No sleep nights recorded.", "The nightly loop writes ", code("experiments/sleep/"), ". Run ", code("ci-lab sleep run --profile fake"), " (or the ", code("sleep-nightly"), " workflow and ", code("ci-lab telemetry pull"), ").");
    }
    const out = [];
    if (s.state) {
        out.push(
            section(
                "State",
                kv([
                    ["Nights", fmtInt(s.state.night)],
                    ["Last night", s.state.lastNightId],
                    ["Last status", s.state.lastStatus ? badge(s.state.lastStatus, stateKind(s.state.lastStatus)) : null],
                    ["Accepted total", fmtInt(s.state.acceptedTotal)],
                    ["Base", s.state.lastBaseSha ? code(shortSha(s.state.lastBaseSha)) : null],
                    ["Tasks reviewed / pending", `${fmtInt(s.tasks.reviewed)} / ${fmtInt(s.tasks.pending)}`],
                    ["Harvested (all nights)", fmtInt(s.harvested)],
                    ["Holdout looks", fmtInt(holdoutLooks)],
                ]),
                s.state.history.length > 1 ? sparkline(s.state.history.map((x) => x.deltaLcb)) : null,
            ),
        );
    }
    if (s.nights.length) {
        out.push(
            section(
                "Nights",
                table(
                    ["Night", "Outcome", "Tasks", "SkillOpt", "ASSERT ΔS", "CI lb", "PR"],
                    [...s.nights].reverse().map((n) => [
                        expLink(n.id),
                        n.outcome ? badge(n.outcome, stateKind(n.outcome)) : "—",
                        n.tasks ? `${fmtInt(n.tasks.total)} (${fmtInt(n.tasks.harvested)} harvested)` : "—",
                        n.gate?.skilloptPassed === null || n.gate?.skilloptPassed === undefined ? "—" : n.gate.skilloptPassed ? badge("pass", "ok") : badge("fail", "bad"),
                        fmtSigned(n.gate?.deltaS),
                        fmtNum(n.gate?.ciLowerBound),
                        n.adoptionPr ? `#${n.adoptionPr}` : "—",
                    ]),
                ),
            ),
        );
    }
    if (s.skillsUpdated.length) out.push(section("Skills updated", h("ul", {}, s.skillsUpdated.map((p) => h("li", {}, code(p))))));
    if (s.pendingPrs.length) out.push(section("Adoption PRs", h("ul", {}, s.pendingPrs.map((p) => h("li", { text: `${p.night}: #${p.pr}` })))));
    return out;
}
function sleepNightCard(e) {
    const g = e.sleep.gate;
    return section(
        "Sleep gate",
        kv([
            ["Night", e.sleep.night],
            ["Tasks", e.sleep.tasks ? `${fmtInt(e.sleep.tasks.total)} (${fmtInt(e.sleep.tasks.reviewed)} reviewed, ${fmtInt(e.sleep.tasks.harvested)} harvested)` : null],
            ["SkillOpt", g.skilloptPassed === null ? null : `${g.skilloptPassed ? "pass" : "fail"} (${fmtNum(g.skilloptScore)} vs ${fmtNum(g.skilloptBaseline)})`],
            ["ASSERT", g.assertPassed === null ? null : `${g.assertPassed ? "pass" : "fail"} ΔS ${fmtSigned(g.deltaS)}, CI lb ${fmtNum(g.ciLowerBound)}`],
            ["Safety violations", g.safetyBaseline === null ? null : `${g.safetyBaseline} → ${g.safetyCandidate}`],
            ["Skill", e.sleep.skillPath ? code(e.sleep.skillPath) : null],
            ["Adoption PR", e.sleep.adoptionPr ? `#${e.sleep.adoptionPr}` : null],
        ]),
    );
}

function viewAspire() {
    const d = got("aspire");
    if (!d) return loading();
    if (d.error) return errorBox(d.error);
    const a = d.value.aspire;
    if (!a.configured) {
        return empty("The Aspire Dashboard is not running.", "Run ", code("ci-lab dashboard up"), " to start it on loopback; local traces under ", code("telemetry/"), " still show in the Traces view without it.");
    }
    return [
        section(
            "Aspire Dashboard",
            kv([
                ["Status", a.reachable ? badge("reachable", "ok") : badge(a.error ?? "unreachable", "bad")],
                ["Process", a.pidAlive === null ? null : a.pidAlive ? "running" : badge("not running", "warn")],
                ["Version", a.version],
                ["Started", a.started],
                ["UI", a.uiUrl ? code(a.uiUrl) : null],
                ["OTLP endpoint", a.otlpUrl ? code(a.otlpUrl) : null],
            ]),
            a.nonLoopbackIgnored ? h("p", { class: "error", text: "Non-loopback dashboard URLs in the state file were ignored." }) : null,
            h(
                "div",
                { class: "toolbar" },
                a.loginAvailable ? h("button", { type: "button", class: "btn primary", on: { click: () => openAspire(null) } }, "Open dashboard") : null,
            ),
            loginFallback(),
            !a.reachable ? h("p", { class: "muted" }, "If the dashboard was stopped, run ", code("ci-lab dashboard up"), " again.") : null,
        ),
        a.resources?.length ? section("Resources", table(["Name", "Traces"], a.resources.map((r) => [r.displayName || r.name, r.hasTraces ? "yes" : "no"]))) : null,
    ];
}

const loading = () => h("p", { class: "muted", role: "status", text: "Loading…" });
const errorBox = (msg) => h("p", { class: "error", role: "alert", text: `Could not load data: ${msg}` });

// ------------------------------------------------------------------ shell
const RENDER = { overview: viewOverview, live: viewLive, experiment: viewExperiment, traces: viewTraces, evals: viewEvals, sleep: viewSleep, aspire: viewAspire };

function renderTabs() {
    const nav = document.getElementById("tabs");
    nav.replaceChildren(
        ...VIEWS.map(([id, label]) =>
            h(
                "button",
                {
                    type: "button",
                    class: `tab ${state.ui.view === id ? "active" : ""}`.trim(),
                    "aria-current": state.ui.view === id ? "page" : undefined,
                    on: {
                        click: () => {
                            if (state.ui.view === id) {
                                if (id === "experiment" && state.ui.experimentId) navigate({ experimentId: null });
                                else if (id === "traces" && state.ui.traceId) navigate({ traceId: null });
                                return;
                            }
                            navigate({ view: id });
                            scheduleLoad(0);
                        },
                    },
                },
                label,
            ),
        ),
    );
}

let lastLoadedKey = null;
function render() {
    renderTabs();
    const key = JSON.stringify([state.ui.view, state.ui.experimentId, state.ui.traceId]);
    if (key !== lastLoadedKey) {
        lastLoadedKey = key;
        scheduleLoad(0);
    }
    const main = document.getElementById("main");
    const scroll = main.scrollTop;
    let content;
    try {
        content = RENDER[state.ui.view]?.() ?? empty("Unknown view.");
    } catch (e) {
        content = errorBox(e.message);
    }
    main.replaceChildren(...[content].flat(Infinity).filter(Boolean));
    main.scrollTop = scroll;
}

function setConn(text, kind) {
    const el = document.getElementById("conn");
    el.textContent = text;
    el.className = `conn ${kind}`;
}

function connect() {
    const es = new EventSource("/events");
    es.addEventListener("hello", (e) => {
        const msg = JSON.parse(e.data);
        setConn("live", "ok");
        applyUi(msg.ui, true);
        state.version = msg.version;
        scheduleLoad(0);
    });
    es.addEventListener("changed", (e) => {
        const msg = JSON.parse(e.data);
        if (msg.version !== state.version) {
            state.version = msg.version;
            scheduleLoad(200);
        }
    });
    es.addEventListener("ui", (e) => applyUi(JSON.parse(e.data), false));
    es.addEventListener("ping", () => setConn("live", "ok"));
    es.onerror = () => setConn("reconnecting…", "warn");
}

function applyUi(ui, force) {
    if (!ui || (!force && typeof ui.seq === "number" && ui.seq <= state.ui.seq)) return;
    state.ui = { ...state.ui, ...ui };
    if (ui.traceId) state.openSpans.clear();
    render();
    document.getElementById("main").focus({ preventScroll: true });
}

setInterval(() => {
    for (const el of document.querySelectorAll(".age[data-epoch]")) el.textContent = fmtAge(serverNow() - Number(el.dataset.epoch));
}, 1000);

document.getElementById("refresh").addEventListener("click", async () => {
    setConn("refreshing…", "info");
    try {
        await post("/api/refresh");
        setConn("live", "ok");
        scheduleLoad(0);
    } catch (e) {
        setConn(e.message, "bad");
    }
});

render();
connect();
