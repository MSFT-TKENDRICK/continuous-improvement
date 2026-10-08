#!/usr/bin/env node
// Browser check for the experiment chat under the dashboard's real CSP (not run in CI).
// Starts dev-preview.mjs with a chat backend, drives headless Microsoft Edge over the DevTools
// protocol (Node's global WebSocket, no extra packages), walks draft → launch → approve, and fails
// on any CSP violation, uncaught exception, console error or non-loopback request. A second pass
// blocks /chat/chat.js and checks the dashboard degrades to an error panel.
//
//   node web/experiment-chat/scripts/csp-check.mjs                         # fake backend (fixture replay)
//   node web/experiment-chat/scripts/csp-check.mjs --command '["uv","run","--no-sync","ci-lab","chat","serve","--profile","fake","--dry-run-launch"]' --repo <checkout with ci_lab.chat>
//   options: --edge <msedge path>  --out <dir> (default web/experiment-chat/.scratch/csp)  --scheme light|dark  --headed
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");
const { values: args } = parseArgs({
    options: {
        command: { type: "string" },
        repo: { type: "string" },
        edge: { type: "string" },
        out: { type: "string" },
        headed: { type: "boolean" },
        scheme: { type: "string" },
        timeout: { type: "string" },
    },
});
const OUT = path.resolve(args.out ?? path.join(HERE, "..", ".scratch", "csp"));
const STEP_MS = Number(args.timeout ?? 120000);
fs.mkdirSync(OUT, { recursive: true });
const say = (m) => process.stdout.write(`${m}\n`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findEdge() {
    const c = [
        args.edge,
        process.env.EDGE_PATH,
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "/usr/bin/microsoft-edge",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ].filter(Boolean);
    const hit = c.find((p) => fs.existsSync(p));
    if (!hit) throw new Error("Microsoft Edge not found; pass --edge <path>");
    return hit;
}

async function startPreview(scratch) {
    const command = args.command
        ? JSON.parse(args.command)
        : [process.execPath, path.join(HERE, "fake-agui.mjs")];
    if (args.command) command.push("--run-dir", path.join(scratch, "runs"), "--chat-dir", path.join(scratch, "chat"), "--ledger-dir", path.join(scratch, "ledger"));
    const env = { ...process.env, CI_CHAT_COMMAND: JSON.stringify(command) };
    delete env.CI_CHAT_DISABLED;
    const child = spawn(process.execPath, [path.join(REPO, ".github", "extensions", "ci-harness-dashboard", "dev-preview.mjs"), "--repo", path.resolve(args.repo ?? REPO), "--view", "chat"], {
        env,
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
    });
    const logs = [];
    child.stderr.on("data", (d) => logs.push(String(d)));
    const url = await new Promise((resolve, reject) => {
        let buf = "";
        const t = setTimeout(() => reject(new Error(`dev-preview did not start:\n${logs.join("")}`)), 30000);
        child.stdout.on("data", (d) => {
            buf += d;
            const m = buf.match(/http:\/\/127\.0\.0\.1:\d+\//);
            if (m) {
                clearTimeout(t);
                resolve(m[0]);
            }
        });
        child.once("exit", (code) => reject(new Error(`dev-preview exited ${code}:\n${logs.join("")}`)));
    });
    return { child, url, logs };
}

async function startEdge(profile) {
    fs.rmSync(profile, { recursive: true, force: true });
    const edge = spawn(
        findEdge(),
        [
            ...(args.headed ? [] : ["--headless=new"]),
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-sync",
            "--disable-features=msEdgeSidebarV2,EdgeCollections,msSmartScreenProtection",
            "--remote-debugging-port=0",
            `--user-data-dir=${profile}`,
            "--window-size=1280,900",
            "about:blank",
        ],
        { stdio: "ignore", windowsHide: true },
    );
    const portFile = path.join(profile, "DevToolsActivePort");
    for (let i = 0; i < 200 && !fs.existsSync(portFile); i++) await sleep(100);
    const port = Number(fs.readFileSync(portFile, "utf8").split(/\r?\n/)[0]);
    let page;
    for (let i = 0; i < 50 && !page; i++) {
        const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        page = list.find((t) => t.type === "page");
        if (!page) await sleep(100);
    }
    return { edge, port, wsUrl: page.webSocketDebuggerUrl };
}

class Cdp {
    constructor(ws) {
        this.ws = ws;
        this.id = 0;
        this.pending = new Map();
        this.handlers = new Map();
        ws.addEventListener("message", (e) => {
            const msg = JSON.parse(e.data);
            if (msg.id && this.pending.has(msg.id)) {
                const { resolve, reject } = this.pending.get(msg.id);
                this.pending.delete(msg.id);
                if (msg.error) reject(new Error(`${msg.error.message} ${msg.error.data ?? ""}`));
                else resolve(msg.result);
            } else if (msg.method) for (const h of this.handlers.get(msg.method) ?? []) h(msg.params);
        });
        ws.addEventListener("close", () => {
            for (const { reject } of this.pending.values()) reject(new Error("CDP socket closed"));
            this.pending.clear();
        });
    }
    static async connect(url) {
        const ws = new WebSocket(url);
        await new Promise((resolve, reject) => {
            ws.addEventListener("open", resolve, { once: true });
            ws.addEventListener("error", reject, { once: true });
        });
        return new Cdp(ws);
    }
    send(method, params = {}, ms = 30000) {
        const id = ++this.id;
        return new Promise((resolve, reject) => {
            const t = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`CDP ${method} timed out`));
            }, ms);
            this.pending.set(id, {
                resolve: (v) => (clearTimeout(t), resolve(v)),
                reject: (e) => (clearTimeout(t), reject(e)),
            });
            try {
                this.ws.send(JSON.stringify({ id, method, params }));
            } catch (e) {
                this.pending.get(id)?.reject(e);
                this.pending.delete(id);
            }
        });
    }
    on(method, fn) {
        if (!this.handlers.has(method)) this.handlers.set(method, []);
        this.handlers.get(method).push(fn);
    }
    async eval(expression) {
        const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
        if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
        return r.result.value;
    }
    async waitFor(expression, label, ms = STEP_MS) {
        const end = Date.now() + ms;
        while (Date.now() < end) {
            if (await this.eval(`(() => { try { return !!(${expression}); } catch { return false; } })()`)) return;
            await sleep(200);
        }
        throw new Error(`timed out waiting for ${label}`);
    }
    async screenshot(name) {
        const { data } = await this.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
        const file = path.join(OUT, `${name}.png`);
        fs.writeFileSync(file, Buffer.from(data, "base64"));
        return file;
    }
}

const CSP_HOOK = `window.__csp = [];
document.addEventListener("securitypolicyviolation", (e) => window.__csp.push({ directive: e.violatedDirective, blocked: e.blockedURI, sample: e.sample, source: e.sourceFile, line: e.lineNumber }));`;

function instrument(cdp, origin, report) {
    cdp.on("Runtime.consoleAPICalled", (p) => {
        if (p.type === "error" || p.type === "assert") report.consoleErrors.push(p.args.map((a) => a.value ?? a.description ?? "").join(" ").slice(0, 500));
    });
    cdp.on("Runtime.exceptionThrown", (p) => report.exceptions.push((p.exceptionDetails.exception?.description ?? p.exceptionDetails.text).slice(0, 500)));
    cdp.on("Log.entryAdded", ({ entry }) => {
        if (entry.level === "error" && !(entry.source === "network" && /favicon\.ico/.test(entry.url ?? ""))) report.logErrors.push(`${entry.source}: ${entry.text}`.slice(0, 500));
    });
    cdp.on("Audits.issueAdded", ({ issue }) => {
        if (issue.code === "ContentSecurityPolicyIssue") report.cspIssues.push(issue.details.contentSecurityPolicyIssueDetails ?? issue.details);
    });
    cdp.on("Network.requestWillBeSent", ({ request }) => {
        const u = request.url;
        report.requests++;
        if (!u.startsWith(origin) && !u.startsWith("data:") && !u.startsWith("blob:") && u !== "about:blank") report.externalRequests.push(u.slice(0, 300));
    });
}

async function enable(cdp) {
    for (const d of ["Page", "Runtime", "Log", "Audits", "Network"]) await cdp.send(`${d}.enable`);
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: CSP_HOOK });
}

async function sendChat(cdp, text) {
    await cdp.waitFor(`document.querySelector(".xc-root textarea")`, "chat input");
    await cdp.eval(`document.querySelector(".xc-root textarea").focus()`);
    await cdp.send("Input.insertText", { text });
    await sleep(150);
    for (const type of ["keyDown", "keyUp"]) await cdp.send("Input.dispatchKeyEvent", { type, key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 });
}

async function main() {
    const scratch = path.join(OUT, "run");
    fs.rmSync(scratch, { recursive: true, force: true });
    fs.mkdirSync(scratch, { recursive: true });
    const preview = await startPreview(scratch);
    const origin = preview.url.replace(/\/$/, "");
    say(`dev-preview: ${preview.url} (backend: ${args.command ? "custom command" : "fake-agui fixture replay"})`);
    const { edge, wsUrl } = await startEdge(path.join(OUT, `profile-${process.pid}`));
    const report = { url: preview.url, cspViolations: [], cspIssues: [], consoleErrors: [], exceptions: [], logErrors: [], externalRequests: [], requests: 0, screenshots: [], steps: [], fallback: null };
    let cdp;
    try {
        cdp = await Cdp.connect(wsUrl);
        instrument(cdp, origin, report);
        await enable(cdp);
        await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
        if (args.scheme) await cdp.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: args.scheme }] });

        await cdp.send("Page.navigate", { url: preview.url });
        await cdp.waitFor(`document.querySelector(".xc-layout") && document.querySelector(".xc-root textarea")`, "chat UI");
        await cdp.waitFor(`/Backend: (ready|starting)/.test(document.querySelector(".chat-status")?.textContent ?? "")`, "backend status");
        report.steps.push("chat rendered");
        report.screenshots.push(await cdp.screenshot("1-loaded"));

        await sendChat(cdp, "Please draft chat-demo: 2 arms, 1 round, local.");
        await cdp.waitFor(`document.querySelector("[data-testid=xc-draft]")?.textContent.includes("chat-demo")`, "draft card");
        report.steps.push("draft card from STATE_SNAPSHOT");
        report.screenshots.push(await cdp.screenshot("2-draft"));

        await sendChat(cdp, "launch chat-demo");
        await cdp.waitFor(`document.querySelector("[data-testid=xc-approval]")`, "approval card");
        report.steps.push("approval card from RUN_FINISHED interrupt");
        report.screenshots.push(await cdp.screenshot("3-approval"));

        await cdp.eval(`[...document.querySelectorAll("[data-testid=xc-approval] button")].find((b) => b.textContent.trim() === "Approve").click()`);
        await cdp.waitFor(`document.querySelector("[data-testid=xc-launches]")?.textContent.includes("chat-demo")`, "launch result");
        report.steps.push("approved resume produced a launch");
        report.screenshots.push(await cdp.screenshot("4-launched"));

        await cdp.eval(`[...document.querySelectorAll("[data-testid=xc-launches] button")].find((b) => /Experiment view/.test(b.textContent)).click()`);
        await cdp.waitFor(`document.querySelector(".tab.active")?.textContent === "Experiment" && !document.getElementById("main").hidden`, "experiment view");
        report.steps.push("launch button switched to the Experiment view");
        report.cspViolations.push(...(await cdp.eval("window.__csp")));

        // Pass 2: the bundle fails to load; the tab shows an error panel and other views still work.
        await cdp.send("Fetch.enable", { patterns: [{ urlPattern: "*/chat/chat.js*" }] });
        cdp.on("Fetch.requestPaused", (p) => cdp.send("Fetch.failRequest", { requestId: p.requestId, errorReason: "Failed" }).catch(() => {}));
        const before = report.logErrors.length;
        await cdp.send("Page.navigate", { url: preview.url });
        await cdp.waitFor(`[...document.querySelectorAll(".tab")].some((b) => b.textContent === "Chat")`, "tabs");
        await cdp.eval(`[...document.querySelectorAll(".tab")].find((b) => b.textContent === "Chat").click()`);
        await cdp.waitFor(`document.querySelector("#chat-host .chat-error")`, "bundle error panel");
        report.screenshots.push(await cdp.screenshot("5-bundle-blocked"));
        await cdp.eval(`[...document.querySelectorAll(".tab")].find((b) => b.textContent === "Overview").click()`);
        await cdp.waitFor(`!document.getElementById("main").hidden && document.getElementById("main").children.length > 0`, "overview after failure");
        report.fallback = "error panel shown; Overview renders";
        // The deliberately failed chat.js request is expected in this pass.
        report.logErrors.splice(before).filter((e) => !/chat\/chat\.js|Failed to fetch dynamically imported module|net::ERR_FAILED/.test(e)).forEach((e) => report.logErrors.push(e));
        report.consoleErrors = report.consoleErrors.filter((e) => !/Failed to fetch dynamically imported module/.test(e));
        report.exceptions = report.exceptions.filter((e) => !/Failed to fetch dynamically imported module/.test(e));
        report.cspViolations.push(...(await cdp.eval("window.__csp")));
    } catch (e) {
        report.error = String(e?.stack ?? e);
        try {
            report.screenshots.push(await cdp?.screenshot("error"));
        } catch {
            /* browser gone */
        }
    } finally {
        try {
            await cdp?.send("Browser.close", {}, 5000);
        } catch {
            /* fall through to kill */
        }
        cdp?.ws.close();
        const exited = edge.exitCode !== null || (await Promise.race([new Promise((r) => edge.once("exit", () => r(true))), sleep(5000).then(() => false)]));
        if (!exited) edge.kill();
        preview.child.stdin.end();
        preview.child.kill();
        await sleep(1000);
        try {
            fs.rmSync(path.join(OUT, `profile-${process.pid}`), { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
        } catch {
            /* Edge helpers may linger briefly; the profile lives in the ignored .scratch dir */
        }
    }
    report.backendLog = preview.logs.join("").split(/\r?\n/).filter((l) => /fake-agui|chat/i.test(l)).slice(-20);
    fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));
    const bad = report.cspViolations.length + report.cspIssues.length + report.exceptions.length + report.consoleErrors.length + report.logErrors.length + report.externalRequests.length + (report.error ? 1 : 0);
    say(`steps: ${report.steps.join(" → ")}`);
    say(`fallback: ${report.fallback ?? "not checked"}`);
    say(`requests: ${report.requests} (external: ${report.externalRequests.length})`);
    say(`CSP violations: ${report.cspViolations.length}, CSP issues: ${report.cspIssues.length}, exceptions: ${report.exceptions.length}, console errors: ${report.consoleErrors.length}, log errors: ${report.logErrors.length}`);
    say(`screenshots: ${report.screenshots.filter(Boolean).join(", ")}`);
    say(`report: ${path.join(OUT, "report.json")}`);
    if (report.error) say(`error: ${report.error}`);
    process.exit(bad ? 1 : 0);
}

await main();
