// Shared test helpers (not a test file).
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const FIX = path.join(here, "fixtures");
export const REPO = path.join(FIX, "repo");
export const API_KEY = "api-key-SECRET-123";
export const BROWSER_TOKEN = "browser-token-SECRET-456";
export const OTLP_KEY = "otlp-key-SECRET-789";
const ASPIRE_FIX = path.join(FIX, "aspire");

/** Scratch dir inside the test folder (never the OS temp dir). */
export function scratch(prefix) {
    return fs.mkdtempSync(path.join(here, `.tmp-${prefix}-`));
}
/** Minimal stand-in for the Aspire Dashboard telemetry API (shapes from the 13.6 spike). */
export async function mockAspire({ key = API_KEY } = {}) {
    const seen = [];
    const server = http.createServer((req, res) => {
        seen.push({ url: req.url, key: req.headers["x-api-key"] });
        if (req.headers["x-api-key"] !== key) {
            res.writeHead(401);
            return res.end();
        }
        const send = (file) => {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(fs.readFileSync(path.join(ASPIRE_FIX, file)));
        };
        if (req.url === "/api/telemetry/resources") return send("resources.json");
        if (req.url === "/api/telemetry/traces") return send("traces.json");
        if (req.url === "/api/telemetry/traces/0af7651916cd43dd8448eb211c80319c") return send("traces.json");
        res.writeHead(404, { "content-type": "application/json" });
        res.end("{}");
    });
    await new Promise((r) => server.listen(0, "127.0.0.1", r));
    const url = `http://127.0.0.1:${server.address().port}`;
    return { url, seen, close: () => new Promise((r) => server.close(r)) };
}

export function writeState(dir, url, extra = {}) {
    const p = path.join(dir, `dashboard-${Math.random().toString(16).slice(2)}.json`);
    fs.writeFileSync(
        p,
        JSON.stringify({ pid: process.pid, version: "13.6.1", ui_url: url, api_url: url, otlp_url: "http://127.0.0.1:4318", started: "2026-01-01T00:00:00Z", browser_token: BROWSER_TOKEN, otlp_key: OTLP_KEY, api_key: API_KEY, ...extra }),
    );
    return p;
}

