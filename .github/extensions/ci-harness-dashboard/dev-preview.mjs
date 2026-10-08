#!/usr/bin/env node
// Dev preview: serve the dashboard for a repo root in a normal browser, without the Copilot SDK.
//   node .github/extensions/ci-harness-dashboard/dev-preview.mjs [--repo <path>] [--port N] [--view <view>]
// Loopback only. The Host check accepts 127.0.0.1:<port> and localhost:<port>.
import path from "node:path";
import { parseArgs } from "node:util";
import { createDashboard, findRepoRoot, existsDir } from "./lib/canvas.mjs";
import { VIEWS } from "./lib/server.mjs";

class PreviewError extends Error {
    constructor(code, message) {
        super(message);
        this.code = code;
    }
}

const { values } = parseArgs({
    options: {
        repo: { type: "string" },
        port: { type: "string" },
        view: { type: "string" },
        help: { type: "boolean", short: "h" },
    },
});
if (values.help) {
    process.stdout.write(`Usage: node dev-preview.mjs [--repo <path>] [--port N] [--view ${VIEWS.join("|")}]\n`);
    process.exit(0);
}
const start = path.resolve(values.repo ?? process.cwd());
if (!existsDir(start)) {
    process.stderr.write(`not a directory: ${start}\n`);
    process.exit(2);
}
const repoRoot = values.repo ? start : await findRepoRoot(start);
const port = values.port ? Number(values.port) : 0;
if (!Number.isInteger(port) || port < 0 || port > 65535) {
    process.stderr.write("--port must be an integer 0-65535\n");
    process.exit(2);
}

const dashboard = createDashboard({
    CanvasError: PreviewError,
    log: (msg, level = "info") => process.stderr.write(`[${level}] ${msg}\n`),
    port,
});
const res = await dashboard.open({ instanceId: "dev-preview", input: { repoRoot, ...(values.view ? { view: values.view } : {}) } });
process.stdout.write(`${res.title}\n  ${res.url}\n  status: ${res.status}\nCtrl+C to stop.\n`);

const stop = () => dashboard.closeAll().finally(() => process.exit(0));
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
