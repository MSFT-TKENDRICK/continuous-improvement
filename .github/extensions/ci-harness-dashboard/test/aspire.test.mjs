import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { AspireClient, statePath } from "../lib/aspire.mjs";
import { API_KEY, BROWSER_TOKEN, OTLP_KEY, mockAspire, writeState, scratch } from "./helpers.mjs";

const tmp = scratch("aspire");
after(() => fs.rmSync(tmp, { recursive: true, force: true }));

const noSecrets = (v) => {
    const s = JSON.stringify(v);
    for (const secret of [API_KEY, BROWSER_TOKEN, OTLP_KEY]) assert.ok(!s.includes(secret), `leaked ${secret}`);
};

test("statePath honours CI_DASHBOARD_STATE and defaults to ~/.ci-lab/dashboard.json", () => {
    assert.equal(statePath({ CI_DASHBOARD_STATE: path.join(tmp, "x.json") }, "H"), path.join(tmp, "x.json"));
    assert.equal(statePath({}, path.join(tmp, "home")), path.join(tmp, "home", ".ci-lab", "dashboard.json"));
});

test("no state file → not configured, with a hint", async () => {
    const c = new AspireClient({ env: { CI_DASHBOARD_STATE: path.join(tmp, "missing.json") } });
    const st = await c.status();
    assert.equal(st.configured, false);
    assert.match(st.hint, /ci-lab dashboard up/);
    assert.equal(await c.traces(), null);
    assert.equal(await c.loginUrl(null), null);
});

test("status, traces and trace use x-api-key and never leak secrets", async () => {
    const m = await mockAspire();
    try {
        const c = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, m.url) } });
        const st = await c.status();
        noSecrets(st);
        assert.equal(st.configured, true);
        assert.equal(st.reachable, true);
        assert.equal(st.version, "13.6.1");
        assert.equal(st.uiUrl, m.url);
        assert.equal(st.pidAlive, true);
        assert.equal(st.loginAvailable, true);
        assert.deepEqual(st.resources, [{ name: "ci-lab.spike", displayName: "ci-lab.spike", hasTraces: true }]);
        const tr = await c.traces();
        assert.equal(tr.data.resourceSpans.length, 1);
        assert.ok(await c.trace("0af7651916cd43dd8448eb211c80319c"));
        assert.equal(await c.trace("ffffffffffffffffffffffffffffffff"), null, "404 → null");
        assert.equal(c.lastError, "not found");
        assert.equal(await c.trace("../../etc"), null, "invalid id never requested");
        assert.ok(m.seen.every((s) => s.key === API_KEY));
        assert.ok(!m.seen.some((s) => s.url.includes("..")));
        assert.equal(await c.deepLink("0AF7651916CD43DD8448EB211C80319C"), `${m.url}/traces/detail/0af7651916cd43dd8448eb211c80319c`);
        noSecrets(await c.deepLink("0af7651916cd43dd8448eb211c80319c"));
    } finally {
        await m.close();
    }
});

test("responses are cached for cacheMs", async () => {
    const m = await mockAspire();
    try {
        let t = 0;
        const c = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, m.url) }, now: () => t, cacheMs: 1000 });
        await c.traces();
        await c.traces();
        assert.equal(m.seen.length, 1);
        t = 2000;
        await c.traces();
        assert.equal(m.seen.length, 2);
    } finally {
        await m.close();
    }
});

test("rejected key maps to our own error wording", async () => {
    const m = await mockAspire({ key: "other" });
    try {
        const c = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, m.url) } });
        const st = await c.status();
        assert.equal(st.reachable, false);
        assert.equal(st.error, "API key rejected");
        noSecrets(st);
    } finally {
        await m.close();
    }
});

test("unreachable dashboard → 'unreachable', no request details", async () => {
    const m = await mockAspire();
    await m.close();
    const c = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, m.url) }, timeoutMs: 1000 });
    const st = await c.status();
    assert.equal(st.reachable, false);
    assert.equal(st.error, "unreachable");
});

test("loginUrl carries the browser token and a returnUrl; only for loopback UIs", async () => {
    const c = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, "http://localhost:18888") } });
    const u = new URL(await c.loginUrl("0af7651916cd43dd8448eb211c80319c"));
    assert.equal(u.origin, "http://localhost:18888");
    assert.equal(u.pathname, "/login");
    assert.equal(u.searchParams.get("t"), BROWSER_TOKEN);
    assert.equal(u.searchParams.get("returnUrl"), "/traces/detail/0af7651916cd43dd8448eb211c80319c");
    assert.equal(new URL(await c.loginUrl(null)).searchParams.get("returnUrl"), null);

    const remote = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, "https://evil.example.com") } });
    assert.equal(await remote.loginUrl(null), null, "never hand a token to a non-loopback URL");
    const st = await remote.status();
    assert.equal(st.uiUrl, null);
    assert.equal(st.nonLoopbackIgnored, true);
    noSecrets(st);
});

test("oversized or malformed state files are ignored", async () => {
    const big = path.join(tmp, "big.json");
    fs.writeFileSync(big, JSON.stringify({ pad: "x".repeat(70 * 1024) }));
    assert.equal((await new AspireClient({ env: { CI_DASHBOARD_STATE: big } }).status()).configured, false);
    const bad = path.join(tmp, "bad.json");
    fs.writeFileSync(bad, "{not json");
    assert.equal((await new AspireClient({ env: { CI_DASHBOARD_STATE: bad } }).status()).configured, false);
});
