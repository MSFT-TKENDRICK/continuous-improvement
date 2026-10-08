// Extension: ci-harness-dashboard — wiring only. All logic lives in ./lib (see docs/canvas.md).
// Never write to stdout here: it carries the JSON-RPC channel to the host.
import { joinSession, createCanvas, CanvasError } from "@github/copilot-sdk/extension";
import { createDashboard, CANVAS_ID, DISPLAY_NAME, DESCRIPTION, OPEN_INPUT_SCHEMA } from "./lib/canvas.mjs";

let session = null;
function log(message, level = "info") {
    try {
        const opts = level === "info" ? { ephemeral: true } : { level };
        session?.log?.(message, opts)?.catch?.(() => {});
    } catch {
        /* logging must never break the canvas */
    }
}

const dashboard = createDashboard({ CanvasError, log });

session = await joinSession({
    canvases: [
        createCanvas({
            id: CANVAS_ID,
            displayName: DISPLAY_NAME,
            description: DESCRIPTION,
            inputSchema: OPEN_INPUT_SCHEMA,
            actions: dashboard.actions,
            open: (ctx) => dashboard.open(ctx),
            onClose: (ctx) => dashboard.onClose(ctx),
        }),
    ],
});
