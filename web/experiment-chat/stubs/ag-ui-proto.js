// Stub: the canvas proxy only speaks SSE; the protobuf transport is not bundled.
export const AGUI_MEDIA_TYPE = "application/vnd.ag-ui.event+proto";
export function encode() { throw new Error("AG-UI protobuf transport is not available in this bundle"); }
export function decode() { throw new Error("AG-UI protobuf transport is not available in this bundle"); }
