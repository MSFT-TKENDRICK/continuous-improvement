// Split point: ReactDOM in its own chunk. A static re-export makes esbuild emit named ESM exports;
// a bare dynamic import of the CommonJS package would only expose `default` once code-split.
export { createRoot } from "react-dom/client";
