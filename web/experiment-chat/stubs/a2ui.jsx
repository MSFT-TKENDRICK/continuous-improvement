// Stub: A2UI generative UI is never enabled here (no runtime, no catalog). Keeps its renderer,
// which injects <style> elements at runtime, out of the bundle under the canvas CSP.
export const A2UI_SCHEMA_CONTEXT_DESCRIPTION = "";
export const DEFAULT_SURFACE_ID = "default";
export const ROOT_COMPONENT_ID = "root";
export const viewerTheme = {};
export class Catalog {}
export function A2UIProvider({ children }) { return children ?? null; }
export function A2UIRenderer() { return null; }
export function buildCatalogContextValue() { return ""; }
export function extractCatalogComponentSchemas() { return []; }
export function filterCatalog(catalog) { return catalog; }
export function initializeDefaultCatalog() {}
export function injectStyles() {}
export function useA2UIActions() { return {}; }
export function useA2UIError() { return null; }
export const defaultTheme = {};
