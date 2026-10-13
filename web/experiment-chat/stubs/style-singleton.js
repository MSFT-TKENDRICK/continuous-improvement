// Stub: react-style-singleton injects <style> elements (blocked by the canvas CSP style-src 'self').
// It only backs the scroll lock of modal menus, which the chat can live without.
export function styleSingleton() { return function Style() { return null; }; }
export function styleHookSingleton() { return function useStyle() {}; }
export function stylesheetSingleton() { return { add() {}, remove() {} }; }
