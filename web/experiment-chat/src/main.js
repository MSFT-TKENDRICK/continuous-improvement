// Entry point loaded by the canvas (ui/chat/chat.js). Kept tiny: React, the AG-UI client and
// CopilotKit load as separate same-origin chunks so every file stays well under 1 MB.
function loadModules() {
  // The extra imports are split points only: each becomes its own chunk shared with app.jsx.
  return Promise.all([
    import("./split/dom.js"),
    import("./app.jsx"),
    import("./split/agui.js"),
    import("./split/core.js"),
  ]);
}

// CopilotKit switches palettes on a `.dark` ancestor; mirror the canvas theme (data-color-mode on
// <html>, else the OS preference) onto the container and follow changes.
function syncTheme(container) {
  const html = document.documentElement;
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");
  const apply = () => {
    const mode = html.getAttribute("data-color-mode");
    container.classList.toggle("dark", mode ? mode === "dark" : Boolean(media?.matches));
  };
  apply();
  const observer = new MutationObserver(apply);
  observer.observe(html, { attributes: true, attributeFilter: ["data-color-mode"] });
  media?.addEventListener?.("change", apply);
  return () => {
    observer.disconnect();
    media?.removeEventListener?.("change", apply);
  };
}

/**
 * Mount the experiment chat into `container`.
 * Options: token (canvas session token), endpoint (default "/agui"), onShowCampaign(cid), onError(message).
 * Resolves to an unmount function.
 */
export async function mount(container, { token, endpoint = "/agui", onShowCampaign, onError } = {}) {
  if (!container) throw new Error("mount: container is required");
  const [{ createRoot }, { Root }] = await loadModules();
  container.classList.add("xc-root");
  const stopTheme = syncTheme(container);
  const root = createRoot(container);
  const reportError = typeof onError === "function"
    ? (event) => onError(String(event?.error?.message ?? event?.code ?? "chat error"))
    : undefined;
  root.render(
    Root({
      token: String(token ?? ""),
      endpoint,
      onShowCampaign: typeof onShowCampaign === "function" ? onShowCampaign : () => {},
      onError: reportError,
    }),
  );
  return () => {
    stopTheme();
    root.unmount();
  };
}
