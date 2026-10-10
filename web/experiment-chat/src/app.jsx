import { Component, StrictMode, useMemo } from "react";
import {
  CopilotChat,
  CopilotKitProvider,
  HttpAgent,
  useAgent,
  useConfigureSuggestions,
  useInterrupt,
} from "@copilotkit/react-core/v2";
import { ApprovalCard, DraftCard, LaunchList } from "./cards.jsx";
import { normalizeLaunches } from "./state.js";

export const AGENT_ID = "experiment_designer";

const SUGGESTIONS = [
  { title: "Improve the harness", message: "Draft a 1-round, 2-arm fake campaign to improve the harness." },
  { title: "Inspect harness metrics", message: "Summarize current harness campaigns, metrics, and traces." },
  { title: "Explain A/A calibration", message: "Explain A/A calibration and when I need it." },
];

const LABELS = {
  chatInputPlaceholder: "Describe the harness improvement to test…",
  welcomeMessageText: "Formulate a harness RRSI campaign as an OES experiment. Launching always asks for approval.",
  modalHeaderTitle: "Experiment chat",
  chatDisclaimerText: "Drafts are proposals; nothing launches without your approval.",
};

function ExperimentPanel({ onShowCampaign }) {
  const { agent } = useAgent({ agentId: AGENT_ID });
  const state = agent?.state ?? {};
  useConfigureSuggestions({ suggestions: SUGGESTIONS, available: "always", consumerAgentId: AGENT_ID });
  useInterrupt({
    agentId: AGENT_ID,
    renderInChat: true,
    render: ({ interrupt, interrupts, resolve }) => (
      <ApprovalCard
        interrupt={interrupt}
        count={interrupts.length}
        onDecide={(approved) => resolve({ approved }, interrupt?.id)}
      />
    ),
  });
  return (
    <div className="xc-layout">
      <section className="xc-chat" aria-label="Experiment designer chat">
        <CopilotChat agentId={AGENT_ID} labels={LABELS} />
      </section>
      <aside className="xc-side" aria-label="Campaign draft and launches">
        <DraftCard draft={state.draft} />
        <LaunchList launches={normalizeLaunches(state.launches)} onShowCampaign={onShowCampaign} />
      </aside>
    </div>
  );
}

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="xc-error" role="alert">
        <strong>Experiment chat failed.</strong>
        <span>{String(this.state.error?.message ?? this.state.error)}</span>
        <button type="button" onClick={() => this.setState({ error: null })}>Retry</button>
      </div>
    );
  }
}

export function App({ token, endpoint, onShowCampaign, onError }) {
  // HttpAgent posts only to the same-origin canvas proxy, which holds the backend secret.
  const agents = useMemo(() => ({
    [AGENT_ID]: new HttpAgent({ url: endpoint, agentId: AGENT_ID, headers: { "x-ci-token": token } }),
  }), [endpoint, token]);
  return (
    <ErrorBoundary>
      <CopilotKitProvider agents__unsafe_dev_only={agents} enableInspector={false} onError={onError}>
        <ExperimentPanel onShowCampaign={onShowCampaign} />
      </CopilotKitProvider>
    </ErrorBoundary>
  );
}

export function Root(props) {
  return (
    <StrictMode>
      <App {...props} />
    </StrictMode>
  );
}