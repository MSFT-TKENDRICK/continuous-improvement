import { useState } from "react";
import { hyperRows, interruptCall, normalizeDraft } from "./state.js";

function Field({ label, children }) {
  return (
    <div className="xc-field">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export function DraftCard({ draft }) {
  const d = normalizeDraft(draft);
  if (!d) {
    return (
      <div className="xc-card xc-card--empty">
        <h3>Draft</h3>
        <p>No campaign drafted yet. Ask the designer for one.</p>
      </div>
    );
  }
  const rows = hyperRows(d.hyper);
  return (
    <div className="xc-card" data-testid="xc-draft">
      <h3>Draft <code>{d.cid ?? "(unnamed)"}</code></h3>
      <dl className="xc-fields">
        <Field label="Domain">{d.domain}</Field>
        <Field label="Target">{d.target ?? "—"}</Field>
        <Field label="Rounds">{d.rounds ?? "—"}</Field>
        <Field label="Est. evaluations"><span title={d.formula}>{d.estimatedEvaluations ?? "—"}</span></Field>
      </dl>
      {d.formula && <p className="xc-muted xc-formula">{d.formula}</p>}
      {d.warnings.length > 0 && (
        <ul className="xc-warnings" aria-label="Draft warnings">
          {d.warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}
      {rows.length > 0 && (
        <table className="xc-hyper">
          <caption>Hyperparameters</caption>
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      {d.rationale && <p className="xc-rationale">{d.rationale}</p>}
    </div>
  );
}

export function ApprovalCard({ interrupt, count, onDecide }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const { name, args } = interruptCall(interrupt);
  const decide = async (approved) => {
    setBusy(true);
    setError(null);
    try {
      await onDecide(approved);
    } catch (e) {
      setError(String(e?.message ?? e));
      setBusy(false);
    }
  };
  const entries = Object.entries(args);
  return (
    <div className="xc-card xc-approval" role="group" aria-label="Launch approval" data-testid="xc-approval">
      <h3>Approval required</h3>
      <p>{interrupt?.message ?? `Approve running ${name}?`}</p>
      {entries.length > 0 && (
        <dl className="xc-fields">
          {entries.map(([k, v]) => (
            <Field key={k} label={k}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</Field>
          ))}
        </dl>
      )}
      {count > 1 && <p className="xc-muted">{count - 1} more pending approval(s) follow.</p>}
      <div className="xc-actions">
        <button type="button" className="xc-btn xc-btn--primary" disabled={busy} onClick={() => decide(true)}>Approve</button>
        <button type="button" className="xc-btn" disabled={busy} onClick={() => decide(false)}>Reject</button>
      </div>
      {error && <p className="xc-error-text" role="alert">{error}</p>}
    </div>
  );
}

export function LaunchList({ launches, onShowCampaign }) {
  if (!launches.length) return null;
  return (
    <div className="xc-card" data-testid="xc-launches">
      <h3>Launches</h3>
      <ul className="xc-launches">
        {launches.map((l, i) => (
          <li key={`${l.cid}-${i}`}>
            <span><code>{l.cid}</code> · {l.target ?? "?"} · <span className={`xc-status xc-status--${l.status}`}>{l.status}</span></span>
            {l.runUrl && <a href={l.runUrl} target="_blank" rel="noopener noreferrer">workflow run</a>}
            <button type="button" className="xc-btn" onClick={() => onShowCampaign(l.cid)}>Open in Experiment view</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
