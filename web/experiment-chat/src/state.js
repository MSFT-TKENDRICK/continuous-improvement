// Tolerant readers for the experiment_designer shared state ({draft, launches}).

const pick = (obj, ...keys) => {
  for (const k of keys) if (obj && obj[k] !== undefined && obj[k] !== null) return obj[k];
  return undefined;
};

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

/** Normalise a draft into display fields; returns null when there is no draft. */
export function normalizeDraft(draft) {
  if (!isObj(draft)) return null;
  const spec = isObj(draft.spec) ? { ...draft.spec, ...draft } : draft;
  const hyper = pick(spec, "hyper", "hyperparameters", "hyperparams");
  const estimate = isObj(spec.estimate) ? spec.estimate : {};
  const evals = pick(estimate, "evaluations") ?? pick(spec, "estimated_evaluations", "estimatedEvaluations", "evaluations");
  return {
    cid: str(pick(spec, "cid", "campaign_id", "campaignId", "id")),
    domain: str(pick(spec, "domain")) ?? "harness",
    target: str(pick(spec, "target")),
    rounds: num(pick(spec, "rounds", "n_rounds")),
    estimatedEvaluations: num(evals),
    formula: str(pick(estimate, "formula")),
    warnings: (Array.isArray(spec.warnings) ? spec.warnings : []).map(str).filter(Boolean),
    rationale: str(pick(spec, "rationale", "reason", "summary")),
    hyper: isObj(hyper) ? hyper : null,
  };
}

/** Normalise the launches list; drops entries without a campaign id. */
export function normalizeLaunches(launches) {
  if (!Array.isArray(launches)) return [];
  return launches.filter(isObj).map((l) => ({
    cid: str(pick(l, "cid", "campaign_id", "campaignId")),
    target: str(pick(l, "target")),
    status: str(pick(l, "status")) ?? "unknown",
    pid: num(pick(l, "pid")),
    log: str(pick(l, "log")),
    runUrl: safeUrl(pick(l, "run_url", "runUrl")),
  })).filter((l) => l.cid);
}

/** Flatten a hyperparameter object into [key, displayValue] rows, skipping unset values. */
export function hyperRows(hyper) {
  if (!isObj(hyper)) return [];
  return Object.entries(hyper)
    .filter(([, v]) => v !== null && v !== undefined && !(isObj(v) && Object.keys(v).length === 0))
    .map(([k, v]) => [k, display(v)]);
}

/** Pull the launch_campaign arguments out of an AG-UI tool_call interrupt. */
export function interruptCall(interrupt) {
  const fc = interrupt?.metadata?.agent_framework?.function_call;
  let args = fc?.arguments ?? {};
  if (typeof args === "string") {
    try { args = JSON.parse(args); } catch { args = {}; }
  }
  return { name: str(fc?.name) ?? "tool", args: isObj(args) ? args : {} };
}

// Only GitHub https links are rendered (workflow run pages); anything else is dropped.
function safeUrl(v) {
  const s = str(v);
  return s && /^https:\/\/github\.com\/[^\s"<>]+$/.test(s) ? s : undefined;
}

function display(v) {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.map(display).join(", ");
  if (isObj(v)) return JSON.stringify(v);
  return String(v);
}

function str(v) {
  return v === undefined || v === null || v === "" ? undefined : String(v);
}

function num(v) {
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : undefined;
}
