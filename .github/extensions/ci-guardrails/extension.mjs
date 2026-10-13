// ci-guardrails: Copilot CLI extension enforcing the dev-loop guardrails in policy.mjs at PreToolUse.
// Never write to stdout here (it carries JSON-RPC); use session.log.

import { execFileSync } from "node:child_process";
import { joinSession } from "@github/copilot-sdk/extension";
import { evaluate, toHookOutput } from "./policy.mjs";

function currentBranch(cwd) {
  try {
    return execFileSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], {
      cwd: cwd || process.cwd(),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5000,
    }).trim();
  } catch {
    return undefined;
  }
}

const session = await joinSession({
  hooks: {
    onPreToolUse: async (input) => {
      const decision = evaluate(input, {
        env: process.env,
        currentBranch: () => currentBranch(input.workingDirectory),
      });
      if (decision) {
        await session.log(`ci-guardrails: denied ${input.toolName} (${decision.rule})`, { level: "warning" });
      }
      return toHookOutput(decision);
    },
  },
});
