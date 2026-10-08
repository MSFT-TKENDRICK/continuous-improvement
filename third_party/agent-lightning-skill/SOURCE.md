# Vendored: Agent Lightning skill

- Upstream: https://github.com/microsoft/agent-lightning (MIT, see `LICENSE`)
- Tag: `v1.0.2` (tag object `ec89537a3b1ac861de3a1112717c46c86f91ea6c`)
- Commit: `d381995396274039f2bb1cbe5ff42ac8067f4e47`
- Source dir: `skills/agent-lightning/` at that commit
  (https://github.com/microsoft/agent-lightning/tree/d381995396274039f2bb1cbe5ff42ac8067f4e47/skills/agent-lightning)
- Fetched from `https://raw.githubusercontent.com/microsoft/agent-lightning/d381995396274039f2bb1cbe5ff42ac8067f4e47/<path>`
  (`gh api` returned 403 because of SAML enforcement; anonymous raw access works).

| File | Bytes | SHA-256 (as fetched) |
|---|---|---|
| `skills/agent-lightning/SKILL.md` | 6641 | `c5e876cb68fffbe94c949c99a39c706c35b774961239b69089ea25055a09ae0a` |
| `skills/agent-lightning/.claude-plugin/plugin.json` | 405 | `bdd4f6f09304ce555b6f062539e911b957b262b37fa23fa518474a3940bc0d45` |
| `LICENSE` (repo root) | 1046 | `8f71659370c5268d9a1dc962a46232540e8fca63462586d8efaa95aab492a208` |

The skill directory upstream contains only these two files; `SKILL.md` references no
other files. Files are unmodified. The skill mentions `/artifacts/cost_budget.json`; in
ci_lab the proposer's budget comes from its brief instead (see
`src/ci_lab/meta/specs/prompts/proposer.md`).

Used by the proposer meta agent via `skills_paths` (`third_party/agent-lightning-skill/skills`).
To update: re-fetch the same paths at a new tag's commit, update this table, re-run
`tests/ci_lab/meta`.
