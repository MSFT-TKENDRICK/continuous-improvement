# Vendored schemas

## `openexperiment-0.1.0.schema.json` — Open Experiment Standard (OES) 0.1.0 core

| Field | Value |
|---|---|
| Source URL | https://www.openexperiment.org/schema/openexperiment-0.1.0.schema.json |
| Schema `$id` | `https://openexperiment.org/schema/openexperiment-0.1.0.schema.json` |
| Fetched | 2026-10-08 (HTTP 200, `Content-Type: application/schema+json`, `Last-Modified: Thu, 08 Oct 2026 03:17:59 GMT`, ETag `c34d8e13f34c9ece4b7014ca738e21eb`) |
| sha256 | `3c709822a2a29f7ce21c93aad31992cb65b219c8bdc106ea41aa4f3efe5eae15` |
| Size | 14983 bytes |
| Dialect | JSON Schema draft 2020-12 |
| Status | Official file, byte-for-byte as served (NOT reconstructed). Do not edit. |
| License | **Not stated.** openexperiment.org publishes no license (`/LICENSE` is 404, no license text on `/`, `/schema` or in the file) and the `github.com/openexperiment` org has no public repositories. The schema is published as an open, vendor-neutral standard for validation ("Documents that validate against this schema are conforming v0.1.0 documents"); it is vendored unmodified solely to validate our documents offline. Revisit if a license is published. |

Verify: `python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('schemas/oes/openexperiment-0.1.0.schema.json').read_bytes()).hexdigest())"`
(`tests/ci_lab/oes` pins this digest).

Notes on 0.1.0 (see `docs/oes.md`): `design.multipleTestingPolicy` has no "exploratory" value (we use
`custom` + the rrsi extension's `multipleTesting`), there is no non-inferiority margin, no cost metric
type and no supersedes/lineage field — those live in our extensions below.

## Extension schemas (ours; same terms as this repository)

| File | Envelope key | Purpose |
|---|---|---|
| `ext-com.microsoft.ci.rrsi.schema.json` | `extensions["com.microsoft.ci.rrsi"]` | RRSI campaign/round/calibration/confirmation record |
| `ext-com.microsoft.ci.sleep.schema.json` | `extensions["com.microsoft.ci.sleep"]` | SkillOpt-Sleep nightly consolidation record |

Both are draft 2020-12, versioned by their required `version` field (currently `0.1.0`), closed
(`additionalProperties: false`) so typos are caught by `ci-lab oes validate`.
