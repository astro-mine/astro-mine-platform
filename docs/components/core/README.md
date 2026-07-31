# astro-mine-core

**The narrow waist of [Astro-Mine](https://github.com/astro-mine).** The thin,
stable contract layer every other package and third-party plugin speaks to — small,
slow-changing, and guarded jealously against bloat. *"If only one thing is designed
superbly, it must be Astro-Mine-Core."*

> **Status:** Phase 1 — the v0.1 narrow-waist interfaces (SADF, Environment/Policy
> APIs, message schemas, registry) are implemented, consumable, and evolving
> **append-only** under the frozen `0.1.0` interface version; consumers now get Core by
> installing the platform rather than by pinning a tag. See the
> [charter](https://github.com/astro-mine/docs/blob/main/charter/Swarm_Exploration_ISRU_Orchestrator_OSS_Project.md),
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/core.md),
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## What Core defines (and only this)

- **SADF** — the Swarm Asset Description Format (`astro_mine.core.sadf`)
- **Environment API** — observe/act on a simulatable world (`astro_mine.core.env`)
- **Policy / Planner API** — compute and compose decisions (`astro_mine.core.policy`)
- **Message schemas** — the typed cross-component vocabulary, incl. `ObjectiveSpec`
  (`astro_mine.core.messages`, `astro_mine.core.objective`)
- **Plugin manifest & registry** — discovery, version negotiation, capability tags
  (`astro_mine.core.registry`, `astro_mine.core.compat`)
- **Units / frames / time** — SI, SPICE-backed frames, TDB/ET epochs (`astro_mine.core.units`)

Core contains **no** physics, solvers, learning, or UI, and depends only on
schema/serialization runtimes. If it can live in a plugin, it must not live in Core.

## Validate authored documents — `astro-mine core validate`

Core ships the **types and validators** for its hand-authored formats; the command line over them
lives in [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli), which depends on this
package. Together they check a document is valid without writing Python. Validation adds no new
dependency (`jsonschema` + `pydantic` + `pyyaml` are already Core deps) and works offline; cross-file
`$ref`s (e.g. a `mission` referencing the units vocabulary) resolve from the installed package.

```bash
# validate one or more documents; the format is inferred from a declared $schema, or named:
astro-mine core validate my-swarm.sadf.yaml --kind sadf
astro-mine core validate examples/mission/*.mission.yaml --kind mission
astro-mine core --json validate plan.json --kind plan      # machine-readable for CI/editors

# list the formats it knows and their schema $ids (derived from the registry — never stale):
astro-mine core kinds
```

The validatable document formats: `sadf`, `objective`, `mission`, `plan`, `manifest` (plugin
manifest), `policy_package`, `run_provenance`, and the message documents `action_batch` /
`contact_plan`. Errors name the JSON-Pointer path, the offending value, and what was expected;
a document is **never** validated against a guessed schema — an ambiguous one fails with the list of
known kinds. The `units` schema is a *referenced vocabulary*, not a standalone document, so it is not
a validate target.

The router `astro-mine validate` (RFC-0011 §6) reaches the same dispatcher without naming a
component: it asks each format owner "is this document yours?" and refuses to guess when none
claims it. Both spellings take the same arguments, with one deliberate difference in where `--json`
hangs — `astro-mine core --json validate …` (the component owns the flag) versus
`astro-mine validate --json …` (the router has no component to hang it from).

## Layout

```
src/astro_mine/core/   # the package (import path: astro_mine.core)
  sadf/ env/ policy/ messages/ objective/ registry/ units/ compat/
schemas/               # canonical .proto and JSON Schema sources (repo root; Core owns them)
tests/core/            # mirrors the package layout
```

## Development

Core is part of the [`astro-mine-platform`](../../../README.md) distribution — one repository, one
environment, one test suite. See [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup, then run
this component's suite with its own CI selection:

```bash
python scripts/test.py core
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full
workflow.

## Distribution

Core is not separately distributed. It installs as part of `astro-mine-platform`
(`docs/CONSOLIDATION_PLAN.md`) — there is no `astro-mine-core` wheel, no per-component tag series,
and no `[tool.uv.sources]` Git pin to copy. `import astro_mine.core` is unchanged, which is the
part that was load-bearing.

The **schemas** are still pinnable independently of the code, and that has not changed: the
content-addressed **schema bundle** is addressed by digest, so a Bench run reproduces against an
exact Core schema version. Build one locally with:

```bash
uv run python scripts/build_schema_bundle.py     # -> dist/schema-bundle/
```

Publishing it to GHCR was a per-repo `publish-schemas` workflow that did not come across the
consolidation; this repo's CI is `ci.yml` alone. Until that lands, the bundle is a local build and
the digest is the contract.

The interface version stays frozen at `0.1.0` and evolves **append-only**
([`docs/VERSIONING.md`](https://github.com/astro-mine/docs/blob/main/VERSIONING.md)) — the promise
Core makes to its consumers, which one repository does not change.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
