# astro-mine-core

**The narrow waist of [Astro-Mine](https://github.com/astro-mine).** The thin,
stable contract layer every other package and third-party plugin speaks to — small,
slow-changing, and guarded jealously against bloat. *"If only one thing is designed
superbly, it must be Astro-Mine-Core."*

> **Status:** Phase 1 — the v0.1 narrow-waist interfaces (SADF, Environment/Policy
> APIs, message schemas, registry) are implemented, consumable, and evolving
> **append-only** under the frozen `0.1.0` interface version; downstream repos pin
> Core by Git tag. See the
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

## Validate authored documents — `astro-mine-core validate`

Core ships the **types and validators** for its hand-authored formats. `astro-mine-core validate`
is the shell over them — check a document is valid without writing Python. It adds no new
dependency (`jsonschema` + `pydantic` + `pyyaml` are already Core deps) and works offline; cross-file
`$ref`s (e.g. a `mission` referencing the units vocabulary) resolve from the installed package.

```bash
# validate one or more documents; the format is inferred from a declared $schema, or named:
astro-mine-core validate my-swarm.sadf.yaml --kind sadf
astro-mine-core validate examples/mission/*.mission.yaml --kind mission
astro-mine-core validate --json plan.json --kind plan      # machine-readable for CI/editors

# list the formats it knows and their schema $ids (derived from the registry — never stale):
astro-mine-core kinds
```

The validatable document formats: `sadf`, `objective`, `mission`, `plan`, `manifest` (plugin
manifest), `policy_package`, `run_provenance`, and the message documents `action_batch` /
`contact_plan`. Errors name the JSON-Pointer path, the offending value, and what was expected;
a document is **never** validated against a guessed schema — an ambiguous one fails with the list of
known kinds. The `units` schema is a *referenced vocabulary*, not a standalone document, so it is not
a validate target.

The umbrella `astro-mine validate` (RFC-0011) routes into this same dispatcher — it is wired, not
merely planned, and needs only [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli)
alongside Core. Both surfaces take the same arguments, with one deliberate difference: `--json` is a
top-level flag here (`astro-mine-core --json validate …`) and a per-verb one there
(`astro-mine validate --json …`), because the umbrella gives a component no top level to hang it
from.

## Layout

```
src/astro_mine/core/   # the package (import path: astro_mine.core)
  sadf/ env/ policy/ messages/ objective/ registry/ units/ compat/
schemas/               # canonical .proto and JSON Schema sources
tests/                 # mirrors the package layout
```

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**.

```bash
conda create -n astro-mine-core python=3.12
conda activate astro-mine-core
uv sync               # install runtime + dev deps
uv run pytest         # tests
uv run ruff check .   # lint
uv run mypy src       # type-check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Distribution

During private incubation Core is **not** on PyPI. It is versioned by **Git tag**
(`hatch-vcs`) and distributed two ways (policy:
[`docs/VERSIONING.md`](https://github.com/astro-mine/docs/blob/main/VERSIONING.md)):

- **Consume the package** via a tag-pinned `uv` Git source, resolved with
  `uv sync --locked` and a read-scoped PAT (`CORE_REPO_TOKEN`) in CI:

  ```toml
  [tool.uv.sources]
  astro-mine-core = { git = "https://github.com/astro-mine/astro-mine-core.git", tag = "v0.2.0" }
  ```

  Copy-paste pattern: [`examples/downstream-consumer/`](examples/downstream-consumer/).

- **Pin the schemas** via the content-addressed **schema bundle** published to private
  GHCR by the `publish-schemas` workflow — pullable by digest, so a Bench run reproduces
  against an exact Core schema version. Build it locally with
  `uv run python scripts/build_schema_bundle.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
