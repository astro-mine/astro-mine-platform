# Architecture

`astro-mine-hub` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/hub.md`](https://github.com/astro-mine/docs/blob/main/architecture/hub.md):
  the artifact registry — OCI-backed, content-addressed, signed and gated store/index for
  worlds, assets, policies, surrogates, and plugins (the supply-chain trust boundary);
  purpose, principles, runtime, data, integration, security, and roadmap alignment.
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

> **Phase-1 scope.** This repo ships the full Hub Phase-1 subsystem (`RM-P1-HUB-01…06`):
> the content-addressed OCI registry — a local layout **and** the OCI Distribution-Spec transport
> against any remote registry (ghcr/Zot/Harbor) — the verify-twice supply chain (config *and*
> payload layers), the Core-manifest index (PostgreSQL + pgvector; SQLite offline) + discovery
> behind a swappable embedding provider, the SemVer/interface resolver, license/export-control
> gating (a pure-Python evaluator *and* an OPA/Rego bundle, conformance-tested against each other)
> + curation, the FastAPI façade, the `astro-mine-hub` client/CLI, and the React web UI —
> tier-1-local-first (client + local registry need no hosted Hub, no Postgres, no OPA). Deferred to
> follow-ups: the hosted deployment (Helm/K8s), live keyless cosign (Fulcio/Rekor), OIDC/Keycloak
> auth, and NATS eventing; multi-region mirrors are P2 and mission-architecture artifact kinds are
> P3.

> **Web UI.** `ui/` follows the platform front-end baseline
> ([`conventions.md` §2.1](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)):
> TypeScript + React on Vite, Vitest + Playwright, **pnpm `11.10.0`**. It was the last tree on npm
> and moved with astro-mine/docs#49; the CI job installs with `--frozen-lockfile` like every other
> front end.
>
> **Deviation: Pico CSS.** The UI is styled by [Pico](https://picocss.com/) rather than the platform
> design system, and leans on Pico's bare-element defaults — `article`, `dl`, `table`, `form`,
> `mark` are almost all unclassed, and the repo carries **zero hardcoded colours** as a result.
> That makes it the cheapest front end to restyle and the most exposed while doing it: deleting the
> Pico import strips essentially all styling, so the conversion to `@astro-mine/ui` must land as one
> change once the table, panel, description-list, form-control and badge primitives exist
> (`astro-mine-hub#31`, Wave 24). It is not a gradual strangle.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).
