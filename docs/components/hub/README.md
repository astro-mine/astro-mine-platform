# astro-mine-hub

**Artifact registry for [Astro-Mine](https://github.com/astro-mine).** An OCI-backed,
content-addressed, signed and gated store and index for the things the community
produces — plugins, worlds, SADF assets, policies, and surrogate models — each indexed by
its [Core](https://github.com/astro-mine/astro-mine-core) plugin manifest. It is the
supply-chain trust boundary: publish once, and the contribution is discoverable,
versioned, signed, and reusable across design, training, operations, and benchmarks. Hub
distributes artifacts; it never executes them.

> **Status:** Phase 1 — the full subsystem (`RM-P1-HUB-01…06`). Tier-1-local-first: the
> client + a local OCI-layout registry need no hosted Hub. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/hub.md) and
> [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Subsystems

```
src/astro_mine/hub/         # import path: astro_mine.hub
├── registry/     content-addressed OCI store — a local layout OR any remote OCI registry
│                 (Distribution Spec over HTTP: ghcr.io / Zot / Harbor); referrers; GC
├── supply_chain/ cosign-keyed sign/verify-twice; SLSA + SBOM attestations (fail closed)
├── index/        Core-manifest catalog (InMemory / SQLAlchemy + pgvector); ingest + facets
├── search/       faceted + full-text + capability + semantic (pluggable embedding provider)
├── resolve/      SemVer + Core-interface-range solver → pinned digests
├── policy/       license + export-control gating (Python or OPA/Rego) + append-only audit log
│   └── rego/     the versioned policy bundle: download.rego + data.json
├── curation/     open/curated/verified namespaces; yank/deprecate/promote
├── api/          FastAPI/OpenAPI façade + ASGI entrypoint          ([service] extra)
└── client/       the astro-mine-hub SDK (resolve/verify/pull/publish/cache) + CLI
                  (publish/search/resolve/pull/verify/keygen)
ui/                @astro-mine/hub-ui — the Hub console surface (RFC-0010; own Node toolchain)
```

## Using the client (no hosted Hub)

`--registry` takes **either** a local OCI-layout directory (fully offline) **or** any remote OCI
registry — `ghcr.io/astro-mine`, a private Zot/Harbor, `http://localhost:5000`. Credentials come
from the standard Docker sources (`docker login`, credential helpers, `GITHUB_TOKEN` for ghcr).

```bash
astro-mine-hub keygen  --out ./keys                  # the one signing-key command: cosign.key + cosign.pub
astro-mine-hub publish --registry ./reg --name pol --version 1.0.0 \
    --kind policy --manifest manifest.json --key ./keys/cosign.key --layer policy.onnx
astro-mine-hub search  --registry ./reg --semantic "excavation"
astro-mine-hub resolve --registry ./reg --name pol --spec ">=1.0.0,<2.0.0"
astro-mine-hub verify  --registry ./reg pol:1.0.0    # re-verify the supply chain, fail closed

# the same commands against a real registry — no hosted Hub, no `oras` binary
astro-mine-hub pull --registry ghcr.io/astro-mine pol:1.0.0                  # the Core manifest
astro-mine-hub pull --registry ghcr.io/astro-mine pol:1.0.0 --payload --out ./artifacts
```

Every pull **re-verifies before it returns bytes** — the manifest's signature/SLSA/SBOM *and* each
payload layer's content address — so a tampered artifact fails closed rather than reaching a caller.
In Python, `HubClient.pull_payload()` / `.materialize()` return verified layer bytes/paths; nothing
needs to reach past the client to `registry.pull_blob()`. Publishing is symmetric: `publish`
**verifies at admission** — the artifact must prove it verifies fail-closed *before* it is indexed,
not only at a later pull (the "verify twice" boundary, hub.md §9).

### Two vocabularies, on purpose

`--kind` is Hub's **container** vocabulary — what *shape* of payload an artifact carries
(`policy`, `world`, `asset`, `surrogate`, `plugin`, `schema`, `design`, `campaign`), which becomes
its `application/vnd.astro-mine.<kind>.v1` media type. It is Hub's own and is **not** derived from
Core's `PluginKind`, which enumerates the *interfaces* a plugin implements.

They overlap on four names and diverge everywhere else, and they cannot be unified: a served
surrogate's Core kind is `field_model` or `regime_engine` depending on its physics domain, so no
total map exists. `plugin` is the deliberate generic container for payloads with no more specific
shape — Link's comms model and Prospect's priors both use it.

A catalog entry carries **both**, as separate facets, and each is independently queryable:

```bash
curl 'localhost:8000/search?artifact_kind=world'   # container shape — Hub's axis
curl 'localhost:8000/search?kind=world_provider'   # Core interface — the contract
```

The container kind is recovered from the stored OCI `artifactType` at admission, not taken from
the publisher's request, so it cannot drift from the bytes. Read `kind` when you want to know what
an artifact *is*; read `artifact_kind` when you want to know how it is *packaged*.

Admission is one gate shared by the library, `POST /publish`, and curation, so a check cannot exist
on one path and be missing from another. It proves the digest exists and its bytes are its address,
that the manifest offered for indexing is the one actually stored, and that the signature/SLSA/SBOM
chain verifies — and it indexes nothing if any of that fails. **Signing is required**: `hub.md` §9
defines no namespace tier for unsigned content, and a trust tier above `open` is granted only by an
audited promotion that re-verifies the evidence, never claimed in a publish request.

## Publishing content

Published platform content lives at **`ghcr.io/astro-mine`** — that is where you get the anchor
scenario's world, fleet assets, resource prior, and contact plan. Producers own their own artifacts:
each component's `publish` builds, signs (with a **supplied** key — mandatory), and pushes its own
artifact through the shared Hub client; there is no separate publishing service.

Consumers verify against a **pinned org public key** — [`anchor-signing.pub`](anchor-signing.pub),
committed here (a public key is not secret; the private half stays in the org's secure store) — not
one carried alongside the artifact, so a tampered registry cannot swap both. Pass it with
`astro-mine-hub verify … --trusted-key anchor-signing.pub`. For the manual, by-component procedure
that publishes the nine-artifact anchor set (and the org-key / trust-anchor prerequisites), see
[docs/publishing-the-anchor-content-set.md](docs/publishing-the-anchor-content-set.md).

## Optional backends

The offline tier-1 path (client + local registry) needs **none** of these and never grows a
dependency on them:

| Backend | Why | How |
|---|---|---|
| **PostgreSQL + pgvector** | catalog facets in SQL + HNSW semantic top-k at 10^5–10^6 scale | `[service]` extra; SQLite is the fallback |
| **A served embedding model** | real (non-hashing) vectors for "find something like this" | `HUB_EMBEDDING_URL` (OpenAI-compatible `/v1/embeddings`) |
| **OPA + the Rego bundle** | license/export-control rules evolve as a bundle release, not a code change | `HUB_OPA_URL` (sidecar) or `HUB_POLICY_ENGINE=opa` (binary) |

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**; a polyglot repo — the
`ui/` React app has its own Node toolchain (`ui/README.md`). The offline gates run on one
workstation (no cloud/account beyond the CI token that pulls private Core).

```bash
uv sync --extra service                 # + the FastAPI/SQLAlchemy service tier
uv run ruff check . && uv run mypy src && uv run pytest -m "not integration" --cov
(cd ui && npm ci && npm run lint && npm run test -- --run && npm run build)
```

The **integration** tests (`-m integration`) exercise the real backends — PostgreSQL/pgvector, a
real **Zot** OCI registry, and **OPA** evaluating the Rego bundle; `docker compose up -d` brings up
the backing stack (see `docker-compose.yml`). Serve the
hosted API with `uvicorn --factory astro_mine.hub.api._asgi:make_app`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
