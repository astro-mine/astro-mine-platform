# astro-mine-studio

**The design front door for [Astro-Mine](https://github.com/astro-mine).** Turns
intent into a scored [Campaign](https://github.com/astro-mine/docs/blob/main/architecture/core.md):
capture an objective, run a trade study across the autonomy stack, and hand back the
best-scored design — with optional, provider-abstracted LLM-assisted intent capture.
Studio also ships a React/TypeScript front end over a FastAPI service; both arrive with
the `RM-P1-STUDIO-*` feature issues (this seed stands up the Python library only).

> **Status:** Phase 1. Importable `astro_mine.studio` library (intent capture, trade-study engine,
> design loop, campaign hand-off, reproducibility, Hub publish/terrain seams) plus the React front
> end that embeds View for 3D inspection — RM-P1-STUDIO-01..07. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/studio.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Quickstart

One command takes a clone to a running Studio — **offline, no account, no cluster** (CX-LOCAL):

```bash
# 1. Install the library + the local server (uvicorn) + the Hub client, in one extra.
uv sync --extra serve            # or: pip install "astro-mine-studio[serve]"

# 2. Build the standalone web UI once (Node ≥ 20.19, pnpm). `build:harness` emits the browsable
#    app (ui/dist-harness); plain `pnpm build` emits the library the console composes, not a page.
cd ui && pnpm install && pnpm build:harness && cd ..

# 3. Serve. Point --registry at a local OCI-layout Hub registry holding your content.
astro-mine-studio serve --registry /path/to/hub-registry
#   → http://127.0.0.1:8000   (open it in a browser)
```

`serve` composes the backend with its Hub seams **wired from the local registry**, mounts the built
UI at `/`, and prints an honest startup banner naming every seam and its state:

```
  Astro-Mine Studio
  → http://127.0.0.1:8000
  UI:   mounted from ui/dist-harness
  Seams:
    ✓ publishing: signing with anchor-dev.key.pem
    ✓ terrain: verifying against anchor-dev.pub.pem
    ✓ catalog: listing assets from /path/to/hub-registry
    ✓ asset preview: verifying against anchor-dev.pub.pem
  Seed: …
```

**Content is verified, not trusted.** Reads (terrain, catalog, asset preview) verify pulled bytes
against a **trusted public key**; published campaigns are signed with a **signing key**. Both default
to the `anchor-dev` dev keypair a workspace registry ships under `<registry>/keys/` — override with
`--trusted-key` / `--signing-key` (or `ASTRO_MINE_STUDIO_TRUSTED_KEY` / `_SIGNING_KEY`).

Common flags: `--host`/`--port`, `--registry` (or `$ASTRO_MINE_HUB_REGISTRY`), `--ui-dir`,
`--no-ui`, `--cache-dir`, `--no-seed`. `astro-mine-studio serve --help` lists them all.

**Degrades honestly, never blank.** Run with no `--registry` and the intent/study/comparison routes
still work; the 5 Hub-backed routes answer `503` with a reason the UI shows. Run before `pnpm build`
and the root URL explains how to build the UI rather than 404-ing. Nothing crashes.

## Layout

```
src/astro_mine/studio/      # import path: astro_mine.studio
  cli.py                    #   `astro-mine-studio serve` — compose + serve the local tier
  hub/                      #   ->Hub / <-Hub seams: publish a design, materialize a world
  compare.py                #   the Pareto/trade-off view, with uncertainty
tests/                      # mirrors the package layout
ui/                         # the React front end: comparison plots + embedded View (see ui/README.md)
```

## Publishing to Hub

A validated `Campaign` (or `TradeStudy`) publishes as a **content-addressed, signed** OCI artifact
whose config is a Core `PluginManifest` of kind `campaign` / `design`
([RFC-0008](https://github.com/astro-mine/docs/blob/main/rfc/0008-design-campaign-artifact-kinds.md)).
Hub indexes it by that manifest — Studio invents no private schema — while the payload rides as a
layer whose bytes Core never parses.

```python
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Registry
from astro_mine.studio.campaign import freeze_campaign
from astro_mine.studio.hub import HubArtifactPublisher, HubCapabilityResolver

registry = Registry("/path/to/registry")
publisher = HubArtifactPublisher(
    HubClient(registry, trusted_public_key_pem=public_pem),
    capability_resolver=HubCapabilityResolver(registry),
    private_key_pem=private_pem,
)
ref = publisher.publish_campaign(freeze_campaign(campaign), name="lunar-ice", version="0.1.0")
same = publisher.pull_campaign(ref.digest)   # re-verified, fail-closed
```

Signing, SLSA provenance, and the SBOM are Hub's (`astro_mine.seal`); Studio holds no crypto.
`pull_campaign` re-verifies the signature, re-hashes every blob, **and** re-derives the payload's
digest from the bytes — so a layer swapped under a valid signature is still refused.

**Capability tags are inherited, never invented.** Every `sadf_ref` in the swarm is resolved against
Hub and its tags folded into the manifest. A ref that will not resolve is a hard error: publishing a
design that understates what its swarm can do would route it past the export-control gate those tags
feed (`studio.md` §9).

## Serving terrain to the embedded View

`<GlobeScene world={{ manifestUrl }}>` fetches a Worlds bundle over HTTP, and the Phase-2 View
Gateway's tiles proxy does not exist. Studio pulls the bundle **by digest**, re-verifies it, unpacks
it into a digest-keyed cache, and static-mounts it. Studio stores the *reference*; the cache is
disposable and nothing is copied authoritatively (`studio.md` §5).

```bash
GET /worlds/shackleton-de-gerlache-v1:0.2.0
  -> {"digest": "sha256:...", "manifest_url": "/worlds/files/sha256-.../world.json"}
```

The bundle must carry the `tiles_anchor` that `RM-P1-WORLDS-16` publishes; View refuses one without
it rather than mis-placing the terrain.

**Two static mounts.** `serve` (and any deployment that passes the cache dirs) mounts the two
digest-keyed content caches — **not** the UI — at fixed prefixes: `WORLD_STATIC_PREFIX =
/worlds/files` and `ASSET_STATIC_PREFIX = /assets/files`. These serve the verified world bundles and
asset geometry the embedded View fetches. The built UI is a separate mount at `/` (see Quickstart).

## Trade-study backends

Search backends are swappable behind one interface (`designspace.search.SearchBackend`); a study
picks one by name. The built-in **NSGA-II** (`nsga2`, `nsga2-lhs`) needs no extra. **Optuna** is the
first external engine (`studio.md` §11 names Optuna · Ax/BoTorch · pymoo · Ray Tune) and ships behind
`[optuna]`:

```python
from astro_mine.studio.designspace import registered_backends, run_trade_study

registered_backends()          # ['nsga2', 'nsga2-lhs', 'optuna', 'optuna-nsga2', 'optuna-tpe']
study = run_trade_study(objective, space, clients=clients, backend="optuna-tpe", seeds=(7,))
```

The name is always discoverable; only *instantiating* a backend imports its engine, so without the
extra `get_backend("optuna")` raises `MissingBackendExtra` naming the install rather than failing
obscurely.

**Determinism.** A proposal is a pure function of `(ranked, senses, seed)` — Optuna's sampler is
rebuilt and re-seeded from Studio's seed on every call, touches no global RNG, and the study is
in-memory and fresh, so a re-run reproduces the identical Pareto front (the STUDIO-02 determinism
gate runs against `optuna` too). See `designspace/optuna_backend.py` for the full seed policy.

Ax/BoTorch, pymoo, and Ray Tune remain **documented seams**, not silent omissions — `search.py`'s
`DEFERRED_BACKENDS` records what each is for and how it plugs in (an adapter module + an extra + a
`register_backend` call, exactly as Optuna did).

## Fanning a design loop out to Cloud

`run_batch` holds a `JobDispatcher`. `LocalDispatcher` runs the loop in-process; `CloudDispatcher`
runs the *same* loop over `cloud.submit()` (`[cloud]` extra), for the local and docker backends —
and the cluster backend once a `ClusterClient` is supplied:

```python
from astro_mine.studio.orchestrate import CloudDispatcher, run_batch

records = await run_batch(
    candidates, objective,
    dispatcher=CloudDispatcher(backend="docker", image="ghcr.io/astro-mine/studio@sha256:..."),
    seeds=(1, 2), store=job_store, cache=result_cache,
)
```

The durable / cancelable / resumable guarantees are **structural** — they live in `run_batch` and the
`JobStore`/`ResultCache` seams, above the dispatcher — so they hold unchanged. Each job runs
`python -m astro_mine.studio.orchestrate.worker`, which calls the same `evaluate_candidate` with the
same seed: a Cloud-evaluated candidate is byte-identical to a locally-evaluated one, which is what
makes the dispatcher a drop-in rather than a second code path.

Two things are stated honestly rather than papered over. Cloud's `submit()` is blocking and returns
no handle, so **an in-flight job cannot be canceled** — cancellation is the same cooperative
pre-dispatch checkpoint it is locally, and no stronger claim is made. And **gRPC sibling fan-out**
(talking to Sim/Learn/Mind/Allocate/Guard/Bench as *services*) is a separate service-level
integration tracked in `astro-mine-sim` ("[gap] gRPC `EnvironmentService` + Ray-actor service skin")
— out of scope here; it changes what a worker binds its clients to (`ASTRO_MINE_STUDIO_CLIENTS`), not
how batches are fanned out.

## Extras — what each unlocks

The base wheel imports only Core (+ FastAPI) — no sibling packages. Each optional dependency is
imported lazily, so importing `astro_mine.studio` never pulls it, and a command that needs one fails
with an install hint rather than an `ImportError`:

| Extra | Unlocks |
|---|---|
| `[serve]` | `astro-mine-studio serve` — `uvicorn` + the `[hub]` client, so one install yields a fully-wired offline Studio (nothing 503s by default). |
| `[hub]` | The publish / terrain / catalog / preview seams. Without it those 5 routes answer `503`; `tests/test_api_hub.py` asserts the API layer never imports `astro_mine.hub`. |
| `[optuna]` | The Optuna trade-study backend (the built-in NSGA-II needs no extra). |
| `[cloud]` | `CloudDispatcher` — fan a design loop out over `cloud.submit()`. |
| `[llm]` | The optional, provider-abstracted LLM-assisted intent path. |

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**. Local-first
(conventions §7 tier-1): the library imports and runs on one workstation against the
Core narrow waist.

```bash
conda create -n astro-mine-studio python=3.12
conda activate astro-mine-studio
uv sync --extra serve && uv run pytest    # --extra serve pulls the [hub] client the API tests bind
uv run ruff check . && uv run mypy src
```

For the web UI, see [`ui/README.md`](ui/README.md). The one-command `astro-mine-studio serve` in the
[Quickstart](#quickstart) is the supported way to run the whole thing end to end.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
