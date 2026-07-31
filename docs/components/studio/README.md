# astro-mine-studio

**The design front door for [Astro-Mine](https://github.com/astro-mine).** Turns
intent into a scored [Campaign](https://github.com/astro-mine/docs/blob/main/architecture/core.md):
capture an objective, run a trade study across the autonomy stack, and hand back the
best-scored design — with optional, provider-abstracted LLM-assisted intent capture.
Studio also has a React/TypeScript front end over a FastAPI service — neither of which is in this
repository; see [What is here, and what is not](#what-is-here-and-what-is-not).

> **Status:** Phase 1. The importable `astro_mine.studio` library — intent capture, trade-study
> engine, design loop, campaign hand-off, reproducibility, Hub publish/terrain seams — is what this
> distribution provides (RM-P1-STUDIO-01..07). The REST surface and the View-embedding front end
> ship elsewhere. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/studio.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## What is here, and what is not

**The library is here. The server and the UI are not.** Studio's REST surface
(`astro_mine.studio.api`) and its React front end were deliberately left out of the consolidation
(`docs/CONSOLIDATION_PLAN.md` §"Not migrated"); the REST tier is destined for `astro-mine-api`,
which is not stood up yet (roadmap `RM-DIST-03`). So there is **no `serve` you can run from this
repo, and nothing to install that would give you one** — no `astro-mine-studio` distribution, and
no `[serve]` extra: it was dropped along with the routes it existed to pull in.

`astro-mine studio serve` still exists as a command, and says exactly that before exiting non-zero:

```console
$ astro-mine studio serve
astro-mine studio serve needs the Studio REST surface (astro_mine.studio.api), which is not
included in astro-mine-platform.
  The REST tier ships in astro-mine-api (docs: architecture/api.md), which is not stood up yet
  — roadmap RM-DIST-03.
  No released distribution provides it today, so there is nothing to install.
```

The verb is kept rather than dropped because that message is the useful behaviour: removing it
would make `astro-mine --help` claim the platform has less than it does, and a reader following an
older tutorial would get "unknown component" instead of one line telling them where the surface
lives. It exits `1` — nothing was served, so it does not report success.

## Quickstart — the library

Everything the trade-study engine does is reachable from Python, offline, with no account and no
cluster (CX-LOCAL). That half came across whole:

```python
from astro_mine.studio.intent import capture_intent      # objective capture + its validation gate
from astro_mine.studio.designspace import run_trade_study  # the search across the autonomy stack
from astro_mine.studio.compare import build_comparison   # the Pareto/trade-off view
from astro_mine.studio.campaign import author_campaign   # freeze the winner into a Campaign
```

**Content is verified, not trusted.** The Hub seams read and write through a local OCI-layout
registry: reads (terrain, catalog, asset preview) verify pulled bytes against a **trusted public
key**, and published campaigns are signed with a **signing key**. Both default to the `anchor-dev`
dev keypair a workspace registry ships under `<registry>/keys/`, and are settable through
`ASTRO_MINE_STUDIO_TRUSTED_KEY` / `ASTRO_MINE_STUDIO_SIGNING_KEY` — the same environment variables
the eventual server reads, so the wiring is unchanged when it arrives.

## Layout

```
src/astro_mine/studio/      # import path: astro_mine.studio
  intent/ designspace/      #   objective capture; the trade-study search
  campaign/ orchestrate/    #   freeze a winner; fan the design loop out
  hub/                      #   ->Hub / <-Hub seams: publish a design, materialize a world
  compare.py                #   the Pareto/trade-off view, with uncertainty
tests/studio/               # mirrors the package layout
```

Neither `api/` nor `ui/` is here — the FastAPI routes and the React front end are the two halves
the consolidation did not migrate. They live in the
[`astro-mine-studio`](https://github.com/astro-mine/astro-mine-studio) repo until `astro-mine-api`
stands up.

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

**Two static mounts.** Any deployment that passes the cache dirs mounts the two digest-keyed content
caches — **not** the UI — at fixed prefixes: `WORLD_STATIC_PREFIX = /worlds/files` and
`ASSET_STATIC_PREFIX = /assets/files`. These serve the verified world bundles and asset geometry the
embedded View fetches; the built UI is a separate mount at `/`. The prefixes are stated here because
the cache layout is this package's, even though the server that mounts them is not.

## Trade-study backends

Search backends are swappable behind one interface (`designspace.search.SearchBackend`); a study
picks one by name. The built-in **NSGA-II** (`nsga2`, `nsga2-lhs`) needs no extra. **Optuna** is the
first external engine (`studio.md` §11 names Optuna · Ax/BoTorch · pymoo · Ray Tune) and ships behind
`[studio-optuna]`:

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
runs the *same* loop over `cloud.submit()` (`[studio-cloud]` extra), for the local and docker backends —
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

Each optional dependency is imported lazily, so importing `astro_mine.studio` never pulls it, and a
path that needs one fails with an install hint rather than an `ImportError`. Extras are
component-prefixed platform-wide, because bare names collided across the 18 merged pyprojects:

| Extra | Unlocks |
|---|---|
| `[studio-hub]` | The publish / terrain / catalog / preview seams. Without it those five seams degrade to a stated reason rather than a blank. |
| `[studio-optuna]` | The Optuna trade-study backend (the built-in NSGA-II needs no extra). |
| `[studio-cloud]` | `CloudDispatcher` — fan a design loop out over `cloud.submit()`. |
| `[studio-llm]` | The optional, provider-abstracted LLM-assisted intent path. |

`[serve]` is **gone**, not renamed: it existed to pull `uvicorn` in for the REST surface, and that
surface is not part of this distribution. `fastapi` was dropped from the base dependencies for the
same reason.

## Development

Studio is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup. Studio is local-first
(`conventions.md` §7 tier 1): the library imports and runs on one workstation against the Core
narrow waist.

```bash
python scripts/test.py studio
```

The React front end is not in this repo — see
[What is here, and what is not](#what-is-here-and-what-is-not).

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
