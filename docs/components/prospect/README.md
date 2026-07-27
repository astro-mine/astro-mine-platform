# astro-mine-prospect

**Probabilistic resource-field models for [Astro-Mine](https://github.com/astro-mine).**
Water ice and mineral concentration as geostatistical distributions with explicit
uncertainty — a sealed ground-truth field and an agent-facing belief field that the
swarm updates by observing. Uncertainty is the product, not a footnote.

> **Status:** Phase 0 complete; **Phase 1 in progress**. The `ResourceField` contract, the GP / grid
> / GMRF / deep-generative backends, dataset-derived priors (parametric + real PDS raster ingest),
> sealed ground truth with spatially-correlated realizations, Bayesian belief updating with
> per-instrument sensor likelihoods, information-gain and EVPI/ISRU objectives, the calibration gate,
> Zarr field storage, Hub publishing, and the distributed gRPC field service have all landed. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/prospect.md) and the
> [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

> **Command renamed.** This CLI is `astro-mine-prospect`; the old name `prospect` still works for one
> deprecation cycle, printing a one-line notice to stderr, and is removed at the first
> public-benchmark milestone. The prefix is normative ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md) §5) — it ends the
> `PATH` land-grab of generic names and makes the package↔command mapping guessable.

## The shipped Shackleton priors

Prospect ships the canonical Shackleton–de Gerlache belief prior as package data in
`src/astro_mine/prospect/priors/catalog.py` — the same prior the anchor benchmark scenario pins
(`shackleton_water_ice_v1`). It is a **belief**, not ground truth: the sealed `GroundTruthField` a
run's sensors read against is realized per seed at runtime and is never published.

```python
from astro_mine.prospect.priors.catalog import (
    CITATIONS,
    SHACKLETON_CRS,
    SHACKLETON_PRIOR_GRID,
    SPECIES,
    UNIT,
)

print(SHACKLETON_CRS)
# body='MOON' body_fixed_frame='MOON_ME' reference_radius_m=1737400.0
# projection='+proj=stere +lat_0=-90 +R=1737400' datum=None

print(SHACKLETON_PRIOR_GRID)
# min_x_m=-30000.0 min_y_m=-30000.0 max_x_m=30000.0 max_y_m=30000.0 n_rows=240 n_cols=240

print(SPECIES, UNIT)
# water_equivalent_hydrogen mass_fraction
```

A 60 km × 60 km square centred on the pole at 250 m resolution, in the same lunar south-polar
stereographic CRS as the anchor world bundle — the two are grid-aligned by construction, which is
what lets a rover's position index both the terrain and the belief without a resampling step.

**Provenance is shipped with it.** `CITATIONS` carries a `DatasetCitation` per source dataset —
instrument, mission, PDS product id, literature reference, and the specific role that dataset plays
in the prior:

| Dataset | Role in the prior |
|---|---|
| **LOLA** (LRO) | south-polar topography → PSR / illumination geometry (cold-trap siting) |
| **Diviner** (LRO) | cold-trap temperatures (<~110 K) gating surface water-ice stability |
| **LEND** (LRO) | epithermal-neutron suppression → bulk water-equivalent hydrogen magnitude |
| **M³** (Chandrayaan-1) | surficial OH / H₂O absorption → near-surface ice presence at the poles |
| **LCROSS** | ground-truth magnitude anchor: 5.6 ± 2.9 wt% water in Cabeus ejecta |

A resource prior that cannot say where its numbers came from is not usable in a commons — anyone
scoring against it needs to be able to check the reasoning, not just the raster.

Alongside the catalog, `priors/` ships `RECIPE.md` (how the prior is derived), `ingest.py` and
`pds.py` (PDS raster conditioning), `provenance.py`, and `recipe.py`.

> **Authoring a prior is Python, not a file format.** There is no `prior.yaml` — no `from_yaml`
> exists anywhere in `src/`. Publishing a prior bundle works today (`astro-mine-prospect publish`);
> whether a hand-authored format *should* exist is an open design question, tracked as **G2.15** in
> the platform's UX gap report.

## Layout

```
src/astro_mine/prospect/    # import path: astro_mine.prospect
  field/        # the Core ResourceField contract + metadata (species/unit/CRS/grid)
  backends/     # GP (GPyTorch) · grid · GMRF/SPDE · deep-generative — one contract, four engines
  priors/       # dataset-derived, provenance-cited priors (parametric + PDS raster ingest)
  belief/       # sealed GroundTruthField + the replayable BeliefField update chain
  sensors/      # per-instrument likelihoods (neutron/NIR/GPR/drill), shared with Sim's sensor model
  isolation/    # the ground-truth seal: capability gate + reachability contract test
  infogain/     # active-perception objectives (variance/MI; EVPI on ISRU yield)
  calibration/  # the coverage gate + geostatistical sanity (variogram, LOO kriging CV)
  publish/      # content-addressed field bundles: Zarr store + .npy tar, Hub-published
  service/      # the optional distributed gRPC field service (TLS + OIDC + RBAC)
tests/                      # mirrors the package layout
```

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**.

```bash
conda create -n astro-mine-prospect python=3.12
conda activate astro-mine-prospect
uv sync && uv run pytest
```

Optional extras: `[zarr]` (Zarr field storage), `[service]` (the gRPC field service), `[publish]`
(Hub publishing), `[ingest]` (GDAL/rasterio PDS raster ingest). The importable core — fields, priors,
belief, info-gain — needs none of them, so the offline local tier runs on Core + numpy alone.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Running the field service

The service is **authenticated by default** (prospect.md §9): TLS transport, OIDC bearer tokens, and
per-method RBAC, with the ground-truth-adjacent `SubmitObservations` additionally gated on the Core
`ground_truth_access` capability grant. There is no unauthenticated default.

```python
from astro_mine.prospect.service import JwtVerifier, ServerTls, ServiceAuth, serve

auth = ServiceAuth(
    tls=ServerTls.from_files("server.crt", "server.key", client_ca="ca.crt",
                             require_client_auth=True),          # mTLS (conventions.md §9)
    verifier=JwtVerifier.from_jwks_uri(
        "https://idp.example/realms/astro-mine/protocol/openid-connect/certs",
        issuer="https://idp.example/realms/astro-mine", audience="astro-mine-prospect",
    ),
)
with serve({"shackleton_water_ice": prior}, auth=auth, address="0.0.0.0:50051") as (server, addr):
    server.wait_for_termination()
```

For local development only, an explicitly-opt-in cleartext mode exists. It requires **all three** of:
passing `InsecureDevAuth()`, setting `ASTRO_MINE_PROSPECT_INSECURE_DEV=1`, and binding loopback —
anything else raises, so a deployment cannot fall into it by omission.

```bash
export ASTRO_MINE_PROSPECT_INSECURE_DEV=1   # local dev only; never off-host
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
