# astro-mine-worlds

**Celestial-body environment models for [Astro-Mine](https://github.com/astro-mine).**
Real planetary data in, simulatable world out: terrain from DEMs, SPICE-driven
illumination with permanently-shadowed-region (PSR) detection, surface thermal, and
regolith terramechanics parameters — the physical substrate every scenario runs on.

> **Status:** Phase 0 — terrain ingest (RM-P0-WORLDS-01), the SPICE
> frames/epochs/geometry backbone (RM-P0-WORLDS-02), illumination/PSR detection
> (RM-P0-WORLDS-03), and the regolith terramechanics parameter field (RM-P0-WORLDS-05) are
> live; thermal, the Env-API provider, and the WorldSpec bundle are in progress. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/worlds.md)
> and [Phase-0 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-0-commons-seed.md).
> Phase 0 builds the **lunar south-polar (Shackleton–de Gerlache)** world only.

> **Where the commands live.** This package ships no console scripts. Worlds's commands are
> `astro-mine worlds <verb>`, provided by
> [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli) — a separate distribution that
> depends on this one. There is one executable and one grammar
> ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md)); the earlier
> `astro-mine-worlds` binary and its bare alias are both retired.

## Layout

```
src/astro_mine/worlds/      # import path: astro_mine.worlds
  crs/ terrain/ fields/ gravity/ illumination/ thermal/ regolith/
  ingest/ bodies/ provider/ spec/
tests/worlds/               # mirrors the package layout
validation/                 # committed published references + error budgets (worlds.md §10)
```

SPICE frames/epochs/geometry are **not** in this package: they come from the shared
`astro-mine-spice` companion
(`import astro_mine.spice`), which [RFC-0002](https://github.com/astro-mine/docs/blob/main/rfc/0002-shared-spice-foundation.md)
factored out so Worlds/Link/Sim/Transit share one SPICE implementation.

## Terrain ingest

Reproject any GDAL-readable polar DEM to the explicit lunar CRS (Core's `PlanetaryCRS`),
derive slope/aspect/roughness, carry vertical/void uncertainty, and write a
content-addressed COG product:

```python
from astro_mine.worlds import terrain
product = terrain.ingest_dem("ldem_80s_20m.tif", "out/shackleton", resolution_m=20.0)
print(product.terrain_hash)                 # reproducible from input + pinned toolchain
terrain.TerrainModel.open(product).sample(x, y)   # elevation/slope/aspect/normal/void
```

The real LOLA DEM is **not** in the repo or CI; fetch it with
`scripts/fetch_shackleton_dem.py`. The `ray_intersect`/Env-API provider that consumes this
product is RM-P0-WORLDS-06.

### No DEM to hand? Synthesize one

`ingest_dem` needs a raster, which made the shipped `synthetic_polar.world.yaml` example a
spec you could validate and then **not build** (issue #60). `synthesize_dem` writes one —
in the target CRS, over the region you name, defaulting to exactly that example's:

```python
dem = terrain.synthesize_dem("out/synthetic-dem.tif")     # 10 km x 10 km, 20 m, no network
product = terrain.ingest_dem(dem, "out/synthetic", resolution_m=20.0)
```

A **stand-in, not science**: an analytic basin (bowl, rim, two craters, seeded fine noise)
chosen to exercise slope / aspect / roughness / void-fill, not to resemble a real place.
Declare `terrain.SYNTHETIC_SOURCE_ID` as your spec's `source_dem.id` so the provenance says
so. Byte-reproducible for a given set of arguments, so a bundle built from it has a stable
`world_hash`.

## SPICE geometry

Worlds consumes the shared **`astro-mine-spice`** package (RFC-0002) rather than owning a
SPICE backbone: furnish a meta-kernel (**fail-loud** — a missing kernel raises, never a
silent default), then query topocentric Sun/Earth elevation/azimuth/range at a surface site.

```python
from astro_mine.spice import kernel_pool, Site, epoch_from_utc, sun_geometry
with kernel_pool("data/spice/metakernel.tm"):
    site = Site.lunar_from_latlon(-89.9, 0.0)          # FrameClass.TOPOCENTRIC
    sun = sun_geometry(site, epoch_from_utc("2025-06-21T00:00:00"))
    print(sun.elevation_deg, sun.azimuth_deg, sun.range_m)
```

Raw primitives — `body_position` (`spkpos`), `frame_transform` (`pxform`), and
`epoch_range` — are exposed for consumers that drive their own window search over the
furnished pool (illumination/PSR horizon maps in RM-P0-WORLDS-03; Link's contact-window
geometry in RM-P0-LINK-01). The default apparent-direction correction is `LT+S`. Real
NAIF kernels are **not** in the repo or CI (which runs against a synthetic kernel set);
fetch them with `scripts/fetch_spice_kernels.py`.

## Illumination & PSR

Precompute per-azimuth **horizon maps** from the terrain skyline, then answer Sun
visibility at any cell/epoch in O(1) by thresholding the SPICE Sun elevation against the
horizon in the Sun's azimuth — and derive **permanently-shadowed-region (PSR) masks** over
an epoch window. The comms/sun-denied core of the anchor scenario (`LUNAR-FR-001`):

```python
from astro_mine.worlds import terrain
from astro_mine.spice import kernel_pool, epoch_from_utc
from astro_mine.worlds.illumination import IlluminationModel, PsrEpochSemantics

product = terrain.ingest_dem("ldem_80s_20m.tif", "out/shackleton", resolution_m=20.0)
with kernel_pool("data/spice/metakernel.tm"):
    illum = IlluminationModel(product)                  # builds the horizon map once
    lit = illum.sun_visible(x, y, epoch_from_utc("2025-06-21T00:00:00"))   # O(1)
    psr = illum.psr_mask(window, step_s, semantics=PsrEpochSemantics.SEASONAL)
    print(illum.illumination_hash, psr.ever_lit_fraction)   # content-addressed, reproducible
```

The horizon map is computed in **world (grid) azimuth**; the topocentric SPICE Sun azimuth
is converted via the south-polar-stereographic grid convergence, so the model requires that
CRS and **fails loudly** on any other (the rigorous per-cell topocentric horizon is
[RM-P1-WORLDS-12](astro-mine-worlds#11)). PSR
*permanence* is explicit on the result (`PsrEpochSemantics`: diurnal / seasonal / mission).
Illumination/PSR is regression-tested in CI against analytic terrain with explicit error
budgets; the real-LOLA validation against published references runs outside CI via
`scripts/validate_illumination.py`.

### Illumination backends are plugins

Which backend answers Sun visibility is a **`WorldSpec` choice**, not a code path
(`worlds.md` §3/§11). `build_illumination_model` resolves the selector; `known_backends()` lists
what is selectable and `available_backends()` narrows that to what actually resolves here:

```python
from astro_mine.worlds.illumination import build_illumination_model, known_backends

known_backends()          # ('horizon', 'raycast_cpu', 'raycast_gpu', 'surrogate', ...)
model = build_illumination_model(product, backend="raycast_cpu", max_radius_m=8000.0)
```

`horizon` is the precomputed default, `raycast_cpu`/`raycast_gpu` the fine on-demand path (the GPU
selector degrades to CPU when CuPy is absent, with the same numbers), and `surrogate:<name>` a
learned model loaded from its published artifacts.

**The group is open.** A third-party field model becomes selectable by advertising an entry point —
no PR to Worlds, no subclassing:

```toml
[project.entry-points."astro_mine.field_models"]
acme-illum = "acme_illum.backend:build"
```

The entry point's **name** is the backend id (`build_illumination_model(..., backend="acme-illum")`)
and its value resolves to a factory `(terrain, **kwargs) -> SunVisibilityModel` — the structural
contract in `illumination/_backend.py`, which the built-ins satisfy and a plugin may satisfy without
inheriting anything of Worlds'. Two rules worth knowing:

- **Listing never imports.** `known_backends()` reads entry-point *names* only, so a heavyweight or
  broken plugin cannot slow or break discovery; a plugin that fails to load is simply excluded from
  `available_backends()`.
- **A built-in id may not be reused.** A backend id is provenance — it is folded into
  `illumination_hash` and stamped into the published `field_model` manifest — so a plugin claiming
  `horizon`, `raycast_cpu`, `raycast_gpu`, or `surrogate` is a hard error naming both claimants
  rather than a silent precedence rule. Pick your own id.

## Gravity

Point-mass + **low-order spherical-harmonic** gravity (worlds.md §11/§12), evaluated by one shared
zonal kernel of selectable degree — so the Moon and Mars do not carry duplicate implementations:

```python
from astro_mine.worlds.bodies import MOON_PACK
from astro_mine.worlds.gravity import MOON_GRAVITY

MOON_GRAVITY.zonals            # (J2, J3, J4) from GRAIL GRGM1200A
MOON_GRAVITY.magnitude(1_737_400.0, -89.9)   # m/s^2 at a radius + latitude
MOON_PACK.gravity(position)                  # -> Core Vector in the local surface frame
```

Lunar coefficients are the archived, 4pi-normalized Stokes coefficients read straight from the
GRAIL **GRGM1200A** record on the PDS Geosciences Node (tide-free), unnormalized in-code by
`J_n = -C_n0 * sqrt(2n+1)`; `GM` and the coefficients' reference radius (1738.0 km — **not** the
1737.4 km CRS datum) travel with them. Only the **radial** component is returned; the dropped
latitudinal term is O(J2 * g) ~ 5e-4 m/s^2 and documented in `gravity/_zonal.py`. The regression
against the published field and its error budget is `validation/grail_lunar_gravity.reference.json`.

## Field layers, horizon maps, and the world bundle

Per worlds.md §5, **COG** carries the 2-D rasters (DEM, slope/aspect/roughness, the PSR mask) and
**Zarr** the chunked N-D field layers. The `(H, W, n_azimuth)` horizon map is persisted as
`illumination/horizon.zarr` inside the bundle — so a pulled world **skips** the whole skyline
rebuild — and the per-class diurnal temperature curves as `thermal/curves.zarr`:

```python
model.write_horizon_zarr("out/horizon.zarr")             # persist
IlluminationModel(terrain, horizon_store="out/horizon.zarr")   # adopt (no recompute)
```

Each store is chunked for range reads, keeps a consolidated metadata index, and is
content-hashed **as it lands on disk**; those hashes fold into `world_hash`, so a stored-vs-
recomputed horizon is provenance-honest and a tampered chunk moves the world hash. A bundle
with no persisted map simply recomputes in-process — the same `illumination_hash` as before.

The published anchor (`shackleton-de-gerlache-v1:0.4.0`) is the first world to ship its skyline.
It is worth being concrete about why (issue #46): the map is `(1264, 1264, 120)` — a
**192-million-entry** ray-march — and until 0.4.0 *every* consumer re-derived it from the packaged
DEM on *every* load. Loading the anchor from Hub now takes **~3 s** instead of the better part of an
hour. The cost is the bundle: it grows **38 MB → 438 MB**, because the skyline is float32 angles
whose mantissas are effectively noise, so it compresses only ~1.9× and no lossless codec does
better (zstd-9 + byte-shuffle lands within 1 MB of the default). That is a deliberate trade — a
one-time pull against a per-load hour, which is the trade `LUNAR-TR-004` cares about.

### A content hash covers content, not the toolchain

Worlds hashes cover the arrays and the parameters that determine them. The **toolchain** is recorded
in every manifest and deliberately *not* hashed (`worlds/_hashing.py`). It used to be, and because the
per-repo version was hatch-vcs-derived — so it tracked git commit distance — a bit-identical world
rebuilt one commit later minted a different `terrain_hash` and `world_hash`.
Content-addressing was really commit-addressing. Nothing is lost by excluding it: every hash covers
its own bytes (each Zarr store's `store_hash` folds into `world_hash`), so a toolchain that writes
different bytes still moves the hash — through the bytes.

### Rebuilding and republishing the anchor

```bash
python scripts/build_shackleton_anchor.py --convert \
    --raw-dem data/dem/ldem_875s_5m_float.lbl --metakernel data/spice/metakernel.tm \
    --out out/shackleton --version 0.4.0 \
    --resolution-m 120 --n-azimuth 120 --max-radius-m 30000 \
    --horizon-frame grid --abcorr NONE \
    --psr-start 2025-01-01T00:00:00 --psr-days 365 --psr-step-hours 12 --psr-semantics seasonal

astro-mine worlds publish out/shackleton/bundle --registry <hub-registry> --key <cosign.key>
```

Those flags are not arbitrary: they are exactly the harness `validation/shackleton_psr.reference.json`
grades against, so the published bundle carries the numbers that were validated. The build is
CPU-bound and takes hours — almost all of it the skyline.

## Validation against published references

`validation/` holds the committed published references, their **explicit error budgets**, and the
result artifacts of real runs (worlds.md §10; conventions.md §11):

| Reference | Budget | Result |
|---|---|---|
| `shackleton_psr.reference.json` — LOLA GDR `LPSR_85S_060M` PSR map (method of Mazarico et al. 2011) over the anchor footprint | 0.05 absolute on PSR area fraction | `shackleton_psr.result.json` |
| `grail_lunar_gravity.reference.json` — GRAIL GRGM1200A zonal coefficients (PDS) | 1e-9 rel. on J_n; 0.5% on mean surface gravity | `tests/test_gravity.py` |

The PSR harness runs on the **real** LOLA DEM + NAIF kernels, which are not in the repo or in CI:

```bash
python scripts/validate_illumination.py --terrain out/shackleton/terrain \
    --metakernel data/spice/metakernel.tm \
    --horizon-store out/shackleton/bundle/illumination/horizon.zarr \
    --reference validation/shackleton_psr.reference.json \
    --report-json validation/shackleton_psr.result.json

# the same comparison as a test (skipped by default; CI stays offline)
ASTRO_MINE_WORLDS_REAL_TERRAIN=... ASTRO_MINE_WORLDS_METAKERNEL=... uv run pytest -m realdata
```

`--horizon-store` lets the harness adopt the skyline the bundle already ships rather than re-derive
it, which turns the run from ~90 minutes into ~18 seconds. It is a cache, not a shortcut: the store
is validated against the parameters resolved for the run and rejected if they disagree. Omit it to
force a from-scratch recompute. The committed result's `illumination_hash` is the digest a consumer
gets loading the published bundle — the two are the same object now that hashes no longer fold in
the toolchain.

## Regolith parameters

A grid-aligned spatial field of the five regolith terramechanics parameters — bulk density,
cohesion, friction angle, bearing capacity, thermal inertia — each with a **companion
uncertainty** layer, content-addressed and georeferenced to a terrain product so Sim
consumes it **without re-projection** (`LUNAR-FR-003`). Parameters only — the constitutive
contact/excavation law is Sim's (RM-P0-SIM-03):

```python
from astro_mine.worlds import terrain
from astro_mine.worlds.regolith import build_regolith_field, RegolithField

product = terrain.ingest_dem("ldem_80s_20m.tif", "out/shackleton", resolution_m=20.0)
reg = build_regolith_field(product, "out/shackleton-regolith")   # 5 mean + 5 uncertainty COGs
field = RegolithField.open(reg)
field.params(x, y)        # -> Core RegolithParams (means) — the Sim-facing contract
field.uncertainty(x, y)   # -> per-parameter 1-sigma, same fields
print(reg.regolith_hash)  # content-addressed, reproducible
```

The Phase-0 mean field is the documented lunar prior (`DEFAULT_LUNAR_PRIOR`), spatially
uniform — there is no per-pixel regolith map yet — so spatial structure lives in the
uncertainty, inflated where the DEM is void (an off-by-default `slope_sensitivity` hook can
modulate means by terrain slope). Nominal values are illustrative baselines with uncertainty
(Lunar Sourcebook ranges); validating the constitutive law against analytic/lab cases is
Sim's job (RM-P0-SIM-10).

## Development

Worlds is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup.

```bash
python scripts/test.py worlds
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
