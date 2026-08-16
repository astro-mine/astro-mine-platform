# Water-ice / hydrogen prior — fit recipe & provenance

> Backlog: **RM-P0-PROSPECT-03** · Architecture: `docs/architecture/prospect.md` §2.4, §6, §12 ·
> Principle: *"Priors are sourced and cited, not invented"* (prospect.md principle 4).

This documents how the Phase-0 default prior `shackleton_water_ice_v1` is derived, so it is
**reconstructable from public inputs** and every dataset is **cited with provenance** (the issue's
acceptance criteria). The recipe is implemented in [`recipe.py`](recipe.py); the cited datasets and
numeric anchors are constants in [`catalog.py`](catalog.py).

## What it produces

A `Prior` over the lunar south-polar **Shackleton–de Gerlache** CRS/grid (`SHACKLETON_CRS` /
`SHACKLETON_PRIOR_GRID`, a 60×60 km polar-stereographic box at 250 m/px about the pole — aligned to
the Worlds reprojected grid, RM-P0-WORLDS-01): a per-cell Gaussian field of **water-equivalent
hydrogen** (species `water_equivalent_hydrogen`, unit `mass_fraction`) with an explicit per-cell
mean **and** variance. It seeds the belief field before any observation (RM-P0-PROSPECT-04) and
realizes as a Core `ResourceField` via `Prior.as_field()`.

## Phase-0 scope — a *cited parametric* prior, not raster ingest

The Phase-0 local tier must run **offline, deterministically, with no account** (`LUNAR-TR-004`,
conventions §7 tier-1), and the conditioning rasters (Diviner temperature, LEND neutron, M³) are
**not ingested upstream by P0 Worlds** (RM-P0-WORLDS-01 ingests only the LOLA DEM). So this recipe
does **not** read PDS rasters. Instead it encodes the **published characterizations** of the
datasets below as a transparent parametric fit, and is flagged as such (honest uncertainty,
prospect.md §9). Real PDS raster ingest is delivered by the **Phase-1** recipe
`shackleton_water_ice_pds_v1` — **RM-P1-PROSPECT-12**
(astro-mine-prospect#11; see the section below) — which
registers behind the same `load_prior` registry with no consumer change; this parametric recipe
stays the **offline default**.

## Cited datasets (provenance)

| Dataset | Product | Role in the fit | Reference |
|---|---|---|---|
| **LOLA** (LRO) | `LRO-L-LOLA-4-GDR-V1.0` polar DEM | south-polar topography → PSR / cold-trap geometry | Smith et al. (2010), *Space Sci. Rev.* 150, 209–241 |
| **Diviner** (LRO) | `LRO-L-DLRE-4-RDR-V1.0` bolometric T | cold-trap temperatures (<~110 K) gating ice stability | Paige et al. (2010), *Science* 330, 479–482 |
| **LEND** (LRO) | `LRO-L-LEND-4-RDR-V1.0` epithermal neutrons | neutron suppression → bulk WEH magnitude | Mitrofanov et al. (2010), *Science* 330, 483–486; Sanin et al. (2017), *Icarus* 283, 20–30 |
| **M³** (Chandrayaan-1) | `CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0` | surficial OH/H₂O absorption → near-surface ice | Pieters et al. (2009), *Science* 326, 568–572; Li et al. (2018), *PNAS* 115, 8907–8912 |
| **LCROSS** | Cabeus impact-plume spectroscopy | magnitude anchor: **5.6 ± 2.9 wt%** water | Colaprete et al. (2010), *Science* 330, 463–468 |

## The fit

1. **Cold-trap weight** `coldness(cell) ∈ [0, 1]` — a proxy for the Diviner cold-trap / PSR field
   over LOLA polar geometry. The parametric default is Gaussian-decaying from the pole (grid origin)
   with a 12 km length scale: ice concentrates in the deep polar cold traps. *(Conditioning hook:
   pass a real grid-shaped `coldness` layer — e.g. from Worlds — to override the default; this is
   the seam RM-P1-PROSPECT-12 drives with ingested Diviner/PSR rasters.)*
2. **Mean** blends the LEND broad-polar background WEH (~0.5 wt%) up to the LCROSS Cabeus water
   anchor (5.6 wt%) by the cold-trap weight:
   `mean = background + (peak − background) · coldness`.
3. **Variance** scales its standard deviation with the same weight (most ice ⇒ most uncertain, per
   the LCROSS spread — uncertainty stays honest): `σ = σ_bg + (σ_peak − σ_bg) · coldness`,
   `variance = σ²`, with `σ_peak = 2.9 wt%`.

All anchors are named constants in `catalog.py`; all knobs are recorded in `Provenance.params`, so
the fit is **deterministic and reconstructable** — re-running the recipe reproduces the field and
its `content_hash` byte-for-byte (conventions §5).

## Reproducing

```python
from astro_mine.prospect.priors import load_prior
prior = load_prior("shackleton_water_ice_v1")      # default Shackleton grid
prior.provenance.content_hash                       # stable address; cites all five datasets
field = prior.as_field()                            # a Core ResourceField (GridField)
```

---

# Phase-1 — the real PDS raster-ingest recipe (`shackleton_water_ice_pds_v1`)

> Backlog: **RM-P1-PROSPECT-12** (astro-mine-prospect#11)
> · Architecture: `docs/architecture/prospect.md` §2.4, §3, §4, §6, §12 · depends on
> **RM-P1-WORLDS-14** (conditioning layers) + Hub.

This recipe replaces the parametric *radial* cold-trap proxy with the **measured cold-trap
geometry**, fitted from **real public planetary rasters** reprojected onto the Shackleton prior
grid, with **per-product content-addressed provenance**. It is implemented in [`pds.py`](pds.py);
the reproject/materialize pipeline is in [`ingest.py`](ingest.py) (the `[ingest]` extra). It is
**additive** — the parametric `shackleton_water_ice_v1` stays the offline default; nothing on the
consumer side changes.

## Ingested rasters

| Dataset | Product ingested | Role → ice-favorability | CRS / resolution |
|---|---|---|---|
| **LOLA + SPICE** | PSR mask (Worlds `shackleton-de-gerlache-v1` illumination component) | permanently-shadowed fraction → cold-trap | polar-stereo, 120 m |
| **Diviner** (LRO) | GDR L3 south-polar bolometric temperature (Tbol, K) | colder than ~110 K → cold-trap | polar-stereo, 240 m |
| **LEND** (LRO) | Sanin et al. 2018 (PSS 162) 3° CSETN epithermal grid | neutron suppression → bulk WEH | 3° global (rasterized) |
| **M³** (Chandrayaan-1) | Lu et al. 2024 (GRL; Zenodo 10608904) hydration band-depth mosaic | 2.8–3.0 µm band depth → surficial water | polar-stereo, ~140 m |

LCROSS stays the magnitude anchor (5.6 ± 2.9 wt%; no raster, so `source_hash=None`). Only real
public, open-commons datasets are ingested; no operational/mission-sensor data (that is **P2**).

## Pipeline (offline after a one-time fetch)

1. **Fetch** the public rasters once — `scripts/fetch_pds_conditioning.py` (multi-MB, documented,
   cached; forbidden in CI by `LUNAR-TR-004`).
2. **Ingest / reproject** each source onto `SHACKLETON_PRIOR_GRID` (`ingest.py`,
   `rasterio.warp.reproject`), recording each raster's `source_hash`.
3. **Materialize** a small, deterministic, **content-addressed conditioning bundle**
   (`conditioning.npz` + `manifest.json`, hashed) — the arrays are ~240×240, so the bundle is
   < 1 MB and the offline fit reads it with **numpy alone** (no GDAL).
4. **Fit** (`build_pds_prior`) — a coverage-weighted **ice-favorability** `w ∈ [0,1]` combines the
   four normalized layers (colder Diviner, more PSR, more neutron suppression, deeper M³ band →
   higher `w`), dropping layers that don't cover a cell. The mean/variance blend is the parametric
   recipe's honest-uncertainty scheme driven by `w`: `mean = bg + (peak−bg)·w`,
   `σ = σ_bg + (σ_peak−σ_bg)·w`, **inflated where conditioning coverage is thin**. All knobs are in
   `Provenance.params`; the per-product `source_hash`es make the fit reproducible from cited public
   inputs.
5. **Publish** to Hub (`publish_prior` / `scripts/build_pds_prior.py --registry`) as a signed,
   content-addressed `resource_field_backend` artifact; consumers resolve it by digest and reopen a
   live `GridField` via the `from_bundle` entry point (never importing `astro_mine.prospect`).

## Reproducing

```bash
python scripts/fetch_pds_conditioning.py --dir data/pds
python scripts/build_pds_prior.py \
    --diviner data/pds/diviner/dgdr_tbol_avg_pols_20100107n_240_img.lbl \
    --m3       data/pds/m3/0416_OP2A_out_magnetotail_mosaic.img \
    --lend     data/pds/lend/lend_sanin2018_pss.txt \
    --psr      <worlds bundle>/illumination/psr_mask.tif \
    --out      data/pds/conditioning --registry files/hub-registry
```

```python
import os
os.environ["ASTRO_MINE_PROSPECT_CONDITIONING"] = "data/pds/conditioning"
from astro_mine.prospect.priors import load_prior
prior = load_prior("shackleton_water_ice_pds_v1")   # fits from the materialized bundle (numpy only)
```

Like the world hash, the ingested-prior digest is **toolchain-reproducible** (pinned GDAL/PROJ/
numpy); consumers reproduce it by **pull-by-digest** from Hub, not by re-ingesting the rasters.

> **There is currently one copy of this artifact.** `shackleton_water_ice_pds_v1:1.0.0`
> (bundle `sha256:d44dd824…`) exists in the workspace store and nowhere else — `ghcr.io/astro-mine`
> holds the nine anchor packages and this is not one of them. So "pull-by-digest from Hub" is advice
> a reader outside this workspace cannot yet follow, and the re-ingest path above needs the raw
> Diviner/M3/LEND products, which no repository carries.
>
> This is the same exposure that made the 2026-08-08 prune of `excavation-gns:0.2.0`–`0.5.0`
> unrecoverable. It is recorded in `registry-inventory.json` and tracked by
> [astro-mine-platform#41](https://github.com/astro-mine/astro-mine-platform/issues/41), which
> mirrors this artifact to `ghcr.io/astro-mine`. Until that lands, do not prune it.
