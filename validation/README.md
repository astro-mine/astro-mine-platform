# Validation against published references

worlds.md §10 and the Phase-0 Worlds exit criteria require illumination/PSR regression against
**published lunar references with explicit error budgets**; conventions.md §11 requires physics
validation against external oracles with the same. worlds.md §9 says why it is not optional: *"the
PSR mask, illumination, and slope/bearing fields it produces feed safety-relevant decisions …
therefore data correctness is a first-class safety concern."*

This directory holds the machine-checkable form of that: each reference is a committed document
carrying the **published value**, its **citation**, the **error budget**, and the exact harness
configuration the value is comparable under. Result artifacts of real runs are committed beside
them. `tests/test_validation_psr.py` and `tests/test_gravity.py` regress against these.

| File | What |
|---|---|
| `shackleton_psr.reference.json` | Published LOLA-derived PSR fraction for the anchor footprint + budget |
| `shackleton_psr.result.json` | Result of running the harness on the **real** LOLA DEM + NAIF kernels |
| `grail_lunar_gravity.reference.json` | GRAIL GRGM1200A zonal coefficients (from PDS) + budget |

## Illumination / PSR — Shackleton–de Gerlache (`RM-P0-WORLDS-03`, `LUNAR-FR-001`)

**Reference.** No paper tabulates a PSR area for this sub-region, so the reference is read directly
off the *published product*: LOLA GDR `LPSR_85S_060M_201608.IMG` (PDS `LRO-L-LOLA-4-GDR-V1.0`; NASA
PGDA "Lunar Polar Illumination", product 69) — a binary permanently-shadowed flag map at 60 m/px
whose PSR determination is the method of **Mazarico et al. (2011)**, *Icarus* 211, 1066–1081
(doi:10.1016/j.icarus.2010.10.030), horizon-based illumination over several 18.6-year lunar
precession cycles. Masked to the anchor grid's footprint (±75 840 m about the south pole in the
lunar polar-stereographic frame), it gives **0.1864** — 4 289 km².

**Result** (`shackleton_psr.result.json`, run on the real 5 m LOLA DEM downsampled to 120 m, real
DE440 kernels, 1264×1264 grid, 730 epochs over 365 d):

| | |
|---|---|
| Computed PSR area fraction | **0.1464** (3 368 km², 233 911 cells) |
| Published reference | **0.1864** |
| \|error\| | **0.0400** |
| Error budget | **0.0500** (absolute) |
| Verdict | **PASS** |

**The error budget, and why the model sits 4 points low.** Every approximation in the Phase-0 model
biases the PSR area the *same* way — down — so the budget is sized to cover their sum rather than to
flatter the model:

1. **Horizon truncation (dominant).** The skyline is searched to `max_radius_m = 30 km`, and no
   terrain outside the 151.68 km tile exists to block the Sun at all. Barker et al. (2023),
   *PSJ* 4, 183 (doi:10.3847/PSJ/acf3e1) compute the horizon out to **310 km** with nested maps
   precisely because distant topography still shadows the poles. Missing far-field horizon ⇒ too
   much light ⇒ too little PSR.
2. **DEM resolution.** 120 m/px against a 60 m/px reference. The published sensitivity is large and
   monotone, all poleward of 80°S: **16 055 km²** at 240 m/px (Mazarico et al. 2011), **20 900 km²**
   at 30 m/px (O'Brien & Byrne 2022), **20 800 km²** at 20 m/px (Barker et al. 2023). Coarser grids
   under-resolve small PSRs.
3. **Grid-azimuth horizon.** `RM-P0-WORLDS-03` computes the skyline in projected-grid azimuth with a
   grid-convergence correction, and applies one region-centre Sun across the raster. The rigorous
   per-cell topocentric horizon is `RM-P1-WORLDS-12`.

Two terms bias the other way and are much smaller: the **point-Sun** approximation (no finite solar
disc — worlds.md §11's open question) and the **365-day** sampling window versus the 18.6-year
precession cycle the published product integrates. Both *add* shadow.

Tightening the budget is what `RM-P1-WORLDS-12` and a full-extent horizon radius buy.

## Reproducing

Neither the 3.7 GB LOLA DEM nor the NAIF kernels are in the repo or in CI (which runs offline
against the synthetic fixture in `tests/conftest.py`). Fetch them, build the anchor terrain, then:

```bash
python scripts/validate_illumination.py \
    --terrain out/shackleton/terrain \
    --metakernel data/spice/metakernel.tm \
    --reference validation/shackleton_psr.reference.json \
    --report-json validation/shackleton_psr.result.json
```

The same comparison runs as a marker-gated test, skipped by default so CI stays offline and green:

```bash
ASTRO_MINE_WORLDS_REAL_TERRAIN=out/shackleton/terrain \
ASTRO_MINE_WORLDS_METAKERNEL=data/spice/metakernel.tm \
    uv run pytest -m realdata
```

## Gravity — GRAIL GRGM1200A (worlds.md §3, §11, §12)

`grail_lunar_gravity.reference.json` carries the archived 4π-normalized zonal Stokes coefficients
C̄(2,0), C̄(3,0), C̄(4,0), `GM`, and the 1738.0 km reference radius, read directly from the GRAIL
**GRGM1200A** spherical-harmonic data record on the PDS Geosciences Node (`gggrx_1200a_sha.tab`) —
not a secondary table — under the tide-free convention. `tests/test_gravity.py` regresses
`astro_mine.worlds.gravity` against them: the unnormalization `J_n = -C̄_n0 · √(2n+1)` must
reproduce the published J₂/J₃/J₄ to 1e-9 relative, and the field must reproduce the published
1.62 m/s² mean surface gravity to 0.5%. The degree-4 truncation costs < 1e-3 relative against a
J₂-only field; the dropped latitudinal component is O(J₂·g) ≈ 5e-4 m/s² and is documented in
`gravity/_zonal.py`.
