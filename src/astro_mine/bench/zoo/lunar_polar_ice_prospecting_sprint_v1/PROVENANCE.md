# Provenance — `lunar-polar-ice-prospecting-sprint-v1`

A short-horizon, prospecting-only Bench task on the anchor's world, traceable to
`docs/scenarios/1-lunar-polar-ice-prospecting.md` and `docs/architecture/bench.md §§3, 8`.

## Content pins — all **reused** from the anchor (no republish)

Every `content.*.content_hash` in `scenario.json` is a **real Hub artifact digest** — the *same*
signed OCI image-manifest digest the anchor pins, published by `RM-P1-WORLDS-15` (world) /
`RM-P1-FLEET-10` (fleet) / `RM-P1-PROSPECT-13` (prospect). The full production recipe for each digest
is recorded in `../lunar_polar_ice_prospecting_v1/PROVENANCE.md`; this task does not rebuild or
republish anything — it *reuses* the anchor's published content under a new task definition, which is
exactly how the zoo grows (bench.md §8: the zoo grows by adding immutable specs, never by mutating).

This scenario pins the subset the prospecting sprint exercises: the world, the relay orbiter, the
cargo lander, the active-perception scout, and the water-ice belief prior. It omits the excavation /
haul / ISRU fleet the anchor uses for the full extraction chain.

## What makes it a distinct task

Same content, different **task identity** (its `spec_hash` differs because the pinned inputs differ):

- **Horizon** — a single 7-day surface campaign (`10080` steps @ 60 s), vs. the anchor's 30 days.
- **Metrics** — the belief/discovery subset only (`information_gain`, `psr_area_characterized`,
  `discovery_latency`, `comms_robustness`); ranked by `information_gain` (the primary metric).
- **Seeds** — a fresh public set (`2001…2005`); no held-out commitment (a public dev/eval task).
- **Budgets** — bounded by a 1-hour wall-clock and 100 compute-unit cap.

## Reproducibility

Consumers **pull** the pinned bundles by digest; they do not rebuild them (see the anchor's
`PROVENANCE.md` for the world-toolchain caveat). Two clean checkouts resolve the identical scenario.
