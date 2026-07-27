# Provenance — `lunar-polar-ice-endurance-v1`

A survival-weighted, full-chain Bench task on the anchor's world, traceable to
`docs/scenarios/1-lunar-polar-ice-prospecting.md` and `docs/architecture/bench.md §§3, 8`.

## Content pins — all **reused** from the anchor (no republish)

Every `content.*.content_hash` in `scenario.json` is a **real Hub artifact digest** — the *same*
signed OCI image-manifest digest the anchor pins, published by `RM-P1-WORLDS-15` (world) /
`RM-P1-FLEET-10` (fleet) / `RM-P1-PROSPECT-13` (prospect). The full production recipe for each digest
is recorded in `../lunar_polar_ice_prospecting_v1/PROVENANCE.md`; this task reuses the anchor's
published content under a new task definition (bench.md §8: the zoo grows by adding immutable specs).

This scenario pins the **full** heterogeneous fleet (orbiter, lander, scout, excavator, hauler, ISRU
plant) — the whole excavate → haul → extract → purify → store chain.

## What makes it a distinct task

Same content, different **task identity** (its `spec_hash` differs because the pinned inputs differ):

- **Metrics** — the survival/yield subset (`nights_survived`, `water_mass`, `energy_per_kg`,
  `comms_robustness`); ranked by `nights_survived` (the primary metric), not by discovery.
- **Seeds** — a fresh public set (`3001…3006`, six seeds); no held-out commitment.
- **Budgets** — a fixed 5000 compute-unit cap, so the campaign is scored under bounded compute.
- **Horizon** — the anchor's 30-day span (`43200` steps @ 60 s), reused deliberately so endurance is
  measured over the same night-spanning campaign length.

## Reproducibility

Consumers **pull** the pinned bundles by digest; they do not rebuild them (see the anchor's
`PROVENANCE.md` for the world-toolchain caveat). Two clean checkouts resolve the identical scenario.
