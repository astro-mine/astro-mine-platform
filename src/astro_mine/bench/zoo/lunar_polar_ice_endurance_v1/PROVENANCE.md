# Provenance — `lunar-polar-ice-endurance-v1`

A survival-weighted, full-chain Bench task on the anchor's world, traceable to
`docs/scenarios/1-lunar-polar-ice-prospecting.md` and `docs/architecture/bench.md §§3, 8`.

## Spec 0.6.0 — every content pin re-published under its conforming name

`0.6.0` moves every `content.*` reference onto the artifact names `conventions.md` §13 requires:
bare kebab-case, no component prefix, no version in the name.

**Nothing about the task changed.** The world, the fleet and the field are the same content; what
moved is what they are *called*. A run of `0.5.0` and a run of `0.6.0` resolve byte-identical
inputs.

Registry names are immutable, so this was a **re-publish, not a rename** — each artifact carries a
new digest, and every name this scenario pinned before is still published and still resolvable. That
is what keeps results scored against `0.5.0` valid for `0.5.0`; they are not comparable to `0.6.0`
and were never meant to be, which is why the pin change is a new spec version rather than an edit
(`bench.md` §5, §8).

§13 requires the migration to run as **one sweep**, so it is two scripts rather than a runbook:
`scripts/hub/migrate_artifact_names.py` re-publishes and records the digests, and
`scripts/hub/repin_zoo_to_conforming_names.py` applies them here.

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
