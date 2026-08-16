# Provenance — `lunar-polar-ice-prospecting-sprint-v1`

A short-horizon, prospecting-only Bench task on the anchor's world, traceable to
`docs/scenarios/1-lunar-polar-ice-prospecting.md` and `docs/architecture/bench.md §§3, 8`.

## Spec 0.5.0 — every content pin re-published under its conforming name

`0.5.0` moves every `content.*` reference onto the artifact names `conventions.md` §13 requires:
bare kebab-case, no component prefix, no version in the name.

**Nothing about the task changed.** The world, the fleet and the field are the same content; what
moved is what they are *called*. A run of `0.4.0` and a run of `0.5.0` resolve byte-identical
inputs.

Registry names are immutable, so this was a **re-publish, not a rename** — each artifact carries a
new digest, and every name this scenario pinned before is still published and still resolvable. That
is what keeps results scored against `0.4.0` valid for `0.4.0`; they are not comparable to `0.5.0`
and were never meant to be, which is why the pin change is a new spec version rather than an edit
(`bench.md` §5, §8).

§13 requires the migration to run as **one sweep**, so it is two scripts rather than a runbook:
`scripts/hub/migrate_artifact_names.py` re-publishes and records the digests, and
`scripts/hub/repin_zoo_to_conforming_names.py` applies them here.

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
