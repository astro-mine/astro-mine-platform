# Provenance — `lunar-polar-ice-excavation-fidelity-v1`

The **surrogate-fidelity** task, traceable to `docs/architecture/surrogate.md §8, §12`
(RM-P1-SURR-04 / RM-P1-SIM-03), `docs/architecture/bench.md §§3, 8`, and `LUNAR-TR-002`.

> **This scenario's headline result is the speedup, not the scorecard.**
>
> The Scorecard this task produces is a byproduct: a single excavator digging for ten seconds stores
> no water, so `water_mass` is 0 and `energy_per_kg` is *not applicable*. The metric set exists
> because a `ScenarioSpec` must pin at least one metric — not because the metrics are the point.
>
> The point is the speedup, and it has **two** faces, produced by two scripts, because a surrogate
> is only worth having if it is *faster than the solver it replaces* **and** *faster at a stated
> error bound* — two claims, and neither is a single number:
>
> - **`CROSSOVER.md` (`crossover.json`) — the cost curve.** How the DEM-vs-surrogate speedup scales
>   with bed size, on this scenario's own pinned content. A DEM contact solver is O(N²) and the
>   served surrogate is O(N·k), so the ratio is not a constant — it is a curve, and a single number
>   without the N it was measured at hides exactly the failure this curve was built to catch (a
>   flat ~2× when the served graph was accidentally O(N²); astro-mine-surrogate#24). This is a
>   **cost** result and carries no `is_claim`.
> - **`RESULTS.md` (`results.json`) — the claim.** The speedup *at the error bound the substitution
>   actually held to*, produced by `measure_surrogate_speedup.py`, which refuses to publish a
>   number when the bound does not hold.

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

`excavation-gns` is untouched. It already conformed before the rule existed, which is why it was
deliberately absent from the legacy set — the proof that §13 describes something achievable rather
than an aspiration. It is also not a content pin (see below).

## Why this task exists

A surrogate physics tier is only worth having if it is **faster than the solver it replaces, at a
stated error bound**. Neither half of that sentence is optional, and the platform had no task that
measured either. This is that task.

It is deliberately *small*: one robot, one physics process, ten seconds. A speedup claim is a claim
about a **cost ratio between two solvers**, and every agent in the scenario that the surrogate has
nothing to do with is a term that dilutes the ratio toward 1.0 while telling you nothing. So the
scenario pins exactly the excavator, and Sim's `FidelitySpeedupRunner` scopes the comparison to the
granular agents (`sim/bench/_speedup.py::_GranularRunner`).

## Content pins — reused from the anchor, except the excavator

Every `content.*.content_hash` is a **real Hub artifact digest**.

| pin | version | source |
|---|---|---|
| `shackleton-de-gerlache-v1` | **0.4.0** | **reused** from the anchor (`RM-P1-WORLDS-15`) — the revision that ships its horizon map (astro-mine-worlds#46) |
| `shackleton_water_ice_v1` | **1.0.0** | **reused** from the anchor (`RM-P1-PROSPECT-13`) — 1.0.0 corrects the GMRF SPDE field math (astro-mine-prospect#39); the belief prior does not enter this task's excavation-speedup measurement |
| `astro-mine.fleet.excavator` | **0.2.0** | **republished** — see below |

Reusing published content under a new task definition is exactly how the zoo grows (bench.md §8:
the zoo grows by adding immutable specs, never by mutating them). The full production recipe for the
world and the prior is in `../lunar_polar_ice_prospecting_v1/PROVENANCE.md`; this task rebuilds
neither.

### The excavator had to be republished (astro-mine-fleet#37)

The anchor pins excavator **0.1.0**, which declares one contact element: a `wheel`.

A `tool` contact element is the *only* declaration a physics engine reads to decide that an asset's
contact with the ground is a **cutting** interaction rather than a rolling one — and so the only
thing that routes it to a granular (DEM / learned-surrogate) contact model instead of a wheel-soil
one (`sim/bench/_scenario.py::dynamics_for_asset`). Without it, the excavator was an excavator in
name — `kind: excavator`, `excavation.bucket` — and a **rover in physics**. `grep "kind: tool"` over
the whole Fleet library returned nothing: the DEM/surrogate ladder was unreachable from every
published asset on the platform, and Sim's own speedup runner refused such a scenario outright
(*"pins no excavator (no asset declares a TOOL contact element)"*).

So excavator **0.2.0** declares the blade. Two of its dimensions are load-bearing rather than
descriptive, because Sim reads them to build the particle bed:

- `dimensions_m.x` = **0.40 m**, the cutting width → the DEM bed is `max(2 x width, 0.6) = 0.80 m`.
- `dimensions_m.z` = **0.15 m**, the blade height → the DEM tool height.

An engine that finds no `tool` element does not fail — it *silently falls back to its own defaults*
(`sim/runtime/content.py::asset_tool_geometry`), which is precisely how a missing blade stays
invisible.

**0.1.0 is untouched and still published.** Registry digests are immutable, and the anchor
benchmark still pins it — correctly, because 0.1.0 is the asset the anchor's historical results were
actually scored under (wheel physics, no granular tier). Pinning content by hash is what makes that
distinction expressible instead of a silent rewrite of history.

## The surrogate under test is **not** a content pin

`ContentPins` has slots for `world | fleet | prospect | link` and **no slot for a surrogate**. That
is correct and should stay that way: the surrogate is the *thing being measured*, not part of the
task definition. Two different surrogates measured against this scenario are two results for **one
task** — which is the entire point of a benchmark. Folding the tier into the spec would give each
candidate a different `spec_hash` and make their numbers incomparable.

The tier is therefore an argument to the runner, and it is pinned in `results.json` (and recorded in
`pins.json` for traceability) rather than in `scenario.json`:

| artifact | version | digest |
|---|---|---|
| `excavation-gns` | 0.6.0 | see `results.json` → `surrogate.content_hash` |

### Why that surrogate had to be retrained (astro-mine-surrogate#17)

Sim queries a granular surrogate with `friction = tan(radians(friction_angle_deg))` — a **friction
coefficient**. The anchor world's regolith prior is `friction_angle = 40 +/- 5 deg`, so the
coefficient Sim asks about on this world is **tan(40 deg) = 0.839**.

The shipped tier declared its trust region as `friction: [0.4, 0.7]`. That is a coefficient band
covering 21.8-35.0 degrees — **it did not contain the lunar soil it was meant to model**, and 0.839
overshot its ceiling by 20%.

The consequence was not a bad number; it was a *meaningless* one. Escalation is permanent per agent
(`sim/engines/surrogate/_engine.py`): the first out-of-domain query escalates that agent to DEM for
the rest of the episode. The config vector is built once. So tick 1 went out of domain, the whole
run was pure DEM, and the "speedup" would have been **1.0 — the reference solver against itself**.
Sim's own loader says so in as many words: *"Benchmarking a surrogate outside its trust region
measures nothing."*

The fix was not to widen a constant. A surrogate's trust region is **derived** — it is the tightest
box enclosing the configs its training fixture actually swept — so the sampling box silently *is*
the tier's contract with the platform, and it was living in module-level constants that no artifact
recorded and no hash covered. The tier is trained on a fixture swept from a declarative,
content-hashed `SamplingPolicy` whose hash is pinned in the published manifest, over a friction band
of **[0.4, 1.0]** — `tan(45 deg)`, covering the prior's mean, its full uncertainty, and leaving
0.839 comfortably interior rather than on the boundary. Every revision since — the `0.6.0` this
result was measured against included — inherits that same sampling box (the run's loaded trust
region is `friction: [0.4, 1.0]`, unchanged).

### Why the tier is 0.6.0 and not the shipped 0.2.0 (astro-mine-surrogate#21/#23/#27)

A correct trust region was necessary but not sufficient: the tier also has to declare an error
**budget** the consumer can actually hold it to, and the early revisions did not.

- **The statistic was wrong (#21).** `0.2.0`-era tiers derived the budget from an **RMSE**, but Sim
  re-validates with `abs(surrogate - reference).max()` over the bed. An RMSE bounds nothing — half a
  90-particle bed exceeds it by construction — so the number could never be satisfied, the tier
  escalated on its first re-validation, and the benchmark produced **no claim**.
- **The horizon was wrong (#23).** The next fix declared a *max* but calibrated it over a single
  step. A step surrogate feeds its own output back in, so its error compounds; a bound that held on
  step 1 was breached a few steps into the rollout. `0.5.0` over-corrected and declared a horizon of
  **4** it could not hold — the drift blows DEM up in the deep-blade regime well before then.
- **`0.6.0` (#27)** declares the budget as a **max over a 2-step rollout** — the longest horizon the
  model empirically holds — and Sim re-validates at exactly that cadence (`revalidate_every` defaults
  to the tier's declared `budget_horizon_steps`; it *refuses* a coarser one). That is what turned a
  chronic **0/5** no-claim into the **5/5** result in `RESULTS.md`: every seed was admitted, held its
  tolerance, and never escalated. `0.2.0`-`0.5.0` remain published and immutable; nothing pins them.

## The cadence is the task's, not the runner's

`episode.max_sim_seconds / episode.horizon_steps` = `10.0 / 200` = **0.05 s**, which is exactly
Sim's contact-scale benchmark tick (`sim/bench/_speedup.py::_CONTACT_DT_S`). The agreement is the
point: the task *declares* the cadence it is scored at rather than having the runner quietly
override it.

That cadence cannot be the anchor's. Excavation is a **contact-scale** process: a DEM bed's stable
internal timestep is ~0.8 ms, so at the anchor's 60 s mission tick a single step would sub-step
~78,000 times per agent — not slow, but unrunnable. A granular benchmark is run at the scale of the
physics it benchmarks.

`horizon_steps: 200` is chosen so the measurement is not dominated by noise. Bed construction and
particle settling (1200 substeps) happen at engine build, **outside** the timed bracket — Sim's
`TimedEngine` brackets `advance` and nothing else — so setup cannot pollute the ratio at any
horizon. What 200 steps buys is a DEM `advance` cost of several seconds per episode: comfortably
above timer noise, while keeping five seeds at two tiers cheap.

## Reproducing the number

The measurement is **not** in Bench's CI, and cannot be: Bench is `core` + `pydantic` and never
imports Sim; there is no onnxruntime and no DEM solver in that environment, and the run is CPU-bound
besides. It is produced by `scripts/measure_surrogate_speedup.py`, which lives outside `src/` and
lazily imports Sim — the same optional-component pattern as `scripts/determinism_gate.py`.

```
python scripts/measure_surrogate_speedup.py \
    --registry /path/to/hub-registry \
    --surrogate excavation-gns:0.6.0
```

`--revalidate-every` is deliberately left off: it defaults to the tier's declared `budget_horizon_steps`
(2 for `0.6.0`), which is the only cadence the tier's budget was calibrated to hold at.

The world resolves in **~3 s**. It is worth saying why that sentence is short, because until
astro-mine-worlds#46 it was not: the anchor bundle shipped **no** horizon map, so every load
re-derived a 1264 x 1264 x 120 skyline — a 192-million-entry ray-march — from the packaged DEM, and
this measurement's ten world constructions (two tiers x five seeds) would have cost the better part
of a day. World `0.4.0` persists the skyline in the bundle and the load path adopts it.

The script still memoizes the provider across tiers and seeds, because Sim builds a fresh
`ContentResolver` per episode and its by-digest cache does not survive a seed. That is now a
convenience (30 s saved) rather than the difference between running and not. Either way it cannot
flatter the result: provider construction is *setup*, entirely outside the `advance` bracket that
Sim's `TimedEngine` measures and that the speedup is a ratio of.
