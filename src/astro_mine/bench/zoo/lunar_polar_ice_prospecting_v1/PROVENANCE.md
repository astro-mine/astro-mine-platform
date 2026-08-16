# Provenance — `lunar-polar-ice-prospecting-v1`

The anchor Bench scenario (**"Lunar Polar Water-Ice Prospecting v1"**), traceable to
`docs/scenarios/1-lunar-polar-ice-prospecting.md §13` and `docs/architecture/bench.md §§1, 3, 5`.
This file records *how every pinned value was produced*, so the scenario is reproducible and each
provisional digest can be replaced by a real published one without guesswork.

## Spec 0.9.0 — the held-out seeds are rotated and leave the repository

`0.9.0` carries a new `seeds.heldout_commit`. The seeds it commits to are **not** in this repository:
they moved to the private [`astro-mine/embargo`](https://github.com/astro-mine/embargo), reached
through `$ASTRO_MINE_BENCH_EMBARGO_ROOT` (astro-mine-platform#37).

The old set was committed here in plaintext, deliberately, for CI verifiability, on the standing
assumption that this repository was private. The public flip retires that assumption for **every
commit**, not just `HEAD` — so rotating in place would have republished the same seeds one commit
later. Rotation was necessary and moving the store was what made it sufficient.

The retired set was also `900001`–`900012`: sequential, and guessable from the public set's
`1001`–`1005` without reading the file at all. Its replacement is drawn from `secrets`.

**Results scored against `0.8.0` remain valid for `0.8.0`.** They are not comparable to `0.9.0`
results and were never meant to be — a new commitment is a new scenario version precisely so the two
cannot be silently pooled.

`0.9.0` also re-pins **every** content reference to its conforming artifact name (`conventions.md`
§13, astro-mine-platform#34):

| was | is |
|---|---|
| `shackleton-de-gerlache-v1` | `shackleton-de-gerlache` |
| `astro-mine.fleet.relay-orbiter` | `relay-orbiter` |
| `astro-mine.fleet.lander` | `lander` |
| `astro-mine.fleet.prospecting-rover` | `prospecting-rover` |
| `astro-mine.fleet.excavator` | `excavator` |
| `astro-mine.fleet.hauler` | `hauler` |
| `astro-mine.fleet.isru-plant` | `isru-plant` |
| `shackleton_water_ice_v1` | `shackleton-water-ice` |
| `astro-mine.link.lunar-polar-relay-dsn` | `lunar-polar-relay-dsn` |

Registry names are immutable, so each is a **re-publish, not a rename**, and each therefore has a new
digest. Every name on the left is still published and still resolvable, which is what keeps results
scored against `0.8.0` valid for `0.8.0`.

Two of these are not just relabelled. The **contact plan was rebuilt**, because its nodes carry the
Fleet SADF `identity.id`s and its provenance pins the world's content hash — both of which moved. It
reproduces byte-identically: the 30-day window searched across 25 node pairs yields
`plan_digest sha256:38d5e507…` and 1108 intervals on repeat runs, the determinism `link.md` claims.
And the **ISRU plant's descriptor version was corrected** from `0.1.0` to `0.2.0` in `pins.json`;
`scenario.json` and the `0.8.0` note above had said `0.2.0` since that re-pin, so the descriptor had
simply drifted from the pin it describes.

**These artifacts are not yet mirrored to `ghcr.io`.** They exist only in the workspace store until
that runs, so a fresh clone cannot resolve this scenario's content — see
`registry-inventory.json` and `docs/hub/publishing-the-anchor-content-set.md`.

## Spec 0.8.0 — the plant declares the tank it fills

`0.8.0` re-pins **one** content reference: `astro-mine.fleet.isru-plant` to `0.2.0`
(`sha256:3b133647…`), the first revision to declare a `water_gauge` — a `resource_storage` sensor
with species `water` and unit `kg`. Every other pin, and the Core schema digest, is unchanged.

Bench scores `water_mass` by matching a reading's `resource_species` and `unit` against the
scenario's water species. The plant declared only `hopper_temp` and `feed_cam`, so it filled a tank
**nothing could read**: the metric summed an empty set to `0.0`, indistinguishable from a swarm that
produced nothing. Measured with astro-mine-sim#64's feedstock coupling in place, the plant held
**16.70 kg of water and 33.4 MJ** at a 5.29% grade after a 12 km haul, and the scorecard still read
zero.

**This was the last of four separate reasons `water_mass` read zero on every Sim-backed run**, each
of which hid the next:

1. the gauge dispatched on `resource` before `kind`, so the tank rendered the ice field
   (astro-mine-sim#61);
2. extraction was gated on a mode string with no feedstock behind it, so nothing legitimately
   produced water (astro-mine-sim#64);
3. the anchor's plant declared no gauge at all (astro-mine-fleet#40, this re-pin);
4. and the parametric ISRU family's gauge declared `si_unit: "mass_kg"` — not a unit token the
   platform knows — so templated plants were invisible too. Fleet's lint now rejects that.

The value chain the scenario describes — prospect → excavate → haul → extract → store — is
therefore measurable end to end for the first time.

The sibling `lunar-polar-ice-endurance-v1` pins the same plant and is re-pinned alongside, at its
own new `spec_version` `0.5.0`. The sprint and excavation-fidelity tasks do not pin the plant and
are untouched.

## Spec 0.7.0 — the sites snap to the pinned DEM instead of carrying Link's elevations

`0.7.0` drops `elevation_m` from all four placement sites. No content pin moves; the latitudes and
longitudes are unchanged.

`0.6.0` adopted astro-mine-link's anchor sites wholesale — lat/lon **and** elevation. The lat/lon
are the right thing to share: they are what Link computed the pinned `ContactPlan` against, so
adopting them is what makes comms geometry and physics describe the same swarm. The elevations were
not. They are hand-authored figures describing Link's own geometry and were **never sampled from
this world bundle**, and measured against the DEM this scenario actually pins they are wrong by
kilometres:

| asset | 0.6.0 pinned | pinned world's DEM | error |
|---|---|---|---|
| `astro-mine.fleet.prospecting-rover` | −3800 m | **+636 m** | **−4436 m (buried)** |
| `astro-mine.fleet.excavator` | −3500 m | −224 m | **−3276 m (buried)** |
| `astro-mine.fleet.hauler` | −800 m | −2417 m | +1617 m (floating) |
| `astro-mine.fleet.isru-plant` | +1800 m | +1499 m | +301 m (floating) |

Two of the four assets would have started **kilometres beneath the terrain**. That is not a cosmetic
offset: an asset below the surface is occluded by the ground it is inside, so `line_of_sight` fails
and comms silently zero — which is precisely the defect `astro-mine-bench#31` fixed for the ring
layout, reintroduced from a different source. The `_layout` docstring in `astro_mine.sim.bench`
already warned about it in as many words; 0.6.0 walked into it anyway.

Omitting `elevation_m` makes each site snap to the pinned DEM, which is what
`SitePlacement.elevation_m = None` exists for. It is also strictly *more* reproducible than a
literal: the terrain is pinned by content hash, so a snapped site cannot disagree with the world the
run uses, whereas a literal can — and did.

**The general lesson, recorded so it is not relearned:** a coordinate is only meaningful against the
model it was measured on. Sharing a *site* between components is right; sharing a *height* across
two different terrain models is not, unless both resolve it from the same pinned raster. Link's
elevations should be read as descriptive of its own anchor, not as a datum this scenario can adopt.

## Spec 0.6.0 — pinned placement + scoring parameters, and a new content-address basis

`0.6.0` moves **no content pin**. The world, six fleet assets, belief prior, link ContactPlan and
`core_schema_digest` are byte-identical to `0.5.0`'s. What changes is the *schema*: the spec can now
express two things it never could, and the way a spec is content-addressed.

**Placement (astro-mine-bench#63).** A `ScenarioSpec` pinned content but not *where the swarm
stands*, so the runner chose — and Sim's choice was a fixed 25 m ring around the body-fixed south
pole with each asset's angle jittered by `sha256(scenario_hash:index)`
(`astro_mine.sim.bench._scenario._layout`). Because `scenario_hash` derives from `spec_hash`, **any
re-pin moved the swarm**, and every position-dependent metric moved with it — `water_mass` through
the resource-field sample under the plant, `discovery_latency`, and anything illumination-dependent.
It also contradicted the ContactPlan this scenario pins, which astro-mine-link computed against a
deliberate siting spanning the shadowed floor and the lit ridge. `0.6.0` adopts those four sites, so
comms geometry and physics finally describe the same swarm:

| asset | lat | lon | elevation | site |
|---|---|---|---|---|
| `astro-mine.fleet.prospecting-rover` | −89.90° | 0° | −3800 m | PSR floor |
| `astro-mine.fleet.excavator` | −89.86° | 90° | −3500 m | PSR floor |
| `astro-mine.fleet.hauler` | −89.78° | 135° | −800 m | lit |
| `astro-mine.fleet.isru-plant` | −89.68° | 204° | +1800 m | lit ridge |

Elevations are **terrain heights**. Link's own `AnchorSurfaceSite.position_m()` adds
`antenna_height_m` because it needs an antenna phase centre; a vehicle body origin is not 1.5 m (or,
for the plant, 4 m) above the ground, so the mast is deliberately not carried over here. The
relay-orbiter and the lander are not sited: the orbiter is on orbit, the lander is a delivery
vehicle, and a scenario places only the assets whose position it depends on.

**Scoring parameters (astro-mine-sim#66).** `psr_area_characterized`, `information_gain` and
`discovery_latency` read parameters that existed only as runner constructor arguments, so the task
could not state them and their defaults were load-bearing. Two of those defaults were actively
wrong: `characterized_variance_threshold = 0.0` is unsatisfiable (no posterior variance is ≤ 0, and
`information_gain` treats a non-positive variance as an error), so the metric reported a confident
`0.0 m²` instead of abstaining; and `discovery_threshold = 0.0` is tripped at tick 0 by any valid
non-negative reading, so a "discovery" was recorded before anything was discovered. The pinned
values and where each comes from:

- **`cell_area_m2 = 62500.0`** — the pinned prior's own grid pitch squared. `SHACKLETON_PRIOR_GRID`
  is a 60×60 km box at 240×240 cells (`astro_mine.prospect.priors.catalog`), i.e. 250 m/px. Without
  it the default of `1.0` reports PSR area in *cell counts* labelled m².
- **`characterized_variance_threshold = 0.00021025`** — the PSR prior sigma halved. The prior's
  sigma runs from 0.004 (background) to `LCROSS_WATER_WT_SIGMA = 0.029` in the cold traps
  (Colaprete et al. 2010), so a PSR cell starts at variance ≈ 8.41e-4; halving the *uncertainty*
  gives (0.029/2)² = 2.1025e-4. **This is a scenario design choice, made here.** The scenario
  document asks for "posterior uncertainty reduced ≥ X% over the target PSR" (§3) and never fixes
  X — this is where X is chosen, and it is deliberately a criterion on sigma rather than variance
  because that is what "uncertainty" means to a reader.
- **`discovery_threshold = 0.01`** — twice the LEND polar background WEH (`0.005`), and well below
  the 0.056 LCROSS Cabeus peak, so a detection means meaningfully more hydrogen than the polar
  background rather than any non-zero reading at all.
- **`psr_region`** — the PSR extent is stated **geometrically** (a lat/lon window over the polar
  cap), not as a set of cell ids. The metric consumes opaque cell ids, but Bench and Prospect share
  no cell-id convention yet (that is settled in astro-mine-sim#66's belief work), and pinning ids
  here would freeze a convention nobody has chosen. A region is convention-independent: whatever
  resolves it to cells is the same code that builds the belief history, so the two agree by
  construction. The window is a *bound* on the area of interest, to be intersected with the pinned
  world's validated PSR mask — not a replacement for it.

**The content-address basis changed.** Through `0.5.0`, `spec_hash` digested a *full* model dump,
which serialized defaulted fields (`"budgets": {"wall_clock_seconds": null, …}`). Under that basis,
appending any optional field to `ScenarioSpec` re-identified **every scenario in the zoo, including
historical ones** — precisely the recomputation `bench.md §8`'s add-only discipline exists to
prevent. From `0.6.0` the canonical form excludes defaults, so a spec that does not exercise an
optional block hashes as though the block did not exist, and future additive growth is hash-stable.

The cost is one-time and is taken here: `0.5.0` and earlier no longer recompute to their recorded
digests (`0.5.0` yields `sha256:6452792773…` under today's basis, not `sha256:a1f4c1c8…`). Those
values are kept in `tests/test_zoo_richer.py` as historical records of what each version addressed
to while it was live — rewriting them would erase the record rather than correct it. Nothing
recomputes them; the immutability test checks only the version the zoo currently ships. This is the
cheapest possible moment for the change: the repos are private, no leaderboard is public, and no
published result is bound to those digests. The consequence to author against is that a field
explicitly set to its own default is now indistinguishable from an omitted one, and **changing a
declared default silently re-identifies every spec that omits that field** — so new optional fields
default to `None`, a sentinel that never carries meaning.

Two committed measurement artifacts recorded the sibling excavation-fidelity task's `spec_hash` and
were re-stamped to the new basis (`crossover.json`, `results.json`). The measurements themselves are
untouched: the same task, the same content, the same physics — only the identity function moved.

Sim does not yet consume either block; it still runs its own layout and its own scoring defaults.
Wiring `placement` into `sim_scenario_from_spec` and `scoring` into the runner's `ScoringContext` is
the follow-up that closes astro-mine-bench#63, and is where the anchor's metrics actually move.

## Spec 0.5.0 — re-pinned fleet + prospect to current source

`0.5.0` re-pins the six fleet assets and the belief prior to the digests current source now
produces, published to `ghcr.io/astro-mine`. Three changes moved them off the `0.4.0` pins: the
excavator gained a `tool` contact element (**0.2.0**, astro-mine-fleet#38); the belief prior's GMRF
SPDE operator was corrected so alpha=2 yields a valid Matern nu=1 field (**1.0.0**,
astro-mine-prospect#39); and re-pinning Core to `v0.3.0` (RFC-0009) re-stamped every producer
manifest — so `0.4.0`'s fleet/prospect digests no longer reproduce from source. The world (`0.4.0`)
and the Core schema digest are unchanged and keep their pins; the link ContactPlan is rebuilt from
its unchanged inputs and republished (below). A re-pin is a new immutable `spec_version`, never an
in-place edit (bench.md §8) — `0.4.0` stays valid for anything scored under it. The same re-pin
propagates to the three sibling tasks that reuse this content (endurance, sprint, excavation
fidelity), so the zoo pins one digest per asset.

## Content pins — all content **real** (Hub-published)

**Every pinned input is published to Hub and Sim resolves it into a runnable Scenario**
(`RM-P1-WORLDS-15` / `RM-P1-FLEET-10` / `RM-P1-PROSPECT-13` / `RM-P1-SIM-01`). So every
`content.*.content_hash` in `scenario.json` is a **real Hub artifact digest** — the OCI
image-manifest digest each producer's signed publish yields, which Sim resolves by content hash
against a local OCI-layout registry (the offline tier-1 path; no hosted Hub).

**Reproducibility.** The fleet and prospect digests are **portably deterministic** — re-running a
producer's publish yields the identical digest on any machine. The **world** digest is deterministic
only **within a pinned toolchain** (GDAL/PROJ/numpy) on one OS: `world_hash` folds those library
versions and the terrain reproject is a GDAL/PROJ computation whose bytes can differ across platform
wheels. That does **not** weaken the pin — the benchmark is reproducible because consumers **pull**
the published bundle by digest, not rebuild it. The world was built on the toolchain pinned in
astro-mine-worlds `uv.lock` (numpy / rasterio+GDAL / spiceypy); a portable-rebuild container is a
deferred follow-up.

**World inputs** (pinned in the WorldSpec `source_dem` + recorded here): LOLA `LDEM_875S_5M` (PDS
LRO-L-LOLA-4-GDR-V1.0; 87.5–90°S at 5 m/px; elevation km on a 1737.4 km sphere), nodata-aware
downsampled to **120 m** and scaled km→m; NAIF **DE440** SPICE kernels (LSK `naif0012`, text PCK
`pck00011`, binary PCK `moon_pa_de440_200625`, FK `moon_de440_250416`, SPK `de440`); PSR sampled over
a **lunar year** (730 epochs, 12 h step) → 12.78% permanently shadowed.

| `content_id` | producer | status | how the pinned digest was obtained |
|---|---|---|---|
| `shackleton-de-gerlache-v1` **0.4.0** | worlds | **real** | `worlds` `scripts/build_shackleton_anchor.py --convert --raw-dem <LDEM_875S_5M.lbl> --metakernel <DE440.tm> --version 0.4.0 --resolution-m 120 --n-azimuth 120 --max-radius-m 30000 --horizon-frame grid --abcorr NONE --psr-start 2025-01-01T00:00:00 --psr-days 365 --psr-step-hours 12 --psr-semantics seasonal` then `worlds publish <bundle> --registry <reg> --key <key>` |
| `astro-mine.fleet.relay-orbiter` | fleet | **real** | `fleet publish library/orbital/relay-orbiter.sadf.yaml --registry <reg> --sign --key <key>` |
| `astro-mine.fleet.lander` | fleet | **real** | `fleet publish library/orbital/lander.sadf.yaml --registry <reg> --sign --key <key>` |
| `astro-mine.fleet.prospecting-rover` | fleet | **real** | `fleet publish library/surface/prospecting-rover.sadf.yaml --registry <reg> --sign --key <key>` |
| `astro-mine.fleet.excavator` **0.2.0** | fleet | **real** | `fleet publish library/manipulation/excavator.sadf.yaml --registry <reg> --sign --key <key>` — 0.2.0 declares the `tool` contact element (astro-mine-fleet#38); supersedes 0.1.0, which stays published and immutable |
| `astro-mine.fleet.hauler` | fleet | **real** | `fleet publish library/logistics/hauler.sadf.yaml --registry <reg> --sign --key <key>` |
| `astro-mine.fleet.isru-plant` | fleet | **real** | `fleet publish library/isru/isru-plant.sadf.yaml --registry <reg> --sign --key <key>` |
| `shackleton_water_ice_v1` **1.0.0** | prospect | **real** | `prospect publish --registry <reg> --name shackleton_water_ice_v1 --private-key <key>` (belief prior only; the sealed `GroundTruthField` is never published — `RM-P0-PROSPECT-05`) — 1.0.0 squares the GMRF SPDE operator (astro-mine-prospect#39); supersedes 0.1.0 |
| `astro-mine.link.lunar-polar-relay-dsn` | link | **real** | `link` `scripts/build_anchor_contact_plan.py --metakernel <DE440.tm> --relay-spk <relay_orbiter.bsp> --world-registry <reg> --world-ref shackleton-de-gerlache-v1:0.4.0 --key <key>` (builds *and* publishes; `--dry-run` prints the digest without publishing). Run from the Worlds venv (`uv run --with ../astro-mine-link`) — the build needs Link *and* the Worlds `world_provider` entry point, and no single venv has both |

The content **ids** are real and final (Fleet `identity.id` values; the Prospect prior recipe name;
a chosen versioned `world_id`; the Link `ANCHOR_ARTIFACT_NAME`). The machine-readable descriptors live
in `pins.json` (this directory); `tests/test_zoo_anchor.py` asserts every pin is a well-formed **real
Hub digest** distinct from the provisional-pin derivation of its `pins.json` descriptor, so no pin can
drift silently.

### The `link` pin — the comms model

`link` was `null` through spec `0.1.1`: LINK-04 content-addressed a `ContactPlan` by digest but exposed
no stable *id* to pin it under (a `ContactPlan` has no id field), so per the `ContentPins` contract the
comms plan stayed unpinned and the comms environment was realized at run time. Link now publishes the
anchor plan as a signed, content-addressed Hub artifact under a stable id
(`ANCHOR_ARTIFACT_NAME = astro-mine.link.lunar-polar-relay-dsn`), so the comms-denied dimension of the
flagship benchmark is content-pinned like every other input (`LUNAR-TR-003`).

The pinned digest is the **OCI image-manifest digest** of that artifact — the same kind of digest every
other pin here carries — *not* the `ContactPlan`'s own wire-form `plan_digest`, which the artifact
records internally as its `provenance.digest`. The artifact is a `plugin` in Hub's vocabulary
(`application/vnd.astro-mine.plugin.v1`); its Core manifest carries `PluginKind.COMMS_MODEL` and binds
it to this scenario via `attributes.scenario_id = "lunar-polar-ice-prospecting-v1"`.

**Production recipe** — the four inputs that determine the digest:

- **Kernels.** The same NAIF **DE440** set the world is built against (LSK `naif0012`, text PCK
  `pck00011`, binary PCK `moon_pa_de440_200625`, FK `moon_de440_250416`, SPK `de440`), plus the relay
  SPK below.
- **Relay SPK.** A notional relay spacecraft, NAIF id **-90001**, generated by `link`
  `scripts/build_relay_spk.py` from pinned elements: circular **polar** orbit at **500 km** altitude
  (ecc 0.0, inc 90°, RAAN/argp/M 0°, `MOON_GM = 4902.800118 km³/s²`), sampled on a 60 s grid as a
  type-9 degree-7 segment.
- **DEM / world.** Occlusion is resolved against the **pinned world artifact**
  `shackleton-de-gerlache-v1:0.4.0` — pulled by digest through the Core `world_provider` contract, so
  the plan is computed against exactly the terrain this scenario pins. The published artifact records
  this as `provenance.source_content_hashes.terrain`, and that value is **byte-identical to the `world`
  digest above** — an independent check that plan and terrain agree.
- **Node set & fidelity config.** 8 nodes: 4 surface (`astro-mine.fleet.prospecting-rover`,
  `astro-mine.fleet.excavator` — both in PSR; `astro-mine.fleet.hauler`,
  `astro-mine.fleet.isru-plant` — on the lit ridge), `astro-mine.fleet.relay-orbiter`, and 3 DSN
  stations (`DSS-14-Goldstone`, `DSS-63-Madrid`, `DSS-43-Canberra`). The robots carry their **Fleet
  SADF `identity.id`**, which is the whole point of plan `0.2.0` (astro-mine-link#30): Sim binds a
  contact node to an agent by **exact id match**, and the `0.1.0` plan's bare names (`excavator`, …)
  never intersected the ids Sim derives from the fleet pins — so no observation was masked and
  `comms_robustness` scored *not applicable* on a scenario that pins a complete contact plan. The DSN
  stations keep their catalogue ids and bind to no agent, which is correct: they are not robots in the
  swarm. Window search steps at **60 s** and refines to **5 s**, with a
  **3 dB** link margin and `link_surface_to_ground = True`, over a **30-day** epoch window from TDB
  **946 728 000** → **949 320 000** (2030-01-01, the full mission the episode runs — plan `0.3.0`,
  astro-mine-link#34). Yields **1 108** contact intervals.

## When the episode runs — `episode.start_epoch`

A `ContactPlan` is a plan **over a window of time**, so whether it applies at all depends on *when*
the episode runs — and until `spec_version` 0.3.0 nothing said. `EpisodeSpec` carried only a horizon
and a sim-time cap, so Sim fell back to its own default (`J2000_EPOCH`, TDB 0.0) while this
scenario's plan covers 24 h at **TDB 946 728 000** (2030-01-01). Thirty years apart. Every contact
interval was inactive at every tick, `earth_contact` was false forever, and `comms_robustness`
scored a confident **0.0** — not `not applicable`, which at least announces itself, but a number
with the shape of a result and none of the content.

So the episode now declares its epoch, pinned to the plan's window start, and Sim refuses a plan
that is never in contact anywhere in the episode rather than silently scoring zero. It is the
temporal twin of the node-id guard: a plan whose *nodes* name no agent masks nothing and scores
*not applicable*; a plan whose *window* does not reach the episode masks everything and scores a
confident zero. Both are vocabulary errors — one in the id namespace, one on the clock.

> **Resolved in plan `0.3.0` (astro-mine-link#34).** Through plan `0.2.0` the window was 24 h
> (link's `ANCHOR_EPOCH_WINDOW`: one Earth rotation, ~12 relay periods — a well-chosen
> *representative* window) while this episode is **30 days** (`43 200 x 60 s`). Beyond the first day
> the plan said nothing, and Sim's mask reads that as *no contact*, so `comms_robustness` over the
> full horizon was silently diluted by the 29 days the plan did not model — the third and last of the
> anchor's comms defects, after the node-id mismatch (link#30) and the J2000-vs-2030 epoch mismatch
> (bench#48). Plan `0.3.0` extends the window to the full mission (TDB **946 728 000** →
> **949 320 000**), so `comms_robustness` now means "connected over the mission". The 24 h reasoning
> survives as the *representative sub-window* the geometry was chosen around; the mission is ~30 such
> rotations.

## The Core contract pin — `core_schema_digest`

`core_interface` pins each Core interface at `0.1.0`, and `VERSIONING.md` §4 freezes those versions
for every interface **through Phase 3**. They are therefore *constant across every Core revision* and
carry no information: two Core revisions with materially different schemas satisfy the identical
`core_interface` block. Version negotiation is a deliberate no-op while the version is frozen.

`core_schema_digest` is what actually pins the contract — `astro_mine.core.SCHEMA_DIGEST`, a content
hash over the exact Core schema set (the JSON Schemas, the Capnp observation schema, the units
conformance table, and every `.proto`). It is a *contract* pin, and the piece that makes CX-REPRO's
byte-for-byte guarantee real:

- **`resolve_scenario()` verifies it and fails loud** (`IncompatibleCoreSchema`) when the installed
  Core's schemas differ from the pin — the check `assert_core_compatible()` cannot perform while the
  interface version is frozen.
- It is folded into `spec_hash` (hence `scenario_hash`), so a scenario resolved against a different
  Core contract is a *different task*, not silently the same one.
- It is recorded in run provenance (`Result`, `ProvenanceBundle`), so a leaderboard entry can be
  audited against the exact contract it validated under.
- It is language-neutral: a non-Python binding (the Rust validator today) has no `uv.lock` to appeal
  to, but it can assert it validated against this digest.

| pinned value | source |
|---|---|
| `sha256:2ebc6353…7980` | `astro_mine.core.SCHEMA_DIGEST` (Core `v0.3.0`, the tag this repo pins; the value is unchanged at `v0.4.0` and on `main`) |

**Re-pinning.** When Core's schemas change, its `SCHEMA_DIGEST` changes and every scenario pinning
the old one stops resolving — by design, loudly. The fix is to re-author the scenario against the new
schemas under a **new `spec_version`**, never to edit a published version in place.

## Seeds

- **Public dev seeds** (`seeds.public`): `[1001, 1002, 1003, 1004, 1005]` — disclosed for development.
- **Held-out seeds** (anti-gaming, `bench.md §9`): sealed in the repo-root
  `embargo/lunar-polar-ice-prospecting-v1/heldout_seeds.json`, which is **excluded from the packaged
  zoo artifact** (the wheel ships only `src/astro_mine`). They are bound into the public spec by
  `seeds.heldout_commit` — a `sha256:` over the sealed `{salt, seeds}` payload — so they influence the
  spec hash without being disclosed. Recompute/verify with
  `python scripts/seal_heldout_seeds.py lunar-polar-ice-prospecting-v1`.
- **SECURITY (CX-SEC follow-up):** these seeds are committed to the *private* repo. **Rotate them
  before the repo flips public** (a rotation is a new spec version). Phase-0 has no encrypted
  eval-time disclosure yet; artifact-exclusion is the Phase-0 embargo mechanism.

## Episode, termination, metrics

- **Episode** spans one full lunar day/night cycle with margin (`§13`): `max_sim_seconds =
  2_592_000` (30 Earth-days; synodic month ≈ 29.53 d) and `horizon_steps = 43_200` at an illustrative
  nominal **60 s** Bench decision tick (Sim substeps are finer). Illustrative baseline, not a
  commitment (see `scenarios/README.md` honesty note).
- **Termination conditions** are named predicates the reproducibility harness (BENCH-04) evaluates,
  drawn from the scenario's hard constraints (`§12.1` LUNAR-FR-006): `critical_asset_loss` (drives
  *nights survived*), `power_floor_violation`, `thermal_ceiling_violation`.
- **Metrics** are the seven `§13` metrics, pinned by name + interface version; the implementations are
  BENCH-03 plugins (`metrics/`, not yet built).
- **Budgets** are unbounded — the always-works local tier (`LUNAR-TR-004`; `bench.md §6`). The
  standard-budget leaderboard SLO (50 seeds, `bench.md §8`) is a BENCH-06 concern, not a per-spec cap.

## Regenerating the whole spec

The content ids, seeds, and parameters above are the source of truth; the derived literals in
`scenario.json` are the **real Hub artifact digests** for every content pin (from the producers'
signed publish) and `content_hash({salt, seeds})` for `heldout_commit`. Any edit to a pinned input
changes `ScenarioSpec.spec_hash` — by design.
