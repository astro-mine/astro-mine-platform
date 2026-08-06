# astro-mine-sim

**The multi-physics engine and scenario runtime for [Astro-Mine](https://github.com/astro-mine).**
One engine couples orbital dynamics, surface mobility/contact, manipulation and granular
excavation, power/thermal evolution, and sensor models behind a single Core Environment API —
engine-pluralist, contract-singular. The beating heart.

Same-inputs-same-seed-same-output is a hard requirement, not a default: every run prints a
`Trace.content_hash` — the determinism key — so a result is checkable from its logs alone.

## Install

Sim installs as part of `astro-mine-platform`. The base install is small and offline — reduced-order
engines, no GPU, no cloud, no Bench. Heavier tiers are opt-in extras.

```bash
uv pip install -e .                 # the platform, base runtime
uv pip install astro-mine-cli       # the command line, if you want one
astro-mine sim --help
```

## The reference environment — zero content, zero setup

The smallest runnable thing this package ships, and the one that needs no document of your own: a
synthetic three-agent scenario (a wheeled rover, a tracked excavator, an orbiting relay) carried as
package data. No content store, no registry, no network, no extra.

```python
from astro_mine.sim.reference import make_reference_env

env = make_reference_env()          # a Core Environment, ready to reset()
env.reset()
```

Construction plus a full 16-tick episode takes single-digit milliseconds, so it suits a quickstart,
a smoke test, or a CI fixture. The anchor scenario is a *benchmark* — ~461 MB of pinned content over
a 43 200-tick lunar month — and is the wrong tool for all three.

A multi-agent RL consumer also needs the SADF describing those agents, to derive per-agent
observation and action spaces from. `make_reference_env_and_assets` hands over both halves:

```python
from astro_mine.sim.reference import make_reference_env_and_assets

env, assets = make_reference_env_and_assets()   # (Core Environment, {AgentId: Core Asset})
```

Both halves are **Core types**, so a consumer wraps them without this package importing the
consumer — and without the consumer importing Sim (`conventions.md` §1.1). The bridge is the
`module:attr` string the consumer already resolves at runtime, e.g. an env factory pointed at
`astro_mine.sim.reference:make_reference_env_and_assets`.

Two properties worth knowing before you train against it. **Rewards are the consumer's**:
`StepResult.rewards` stays empty, because reward shaping is a training concern and not a physics
one. **Actions are not** — the two surface agents declare `MobilityDynamics` and the env is built
with `coupled_engine_factory`, so they route to the mobility engine, which honours `MODE`, a
`VELOCITY` setpoint **and** a `GotoTask`. Different action sequences produce different
trajectories.

That last kind matters more than it looks. Sim's default kinematic engine ignores `TASK` actions
entirely, while an RL adapter typically encodes its mobility modality *as* a `GotoTask` and never
emits `VELOCITY` — so on the default engine such a policy moves nothing, any pose-derived reward
is flat with respect to it, and training runs while learning nothing. The tests pin both action
kinds. (Being action-responsive is what makes an env trainable; whether a given reward and
training budget actually converge is the consumer's business, not Sim's.)

## Quickstart — record a Sim `Scenario`

The always-available local path: run a self-contained Sim `Scenario` document and record it to
[MCAP](https://mcap.dev/). No account, no cloud, no content resolution.

```bash
astro-mine sim record --scenario-file my_scenario.json --seed 7 --out run.mcap
# prints the run's Trace.content_hash (the determinism key); re-running the same inputs prints it again
```

The MCAP holds the timestamped, schema-tagged state/observation stream — replay it in the
application's `/bench` replay view
([`astro-mine-ui`](https://github.com/astro-mine/astro-mine-ui)), or score it with `astro-mine
bench`.

Both verbs come from [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli), a separate
distribution that depends on this one; this package ships no console scripts. There is one address
for every command on the platform — `astro-mine <component> <verb>` — so it is `astro-mine sim run`
and `astro-mine sim record`, and no bare `astro-mine run` or `astro-mine-sim`.

## The two scenario schemas — know which you have

`scenario` is an overloaded word across the platform. The CLI keeps the two apart, and so should
you:

| | Sim `Scenario` | Bench `ScenarioSpec` |
|---|---|---|
| **What** | A *materialized, runnable episode* — concrete agents, `dt_s`, horizon, fidelity | A *declarative benchmark task* — pins world/fleet/prospect/link content **by hash**, plus seeds, metrics, budgets |
| **Identified by** | a **file path** (a JSON document) | an **id** (e.g. `lunar-polar-ice-prospecting-v1`, from `astro-mine bench list`) |
| **Owned by** | this repo (`astro_mine.sim.runtime`) | [Bench](https://github.com/astro-mine/astro-mine-bench) (`astro_mine.bench.scenario`) |
| **CLI** | `astro-mine sim record --scenario-file …` | `astro-mine sim run <id>` |
| **Needs** | nothing beyond the base install | the `[sim-bench]` + `[sim-hub]` extras and fetched content |

`run` bridges the two: it loads the Bench `ScenarioSpec` by id, resolves its pinned content from a
local Hub registry, and materializes a Sim `Scenario` via `sim_scenario_from_spec`
([RM-P1-SIM-01](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-and-studio.md)).

```bash
uv sync --extra sim-bench --extra sim-hub
# resolve the anchor's pinned content into a local registry first (see `astro-mine bench fetch`),
# then point --registry (or $ASTRO_MINE_HUB_REGISTRY) at it:
astro-mine sim run lunar-polar-ice-prospecting-v1 --registry ./hub-registry --seed 1001 --out anchor.mcap
```

### Content is not code — you need both

`astro-mine bench fetch <scenario>` obtains the anchor's pinned content by digest and prints the
store path. It fetches **content**. Rebuilding a world bundle back into a live `WorldProvider` is
Worlds' job, and Sim reaches it through the `astro_mine.providers` entry-point group rather than by
importing it — so the producers have to be **present**. Since consolidation they always are: all
three are modules in the one distribution, and there is nothing extra to install.

| pin | producer | without it |
|---|---|---|
| world | `astro_mine.worlds` | no terrain/illumination — `nights_survived` cannot score |
| prospect | `astro_mine.prospect` | no sealed field — `discovery_latency` never trips, ISRU sees no abundance |
| link | `astro_mine.link` | no contact plan — `comms_robustness` cannot score |

The table is kept because the *failure modes* are unchanged — a pin can still fail to rebuild, and
this is what each costs when it does.

A Sim-backed anchor run also needs a **SPICE metakernel** for the body-fixed frames the world
resolves against. Kernels are not shipped — obtain SPK/PCK/FK/LSK kernels from
[NAIF](https://naif.jpl.nasa.gov/naif/data.html) and list them in a `.tm` — and supply it either way:

```bash
astro-mine sim run <scenario-id> --metakernel /kernels/lunar.tm
export ASTRO_MINE_SPICE_METAKERNEL=/kernels/lunar.tm    # or once, for the shell
```

The env var is also how the **scoring** path gets a pool: `astro-mine bench score --runner sim`
passes Sim a content store and nothing else — Bench has no vocabulary for SPICE and must not grow
one (conventions.md §1.1) — so the `sim` runner reads `$ASTRO_MINE_SPICE_METAKERNEL` itself, exactly
as it already resolves its store from `$ASTRO_MINE_HUB_REGISTRY`.

Kernels are furnished once the scenario is materialized, so its epoch window is validated against
the SPK pool **up front** (`spice.md` §10): a kernel set that stops short of a 30-day lunar episode
fails in the first second rather than ~18,000 ticks in. A run that needs no geometry — `record` on a
self-contained scenario — needs no kernels and no configuration.

**Sim tells you when this is wrong rather than scoring blind.** `astro-mine bench score --runner
sim` **refuses** a scenario whose pinned providers did not rebuild — a scorecard is a claim, and
there is no honest use for scoring the anchor against a world that was never loaded.
`astro-mine sim run` warns and proceeds, because recording a partial run is a legitimate ask at the library tier,
and a run that *was* blind records that fact in its own provenance. Pass
`SimEpisodeRunner(allow_unresolved_content=True)` to score anyway, deliberately.

`record` on a Sim `Scenario` file needs none of this — it is self-contained and works offline.

## The anchor baseline

`astro-mine bench score --runner sim` scores a **capability-aware mode policy**
(`astro_mine.sim.bench._policy`). Bench asks this package for it through its optional
`DefaultPolicyProvider` seam, so the baseline is chosen by the runner that resolves the content, not
guessed by a CLI that cannot read a SADF document.

Each agent is held in a mode derived from its own SADF — its capability tags say what it is for, its
`power.loads_by_mode` says which mode names it actually publishes a power draw for. On the anchor
roster that is:

| asset | mode |
|---|---|
| `prospecting-rover` | `prospect` |
| `excavator` | `excavate` |
| `hauler` | `drive_empty` |
| `relay-orbiter` | `downlink` |
| `lander` | `idle` |
| `isru-plant` | `idle` |

**What it is not.** It does not plan, allocate, navigate, or react to anything it observes — a
*replaceable example*, the conformance floor for the Sim-backed path, exactly as Bench's
`BaselinePolicy` is for the fixture path. A leaderboard needs something to beat, not something
strong.

**What it produces.** Scored over the anchor (`lunar-polar-ice-prospecting-v1`, seed 1001):

```
scenario:  lunar-polar-ice-prospecting-v1
runner:    astro-mine-sim/0.1.0
scorecard: sha256:5d31193541cd3f43a6bf409358a7662e0db12e05eb647e9522b176f82bb8bfa2

  water_mass                    47.612 kg           (up-better, n=1)
  energy_per_kg            2.53768e+06 J/kg         (down-better, n=1)
  information_gain             9000.68 nat          (up-better, n=1)
  psr_area_characterized    1.7975e+08 m^2          (up-better, n=1)
  nights_survived                    0 dimensionless (up-better, n=1)
  comms_robustness            0.429576 dimensionless (up-better, n=1)
  discovery_latency                  0 s            (down-better, n=1)
```

All seven metrics score. `water_mass` is non-zero because since
[#64](https://github.com/astro-mine/astro-mine-sim/issues/64) extraction consumes regolith that was
dug, carried and delivered — an ISRU plant is no longer excluded from extraction modes, so the
stored mass is earned by the value chain rather than conjured from a mode string. `information_gain`
and `psr_area_characterized` score because [#72](https://github.com/astro-mine/astro-mine-sim/pull/72)
conditions a real belief ([#66](https://github.com/astro-mine/astro-mine-sim/issues/66)).

The numbers are the conformance floor, not a target: `energy_per_kg` is enormous and
`nights_survived` is `0`. That is what a leaderboard is for.

**`discovery_latency` scores `0.0`** because Bench's `discovery_threshold` defaults to `0.0`, so the
first valid reading trips it. A real threshold belongs in the `ScenarioSpec`, which cannot yet
express one — this is the one remaining caveat in this section.

## Container entrypoint

The workload image (`docker/Dockerfile`) runs `python -m astro_mine.sim`; a
[Cloud](https://github.com/astro-mine/astro-mine-cloud) job appends `--scenario … --seed … --out …`.
That flat form routes to `record` (a laptop run and a cluster run are the same run, cloud.md §4) —
`--scenario` is a deprecated alias for `record`'s `--scenario-file`.

## Extras

Each engine tier and integration is opt-in, so the base wheel stays lean and the local tier always
works. A scenario that selects a tier without its extra fails with a clear message naming it.

| Extra | Unlocks |
|---|---|
| `sim-hub` | Resolve content-pinned bundles from a Hub registry (the `run` path) |
| `sim-bench` | The Sim-backed [Bench](https://github.com/astro-mine/astro-mine-bench) runner — real-physics scoring + the determinism gate (`astro_mine.sim.bench`) |
| `sim-dem` | High-fidelity DEM granular-excavation engine (numpy soft-sphere) |
| `sim-surrogate` | The learned [Surrogate](https://github.com/astro-mine/astro-mine-surrogate) fidelity tier (ONNX Runtime) |
| `sim-brax` / `sim-ray` | Brax/MJX GPU-vectorized contact/training tier, and Ray fan-out |
| `sim-mujoco` | The MuJoCo articulated wheel-soil mobility/contact tier |
| `sim-orekit` | The Orekit higher-fidelity orbital propagator (bundled JVM) |
| `sim-report` | Error-budget reports as Parquet (per-tier deviation-vs-reference) |
| `sim-service` | The gRPC `EnvironmentService` skin |

## Layout

```
src/astro_mine/sim/         # import path: astro_mine.sim
  runtime/        # Scenario + load_scenario, SPICE clock, content resolution, the Simulator
  engines/        # the pluggable physics tiers (orbital, mobility, granular/DEM, contact, ...)
  coupling/       # the multi-physics coupler (frame/time bridge via astro_mine.spice)
  scheduler/      # fidelity policy: which tier runs, and when a surrogate may substitute
  sensors/        # sensor models projecting world/field state into the Observation stream
  power_thermal/  # power/thermal evolution (battery, survival across lunar night)
  comms/          # the comms/connectivity environment (contact windows, masks)
  isru/           # in-situ resource extraction/storage (the ISRU model)
  recording/      # record_episode → MCAP; the one output format (replay + Bench scoring)
  service/        # the gRPC EnvironmentService + Ray-actor skin ([service] extra)
  bench/          # the Sim-backed Bench runner ([bench] extra) — Bench never imports Sim
  reference/      # the offline reference Scenario + Core-typed env/asset factories (no extra)
  validation/     # scenario/document validation
tests/sim/                  # mirrors the package layout
```

## Development

Sim is part of the [`astro-mine-platform`](../../../README.md) distribution — one repository, one
environment, one test suite. See [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup, then:

```bash
python scripts/test.py sim
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow, and the
[architecture](https://github.com/astro-mine/docs/blob/main/architecture/sim.md) for the design.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
