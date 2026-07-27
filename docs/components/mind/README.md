# astro-mine-mind

**Hierarchical autonomy framework for [Astro-Mine](https://github.com/astro-mine).**
The three-tier composition — a strategic **mission planner** → per-agent **task-and-motion
planners (TAMP)** → reactive **local controllers** — that turns a stated objective into
actuator-level commands across a heterogeneous swarm. Every tier is a pluggable, swappable
implementation of Core's Policy/Planner contract; behavior trees run throughout and every
emitted action is wrapped by Guard. Mind orchestrates the Core interfaces; it does not own
them.

> **Status:** Phase 1 — the autonomy stack (RM-P1-MIND-01…07) is implemented: declarative stack
> spec → composed hierarchy graph, the executive tick loop, behavior-tree execution, the
> mission/TAMP/control backends (reference **and** native engines), Allocate delegation,
> mandatory Guard-wrapping, degrade-not-collapse contingent plans, and the decision-trace /
> replay surface. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/mind.md) and the
> [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Two invariants, enforced everywhere

- **Guard-wrapped output is the only output.** The executive is the sole `Environment` holder;
  every emitted action leaves through the mandatory shield stage.
- **Determinism on demand.** Seed + pinned plugin set + fixed inputs ⇒ byte-identical golden
  decision traces; CI fails on drift.

## Planner backends

The framework commits to the *interface*, not the backend: a tier binds a plugin **by name** in
the stack spec, so swapping an engine is a spec edit, not a code change.

| Tier | Reference — pure-Python, **bit-exact**, CI-tested default | Native engine — optional extra |
|---|---|---|
| Mission | `mind.mission.pddl` — generates a PDDL problem per replan, solves it deterministically | `mind.mission.up` — **unified-planning** over **Fast Downward** / OPTIC / ENHSP (`[pddl]`) |
| TAMP | `mind.tamp.sampling` — symbolic task + pure-Python RRT | `mind.tamp.ompl` — **OMPL** RRT\*/PRM\*/BIT\* + **FCL** collision (`[native]`) |
| Control | `mind.control.pid`, `mind.control.mpc` | `mind.control.onnx` — ONNX Runtime hosting a Learn policy (`[onnx]`) |

```bash
uv sync --extra pddl --extra native   # install the real engines
uv run pytest -m "pddl or native"     # exercise them (deselected in CI)
```

The generated PDDL problem is **genuinely solvable**: the domain models the assignment decision
itself (`assign` binds a free agent to an unassigned region; `prospect` requires that binding), so
a real engine *derives* the agent→region decomposition from the goal rather than rubber-stamping
one Mind already made.

Native backends are deselected in CI (their wheels bundle native binaries) and are
coverage-omitted; the reference backends carry the coverage gate and are the golden-trace
baseline. OMPL in particular seeds a **process-global** RNG under a wall-clock solve budget, so
its manifest declares `determinism_class: tolerance` — which is exactly why the bit-exact
reference RRT, not OMPL, is the default.

### Behavior trees: the Groot XML dialect, no native BehaviorTree.CPP binding

`bt/` implements the **BehaviorTree.CPP v4 / Groot XML dialect** — parsed, validated, and
round-tripping (`parse(to_xml(t)) == t`) — over a closed node AST, ticked by a deterministic
reactive engine with faithful `success`/`failure`/`running` propagation. The **interop contract is
the XML**: trees author and inspect in Groot.

What is deliberately *not* here is a binding to the BehaviorTree.CPP C++ engine, because **no such
Python binding is distributed** — the project publishes none, and no candidate package exists on
PyPI. Binding it would mean vendoring a CMake + pybind11 build of a C++ library into this wheel,
which is disproportionate for Phase 1 and would break the tier-1 "local install must always work"
rule (`conventions.md §7`) for no behavioral gain over the XML the native engine already speaks.
See [#13](https://github.com/astro-mine/astro-mine-mind/issues/13) for the full rationale.

## The stack spec — the whole idea, in one file

*Swapping an engine is a spec edit, not a code change.* Here is a complete, runnable one — the
shipped `lunar_prospecting` reference stack (`astro_mine.mind.reference`, resolvable from an
installed Mind):

```yaml
stack_spec_version: "0.1"
stack_spec:
  id: reference-lunar-prospecting
  name: Reference lunar-polar prospecting stack
  scenario_ref: lunar-polar-prospecting
  tiers:
    - role: mission                       # strategic: assign each agent a prospect region
      plugin: mind.reference.mission
      validity_horizon_s: 5.0
      replan_triggers:
        - kind: plan_expired
    - role: tamp                          # per-agent: turn the assignment into a GOTO
      plugin: mind.reference.tamp
      replan_triggers:
        - kind: periodic
          every_ticks: 3
    - role: control                       # reactive: close the loop to actuator commands
      plugin: mind.reference.control
  shield:                                 # mandatory — every action leaves through it
    plugin: mind.reference.shield
```

Each `plugin:` binds a tier to an implementation **by name**, discovered at compose time from the
`astro_mine.mind.tier_plugins` entry-point group. Swapping the mission engine to Fast Downward is a
one-line edit — `plugin: mind.mission.up` — with **no framework change**; the native backends are in
the [Planner backends](#planner-backends) table.

Compose it and inspect what bound where — no episode needed:

```python
from astro_mine.mind import compose, TierRegistry
from astro_mine.mind.reference import load_stack_resource

doc = load_stack_resource("lunar_prospecting.yaml")   # a shipped reference stack, by name
registry = TierRegistry.from_entry_points()            # discover installed tier plugins
graph = compose(doc, registry)                         # resolve to a runnable hierarchy graph

for tier in graph.tiers:
    print(f"{tier.role.value:<8} -> {tier.plugin_name} @ {tier.manifest.version}")
# mission  -> mind.reference.mission @ 0.1.0
# tamp     -> mind.reference.tamp @ 0.1.0
# control  -> mind.reference.control @ 0.1.0
```

### The CLI — `astro-mine-mind`

```bash
astro-mine-mind stacks                         # list the shipped reference stacks + manifests
astro-mine-mind validate lunar_prospecting.yaml# schema + a registry check (unregistered plugin = fail)
astro-mine-mind compose  lunar_prospecting.yaml# tier -> plugin @ version, from which entry-point group
```

A bare name resolves against the shipped stacks, so the CLI works without a checkout. `validate`
catches the failure a shape-only check misses — a stack binding a plugin no installed package
registers fails with the entry-point group and the missing name (**the** most common real error).
There is **no `run`**: stepping a stack needs a Core `Environment`, which is Sim's job — Mind must
not import Sim (the narrow waist). Run an episode with `astro-mine-sim run` over a composed stack.

### The reference stacks — a curriculum

Six stacks ship as package data; each is a deliberate variation on the same spine:

| Stack | Demonstrates |
|---|---|
| `lunar_prospecting` | the base three-tier spine over the reference plugins |
| `lunar_prospecting_backends` | swapping in the RM-P1-MIND-03 native engines (`mpc` control, native TAMP/mission) |
| `lunar_prospecting_allocate` | delegating region assignment to Allocate (`mind.allocate.greedy`) |
| `lunar_prospecting_anchor` | the anchor scenario wired to the **real Guard shield** and Allocate |
| `lunar_prospecting_bt` | **behavior-tree** execution instead of tier composition |
| `lunar_prospecting_degrade` | degrade-not-collapse: a contingent fallback under the constraint shield |

The **13 manifests** under `reference/manifests/` are the Core `PluginManifest` descriptors for the
tier/shield plugins each stack binds (`control`, `mpc_control`, `onnx_control`, `pid_control`;
`tamp`, `sampling_tamp`, `ompl_tamp`; `mission`, `pddl_mission`, `up_mission`; `greedy_allocator`;
`shield`, `constraint_shield`) — populating the mission / TAMP / control / shield tier vocabulary.

Both are advertised under the **`astro_mine.mind.tier_plugins`** entry-point group — the hub
Allocate and Guard register their real backends into, as drop-in replacements for these references,
with no dependency back into Mind. The reference plugins in `reference/tiers.py` are the worked
template for writing your own; the full plugin-authoring recipe for that group is the forthcoming
platform guide (Wave 22.5 / G2.8).

## Layout

```
src/astro_mine/mind/        # import path: astro_mine.mind
  reference/stacks/         # the 6 reference stack specs (package data)
  reference/manifests/      # the 13 tier/shield plugin manifests (package data)
tests/                      # mirrors the package layout
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module-by-module map.

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**.

```bash
conda create -n astro-mine-mind python=3.12
conda activate astro-mine-mind
uv sync && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
