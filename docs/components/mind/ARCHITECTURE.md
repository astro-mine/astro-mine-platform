# Architecture

`astro-mine-mind` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/mind.md`](https://github.com/astro-mine/docs/blob/main/architecture/mind.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package (the hierarchical autonomy framework — a three-tier mission planner →
  per-agent TAMP → local controller composition over Core's Policy/Planner API, with
  behavior trees throughout and mandatory Guard-wrapping).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).

## Package layout (`src/astro_mine/mind/`)

The narrow-waist spine (RM-P1-MIND-01) plus the RM-P1-MIND-02…07 autonomy modules, each a
pluggable realization behind a Core Policy/Planner sub-interface (`mind.md §3`):

| Module | Role | Issue |
|---|---|---|
| `spec/`, `compose/`, `registry/` | declarative stack spec → validated hierarchy graph; plugin discovery/gating | MIND-01 |
| `exec/` | the executive tick loop; `strategy.py` (direct composition) + `degrade.py` (comms-aware) + `plan.py` (behavior over Core's RFC-0006 `Plan`/`ContingentPlan`) + `build_strategy` seam | MIND-01, -06 |
| `bt/` | behavior-tree execution — Groot/BT.CPP-v4 XML, node AST, deterministic reactive engine, `BehaviorTreeStrategy`. No native BT.CPP binding: none is distributed (re-scoped, see the package docstring) | **MIND-02** |
| `mission/planner/` | PDDL/temporal mission backend — reference symbolic planner + `native/` **unified-planning** (Fast Downward/OPTIC/ENHSP) via `[mind-pddl]` | **MIND-03** |
| `tamp/` (`task/`, `motion/`) | symbolic task + sampling motion behind the `MotionPlanner` protocol — reference RRT + `native/` **OMPL** (RRT\*/PRM\*/BIT\*) + **FCL** via `[mind-native]` | **MIND-03** |
| `control/` (`policy/`) | classical PID/MPC + ONNX-Runtime learned controller (Core `OnnxPolicy`; `[mind-onnx]`) | **MIND-03** |
| `mission/allocate/` | thin adapter delegating assignment to Allocate over the Core `Allocator` sub-interface | **MIND-04** |
| `guardrail/` | the single, mandatory shield egress + `ReportingShield` intervention provenance | **MIND-05** |
| `coord/` | decentralized gossip/consensus. The validity-horizoned `Plan`/`ContingentPlan` artifacts are **Core-owned** (`astro_mine.core.plan`, RFC-0006); Mind layers behavior over them in `exec/plan.py` and no longer keeps a local copy | **MIND-06** |
| `belief/` | partial-observability tier input (observations + comms mask) | MIND-01 |
| `trace/` | neutral decision-record model, canonical-JSON determinism gate, MCAP serializer, content-hash provenance, plan explanation | MIND-01, **-07** |
| `reference/` | replaceable example tiers/shields/backends + stacks (`conventions.md §1.3`) — swapped via the registry, never privileged |

**Invariants enforced everywhere:** Guard-wrapped output is the only output (the executive is
the sole `Environment` holder; every action passes through `guardrail/`), and
determinism-on-demand (seed + pinned plugin set + fixed inputs ⇒ byte-identical golden decision
traces; CI fails on drift).

**Native backends.** The heavy engines named in `mind.md §4` bind behind these same Core contracts
as optional extras, discovered through the ordinary plugin registry — `unified-planning`
(`[mind-pddl]`, plugin `mind.mission.up`), `OMPL`/`FCL` (`[mind-native]`, plugin
`mind.tamp.ompl`), ONNX Runtime (`[mind-onnx]`), and MCAP (`[mind-recording]`). Their providers defer the heavy import into the
plugin *factory*, so a base install still discovers them and only *binding* one needs the extra.
The pure-Python reference realizations remain the CI-tested default, carry the coverage gate, and
are the golden-trace baseline (the native adapters are coverage-omitted and marker-deselected).
BehaviorTree.CPP is the one exception — no Python binding is distributed, so `bt/` implements its
Groot XML dialect rather than embedding the engine (re-scoped; see `bt/__init__.py`).
