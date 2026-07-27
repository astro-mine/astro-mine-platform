# astro-mine-allocate

**Combinatorial task allocation and scheduling for [Astro-Mine](https://github.com/astro-mine).**
A solver-neutral **Allocation IR** with a CP-SAT backend — who does what, when, and where
— under power, comms, and terrain constraints, computed *anytime* and *explainable*.

> **Status:** Phase 1 — the `RM-P1-ALLOC-*` MVP is implemented and measured. The Allocation IR
> (JSON Schema + Protobuf wire form) with its compiler and independent feasibility verifier, the
> **CP-SAT** backend behind the solver strategy, the power / comms-window / terrain constraint
> builders, the info-gain-vs-ROI objective, the anytime incumbent stream, explainability (binding
> constraints, objective decomposition, an irreducible infeasible set), and seeded-golden
> determinism all ship. The [scale benchmark](#scale-benchmark) solves **25 assets / 252 tasks**
> over a three-day horizon to a proven optimum well inside its deadline. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/allocate.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## What it does

```python
from astro_mine.allocate import AllocationPlanner, compile_request, verify_feasible

allocation = AllocationPlanner(backend="cp-sat").solve(request)   # who does what, when
assert verify_feasible(allocation, compile_request(request))      # re-checked, not trusted
```

An `AllocationRequest` — tasks (kind, location, **disjoint** time windows, precedence, duration,
uncertain value) and assets (SADF capability tags, budgets) — compiles to the solver-neutral
**Allocation IR**, which a backend lowers and which the returned plan is independently re-checked
against. The result is a per-asset, time-ordered, **non-overlapping** schedule with a realized
objective, an optimality gap, the binding constraints, and full reproducibility provenance — or an
honest `INFEASIBLE` carrying an irreducible infeasible set naming *why*.

Allocate is **not** the safety authority: every plan is re-checkable by a party that does not trust
the solver (`verify_feasible`) — which is exactly what
[Guard](https://github.com/astro-mine/astro-mine-guard) does at execution.

## Layout

```
src/astro_mine/allocate/
├── api/          # Core Policy/Planner allocation sub-interface; request/response types; manifest
├── model/        # The Allocation IR (+ JSON Schema/proto), its compiler, verifier, and lowerings
│   ├── ir/       #   the solver-neutral model: variables, constraints, objective, scheduling
│   └── compile/  #   IR -> CP-SAT
├── constraints/  # Power, comms-window, terrain builders + the declared policy / cached cost table
├── solvers/      # Backend plugins behind one Solver strategy (CP-SAT; a no-dependency stub)
│                 #   + the registry third-party backends register into (see below)
├── anytime/      # Incumbent/bound streaming, deadline status, warm-start seam
├── explain/      # Binding constraints, objective decomposition, IIS
└── mind.py       # The optional [mind] tier-plugin binding (see below)
tests/            # mirrors the package layout
```

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**. This is the local
tier — it runs on one workstation, no cloud or account required.

```bash
conda create -n astro-mine-allocate python=3.12
conda activate astro-mine-allocate
uv sync             # resolves against the tag-pinned astro-mine-core (v0.2.0)
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Contributing a solver backend

`solvers/` is a real extension point: a third-party backend reaches the dispatch table through a
Python entry point, with **no PR to Allocate** (conventions.md §7; allocate.md §11).

Implement the `Solver` strategy — lower the IR, search within the budget, stream improving
incumbents — and advertise a factory under `astro_mine.allocate.solvers`. The entry point's
**name is the backend id** recorded in a plan's provenance:

```toml
# your pyproject.toml
[project.entry-points."astro_mine.allocate.solvers"]
my-solver = "my_pkg.backend:MySolver"
```

```python
class MySolver:  # the astro_mine.allocate.solvers.base.Solver protocol
    def __init__(self, *, task_kinds, durations=None): ...

    def solve(self, ir, budget, *, hints=None):
        """Yield Incumbents with monotonically improving bounds; the last carries the status."""
```

`pip install` it and the id resolves everywhere a built-in does:

```python
from astro_mine.allocate import AllocationPlanner, known_backends

known_backends()                                   # ('cp-sat', 'my-solver', 'trivial-stub')
AllocationPlanner(backend="my-solver").solve(request)
```

Notes on the contract:

- **Built-ins and plugins resolve through one path.** `cp-sat` and `trivial-stub` are seeded in
  code only so the base package works from a raw checkout; they get no privileges.
- **Discovery is lazy.** Listing backends reads entry-point *names* and never imports one, so
  installing your plugin costs an import only when someone actually asks for its id.
- **You may not shadow a built-in id.** Advertising `cp-sat` is a hard error naming both
  claimants — which solver produced a plan is provenance (RM-P1-ALLOC-07), so an ambiguous id is
  never resolved silently.
- **Your plan is re-checked.** Every feasible plan, from any backend, is independently verified
  against the IR by `verify_feasible` — Allocate is not the safety authority (allocate.md §9),
  which is exactly what makes accepting a third-party solver safe.
- **Determinism is expected.** Same model + same seed + same pinned backend ⇒ same plan.

Your solver's *public* face is still the Core manifest it advertises itself through; the entry
point is only the in-process discovery mechanism.

## Scale benchmark

The Phase-1 exit criterion — "tens of robots / hundreds of tasks solved to a few-% gap within a
deadline" (allocate.md §8) — ships as a **marker-gated** benchmark, opt-in so a wall-clock
assertion never flakes normal CI:

```bash
uv run pytest -m scale
```

It builds a seeded 25-asset / 252-task lunar-polar instance over a three-day horizon — terrain
keep-outs, comms gating, energy budgets, per-asset no-overlap and disjoint-window disjunctions all
live — and asserts a **proven optimum inside a 60 s deadline** within a 5 % gap bound, plus a small,
promptly-extracted irreducible infeasible set on a localized conflict. Each run records its numbers
(instance size, IR size, wall clock, gap, pinned OR-Tools version) under `benchmarks/`, and the
scheduled `scale-bench` workflow uploads them so the trend is tracked over time.

## The `[mind]` extra — binding into an autonomy stack

Per RFC-0006's sibling-binding convention, Allocate ships the binding that lets
[Mind](https://github.com/astro-mine/astro-mine-mind) delegate assignment to the **real** CP-SAT
planner — with **no `mind → allocate` dependency in either base package**:

```bash
uv pip install "astro-mine-allocate[mind]"
```

The extra registers `allocate.planner` under Mind's `astro_mine.mind.tier_plugins` entry-point
group. Mind's `TierRegistry.from_entry_points()` discovers it, gates its Core manifest, and a stack
spec names it declaratively in the `allocator` tier:

```yaml
stack_spec_version: "0.1"
stack_spec:
  id: lunar-prospecting
  name: Lunar polar prospecting
  tiers:
    - role: allocator
      plugin: allocate.planner        # the real CP-SAT planner, not Mind's greedy stand-in
      params: { backend: cp-sat, deadline_s: 5.0 }
  shield:
    plugin: mind.reference.shield
```

Behind Mind's own `AllocationAdapter`, the plugin reads its request from the shared
`allocation.request` `DecisionContext.extras` key and reports the solve back through Mind's
`AllocationReporter` seam, so a delegated decision carries the solver + seed that reproduce it.
Without the extra, Mind keeps using its in-repo `GreedyReferenceAllocator` stand-in. See
`src/astro_mine/allocate/mind.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
