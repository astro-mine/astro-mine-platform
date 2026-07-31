# Architecture

`astro-mine-allocate` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/allocate.md`](https://github.com/astro-mine/docs/blob/main/architecture/allocate.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package (combinatorial task allocation and scheduling — a solver-neutral
  Allocation IR with a CP-SAT backend, power/comms/terrain constraints, anytime and
  explainable).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).

## What is implemented here

The Phase-1 MVP (`RM-P1-ALLOC-01`…`08`) is in place; the map from the design doc to the code:

| allocate.md | Module | State |
|---|---|---|
| §3 canonical model, `model/ir/` | `astro_mine.allocate.model.ir` | The **Allocation IR** — decision variables, linear constraints, the `NO_OVERLAP`/`CUMULATIVE` scheduling families, objective terms — with a JSON Schema **and** Protobuf wire form, a byte-stable compiler, and an **independent feasibility verifier** (`verify_feasible`, the Guard-recheckable oracle). |
| §3/§4 `model/compile/`, `solvers/` | `astro_mine.allocate.model.compile.cpsat`, `.solvers` | A pure IR → **CP-SAT** lowering behind one `Solver` strategy, plus a no-dependency `trivial-stub` backend so the local tier always works. |
| §6 constraint builders | `astro_mine.allocate.constraints` | **Power**, **comms-window**, and **terrain-traversability** builders reading upstream truth only through Core contracts (no sibling imports), over a declared, content-addressed modeling policy + cached cost table. |
| §3 objective | `astro_mine.allocate.constraints.value`, `.infogain` | Prospect-refined ROI and the **info-gain-vs-ROI** split. |
| §2 anytime | `astro_mine.allocate.anytime` | Streaming incumbents with **monotone** bounds, an explicit gap, an honest deadline status, and the warm-start seam. |
| §9/§10 explain | `astro_mine.allocate.explain` | Binding constraints (which window/floor/keep-out bound the plan), objective decomposition, and an **irreducible infeasible set** (CP-SAT assumptions + a deletion filter). |
| §8 determinism & scale | `tests/test_cpsat_determinism.py`, `tests/test_scale_benchmark.py` | Seeded golden plans/certificates, and the marker-gated **scale benchmark** (25 assets / 252 tasks, three-day horizon) that measures the Phase-1 exit criterion. |
| RFC-0006 sibling binding | `astro_mine.allocate.mind` | The optional **`[allocate-mind]` extra**: registers the real `AllocationPlanner` as Mind's `allocator`-role tier plugin, with no `mind → allocate` base dependency. |

Not yet here (per allocate.md §12, Phase-1-later/Phase-2+): the MILP track and cross-solver
consistency tests, learned `guidance` (GNN warm starts, learning-to-branch), `decompose`
(rolling-horizon / spatial partition), the decentralized auction fallback, and
stochastic/robust formulations. A per-*pair* objective family (value minus traverse cost) is the
notable gap in the objective — today's coefficients are per-task, so any feasible cover is optimal.

> **Runtime dependencies.** `astro-mine-core` (the narrow waist), `protobuf` (the IR wire form),
> and **OR-Tools** CP-SAT — the Phase-1 MVP backend, so it is a core dependency, not an extra
> (Gurobi is the only optional solver, allocate.md §4). Mind's binding is pulled in only by the
> optional `[allocate-mind]` extra.
