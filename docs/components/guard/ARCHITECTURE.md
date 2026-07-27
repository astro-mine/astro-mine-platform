# Architecture

`astro-mine-guard` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/guard.md`](https://github.com/astro-mine/docs/blob/main/architecture/guard.md):
  runtime assurance — the verifiable shield that wraps any policy so declared hard
  constraints cannot be violated (SafetySpec, STL/MTL runtime monitors, CBF/reachability
  shields, simplex backup, the PolicyShield over the Core Policy/Planner API); purpose,
  principles, runtime, data, integration, security, and roadmap alignment for this package.
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

> **Trusted safety core (Rust).** Guard's trusted computing base — the `arbiter`, `shields`,
> `monitors`, `backup`, and `spec` evaluator ([`guard.md` §3–4](https://github.com/astro-mine/docs/blob/main/architecture/guard.md))
> — is a small, deterministic, allocation-free **Rust** crate in [`rust/`](rust/), added in
> **RM-P1-GUARD-02**. It builds and tests standalone (no Python) for the edge, and the wheel is
> **maturin**-built so it bundles the core as the `astro_mine.guard._core` PyO3 extension. The
> Python layer under `src/` (spec authoring, the `wrap`/`models`/`audit` modules, and the
> forthcoming `coord`) is the untrusted orchestration around it.

> **Constraint-source adapters (`models`).** `astro_mine.guard.models` (**RM-P1-GUARD-04**) is the
> **untrusted** layer that resolves a `SafetySpec`'s abstract constraint sources against Core-typed
> **Fleet SADF** budgets (`SadfBudgets`) and **Worlds** terrain/illumination (`WorldsTerrain`) — the
> charging-window key, terrain slope, and surface temperature — and marshals them into the per-tick
> signal vector via the `WorldsFleetSignalResolver` (unresolved ⇒ `NaN` ⇒ verified backup). It reads
> only `astro_mine.core` (no sibling imports) and asserts a planetary body-fixed frame (no implicit
> Earth frame). The safety guarantee stays entirely in the Rust core; a wrong adapter can only
> mis-set a *threshold* or *signal*, caught by review of the content-addressed SafetySpec.
>
> The **night-survival safe behaviours** are three *distinct*, verified-safe backup control laws in
> the trusted core (brake-to-stop · station-keeping **hold** · **retreat**-to-charging-pose), keyed
> by each constraint's `on_uncertain`. The retreat target is an additive `safe_pose` on the
> `SafetySpec` (extends RFC-0004); a missing/invalid pose or any unsafe step degrades to
> brake-to-stop — fail-safe, never fail-open.

## Decision records (deviations from the platform spec)

Where this repo deliberately departs from [`guard.md`](https://github.com/astro-mine/docs/blob/main/architecture/guard.md),
the decision is recorded here rather than left to be rediscovered from the code. ADRs live in
[`docs/adr/`](docs/adr/).

| ADR | Decision | Deviates from |
|---|---|---|
| [ADR-0001](docs/adr/0001-cbf-qp-solver.md) | The CBF-QP shield solves its per-tick program with a bespoke **allocation-free Dykstra projection** inside the TCB, **not** by linking OSQP/Clarabel. The shield *hard-certifies its own output*, so solver error can only cause a fallback, never an unsafe action — which is what makes an ~80-line kernel acceptable, and what keeps the TCB small enough to be **Kani-verifiable** (`rust/src/verify.rs`). Clarabel stays the reference optimizer the in-TCB solver is cross-validated against in CI (`rust/tests/clarabel_crosscheck.rs`). | `guard.md` §4, §11 (which name OSQP/Clarabel as the solver) |
| [ADR-0002](docs/adr/0002-mode-task-allowlist-in-safetyspec.md) | The **MODE/TASK directive allowlist lives in the `SafetySpec`**, not in `CoreConfig`. The gate's effective allowlist is `spec ∩ config`, and a **spec that is silent grants nothing** — configuration may only *narrow* the reviewed grant, never create a permission. A config-only grant was an expressible `passthrough`, on the one actuation path the shield cannot project or correct, in the one artifact with no integrity story. | Ratified as **RFC-0004 Amendment 2** (an additive extension of the contract, not a deviation) |

> **Machine-checked kernels (`rust/src/verify.rs`).** `guard.md` §9.3 makes Kani-style verification
> amenability a design goal of keeping the TCB tiny. The load-bearing kernels — the brake law's
> `|u| ≤ u_max ∧ v·u ≤ 0`, the box projection, the ball projection's minimality, and the
> **fail-closed action gate** (nothing unmodelled is ever certified, under *any* gate configuration)
> — are discharged with **Kani** in the `formal` CI job.
>
> This is not ceremony: the ball-projection proof **found a real IEEE-754 bug** — a subnormal
> underflow of `radius/len` that let the naive speed-ceiling projection escape its ball by 0.4 % —
> which 400 randomized Clarabel cross-checks and the whole proptest suite had both missed. The kernel
> now validates its own quotient and the counterexample is pinned as a regression test. The *numeric*
> in-ball property itself is deliberately **not** a harness (a symbolic `f64` multiply/divide over the
> unbounded exponent range does not terminate in CBMC); it is certified at run time instead, which is
> the same certify-the-output discipline as ADR-0001.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).
