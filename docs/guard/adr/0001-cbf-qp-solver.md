# ADR-0001 — The CBF-QP shield solves its QP with a bespoke Dykstra projection, not OSQP/Clarabel

- **Status:** Accepted
- **Date:** 2026-07-11
- **Component:** `astro-mine-guard` — the Rust safety core (TCB), `rust/src/shield.rs`
- **Traceability:** `RM-P1-GUARD-02`; `guard.md` §2 (principles), §9.1, §9.3, §11; `conventions.md` §2, §11

## Context

[`guard.md` §11][guard-md] recommends, for the "QP / reachability runtime" decision:

> **Offline precompute (HJ value functions) + small online QP** (OSQP/Clarabel) — keeps the safety
> path within the tick budget.

and [§4][guard-md] names **OSQP** (or Clarabel, Rust-native) as the CBF-shielding solver.

The shipped TCB does **not** call either. `rust/src/shield.rs` solves the per-tick program

```
minimise ‖x − x₀‖²   s.t.  aᵢ·x ≥ bᵢ  (one row per keep-out) ,  x ∈ K
```

with a bespoke, fixed-capacity **Dykstra alternating projection** onto the intersection of the
barrier half-spaces and the mode's kinematic set `K` (a box for `EFFORT`, a ball for
`VELOCITY`/`POSITION`). Clarabel appears only as a **dev-dependency**, in
`rust/tests/clarabel_crosscheck.rs`, as an independent optimality oracle.

That is a real deviation from the architecture doc, and until now it was undocumented — which is
the actual defect. This ADR records the decision and its reasoning.

## Decision

**The Dykstra projection is the production solver. Clarabel stays a test-only oracle. `guard.md` §11
should be amended to record the deviation, not the code changed to match it.**

## Rationale

The recommendation in `guard.md` §11 optimises for *"a small, fast, deterministic QP per tick"*.
Every one of those adjectives is satisfied better by the projection than by linking a general solver,
because Guard's QP is not a general QP — and three of `guard.md`'s own architecture principles bind
harder than the solver recommendation does:

1. **Minimal TCB (§9.1: "the TCB's smallness is what makes the guarantee *analyzable*").**
   OSQP is C; Clarabel is ~15 kLOC of Rust with a sparse-linear-algebra stack behind it. Either
   would be *inside* the trusted computing base, on the safety path, in a component whose entire
   value proposition is that its trusted core is small enough to reason about. The projection is
   ~80 lines of arithmetic over pre-sized `f64` buffers. The dependency surface Guard is willing to
   trust on the safety path is the thing being conserved.

2. **No hot-path allocation, statically-bounded work (§2 principle 6, §8).** The projection is
   allocation-free by construction — every buffer is sized at `Shield::new` from the compiled
   model's `ResourceBounds` and never grows — and its worst-case work is exactly
   `max_iter × (m + 1) × dim` flops, a *fixed* budget that gives the tick a static latency bound.
   `rust/tests/no_alloc.rs` asserts **zero** heap allocations per tick with a counting global
   allocator and pins the mean tick at ~4 µs. A general solver's setup/factorisation path is not
   allocation-free, and its iteration count is data-dependent — precisely what the watchdog
   (§9.1, §10) exists to catch, so we would be manufacturing the fault we then have to detect.

3. **Formal-verification amenability (§9.3).** Keeping the numeric kernels tiny and free of
   dependencies is what lets `rust/src/verify.rs` discharge them with **Kani** — and that is not
   hypothetical. Kani proved the brake law, the box projection, and the fail-closed action gate; and
   it *found a genuine bug* in the ball projection (the speed ceiling / step cap): for
   `len ≈ 1.5e172`, `radius ≈ 7.6e-152` the quotient `radius/len` underflows into the **subnormal**
   range, carries ~0.4 % relative error, and the "projected" command escapes the ball. The 400
   randomized Clarabel cross-checks and the whole proptest suite had both missed it. The kernel now
   validates its own quotient (`ball_scale_is_sound`) and collapses to the zero command when the FPU
   betrays it; the counterexample is pinned as a regression test. Neither OSQP nor Clarabel is
   model-checkable at this granularity, and neither would have surfaced that.

   *Scope of the proofs.* The **structural** kernels are proved; the **numeric** in-ball property
   (`len · scale ≤ radius`) is not, because every statement of it puts a symbolic IEEE-754 multiply
   and division in front of CBMC over the unbounded `f64` exponent range, and the propositional
   encoding does not terminate (bounding the magnitudes does not rescue it). It is instead enforced
   at run time on every call, by exactly the validator the failed proof forced into existence — and
   re-checked by `Shield::certify` before anything is certified. Static proof of what is decidable,
   run-time certification of what is not, is the same discipline this whole ADR is about.

**The correctness argument does not rest on trusting the solver.** This is the load-bearing point.
The shield **hard-certifies** its own output before returning it (`Shield::certify`): every barrier
row is re-checked, the kinematic set is re-checked, and — for the kinematic modes — the *realised
next pose* is re-checked against the exact barrier. A solver that returns a wrong answer therefore
cannot cause an unsafe action; it can only cause a **fallback** (§9.1, fail-safe never fail-open).
The solver is on the *performance* path, not on the *guarantee* path. That inverts the usual
"the QP must be right" pressure, and it is why a 80-line projection is an acceptable thing to trust
where it would not be if the certificate came from the solver itself.

**Optimality is still gated, by a reference optimizer.** `rust/tests/clarabel_crosscheck.rs` solves
400 randomized shield QPs with Clarabel — the exact solver `guard.md` §11 recommends — and asserts
the in-TCB projection agrees with it to 1e-3. `rust/tests/proptest_shield.rs` independently checks
the projection variational inequality `⟨x₀ − x*, y − x*⟩ ≤ 0`. So the recommendation is honoured
where it belongs: as the *oracle we are checked against*, not as the code we ship on the safety path.

## Consequences

- **The deviation is now recorded** (this ADR, `ARCHITECTURE.md`, and the `shield.rs` module docs).
- **`guard.md` §11 must be amended** to reflect it. That edit lives in the `astro-mine/docs` repo and
  is *not* made by this PR (a component repo does not edit the platform spec). The proposed wording:

  > | QP / reachability runtime | Online PDE/optimization; offline precompute + online lookup/solve | **Offline precompute (HJ value functions) + a small online projection.** The CBF-QP is solved in the TCB by a bespoke allocation-free Dykstra alternating projection (Guard ADR-0001) rather than by linking OSQP/Clarabel: the program is tiny and fixed-shape, the shield hard-certifies its own output (so solver error can only cause a *fallback*, never an unsafe action), and keeping the kernel dependency-free is what preserves the small, allocation-free, **Kani-verifiable** TCB §9.1/§9.3 call for. **Clarabel** remains the reference optimizer the in-TCB solver is cross-validated against in CI. |

- **Known limitation.** Dykstra's tail converges linearly, so a badly-conditioned solve (a proposal
  far outside a tight speed ceiling with several near-parallel active barriers) may not reach
  optimality within the fixed iteration budget. The consequence is *conservatism*, not unsafety: an
  uncertified iterate falls back. Two mitigations are in place — the loop only stops early once the
  iterate is **feasible** (stopping on "stopped moving" alone once handed the certifier an infeasible
  point and needlessly vetoed a correctable action), and the budget is 512 cycles, which costs
  nothing on the fast path because an easy solve breaks out on the first stall.
- **Revisit if** the safe set grows constraint kinds the projection cannot express (a second-order
  cone, a non-convex reachability set), or if the fallback rate from budget exhaustion becomes
  measurable on the anchor. Promoting Clarabel from oracle to production is then a contained change:
  it is already a dev-dependency, and the certification step in front of it does not move.

[guard-md]: https://github.com/astro-mine/docs/blob/main/architecture/guard.md
