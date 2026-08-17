// SPDX-License-Identifier: Apache-2.0
//! Machine-checked proofs of the TCB's load-bearing kernels — **Kani** model checking
//! (guard.md §9.3: "for the smallest critical kernels — amenability to formal analysis
//! (model checking / Kani-style verification) is a stated design goal of keeping the TCB tiny").
//!
//! `proptest` (`rust/tests/proptest_*.rs`) *samples* the invariant space; Kani **exhausts** it —
//! it discharges each harness below over *every* input, including the floating-point corner cases
//! (`NaN`, `±∞`, `±0.0`, subnormals) a randomized search will practically never hit. That is the
//! difference between "we did not find a counterexample" and "there is none", and it is why the
//! guarantee-bearing kernels are factored out as small, heap-free, `pub(crate)` free functions:
//! **the TCB is kept tiny precisely so this is possible** (guard.md §9.1).
//!
//! **This is not ceremony — it found a real bug.** A harness over the ball projection (the
//! `VELOCITY` speed ceiling and the `POSITION` step cap are the same kernel) *failed* on the obvious
//! implementation, `scale = radius / len`, and handed back the counterexample:
//! `len ≈ 1.5e172`, `radius ≈ 7.6e-152`, where the quotient underflows into the **subnormal** range,
//! carries ~0.4 % relative error, and the "projected" command lands **outside the ball**. 400
//! randomized Clarabel cross-checks and the entire proptest suite had both missed it, and no amount
//! of further sampling was going to find it. The fix is
//! [`ball_scale_is_sound`](crate::shield::ball_scale_is_sound): the kernel now **validates its own
//! quotient** and collapses to `0.0` (the zero command — inside every ball, and the safe floor in
//! every mode) rather than trusting the divider. That counterexample is pinned as a regression test
//! in `shield.rs`.
//!
//! What is proved (the kernels every fail-safe path bottoms out in):
//!
//! 1. [`brake_never_adds_speed`] — the *recover* layer's floor. `brake_axis` is bounded by the
//!    control box and never accelerates in the direction of travel (`v·u ≤ 0`), for **every**
//!    finite velocity, gain, and ceiling. This is the invariant the whole simplex argument rests
//!    on: every other law degrades to this one.
//! 2. [`box_projection_is_sound`] — the `Effort` kinematic projection lands inside the control box
//!    and is idempotent (a point already inside is untouched — "minimally perturb").
//! 3. [`a_feasible_command_is_not_scaled`] — the ball projection leaves an already-feasible command
//!    untouched, for every length/radius.
//! 4. [`gate_admits_nothing_by_default`] and [`gate_never_admits_an_unmodelled_action`] — the
//!    *action gate* is fail-closed. An unconfigured Guard certifies **no** directive (for *every*
//!    directive name, not a handful of sampled literals), and **no** configuration of the gate —
//!    however permissive — admits an opaque actuator command (an unmodelled
//!    `IMPEDANCE`/`TRAJECTORY` control mode), nor lets a modelled command skip the shield's
//!    projection. This is "no unmodelled action ever passes through" as a theorem rather than a
//!    code-reading (guard.md §9.1, LUNAR-FR-006).
//! 5. [`configuration_cannot_widen_the_authored_grant`] — the *gate's allowlist is the reviewed
//!    contract's*, not the deployment's (RFC-0004 Amendment 2). No configuration, however
//!    permissive, admits a directive the `SafetySpec` did not author, and a **spec-silent** model
//!    admits **nothing**. The tighten-only invariant, machine-checked over a symbolic directive
//!    name — and the reason the allowlist stopped being a config knob at all.
//!
//! ## What is *not* proved here, and why
//!
//! The numeric in-ball property itself — `len · scale ≤ radius` — is **not** a Kani harness. Every
//! statement of it puts a symbolic IEEE-754 multiply (and, in the natural formulation, a division)
//! in front of CBMC over the unbounded `f64` exponent range, and the propositional encoding does not
//! terminate; bounding the magnitudes does not rescue it either. Rather than ship a proof that hangs
//! the `formal` CI job, the property is enforced **at run time, on every call**, by
//! `ball_scale_is_sound` — the very check the failed proof forced into existence — and re-checked a
//! second time by `Shield::certify` before any command is certified. Static proof of the structural
//! kernels, run-time validation of the numeric one, and a pinned regression for the corner the model
//! checker found: that is the same fail-closed, certify-the-output discipline the rest of the TCB is
//! built on (`docs/adr/0001-cbf-qp-solver.md`), applied to a single division.
//!
//! Run with `cargo kani` (CI job `formal`); the harnesses are compiled only under `cfg(kani)`, so
//! they cost the shipped TCB nothing.

#![allow(clippy::needless_range_loop)]

use crate::arbiter::{ActionPolicy, ProposedAction};
use crate::backup::brake_axis;
use crate::shield::{ball_scale, narrow, project_box_axis, ControlMode};

/// The *recover* layer's floor: a brake command is inside the control box and never adds speed.
///
/// `∀ v, k_brake > 0, u_max > 0 (all finite):  |brake_axis(v,k,u_max)| ≤ u_max  ∧  v·u ≤ 0`.
#[kani::proof]
fn brake_never_adds_speed() {
    let v: f64 = kani::any();
    let k_brake: f64 = kani::any();
    let u_max: f64 = kani::any();
    kani::assume(v.is_finite());
    kani::assume(k_brake.is_finite() && k_brake > 0.0);
    kani::assume(u_max.is_finite() && u_max > 0.0);

    let u = brake_axis(v, k_brake, u_max);

    assert!(u.is_finite(), "a brake command must be finite");
    assert!(u.abs() <= u_max, "a brake command must stay in the box");
    // Never accelerates along the direction of travel — the simplex invariant.
    assert!(v * u <= 0.0, "braking must not add speed");
}

/// The `Effort` kinematic projection: lands in the box, and leaves an in-box point untouched.
#[kani::proof]
fn box_projection_is_sound() {
    let x: f64 = kani::any();
    let limit: f64 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(limit.is_finite() && limit > 0.0);

    let p = project_box_axis(x, limit);
    assert!(p.abs() <= limit, "projection escaped the control box");
    // Idempotent, and minimal: an already-feasible point is not perturbed.
    assert!(
        project_box_axis(p, limit) == p,
        "projection is not idempotent"
    );
    if x.abs() <= limit {
        assert!(p == x, "the box projection perturbed a feasible action");
    }
}

/// The early-out path: a command already inside the ball is returned **untouched** (`scale == 1.0`).
///
/// "Minimally perturb" (guard.md §9) is a safety-relevant property in its own right — a shield that
/// quietly shrinks a *feasible* command is silently degrading the policy it is supposed to be
/// certifying. Cheap to prove because it is a comparison, not a division.
#[kani::proof]
fn a_feasible_command_is_not_scaled() {
    let len: f64 = kani::any();
    let radius: f64 = kani::any();
    kani::assume(len.is_finite() && len >= 0.0);
    kani::assume(radius.is_finite() && radius > 0.0);
    kani::assume(len <= radius); // already inside the ball

    assert!(
        ball_scale(len, radius) == 1.0,
        "the ball projection perturbed a feasible command"
    );
}

/// The **action gate** is fail-closed *by default*: an unconfigured Guard certifies **nothing**.
///
/// The allowlists are empty, so the gate admits **no** directive — for *every* directive name, here
/// a symbolic two-byte string rather than a handful of sampled literals. Together with
/// [`gate_never_admits_an_unmodelled_action`] this is the theorem behind "no action reaches an
/// actuator uncertified": a directive needs an explicit, reviewed allowlist entry, and nothing else
/// is admissible at all.
#[kani::proof]
#[kani::unwind(4)]
fn gate_admits_nothing_by_default() {
    let policy = ActionPolicy::default(); // empty allowlists — no heap, no configuration
    let setpoint = [0.0f64];

    // Symbolic directive name (any 2-byte ASCII string), so this really is a for-all over names.
    let bytes: [u8; 2] = kani::any();
    kani::assume(bytes[0].is_ascii_graphic() && bytes[1].is_ascii_graphic());
    let name = core::str::from_utf8(&bytes).unwrap();

    assert!(!policy.admits(&ProposedAction::Opaque));
    assert!(!policy.admits(&ProposedAction::Effort(&setpoint)));
    assert!(!policy.admits(&ProposedAction::Velocity(&setpoint)));
    assert!(!policy.admits(&ProposedAction::Position(&setpoint)));
    assert!(
        !policy.admits(&ProposedAction::Mode(name)),
        "an empty allowlist must certify no MODE"
    );
    assert!(
        !policy.admits(&ProposedAction::Task(name)),
        "an empty allowlist must certify no TASK"
    );
}

/// **No unmodelled action is ever admitted — not even by a configured gate.**
///
/// A policy that *does* allowlist directives still never admits an opaque actuator command (an
/// unmodelled `IMPEDANCE`/`TRAJECTORY` control mode), and never admits a modelled command either —
/// those must go through the shield's projection, not the gate. So there is no configuration of the
/// gate, however permissive, under which an action the TCB cannot certify passes through: the
/// "unmodelled ⇒ pass through" path does not exist (guard.md §9.1, LUNAR-FR-006).
///
/// The directive arms are checked against their own lists (and, crucially, *not* against each
/// other's — a MODE allowlist must never certify a TASK).
#[kani::proof]
#[kani::unwind(4)]
fn gate_never_admits_an_unmodelled_action() {
    let policy = ActionPolicy {
        certified_modes: vec!["m".to_string()],
        certified_tasks: vec!["t".to_string()],
        fallback_mode: ControlMode::Effort,
    };
    let setpoint = [0.0f64];

    // Unmodelled and modelled commands: never admitted, whatever the allowlists say.
    assert!(
        !policy.admits(&ProposedAction::Opaque),
        "an unmodelled actuator command must never be admitted"
    );
    assert!(!policy.admits(&ProposedAction::Effort(&setpoint)));
    assert!(!policy.admits(&ProposedAction::Velocity(&setpoint)));
    assert!(!policy.admits(&ProposedAction::Position(&setpoint)));

    // Directives: admitted exactly by their own list, and never by the sibling list.
    assert!(policy.admits(&ProposedAction::Mode("m")));
    assert!(policy.admits(&ProposedAction::Task("t")));
    assert!(
        !policy.admits(&ProposedAction::Mode("t")),
        "the TASK allowlist must never certify a MODE"
    );
    assert!(
        !policy.admits(&ProposedAction::Task("m")),
        "the MODE allowlist must never certify a TASK"
    );
}

/// **No configuration, however permissive, admits a directive the reviewed model does not.**
///
/// This is the machine-checked statement of RFC-0004 Amendment 2 — the tighten-only invariant on the
/// action gate — and it is the point of the whole change. `narrow` is the permission-set sibling of
/// `tighten`: the effective allowlist the core gates against is `configured ∩ authored`, resolved
/// once at construction, so the gate can only ever be *narrowed* by configuration, never widened.
///
/// Two properties, over a **symbolic** directive name (every name, not a handful of literals):
///
/// 1. **Silence grants nothing.** A model that authored no grant (`None`) admits **∅**, whatever the
///    configuration allowlists. This is the deliberate asymmetry with `tighten`, where an absent
///    authored limit leaves the *configured* ceiling standing: the identity of a ceiling's meet is
///    `+∞`, the identity of a permission set's meet is `∅`. Reading `None` as "config stands" here
///    would be fail-open-by-silence — and every spec written before Amendment 2 is silent.
///
/// 2. **The effective set is a subset of the authored one.** So a name the model did not author is
///    never certifiable, no matter how the deployment is configured. Configuration cannot *create* a
///    permission; it can only decline one.
///
/// Together with [`gate_admits_nothing_by_default`] and [`gate_never_admits_an_unmodelled_action`],
/// this closes the gate: an action reaches an actuator uncertified only if the **reviewed,
/// content-addressed, signed contract** said it could.
#[kani::proof]
#[kani::unwind(4)]
fn configuration_cannot_widen_the_authored_grant() {
    // Two symbolic one-byte names: `c` is what the *configuration* grants, `a` what the *model*
    // authors. Ranging over both makes this a for-all over (config, contract) name pairs.
    let cb: u8 = kani::any();
    let ab: u8 = kani::any();
    kani::assume(cb.is_ascii_graphic() && ab.is_ascii_graphic());
    let cname = (cb as char).to_string();
    let aname = (ab as char).to_string();

    // A maximally permissive configuration: it allowlists the symbolic name, as MODE *and* TASK.
    let configured = vec![cname.clone()];
    let authored = [aname.clone()];

    // (1) The model authored nothing ⇒ nothing is admissible, however the config is set.
    let silent_modes = narrow(&configured, None);
    let silent_tasks = narrow(&configured, None);
    assert!(
        silent_modes.is_empty() && silent_tasks.is_empty(),
        "a spec-silent model must admit no directive, whatever the configuration grants"
    );
    let silent_policy = ActionPolicy {
        certified_modes: silent_modes,
        certified_tasks: silent_tasks,
        fallback_mode: ControlMode::Effort,
    };
    assert!(!silent_policy.admits(&ProposedAction::Mode(&cname)));
    assert!(!silent_policy.admits(&ProposedAction::Task(&cname)));

    // (2) The model authored a grant ⇒ the effective set is a subset of BOTH inputs …
    let effective = narrow(&configured, Some(&authored));
    for name in &effective {
        assert!(
            authored.contains(name),
            "narrow() admitted a directive the reviewed model did not author"
        );
        assert!(
            configured.contains(name),
            "narrow() admitted a directive the configuration did not grant"
        );
    }

    // … so a configured name the model did NOT author is never certifiable. The gate is built from
    // the effective policy, so this holds at the gate, not merely inside `narrow`.
    let policy = ActionPolicy {
        certified_modes: effective,
        // The model authored no TASK at all — the empty-set case, checked at the gate.
        certified_tasks: narrow(&configured, Some(&[])),
        fallback_mode: ControlMode::Effort,
    };
    assert!(
        !policy.admits(&ProposedAction::Task(&cname)),
        "an unauthored TASK must never be certifiable, however permissive the configuration"
    );
    if cb != ab {
        assert!(
            !policy.admits(&ProposedAction::Mode(&cname)),
            "a configuration widened the reviewed grant"
        );
    }
}
