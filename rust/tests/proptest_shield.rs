// SPDX-License-Identifier: Apache-2.0
//! Property-based invariants for the CBF-QP shield, in **every** control mode (guard.md §9.3;
//! RM-P1-GUARD-03).
//!
//! For a half-space keep-out the constraint `a·x ≥ b` is known in closed form (`a` = the unit
//! normal), so the test can independently verify that the shield's certified command is the true QP
//! optimum — the Euclidean projection of the proposal onto the feasible set — via the projection
//! variational inequality `⟨x₀ − x*, y − x*⟩ ≤ 0 ∀ feasible y`. Also checks kinematic-set
//! feasibility, determinism, and that an already-safe proposal is left untouched.
//!
//! The kinematic modes carry a **stronger, falsifiable** property than the effort mode's HOCBF row:
//! whenever the shield certifies a `Velocity`/`Position` command, the *realised next pose* is
//! outside the keep-out — checked against the exact barrier, re-derived here, over randomized
//! sphere and half-space geometry. That is the falsification search for the newly-shielded modes.

mod common;

use astro_mine_guard_core::model::{ActionLimits, KeepOutTerm, OnUncertain, Shape};
use astro_mine_guard_core::shield::{ControlMode, Shield, ShieldConfig};
use proptest::prelude::*;

/// The compiled sample period the kinematic modes use as their one-step horizon.
const DT: f64 = 0.05;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm3(v: &[f64]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sphere_term(center: [f64; 3], radius: f64, margin: f64) -> KeepOutTerm {
    KeepOutTerm {
        constraint_id: "c".into(),
        on_uncertain: OnUncertain::Fallback,
        shape: Shape::Sphere,
        margin_m: margin,
        center: center.to_vec(),
        half_extents: vec![],
        radius: Some(radius),
        normal: vec![],
        offset: None,
    }
}

/// The exact safe-set predicate, re-derived independently of the core: outside the sphere by the
/// margin. This is the oracle the falsification properties below try to break.
fn sphere_h(p: &[f64], center: [f64; 3], radius: f64, margin: f64) -> f64 {
    let d = ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2))
        .sqrt();
    d - (radius + margin)
}

/// A reviewed kinematic envelope with a commanded-speed ceiling.
fn limits(v_max: f64) -> ActionLimits {
    ActionLimits {
        max_velocity_mps: Some(v_max),
        max_accel_mps2: None,
    }
}

fn unit(mut v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn halfspace_term(normal: [f64; 3], offset: f64, margin: f64) -> KeepOutTerm {
    KeepOutTerm {
        constraint_id: "c".into(),
        on_uncertain: OnUncertain::Fallback,
        shape: Shape::HalfSpace,
        margin_m: margin,
        center: vec![],
        half_extents: vec![],
        radius: None,
        normal: normal.to_vec(),
        offset: Some(offset),
    }
}

/// Accurate config (many iterations, tight tolerance) so the projection is exact enough to
/// check optimality.
fn accurate_cfg() -> ShieldConfig {
    ShieldConfig {
        max_iter: 2000,
        tol: 1e-12,
        ..Default::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn certified_action_is_the_qp_optimum(
        nx in -1.0f64..1.0, ny in -1.0f64..1.0, nz in -1.0f64..1.0,
        offset in -3.0f64..3.0, margin in 0.0f64..3.0,
        px in -8.0f64..8.0, py in -8.0f64..8.0, pz in -8.0f64..8.0,
        vx in -4.0f64..4.0, vy in -4.0f64..4.0, vz in -4.0f64..4.0,
        ux in -40.0f64..40.0, uy in -40.0f64..40.0, uz in -40.0f64..40.0,
        seed in any::<u64>(),
    ) {
        prop_assume!(nx.abs() + ny.abs() + nz.abs() > 0.2);
        let cfg = accurate_cfg();
        let n = unit([nx, ny, nz]);
        let mut shield = Shield::new(&[halfspace_term(n, offset, margin)], 3, cfg, DT, &ActionLimits::default());

        let p = [px, py, pz];
        let v = [vx, vy, vz];
        let u0 = [ux, uy, uz];

        let mut action = Vec::with_capacity(3);
        let (certified, _h) = shield.solve(&p, &v, &u0, &mut action);
        prop_assume!(certified);

        // Box feasibility.
        for &a in &action {
            prop_assert!(a.abs() <= cfg.u_max + 1e-6, "action {a} outside box");
        }

        // Reconstruct the known constraint  a·u ≥ b.
        let rhs = margin - offset; // unit normal ⇒ rhs = (margin - offset)/‖n‖ with ‖n‖ = 1
        let h = dot(&n, &p) - rhs;
        let b = -(cfg.k1 * dot(&n, &v) + cfg.k0 * h);
        // The returned action satisfies its own constraint.
        prop_assert!(dot(&n, &action) >= b - 1e-4, "action violates the CBF constraint");

        // Variational inequality against sampled feasible points.
        let feasible = |y: &[f64; 3]| dot(&n, y) >= b - 1e-9 && y.iter().all(|c| c.abs() <= cfg.u_max);
        let mut rng = seed;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 };
        let mut checked = 0;
        for _ in 0..40 {
            let y = [
                (next() * 2.0 - 1.0) * cfg.u_max,
                (next() * 2.0 - 1.0) * cfg.u_max,
                (next() * 2.0 - 1.0) * cfg.u_max,
            ];
            if !feasible(&y) { continue; }
            checked += 1;
            let vi = (u0[0] - action[0]) * (y[0] - action[0])
                + (u0[1] - action[1]) * (y[1] - action[1])
                + (u0[2] - action[2]) * (y[2] - action[2]);
            prop_assert!(vi <= 1e-2, "not the projection: ⟨u0-u*, y-u*⟩ = {vi} > 0");
        }
        let _ = checked;
    }

    #[test]
    fn shield_is_deterministic(
        px in -8.0f64..8.0, py in -8.0f64..8.0, pz in -8.0f64..8.0,
        ux in -40.0f64..40.0, uy in -40.0f64..40.0, uz in -40.0f64..40.0,
    ) {
        let cfg = ShieldConfig::default();
        let term = halfspace_term(unit([0.2, 0.3, 1.0]), 0.5, 1.0);
        let mut a = Shield::new(std::slice::from_ref(&term), 3, cfg, DT, &ActionLimits::default());
        let mut b = Shield::new(std::slice::from_ref(&term), 3, cfg, DT, &ActionLimits::default());
        let (p, v, u0) = ([px, py, pz], [0.1, -0.2, 0.3], [ux, uy, uz]);
        let (mut aa, mut bb) = (Vec::new(), Vec::new());
        let ra = a.solve(&p, &v, &u0, &mut aa);
        let rb = b.solve(&p, &v, &u0, &mut bb);
        prop_assert_eq!(ra.0, rb.0);
        for (x, y) in aa.iter().zip(&bb) {
            prop_assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    // --- the kinematic modes (RM-P1-GUARD-03) -------------------------------------------------

    /// **The falsification property for VELOCITY.** Whatever velocity an adversarial policy
    /// proposes, a *certified* command (a) respects the reviewed speed ceiling and (b) leaves the
    /// realised next pose `p + w·dt` outside the keep-out — checked against the exact barrier,
    /// re-derived here. Randomized over sphere geometry, pose, and proposal.
    #[test]
    fn certified_velocity_never_enters_the_keepout(
        cx in -5.0f64..5.0, cy in -5.0f64..5.0, cz in -5.0f64..5.0,
        radius in 1.0f64..12.0, margin in 0.0f64..3.0,
        px in -30.0f64..30.0, py in -30.0f64..30.0, pz in -30.0f64..30.0,
        wx in -20.0f64..20.0, wy in -20.0f64..20.0, wz in -20.0f64..20.0,
        v_max in 0.05f64..5.0,
    ) {
        let cfg = ShieldConfig { max_iter: 500, tol: 1e-11, ..Default::default() };
        let center = [cx, cy, cz];
        let term = sphere_term(center, radius, margin);
        let mut shield = Shield::new(&[term], 3, cfg, DT, &limits(v_max));

        let p = [px, py, pz];
        let w0 = [wx, wy, wz];
        let mut w = Vec::with_capacity(3);
        let (certified, _h) = shield.solve_mode(ControlMode::Velocity, &p, &[0.0; 3], &w0, &mut w);
        prop_assume!(certified);

        // (a) the reviewed kinematic envelope.
        prop_assert!(
            norm3(&w) <= v_max + 1e-6,
            "certified speed {} exceeds the reviewed ceiling {v_max}", norm3(&w)
        );
        // (b) the exact one-step safe-set certificate.
        let next = [p[0] + w[0] * DT, p[1] + w[1] * DT, p[2] + w[2] * DT];
        prop_assert!(
            sphere_h(&next, center, radius, margin) >= -1e-5,
            "certified velocity {w:?} drove the next pose into the keep-out"
        );
    }

    /// **The falsification property for POSITION.** A *certified* target lies inside the reviewed
    /// step ball around the current pose **and** outside the keep-out — so a policy cannot command
    /// a teleport, nor a target inside a forbidden volume.
    #[test]
    fn certified_position_target_is_step_capped_and_safe(
        cx in -5.0f64..5.0, cy in -5.0f64..5.0, cz in -5.0f64..5.0,
        radius in 1.0f64..12.0, margin in 0.0f64..3.0,
        px in -30.0f64..30.0, py in -30.0f64..30.0, pz in -30.0f64..30.0,
        qx in -40.0f64..40.0, qy in -40.0f64..40.0, qz in -40.0f64..40.0,
        v_max in 0.05f64..5.0,
    ) {
        let cfg = ShieldConfig { max_iter: 500, tol: 1e-11, ..Default::default() };
        let center = [cx, cy, cz];
        let term = sphere_term(center, radius, margin);
        let mut shield = Shield::new(&[term], 3, cfg, DT, &limits(v_max));

        let p = [px, py, pz];
        let q0 = [qx, qy, qz];
        let mut q = Vec::with_capacity(3);
        let (certified, _h) = shield.solve_mode(ControlMode::Position, &p, &[0.0; 3], &q0, &mut q);
        prop_assume!(certified);

        let step = norm3(&[q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
        prop_assert!(
            step <= v_max * DT + 1e-6,
            "certified step {step} exceeds the reviewed cap {}", v_max * DT
        );
        prop_assert!(
            sphere_h(&q, center, radius, margin) >= -1e-5,
            "certified target {q:?} lies inside the keep-out"
        );
    }

    /// A kinematic command that is already feasible is passed through untouched — the shield only
    /// intervenes when it must, in every mode (guard.md §9 "minimally perturb").
    #[test]
    fn a_feasible_kinematic_command_is_untouched(
        wx in -0.4f64..0.4, wy in -0.4f64..0.4, wz in -0.4f64..0.4,
    ) {
        let cfg = ShieldConfig { max_iter: 500, tol: 1e-11, ..Default::default() };
        let term = sphere_term([0.0, 0.0, 0.0], 10.0, 2.0);
        let mut shield = Shield::new(&[term], 3, cfg, DT, &limits(1.0));
        // 1000 m from the keep-out: only the (unbinding) speed ball is active.
        let p = [1000.0, 0.0, 0.0];
        let w0 = [wx, wy, wz];
        let mut w = Vec::new();
        let (certified, _h) = shield.solve_mode(ControlMode::Velocity, &p, &[0.0; 3], &w0, &mut w);
        prop_assert!(certified);
        for (a, b) in w.iter().zip(&w0) {
            prop_assert!((a - b).abs() < 1e-6, "the shield perturbed a feasible velocity");
        }
    }
}

/// A proposal that is already deep inside the safe set and within the control box is passed
/// through unperturbed — the shield only intervenes when it must (guard.md §9 "minimally").
#[test]
fn safe_proposal_is_left_untouched() {
    let cfg = ShieldConfig::default();
    let term = halfspace_term([0.0, 0.0, 1.0], 0.0, 1.0); // safe set z >= 1
    let mut shield = Shield::new(&[term], 3, cfg, DT, &ActionLimits::default());
    // Far inside (z = 100), gentle proposal well within the box.
    let (p, v, u0) = ([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [1.0, -2.0, 0.5]);
    let mut action = Vec::new();
    let (certified, _h) = shield.solve(&p, &v, &u0, &mut action);
    assert!(certified);
    for (a, u) in action.iter().zip(&u0) {
        assert!(
            (a - u).abs() < 1e-6,
            "shield perturbed a safe action: {a} vs {u}"
        );
    }
}
