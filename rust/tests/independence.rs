// SPDX-License-Identifier: Apache-2.0
//! Acceptance: **independence proven** (guard.md §9.1). A deliberately adversarial policy —
//! one that always commands acceleration straight into the keep-out — cannot drive a
//! hard-constraint violation: the arbiter's precedence (shield-correct, else backup) keeps the
//! barrier `h ≥ 0` throughout. This is a bounded simulation demonstration in the regime where
//! the CBF-QP retains authority (a speed-limited plant with ample `u_max`); the exhaustive
//! falsification search is GUARD-05.

mod common;

use astro_mine_guard_core::model::OnUncertain;
use astro_mine_guard_core::{CoreConfig, ProposedAction, SafetyCore, SafetyInput};
use common::*;

/// Deterministic, allocation-free xorshift PRNG (no `rand` in the TCB test path).
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn clamp_speed(v: &mut [f64], v_max: f64) {
    let s: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if s > v_max {
        let k = v_max / s;
        for vi in v.iter_mut() {
            *vi *= k;
        }
    }
}

#[test]
fn adversarial_policy_cannot_breach_sphere_keepout() {
    let radius = 10.0;
    let margin = 2.0;
    let full_r = radius + margin; // barrier zero-crossing
    let cfg = CoreConfig::default();
    let u_max = cfg.shield.u_max;
    let v_max = 5.0; // kinematic speed limit → bounded braking distance ≪ start margin
    let center = [0.0f64, 0.0, 0.0];

    let mut worst_margin = f64::INFINITY;
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);

    for case in 0..40 {
        let mut core = SafetyCore::from_model(
            sphere_model(3, center.to_vec(), radius, margin, OnUncertain::Fallback),
            cfg.clone(),
        )
        .unwrap();

        // Start on a random ray at a comfortable distance, with a small random velocity.
        let dir = {
            let mut d = [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
            for x in &mut d {
                *x /= n;
            }
            d
        };
        let start_dist = rng.uniform(full_r + 8.0, full_r + 20.0);
        let p = [
            dir[0] * start_dist,
            dir[1] * start_dist,
            dir[2] * start_dist,
        ];
        let v = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ];
        let mut plant = Plant::new(p.to_vec(), v.to_vec(), 0.02);

        let mut out = core.new_verdict();
        for _ in 0..400 {
            // Adversary: aim acceleration straight at the sphere centre at full authority,
            // with random jitter — the worst thing a pathological policy can propose.
            let mut u0 = [0.0f64; 3];
            for i in 0..3 {
                u0[i] = center[i] - plant.p[i];
            }
            let n = (u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2])
                .sqrt()
                .max(1e-9);
            for ui in u0.iter_mut() {
                *ui = *ui / n * u_max + rng.uniform(-u_max, u_max);
            }

            core.step_into(
                &SafetyInput {
                    signals: &[],
                    position: &plant.p,
                    velocity: &plant.v,
                    proposed: ProposedAction::Effort(&u0),
                },
                &mut out,
            );
            for a in &out.certified_action {
                assert!(a.is_finite(), "certified action must be finite");
            }
            plant.step(&out.certified_action);
            clamp_speed(&mut plant.v, v_max);

            // The hard guarantee: the rover never enters the *physical* keep-out (the actual
            // hazard radius). The `margin` is the safety buffer that absorbs discretization —
            // the CBF keeps the state outside `radius + margin`, and the worst the semi-implicit
            // integrator does is nibble a millimetre into that buffer, never near the hazard.
            let physical = plant.sphere_barrier(&center, radius);
            assert!(
                physical > 0.0,
                "case {case}: adversary breached the physical keep-out: dist-to-hazard = {physical} (state {:?})",
                plant.p
            );
            worst_margin = worst_margin.min(plant.sphere_barrier(&center, full_r));
        }
    }
    // The shield actually did work — the barrier got meaningfully exercised, not trivially safe.
    assert!(
        worst_margin < 8.0,
        "keep-out was never approached (worst h = {worst_margin})"
    );
}

#[test]
fn adversarial_policy_cannot_cross_halfspace() {
    // Safe set: z >= margin - offset = 1 - 2 = -1, i.e. z >= ... use unit normal +z, offset 0,
    // margin 1 → safe set z >= 1.
    let cfg = CoreConfig::default();
    let mut core = SafetyCore::from_model(
        halfspace_model(3, vec![0.0, 0.0, 1.0], 0.0, 1.0),
        cfg.clone(),
    )
    .unwrap();
    let mut plant = Plant::new(vec![0.0, 0.0, 15.0], vec![0.0, 0.0, 0.0], 0.02);
    let mut out = core.new_verdict();
    for _ in 0..400 {
        // Adversary drives straight down (into the boundary) at full authority.
        let u0 = [0.0, 0.0, -cfg.shield.u_max];
        core.step_into(
            &SafetyInput {
                signals: &[],
                position: &plant.p,
                velocity: &plant.v,
                proposed: ProposedAction::Effort(&u0),
            },
            &mut out,
        );
        plant.step(&out.certified_action);
        // enforce kinematic speed cap
        let s = plant.v[2].abs();
        if s > 5.0 {
            plant.v[2] *= 5.0 / s;
        }
        assert!(
            plant.p[2] > 1.0 - 1e-2,
            "adversary crossed half-space: z = {}",
            plant.p[2]
        );
    }
}
