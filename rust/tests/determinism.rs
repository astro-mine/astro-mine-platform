// SPDX-License-Identifier: Apache-2.0
//! Acceptance: **determinism** (guard.md §2 principle 3). Two fresh cores fed the same input
//! stream produce byte-identical verdicts — the seeded golden/determinism gate. (The
//! allocation-free + latency acceptance lives in its own single-test binary, `no_alloc.rs`,
//! so the process-wide allocation counter is not polluted by a concurrently-running test.)

mod common;

use astro_mine_guard_core::{CoreConfig, ProposedAction, SafetyCore, SafetyInput};
use common::*;

/// Drive the combined core over a deterministic input sweep, returning a compact fingerprint
/// of every verdict so two runs can be compared bit-for-bit.
fn run_fingerprint() -> Vec<u64> {
    let mut core = SafetyCore::from_model(combined_model(), CoreConfig::default()).unwrap();
    let mut out = core.new_verdict();
    let mut fp = Vec::new();
    for k in 0..500u64 {
        let t = k as f64 * 0.02;
        let signals = [18.0 + (t).sin(), 130.0 + 5.0 * (0.5 * t).cos()];
        let position = [
            12.0 + (0.3 * t).cos(),
            (0.2 * t).sin(),
            0.5 * (0.1 * t).sin(),
        ];
        let velocity = [-(0.3 * t).sin() * 0.3, (0.2 * t).cos() * 0.2, 0.0];
        let proposed = [-2.0 - t.cos(), 1.0, 0.0];
        core.step_into(
            &SafetyInput {
                signals: &signals,
                position: &position,
                velocity: &velocity,
                proposed: ProposedAction::Effort(&proposed),
            },
            &mut out,
        );
        fp.push(out.layer as u64);
        fp.push(out.reason as u64);
        for a in &out.certified_action {
            fp.push(a.to_bits());
        }
        fp.push(out.min_barrier_margin.to_bits());
    }
    fp
}

#[test]
fn identical_inputs_give_byte_identical_verdicts() {
    let a = run_fingerprint();
    let b = run_fingerprint();
    assert_eq!(a, b, "the safety core is not reproducible");
}
