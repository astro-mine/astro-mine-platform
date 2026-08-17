// SPDX-License-Identifier: Apache-2.0
//! Property-based invariants for the STL/MTL monitors (guard.md §9.3 — the proptest gate).
//!
//! Drives [`Monitor`] directly with arbitrary predicate-robustness streams and checks the
//! bounded past-time robust semantics against an independent reference: `always` = windowed
//! min, `eventually` = windowed max, plus determinism and boundedness.

mod common;

use astro_mine_guard_core::model::{CompiledNode, MonitorAutomaton, OnUncertain, TemporalOp};
use astro_mine_guard_core::monitors::Monitor;
use proptest::prelude::*;

fn window_monitor(op: TemporalOp, window: usize) -> Monitor {
    let root = CompiledNode {
        op,
        predicate_index: None,
        interval_lo_samples: Some(0),
        interval_hi_samples: Some(window),
        args: vec![CompiledNode {
            op: TemporalOp::Predicate,
            predicate_index: Some(0),
            interval_lo_samples: None,
            interval_hi_samples: None,
            args: vec![],
        }],
    };
    Monitor::new(&MonitorAutomaton {
        constraint_id: "c".into(),
        on_uncertain: OnUncertain::Fallback,
        root,
        history_window_len: window,
        node_count: 2,
        predicate_indices: vec![0],
    })
}

proptest! {
    /// `always[0,w]` robustness equals the min of the predicate robustness over the last
    /// `w + 1` samples (bounded past-time semantics).
    #[test]
    fn always_is_windowed_min(values in prop::collection::vec(-100.0f64..100.0, 1..60), window in 0usize..8) {
        let mut mon = window_monitor(TemporalOp::Always, window);
        for (t, &val) in values.iter().enumerate() {
            let verdict = mon.step(&|_| val, 3);
            let lo = t.saturating_sub(window);
            let expected = values[lo..=t].iter().copied().fold(f64::INFINITY, f64::min);
            prop_assert!((verdict.robustness - expected).abs() < 1e-9,
                "tick {t}: got {}, expected {}", verdict.robustness, expected);
            prop_assert_eq!(verdict.violated, verdict.robustness < 0.0);
        }
    }

    /// `eventually[0,w]` robustness equals the windowed max.
    #[test]
    fn eventually_is_windowed_max(values in prop::collection::vec(-100.0f64..100.0, 1..60), window in 0usize..8) {
        let mut mon = window_monitor(TemporalOp::Eventually, window);
        for (t, &val) in values.iter().enumerate() {
            let verdict = mon.step(&|_| val, 3);
            let lo = t.saturating_sub(window);
            let expected = values[lo..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!((verdict.robustness - expected).abs() < 1e-9);
        }
    }

    /// The engine is deterministic: the same stream yields the same robustness trace.
    #[test]
    fn monitor_is_deterministic(values in prop::collection::vec(-50.0f64..50.0, 1..40), window in 0usize..6) {
        let mut a = window_monitor(TemporalOp::Always, window);
        let mut b = window_monitor(TemporalOp::Always, window);
        for &val in &values {
            let ra = a.step(&|_| val, 3);
            let rb = b.step(&|_| val, 3);
            prop_assert_eq!(ra.robustness.to_bits(), rb.robustness.to_bits());
            prop_assert_eq!(ra.violated, rb.violated);
        }
    }

    /// Robustness is always bounded by the extremes actually seen in the window — no value is
    /// invented (a sanity bound on the semantics).
    #[test]
    fn robustness_is_bounded_by_window(values in prop::collection::vec(-30.0f64..30.0, 1..50), window in 0usize..8) {
        let mut mon = window_monitor(TemporalOp::Always, window);
        for (t, &val) in values.iter().enumerate() {
            let verdict = mon.step(&|_| val, 3);
            let lo = t.saturating_sub(window);
            let wmin = values[lo..=t].iter().copied().fold(f64::INFINITY, f64::min);
            let wmax = values[lo..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(verdict.robustness >= wmin - 1e-9 && verdict.robustness <= wmax + 1e-9);
        }
    }
}

/// A monotonically-decreasing robustness trend is flagged *predictively* before it crosses
/// zero (guard.md §9.2 predictive monitoring), giving the arbiter lead time to act.
#[test]
fn predictive_flag_fires_before_violation() {
    let mut mon = window_monitor(TemporalOp::Always, 0); // instantaneous (window 0) → tracks the signal
                                                         // Robustness ramps down 10, 8, 6, 4, 2, ... crossing zero at tick 5.
    let mut fired_before_violation = false;
    for k in 0..6 {
        let val = 10.0 - 2.0 * k as f64;
        let v = mon.step(&|_| val, 3); // predictive horizon 3 samples
        if val > 0.0 && v.predicted {
            fired_before_violation = true;
        }
    }
    assert!(
        fired_before_violation,
        "predictive monitor never warned before the crossing"
    );
}
