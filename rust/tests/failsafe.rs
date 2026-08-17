// SPDX-License-Identifier: Apache-2.0
//! Acceptance: **fail-safe, never fail-open** (guard.md §9.1). Every uncertifiable case
//! resolves to a verified safe fallback whose action is finite and within the control box.

mod common;

use astro_mine_guard_core::model::OnUncertain;
use astro_mine_guard_core::{
    BackupKind, CoreConfig, Intervention, Layer, ProposedAction, Reason, SafetyCore, SafetyInput,
};
use common::*;

fn finite_and_bounded(action: &[f64], u_max: f64) {
    for &a in action {
        assert!(a.is_finite(), "backup action must be finite, got {a}");
        assert!(
            a.abs() <= u_max + 1e-6,
            "backup action {a} exceeds u_max {u_max}"
        );
    }
}

#[test]
fn nan_signal_falls_back() {
    let mut core = SafetyCore::from_model(
        scalar_floor_model(15.0, OnUncertain::Fallback),
        CoreConfig::default(),
    )
    .unwrap();
    let v = core.step(&SafetyInput {
        signals: &[f64::NAN],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::BadInput);
    assert_eq!(v.intervention, Intervention::Fallback);
}

#[test]
fn wrong_signal_arity_falls_back() {
    let mut core = SafetyCore::from_model(
        scalar_floor_model(15.0, OnUncertain::Fallback),
        CoreConfig::default(),
    )
    .unwrap();
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.reason, Reason::BadInput);
    assert_eq!(v.layer, Layer::Backup);
}

#[test]
fn violated_scalar_floor_forces_backup() {
    let cfg = CoreConfig::default();
    let mut core =
        SafetyCore::from_model(scalar_floor_model(15.0, OnUncertain::Hold), cfg.clone()).unwrap();
    // soc below the floor.
    let v = core.step(&SafetyInput {
        signals: &[10.0],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::ScalarViolated);
    assert_eq!(v.backup_kind, Some(BackupKind::Hold));
    assert_eq!(v.fired.len(), 1);
    assert_eq!(core.fired_name(v.fired[0]), "c_floor");

    // soc above the floor: nothing to enforce (no keep-out) → certified primary.
    let v = core.step(&SafetyInput {
        signals: &[20.0],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.layer, Layer::Primary);
    assert_eq!(v.reason, Reason::Certified);
}

#[test]
fn violated_monitor_forces_backup() {
    let cfg = CoreConfig::default();
    let mut core =
        SafetyCore::from_model(always_floor_monitor_model(120.0, 4), cfg.clone()).unwrap();
    // First tick above the floor: satisfied.
    let v = core.step(&SafetyInput {
        signals: &[200.0],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.layer, Layer::Primary);
    // Now below the floor: always-clause violated → backup.
    let v = core.step(&SafetyInput {
        signals: &[100.0],
        position: &[],
        velocity: &[],
        proposed: ProposedAction::Effort(&[]),
    });
    assert_eq!(v.reason, Reason::MonitorFired);
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.fired.len(), 1);
    assert_eq!(core.fired_name(v.fired[0]), "c_always");
}

#[test]
fn degenerate_shield_at_center_falls_back() {
    let cfg = CoreConfig::default();
    let mut core = SafetyCore::from_model(
        sphere_model(3, vec![0.0, 0.0, 0.0], 10.0, 2.0, OnUncertain::Fallback),
        cfg.clone(),
    )
    .unwrap();
    // Position exactly at the sphere centre: the barrier normal is undefined → uncertifiable.
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[0.0, 0.0, 0.0],
        velocity: &[0.0, 0.0, 0.0],
        proposed: ProposedAction::Effort(&[1.0, 0.0, 0.0]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::QpUncertifiable);
    finite_and_bounded(&v.certified_action, cfg.shield.u_max);
}

#[test]
fn safe_state_floor_retreats_toward_charging_pose() {
    // An energy floor keyed on_uncertain=safe_state, with an authored charging pose at (14,0,0)
    // (just outside the 12 m lander keep-out). Below the floor, the arbiter must hand control to
    // the verified retreat law: report backup_kind=SafeState and emit a bounded command that pulls
    // toward the pose (−x from the start at 40 m) — distinct from a pure brake (which, at rest,
    // would be zero).
    let cfg = CoreConfig::default();
    let mut core = SafetyCore::from_model(
        keepout_floor_model(OnUncertain::SafeState, Some(vec![14.0, 0.0, 0.0])),
        cfg.clone(),
    )
    .unwrap();
    let v = core.step(&SafetyInput {
        signals: &[5.0], // below the 15 floor
        position: &[40.0, 0.0, 0.0],
        velocity: &[0.0, 0.0, 0.0],
        proposed: ProposedAction::Effort(&[0.0, 0.0, 0.0]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::ScalarViolated);
    assert_eq!(v.backup_kind, Some(BackupKind::SafeState));
    finite_and_bounded(&v.certified_action, cfg.shield.u_max);
    assert!(
        v.certified_action[0] < 0.0,
        "retreat must pull toward the charging pose (−x), got {:?}",
        v.certified_action
    );
}

#[test]
fn safe_state_without_pose_is_still_fail_safe() {
    // Same floor keyed safe_state but with NO authored pose: the retreat degrades to brake-to-stop
    // (never fail-open). backup_kind still reports the selected safe_state for traceability.
    let cfg = CoreConfig::default();
    let mut core = SafetyCore::from_model(
        keepout_floor_model(OnUncertain::SafeState, None),
        cfg.clone(),
    )
    .unwrap();
    let v = core.step(&SafetyInput {
        signals: &[5.0],
        position: &[40.0, 0.0, 0.0],
        velocity: &[2.0, 0.0, 0.0],
        proposed: ProposedAction::Effort(&[9.0, 0.0, 0.0]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.backup_kind, Some(BackupKind::SafeState));
    finite_and_bounded(&v.certified_action, cfg.shield.u_max);
    // Degraded to brake: decelerating the +x velocity ⇒ −x command, never the +9 proposal.
    assert!(v.certified_action[0] < 0.0);
}

#[test]
fn hold_floor_latches_and_reports_hold() {
    // A floor keyed on_uncertain=hold: the arbiter latches the station-keep anchor and reports
    // backup_kind=Hold with a bounded command.
    let cfg = CoreConfig::default();
    let mut core =
        SafetyCore::from_model(keepout_floor_model(OnUncertain::Hold, None), cfg.clone()).unwrap();
    let v = core.step(&SafetyInput {
        signals: &[5.0],
        position: &[40.0, 0.0, 0.0],
        velocity: &[0.0, 0.0, 0.0],
        proposed: ProposedAction::Effort(&[3.0, 0.0, 0.0]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.backup_kind, Some(BackupKind::Hold));
    finite_and_bounded(&v.certified_action, cfg.shield.u_max);
}

#[test]
fn watchdog_expiry_is_a_safe_mode_trigger() {
    let cfg = CoreConfig::default();
    let mut core = SafetyCore::from_model(
        sphere_model(3, vec![0.0, 0.0, 0.0], 10.0, 2.0, OnUncertain::Fallback),
        cfg.clone(),
    )
    .unwrap();
    let mut out = core.new_verdict();
    core.force_watchdog(
        &SafetyInput {
            signals: &[],
            position: &[20.0, 0.0, 0.0],
            velocity: &[1.0, 0.0, 0.0],
            proposed: ProposedAction::Effort(&[5.0, 0.0, 0.0]),
        },
        &mut out,
    );
    assert_eq!(out.layer, Layer::Backup);
    assert_eq!(out.reason, Reason::WatchdogExpired);
    assert_eq!(out.backup_kind, Some(BackupKind::BrakeToStop));
    finite_and_bounded(&out.certified_action, cfg.shield.u_max);
}
