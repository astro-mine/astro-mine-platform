//! Acceptance: **every path to actuation crosses Guard** (RM-P1-GUARD-03; LUNAR-FR-006,
//! LUNAR-SR-004; guard.md §3, §6, §9.1).
//!
//! The arbiter's *action gate* is what makes that claim true for the whole Core `Action` union, not
//! just for `EFFORT` setpoints. These tests pin the four dispositions and — the load-bearing part —
//! that **no** kind of action can reach an actuator uncertified:
//!
//! - `VELOCITY` / `POSITION` setpoints are projected onto the reviewed kinematic envelope **and**
//!   the keep-out safe set, and the certified command's realised next pose is in the safe set;
//! - an actuator command in an unmodelled control mode (`IMPEDANCE` / `TRAJECTORY` → `Opaque`) is
//!   **rejected**, never passed through;
//! - a `MODE` / `TASK` directive is certified **only** when it is on the reviewed allowlist;
//! - every rejection is answered **in the proposal's own actuation channel** (a velocity proposal
//!   gets a velocity command back), because a safe command in a channel the plant does not read is
//!   a fail-open in disguise.

mod common;

use astro_mine_guard_core::model::{ActionLimits, AdmissibleDirectives, OnUncertain};
use astro_mine_guard_core::{
    ActionPolicy, BackupKind, ControlMode, CoreConfig, Intervention, Layer, ProposedAction, Reason,
    SafetyCore, SafetyInput,
};
use common::*;

/// The lander-zone sphere (r = 10, margin = 2 at the origin), a 1 s tick, and a reviewed 0.5 m/s
/// commanded-speed ceiling — so `step_max = v_max·dt = 0.5 m`.
///
/// The model authors **no** directive grant, so no MODE/TASK is certifiable through this core
/// however permissive `policy` is (RFC-0004 Amendment 2). Use [`kinematic_core_granting`] for the
/// tests that need the *contract* to admit a directive.
fn kinematic_core(policy: ActionPolicy) -> SafetyCore {
    kinematic_core_with(policy, None)
}

/// As [`kinematic_core`], but the **reviewed model** authors `authored` — the contract side of the
/// gate. The effective allowlist the core enforces is `authored ∩ policy`.
fn kinematic_core_granting(policy: ActionPolicy, authored: AdmissibleDirectives) -> SafetyCore {
    kinematic_core_with(policy, Some(authored))
}

fn kinematic_core_with(policy: ActionPolicy, authored: Option<AdmissibleDirectives>) -> SafetyCore {
    let mut model = sphere_model(2, vec![0.0, 0.0], 10.0, 2.0, OnUncertain::Fallback);
    model.sample_period_s = 1.0;
    model.action_limits = ActionLimits {
        max_velocity_mps: Some(0.5),
        max_accel_mps2: None,
    };
    model.admissible_directives = authored;
    let cfg = CoreConfig {
        action_policy: policy,
        ..Default::default()
    };
    SafetyCore::from_model(model, cfg).unwrap()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The exact safe-set predicate, re-derived here (an *external* check, not a restatement of the
/// core's own): outside the sphere by the margin.
fn outside_keepout(p: &[f64]) -> bool {
    norm(p) >= 12.0 - 1e-6
}

// --- VELOCITY: kinematic-limit + keep-out projection -----------------------------------------

#[test]
fn velocity_setpoint_is_capped_by_the_reviewed_kinematic_limit() {
    let mut core = kinematic_core(ActionPolicy::default());
    // Far from the keep-out; only the speed ceiling can bind.
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Velocity(&[9.0, 12.0]), // ‖w‖ = 15 m/s ≫ 0.5
    });
    assert_eq!(v.layer, Layer::Shield);
    assert_eq!(v.intervention, Intervention::Modified);
    assert_eq!(v.certified_mode, ControlMode::Velocity);
    assert!(
        norm(&v.certified_action) <= 0.5 + 1e-9,
        "certified speed {} exceeds the reviewed 0.5 m/s ceiling",
        norm(&v.certified_action)
    );
    // Minimal perturbation: the direction is preserved (a pure radial scaling).
    let a = &v.certified_action;
    assert!((a[0] / a[1] - 9.0 / 12.0).abs() < 1e-6);
}

#[test]
fn velocity_setpoint_into_the_keepout_is_projected_and_the_next_pose_is_safe() {
    let mut core = kinematic_core(ActionPolicy::default());
    // Just outside the barrier (h = 0.2 m) driving straight at the centre at full authority.
    let p = [12.2f64, 0.0];
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &p,
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Velocity(&[-0.5, 0.0]),
    });
    assert_eq!(v.intervention, Intervention::Modified);
    let w = &v.certified_action;
    // The realised next pose (dt = 1 s) is still outside the keep-out — the discrete-time
    // forward-invariance certificate, checked externally.
    let next = [p[0] + w[0], p[1] + w[1]];
    assert!(
        outside_keepout(&next),
        "certified velocity {w:?} drove the next pose {next:?} into the keep-out"
    );
}

#[test]
fn a_safe_velocity_setpoint_is_certified_untouched() {
    let mut core = kinematic_core(ActionPolicy::default());
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Velocity(&[0.3, 0.0]),
    });
    assert_eq!(v.layer, Layer::Primary);
    assert_eq!(v.intervention, Intervention::None);
    assert_eq!(v.reason, Reason::Certified);
    assert!((v.certified_action[0] - 0.3).abs() < 1e-9);
}

// --- POSITION: step-cap + keep-out projection -------------------------------------------------

#[test]
fn position_setpoint_is_step_capped_and_stays_in_the_safe_set() {
    let mut core = kinematic_core(ActionPolicy::default());
    let p = [13.0f64, 0.0];
    // Command a teleport to the keep-out centre.
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &p,
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Position(&[0.0, 0.0]),
    });
    assert_eq!(v.intervention, Intervention::Modified);
    assert_eq!(v.certified_mode, ControlMode::Position);
    let q = &v.certified_action;
    // Inside the step ball (v_max·dt = 0.5 m) …
    let step = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
    assert!(step <= 0.5 + 1e-6, "commanded step {step} exceeds 0.5 m");
    // … and the *target itself* is outside the keep-out.
    assert!(
        outside_keepout(q),
        "certified target {q:?} is in the keep-out"
    );
}

#[test]
fn a_position_target_inside_the_keepout_cannot_be_certified() {
    let mut core = kinematic_core(ActionPolicy::default());
    // The pose is *already* deep inside the keep-out: no target within one step is in the safe set,
    // so the projection is infeasible and the core must fall back rather than certify anything.
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[1.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Position(&[0.0, 0.0]),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::QpUncertifiable);
    assert_eq!(v.certified_mode, ControlMode::Position);
    // The recover layer answered in the *position* channel: stay put.
    assert_eq!(v.certified_action, vec![1.0, 0.0]);
}

// --- OPAQUE: the unmodelled control modes (IMPEDANCE / TRAJECTORY) ----------------------------

#[test]
fn an_unmodelled_actuator_command_is_rejected_not_passed_through() {
    let mut core = kinematic_core(ActionPolicy::default());
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[1.0, 0.0],
        proposed: ProposedAction::Opaque,
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::NotCertifiable);
    assert_eq!(v.intervention, Intervention::Fallback);
    assert_eq!(v.backup_kind, Some(BackupKind::BrakeToStop));
    // Answered in the configured fallback channel (EFFORT by default): a brake opposing the motion.
    assert_eq!(v.certified_mode, ControlMode::Effort);
    assert!(v.certified_action[0] < 0.0);
}

#[test]
fn the_opaque_fallback_channel_is_configurable() {
    // A velocity-tracking plant configures `fallback_mode = Velocity`, so a rejected directive is
    // answered with a *velocity* command its actuator actually reads.
    let mut core = kinematic_core(ActionPolicy {
        fallback_mode: ControlMode::Velocity,
        ..Default::default()
    });
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[1.0, 0.0],
        proposed: ProposedAction::Opaque,
    });
    assert_eq!(v.certified_mode, ControlMode::Velocity);
    assert_eq!(
        v.certified_action,
        vec![0.0, 0.0],
        "the velocity floor is a stop"
    );
}

// --- MODE / TASK: the certified allowlist ------------------------------------------------------

#[test]
fn an_unlisted_mode_is_rejected() {
    let mut core = kinematic_core(ActionPolicy::default()); // empty allowlist
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[2.0, 0.0],
        proposed: ProposedAction::Mode("drill_hard"),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::NotCertifiable);
    assert!(
        v.certified_action[0] < 0.0,
        "a rejected mode must be answered with a verified safe command"
    );
}

#[test]
fn an_allowlisted_mode_is_certified_and_carries_no_command() {
    // Certified only because BOTH admit it: the reviewed model grants `safe_hold`, and so does the
    // configuration.
    let mut core = kinematic_core_granting(
        ActionPolicy {
            certified_modes: vec!["safe_hold".into()],
            ..Default::default()
        },
        grant(&["safe_hold"], &[]),
    );
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[2.0, 0.0],
        proposed: ProposedAction::Mode("safe_hold"),
    });
    assert_eq!(v.layer, Layer::Primary);
    assert_eq!(v.intervention, Intervention::None);
    assert_eq!(v.reason, Reason::Certified);
    assert_eq!(v.backup_kind, None);
    // A directive carries no numeric command — the marshal layer re-emits the proposal untouched.
    assert!(v.certified_action.is_empty());
}

#[test]
fn a_task_is_gated_against_its_own_allowlist() {
    let mut core = kinematic_core_granting(
        ActionPolicy {
            certified_modes: vec!["standby".into()], // the *mode* list must not certify a *task*
            certified_tasks: vec!["prospect".into()],
            ..Default::default()
        },
        grant(&["standby"], &["prospect"]),
    );
    let listed = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Task("prospect"),
    });
    assert_eq!(listed.reason, Reason::Certified);

    let cross_listed = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Task("standby"),
    });
    assert_eq!(
        cross_listed.reason,
        Reason::NotCertifiable,
        "a mode allowlist must never certify a task"
    );
}

#[test]
fn a_fired_detect_layer_outranks_the_allowlist() {
    // A pre-certified directive is still *not* emitted while a hard constraint is violated: the
    // detect layers keep their precedence over the gate (guard.md §9.2).
    let mut model = keepout_floor_model(OnUncertain::Fallback, None);
    model.sample_period_s = 1.0;
    model.admissible_directives = Some(grant(&["safe_hold"], &[]));
    let cfg = CoreConfig {
        action_policy: ActionPolicy {
            certified_modes: vec!["safe_hold".into()],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut core = SafetyCore::from_model(model, cfg).unwrap();
    let v = core.step(&SafetyInput {
        signals: &[1.0], // power floor is 15 → violated
        position: &[50.0, 0.0, 0.0],
        velocity: &[1.0, 0.0, 0.0],
        proposed: ProposedAction::Mode("safe_hold"),
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::ScalarViolated);
}

// --- the effective allowlist is `spec ∩ config` (RFC-0004 Amendment 2) -------------------------

#[test]
fn a_spec_silent_model_certifies_nothing_however_permissive_the_config() {
    // **The headline invariant.** The reviewed contract authored no grant, so NOTHING is
    // certifiable — even though the configuration allowlists both directives by name. Before
    // Amendment 2 this certified both, which made a config `params` dict the only thing standing
    // between an untrusted policy and an uncertified passthrough to an actuator.
    let mut core = kinematic_core(ActionPolicy {
        certified_modes: vec!["safe_hold".into(), "velocity".into()],
        certified_tasks: vec!["prospect".into(), "standby".into()],
        ..Default::default()
    });
    for proposed in [
        ProposedAction::Mode("safe_hold"),
        ProposedAction::Mode("velocity"),
        ProposedAction::Task("prospect"),
        ProposedAction::Task("standby"),
    ] {
        let v = core.step(&SafetyInput {
            signals: &[],
            position: &[100.0, 0.0],
            velocity: &[2.0, 0.0],
            proposed,
        });
        assert_eq!(
            v.reason,
            Reason::NotCertifiable,
            "a spec-silent contract must admit no directive, whatever the configuration grants"
        );
        assert_eq!(v.layer, Layer::Backup);
    }
}

#[test]
fn configuration_can_only_narrow_the_authored_grant() {
    // The contract admits {a, b}; the deployment is configured stricter, for {b, c}. The effective
    // gate is the intersection: only `b` is certifiable. `a` is *authored but not configured*
    // (the legitimate "run stricter than the contract" case) and `c` is *configured but not
    // authored* (the fail-open this whole change removes).
    let mut core = kinematic_core_granting(
        ActionPolicy {
            certified_modes: vec!["b".into(), "c".into()],
            ..Default::default()
        },
        grant(&["a", "b"], &[]),
    );
    let admitted = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Mode("b"),
    });
    assert_eq!(
        admitted.reason,
        Reason::Certified,
        "spec ∩ config must admit b"
    );

    for name in ["a", "c"] {
        let v = core.step(&SafetyInput {
            signals: &[],
            position: &[100.0, 0.0],
            velocity: &[0.0, 0.0],
            proposed: ProposedAction::Mode(name),
        });
        assert_eq!(
            v.reason,
            Reason::NotCertifiable,
            "only the intersection is certifiable — {name} is in exactly one of the two"
        );
    }
}

#[test]
fn an_empty_config_admits_nothing_even_when_the_spec_grants() {
    // The dual of the headline invariant, and the reason the knob survives at all: a deployment
    // may run *stricter* than the reviewed contract. An empty configured allowlist is a total
    // revocation.
    let mut core =
        kinematic_core_granting(ActionPolicy::default(), grant(&["safe_hold"], &["dock"]));
    for proposed in [
        ProposedAction::Mode("safe_hold"),
        ProposedAction::Task("dock"),
    ] {
        let v = core.step(&SafetyInput {
            signals: &[],
            position: &[100.0, 0.0],
            velocity: &[0.0, 0.0],
            proposed,
        });
        assert_eq!(v.reason, Reason::NotCertifiable);
    }
}

#[test]
fn an_authored_mode_grant_never_certifies_a_task() {
    // Re-asserted through the new path: the two allowlists are separate permission sets on both
    // sides of the intersection, so a MODE grant (spec *and* config) cannot certify a TASK of the
    // same name.
    let mut core = kinematic_core_granting(
        ActionPolicy {
            certified_modes: vec!["dock".into()],
            certified_tasks: vec!["dock".into()],
            ..Default::default()
        },
        grant(&["dock"], &[]), // the contract grants the MODE `dock`, and no TASK at all
    );
    let mode = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Mode("dock"),
    });
    assert_eq!(mode.reason, Reason::Certified);

    let task = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Task("dock"),
    });
    assert_eq!(
        task.reason,
        Reason::NotCertifiable,
        "a MODE grant must never certify a TASK, even one configured by the same name"
    );
}

// --- arity / non-spatial: the plant cannot evaluate the setpoint ------------------------------

#[test]
fn a_setpoint_of_the_wrong_arity_falls_back_in_its_own_channel() {
    let mut core = kinematic_core(ActionPolicy::default()); // 2-D plant
    let v = core.step(&SafetyInput {
        signals: &[],
        position: &[100.0, 0.0],
        velocity: &[0.0, 0.0],
        proposed: ProposedAction::Velocity(&[0.1]), // 1-D setpoint in a 2-D frame
    });
    assert_eq!(v.layer, Layer::Backup);
    assert_eq!(v.reason, Reason::BadInput);
    assert_eq!(v.certified_mode, ControlMode::Velocity);
    assert_eq!(v.certified_action, vec![0.0, 0.0]);
}

#[test]
fn a_non_finite_setpoint_never_reaches_an_actuator() {
    let mut core = kinematic_core(ActionPolicy::default());
    for proposed in [
        ProposedAction::Velocity(&[f64::NAN, 0.0]),
        ProposedAction::Position(&[f64::INFINITY, 0.0]),
        ProposedAction::Effort(&[f64::NAN, f64::NAN]),
    ] {
        let v = core.step(&SafetyInput {
            signals: &[],
            position: &[100.0, 0.0],
            velocity: &[0.0, 0.0],
            proposed,
        });
        assert_eq!(v.layer, Layer::Backup);
        assert_eq!(v.reason, Reason::BadInput);
        assert!(
            v.certified_action.iter().all(|x| x.is_finite()),
            "a non-finite proposal produced a non-finite certified command"
        );
    }
}
