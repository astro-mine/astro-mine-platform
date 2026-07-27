//! Shared test fixtures: hand-built compiled models and a double-integrator simulator.

#![allow(dead_code)]
// The integrator steps position/velocity/control arrays in lockstep by index.
#![allow(clippy::needless_range_loop)]

use astro_mine_guard_core::model::{
    ActionLimits, AdmissibleDirectives, CompiledNode, CompiledSafetyModel, KeepOutTerm,
    MonitorAutomaton, OnUncertain, PredicateAtom, PredicateOp, ResourceBounds, ScalarBound, Shape,
    TemporalOp,
};

/// Empty resource bounds (the core does not depend on these counts at runtime — they are the
/// static-analysis witness; the actual pre-allocation is driven by the concrete lists).
pub fn empty_bounds() -> ResourceBounds {
    ResourceBounds::default()
}

/// A model with a single spherical keep-out and nothing else (pure shield exercise).
pub fn sphere_model(
    dim: usize,
    center: Vec<f64>,
    radius: f64,
    margin: f64,
    on_uncertain: OnUncertain,
) -> CompiledSafetyModel {
    let term = KeepOutTerm {
        constraint_id: "c_sphere".into(),
        on_uncertain,
        shape: Shape::Sphere,
        margin_m: margin,
        center,
        half_extents: vec![],
        radius: Some(radius),
        normal: vec![],
        offset: None,
    };
    base_model(dim, vec![], vec![], vec![], vec![term], vec![])
}

/// A model with a half-space keep-out `n·p + offset >= margin`.
pub fn halfspace_model(
    dim: usize,
    normal: Vec<f64>,
    offset: f64,
    margin: f64,
) -> CompiledSafetyModel {
    let term = KeepOutTerm {
        constraint_id: "c_half".into(),
        on_uncertain: OnUncertain::Fallback,
        shape: Shape::HalfSpace,
        margin_m: margin,
        center: vec![],
        half_extents: vec![],
        radius: None,
        normal,
        offset: Some(offset),
    };
    base_model(dim, vec![], vec![], vec![], vec![term], vec![])
}

/// A model with a single scalar floor `signal0 >= floor` (fail-safe = `on_uncertain`).
pub fn scalar_floor_model(floor: f64, on_uncertain: OnUncertain) -> CompiledSafetyModel {
    let atoms = vec![PredicateAtom {
        op: PredicateOp::Ge,
        signal_index: 0,
        threshold: floor,
    }];
    let bounds = vec![ScalarBound {
        constraint_id: "c_floor".into(),
        on_uncertain,
        atom_index: 0,
    }];
    base_model(0, vec!["soc".into()], atoms, bounds, vec![], vec![])
}

/// A model with a single `always[0,window] (signal0 >= floor)` temporal monitor.
pub fn always_floor_monitor_model(floor: f64, window: usize) -> CompiledSafetyModel {
    let atoms = vec![PredicateAtom {
        op: PredicateOp::Ge,
        signal_index: 0,
        threshold: floor,
    }];
    let root = CompiledNode {
        op: TemporalOp::Always,
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
    let monitor = MonitorAutomaton {
        constraint_id: "c_always".into(),
        on_uncertain: OnUncertain::Fallback,
        root,
        history_window_len: window,
        node_count: 2,
        predicate_indices: vec![0],
    };
    base_model(0, vec!["temp".into()], atoms, vec![], vec![], vec![monitor])
}

/// A combined model: a sphere keep-out + a scalar floor + an always monitor. Exercises every
/// layer of the arbiter in one core.
pub fn combined_model() -> CompiledSafetyModel {
    let center = vec![0.0, 0.0, 0.0];
    let term = KeepOutTerm {
        constraint_id: "c_sphere".into(),
        on_uncertain: OnUncertain::Fallback,
        shape: Shape::Sphere,
        margin_m: 2.0,
        center,
        half_extents: vec![],
        radius: Some(10.0),
        normal: vec![],
        offset: None,
    };
    let atoms = vec![
        PredicateAtom {
            op: PredicateOp::Ge,
            signal_index: 0,
            threshold: 15.0,
        }, // power floor
        PredicateAtom {
            op: PredicateOp::Ge,
            signal_index: 1,
            threshold: 120.0,
        }, // thermal floor
    ];
    let bounds = vec![ScalarBound {
        constraint_id: "c_power".into(),
        on_uncertain: OnUncertain::Fallback,
        atom_index: 0,
    }];
    let root = CompiledNode {
        op: TemporalOp::Always,
        predicate_index: None,
        interval_lo_samples: Some(0),
        interval_hi_samples: Some(4),
        args: vec![CompiledNode {
            op: TemporalOp::Predicate,
            predicate_index: Some(1),
            interval_lo_samples: None,
            interval_hi_samples: None,
            args: vec![],
        }],
    };
    let monitor = MonitorAutomaton {
        constraint_id: "c_thermal".into(),
        on_uncertain: OnUncertain::Hold,
        root,
        history_window_len: 4,
        node_count: 2,
        predicate_indices: vec![1],
    };
    base_model(
        3,
        vec!["power".into(), "temp".into()],
        atoms,
        bounds,
        vec![term],
        vec![monitor],
    )
}

/// A spatial model exercising the night-survival recover layer: a sphere keep-out (r=10, margin=2
/// at the origin) plus a single scalar floor `signal0 >= 15` whose `on_uncertain` selects the
/// distinct backup law, with an optional authored `safe_pose` (the retreat target). Used to prove
/// the arbiter routes a fired floor to a genuine Hold / SafeState law end-to-end.
pub fn keepout_floor_model(
    on_uncertain: OnUncertain,
    safe_pose: Option<Vec<f64>>,
) -> CompiledSafetyModel {
    let term = KeepOutTerm {
        constraint_id: "c_lander".into(),
        on_uncertain: OnUncertain::Fallback,
        shape: Shape::Sphere,
        margin_m: 2.0,
        center: vec![0.0, 0.0, 0.0],
        half_extents: vec![],
        radius: Some(10.0),
        normal: vec![],
        offset: None,
    };
    let atoms = vec![PredicateAtom {
        op: PredicateOp::Ge,
        signal_index: 0,
        threshold: 15.0,
    }];
    let bounds = vec![ScalarBound {
        constraint_id: "c_energy_floor".into(),
        on_uncertain,
        atom_index: 0,
    }];
    let mut m = base_model(3, vec!["power".into()], atoms, bounds, vec![term], vec![]);
    m.safe_pose = safe_pose;
    m
}

fn base_model(
    dim: usize,
    signals: Vec<String>,
    atoms: Vec<PredicateAtom>,
    scalar_bounds: Vec<ScalarBound>,
    keep_out_terms: Vec<KeepOutTerm>,
    monitors: Vec<MonitorAutomaton>,
) -> CompiledSafetyModel {
    let spatial_dim = if keep_out_terms.is_empty() {
        None
    } else {
        Some(dim)
    };
    CompiledSafetyModel {
        compiled_version: "0.1".into(),
        spec_id: "test-spec".into(),
        spec_content_hash: "sha256:test".into(),
        sample_period_s: 0.02,
        signals,
        atoms,
        scalar_bounds,
        keep_out_terms,
        monitors,
        resource_bounds: empty_bounds(),
        action_limits: ActionLimits::default(),
        spatial_dim,
        safe_pose: None,
        // No authored directive grant by default — so a fixture model certifies NO MODE/TASK,
        // whatever the CoreConfig allowlist says (RFC-0004 Amendment 2). A test that needs a
        // directive certified authors the grant on the model, exactly as a reviewed spec would.
        admissible_directives: None,
    }
}

/// The reviewed MODE/TASK grant a test model authors — the *contract* side of the action gate.
/// `CoreConfig.action_policy` can only ever narrow this (`spec ∩ config`).
pub fn grant(modes: &[&str], tasks: &[&str]) -> AdmissibleDirectives {
    AdmissibleDirectives {
        modes: modes.iter().map(|s| (*s).to_string()).collect(),
        tasks: tasks.iter().map(|s| (*s).to_string()).collect(),
    }
}

/// A double-integrator point mass: `ṗ = v`, `v̇ = u`. The plant Guard shields.
#[derive(Clone, Debug)]
pub struct Plant {
    pub p: Vec<f64>,
    pub v: Vec<f64>,
    pub dt: f64,
}

impl Plant {
    pub fn new(p: Vec<f64>, v: Vec<f64>, dt: f64) -> Self {
        Self { p, v, dt }
    }

    /// Apply a control (acceleration) for one step (semi-implicit Euler).
    pub fn step(&mut self, u: &[f64]) {
        for i in 0..self.p.len() {
            self.v[i] += u[i] * self.dt;
            self.p[i] += self.v[i] * self.dt;
        }
    }

    /// Signed distance to a sphere surface (`h = ‖p−c‖ − (r+margin)`).
    pub fn sphere_barrier(&self, center: &[f64], radius: f64) -> f64 {
        let d2: f64 = self
            .p
            .iter()
            .zip(center)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        d2.sqrt() - radius
    }
}
