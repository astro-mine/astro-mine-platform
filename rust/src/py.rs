//! PyO3 binding — the in-process bridge from the Python orchestration layer to the trusted
//! Rust core (guard.md §3, §4). The core does all the safety work; this layer only marshals
//! typed inputs in and an auditable verdict out. Built into `astro_mine.guard._core` by
//! maturin; the same core also runs with no Python at all on the edge.
//!
//! The marshalling is deliberately **fail-closed at the boundary**: an `action_kind` string outside
//! the closed vocabulary, or a directive with no name, is marshalled into the core as an *opaque*
//! (uncertifiable) action rather than raising — a malformed action must reach the arbiter's gate and
//! be answered with a verified safe command, not blow up the control loop.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Duration;

use crate::arbiter::{
    ActionPolicy, CoreConfig, Intervention, Layer, ProposedAction, Reason, SafetyCore, SafetyInput,
    SafetyVerdict,
};
use crate::backup::BackupKind;
use crate::shield::{ControlMode, ShieldConfig};

fn layer_str(l: Layer) -> &'static str {
    match l {
        Layer::Primary => "primary",
        Layer::Shield => "shield",
        Layer::Backup => "backup",
    }
}

fn intervention_str(i: Intervention) -> &'static str {
    match i {
        Intervention::None => "none",
        Intervention::Modified => "modified",
        Intervention::Fallback => "fallback",
    }
}

fn reason_str(r: Reason) -> &'static str {
    match r {
        Reason::Certified => "certified",
        Reason::ShieldCorrected => "shield_corrected",
        Reason::ScalarViolated => "scalar_violated",
        Reason::MonitorFired => "monitor_fired",
        Reason::QpUncertifiable => "qp_uncertifiable",
        Reason::BadInput => "bad_input",
        Reason::WatchdogExpired => "watchdog_expired",
        Reason::NotCertifiable => "not_certifiable",
    }
}

fn backup_str(k: BackupKind) -> &'static str {
    match k {
        BackupKind::BrakeToStop => "brake_to_stop",
        BackupKind::Hold => "hold",
        BackupKind::SafeState => "safe_state",
    }
}

fn mode_str(m: ControlMode) -> &'static str {
    match m {
        ControlMode::Effort => "effort",
        ControlMode::Velocity => "velocity",
        ControlMode::Position => "position",
    }
}

/// Parse a control-mode token for the *configured* fallback channel. Unlike the per-tick
/// `action_kind` (which fails closed), a misconfigured core must fail **loud at construction** —
/// never mid-episode (guard.md §9.1).
fn parse_mode(s: &str) -> PyResult<ControlMode> {
    match s {
        "effort" => Ok(ControlMode::Effort),
        "velocity" => Ok(ControlMode::Velocity),
        "position" => Ok(ControlMode::Position),
        other => Err(PyValueError::new_err(format!(
            "fallback_control_mode must be effort|velocity|position, got {other:?}"
        ))),
    }
}

/// The trusted safety core, callable from Python. Holds a reused verdict buffer so repeated
/// `step` calls do no Rust-side allocation on the safety path.
#[pyclass(name = "SafetyCore")]
struct PySafetyCore {
    core: SafetyCore,
    verdict: SafetyVerdict,
}

#[pymethods]
impl PySafetyCore {
    /// Load a core from a `CompiledSafetyModel` protobuf payload (`compiled_to_wire`).
    #[staticmethod]
    #[pyo3(signature = (
        compiled_wire,
        *,
        u_max = 20.0,
        v_max = 2.0,
        k0 = 9.0,
        k1 = 6.0,
        k_brake = 4.0,
        predictive_horizon_samples = 5,
        deadline_us = None,
        max_history_cap = 1_048_576,
        certified_modes = None,
        certified_tasks = None,
        fallback_control_mode = "effort",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_wire(
        compiled_wire: &[u8],
        u_max: f64,
        v_max: f64,
        k0: f64,
        k1: f64,
        k_brake: f64,
        predictive_horizon_samples: usize,
        deadline_us: Option<u64>,
        max_history_cap: usize,
        certified_modes: Option<Vec<String>>,
        certified_tasks: Option<Vec<String>>,
        fallback_control_mode: &str,
    ) -> PyResult<Self> {
        let cfg = CoreConfig {
            predictive_horizon_samples,
            shield: ShieldConfig {
                k0,
                k1,
                u_max,
                v_max,
                ..Default::default()
            },
            k_brake,
            max_history_cap,
            deadline: deadline_us.map(Duration::from_micros),
            unchanged_tol: 1e-6,
            action_policy: ActionPolicy {
                certified_modes: certified_modes.unwrap_or_default(),
                certified_tasks: certified_tasks.unwrap_or_default(),
                fallback_mode: parse_mode(fallback_control_mode)?,
            },
        };
        let core = SafetyCore::from_wire(compiled_wire, cfg)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let verdict = core.new_verdict();
        Ok(Self { core, verdict })
    }

    #[getter]
    fn spatial_dim(&self) -> usize {
        self.core.spatial_dim()
    }

    #[getter]
    fn spec_id(&self) -> &str {
        self.core.spec_id()
    }

    #[getter]
    fn spec_content_hash(&self) -> &str {
        self.core.spec_content_hash()
    }

    /// The enforced commanded-speed ceiling (the reviewed spec's `max_velocity_mps`, else the
    /// configured `v_max`); `None` for a non-spatial model, which has no shield.
    #[getter]
    fn v_max(&self) -> Option<f64> {
        self.core.v_max()
    }

    /// Certify one action. Returns a dict with `certified_action`, `certified_control_mode`,
    /// `layer`, `intervention`, `reason`, `fired` (constraint ids), `backup_kind`, and
    /// `min_barrier_margin`.
    ///
    /// `action_kind` classifies the proposal for the arbiter's action gate:
    /// `effort` | `velocity` | `position` (modelled commands, carried in `proposed_action`),
    /// `mode` | `task` (discrete directives, named by `directive`), or `opaque` (an actuator
    /// command the core has no plant model for). **Any unrecognised token — and a directive with
    /// no name — is treated as `opaque`, i.e. uncertifiable**: the core substitutes a verified safe
    /// command rather than passing an action it cannot reason about (fail-closed, never fail-open).
    ///
    /// With `watchdog=True` the real-time per-tick deadline is armed (a deadline miss forces
    /// the safe fallback); the default deterministic path is used otherwise.
    #[pyo3(signature = (
        signals, position, velocity, proposed_action,
        *, action_kind = "effort", directive = None, watchdog = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        signals: Vec<f64>,
        position: Vec<f64>,
        velocity: Vec<f64>,
        proposed_action: Vec<f64>,
        action_kind: &str,
        directive: Option<String>,
        watchdog: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let name = directive.unwrap_or_default();
        let proposed = match action_kind {
            "effort" => ProposedAction::Effort(&proposed_action),
            "velocity" => ProposedAction::Velocity(&proposed_action),
            "position" => ProposedAction::Position(&proposed_action),
            // A directive with no name can never match an allowlist entry, so it is rejected — but
            // it is still *classified* as a directive so the audit trail says so.
            "mode" => ProposedAction::Mode(&name),
            "task" => ProposedAction::Task(&name),
            // "opaque" — and anything the vocabulary does not know — is uncertifiable.
            _ => ProposedAction::Opaque,
        };
        let input = SafetyInput {
            signals: &signals,
            position: &position,
            velocity: &velocity,
            proposed,
        };
        if watchdog {
            self.core.step_guarded(&input, &mut self.verdict);
        } else {
            self.core.step_into(&input, &mut self.verdict);
        }

        let v = &self.verdict;
        let d = PyDict::new(py);
        d.set_item("certified_action", v.certified_action.clone())?;
        d.set_item("certified_control_mode", mode_str(v.certified_mode))?;
        d.set_item("layer", layer_str(v.layer))?;
        d.set_item("intervention", intervention_str(v.intervention))?;
        d.set_item("reason", reason_str(v.reason))?;
        let fired: Vec<&str> = v.fired.iter().map(|&i| self.core.fired_name(i)).collect();
        d.set_item("fired", fired)?;
        d.set_item("backup_kind", v.backup_kind.map(backup_str))?;
        d.set_item("min_barrier_margin", v.min_barrier_margin)?;
        Ok(d)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafetyCore>()?;
    m.add(
        "__doc__",
        "Astro-Mine-Guard trusted safety core (Rust TCB, RM-P1-GUARD-02).",
    )?;
    Ok(())
}
