// SPDX-License-Identifier: Apache-2.0
//! The arbiter — the decision core of the TCB (guard.md §2, §9.1, §9.2).
//!
//! Every tick it combines the *detect* layers (scalar-bound predicates + STL/MTL monitors), the
//! **action gate** (is this even an action the core can certify?), the *correct* layer (the CBF-QP
//! shield), and the *recover* layer (simplex backup) into **exactly one certified action**, under a
//! strict precedence:
//!
//! 1. **Watchdog** expiry (deadline miss) → backup.
//! 2. **Bad input** (wrong arity, non-finite) → backup.
//! 3. A fired **scalar bound** (floor/ceiling/torque/kinematic violated) → backup.
//! 4. A fired **monitor** (temporal clause violated *or* predicted imminent) → backup.
//! 5. The **action gate** cannot certify the *kind* of action proposed → backup.
//! 6. The **shield** cannot certify the (perturbed) command → backup.
//! 7. Otherwise → the certified command (minimally perturbed, or the proposal itself).
//!
//! No positive certificate ⇒ backup: fail-safe, never fail-open. The monitors are advanced
//! **every** tick (even when an earlier layer forces the backup) so their bounded history
//! stays consistent. After construction the hot path is allocation-free.
//!
//! ## The action gate (RM-P1-GUARD-03; LUNAR-FR-006 "every path to actuation crosses Guard")
//!
//! A Core `Action` is a tagged union — an actuator command in one of five control modes, a discrete
//! `MODE` command, or a `TASK` directive. The gate is what makes "every path to actuation crosses
//! Guard" true rather than aspirational, by classifying **every** proposal into exactly one of:
//!
//! - a **modelled command** ([`ProposedAction::Effort`] / [`Velocity`](ProposedAction::Velocity) /
//!   [`Position`](ProposedAction::Position)) → projected onto the safe set by the shield;
//! - a **discrete directive** ([`Mode`](ProposedAction::Mode) / [`Task`](ProposedAction::Task)) →
//!   certified **only** if it is on the [`ActionPolicy`] allowlist. A continuous projection is
//!   meaningless for a discrete directive; enumeration of a pre-certified safe set is the
//!   corresponding discipline. The allowlist that gates it is the **effective** one —
//!   `reviewed SafetySpec ∩ configuration` (RFC-0004 Amendment 2), resolved once in
//!   [`SafetyCore::from_model`]. Configuration may only *narrow* the reviewed grant, and a spec that
//!   authored none admits **nothing**: an unconfigured *or* unauthored Guard certifies *no*
//!   directive;
//! - an **opaque command** ([`ProposedAction::Opaque`]) — an actuator command in a control mode the
//!   TCB has no plant model for (`IMPEDANCE`, `TRAJECTORY`), or a setpoint the spatial plant cannot
//!   evaluate (wrong arity, non-spatial model). It carries no certifiable meaning, so it is
//!   **rejected**: the arbiter substitutes a verified safe command. There is deliberately **no**
//!   "unmodelled ⇒ pass through" path — that would be exactly the fail-open the whole component
//!   exists to prevent (guard.md §9.1).
//!
//! A rejected proposal falls back **in its own actuation channel** where one exists (a `VELOCITY`
//! proposal is answered with a certified zero-velocity command, not with an `EFFORT` brake the
//! downstream actuator would ignore), and in [`ActionPolicy::fallback_mode`] otherwise.

use std::time::{Duration, Instant};

use crate::backup::{Backup, BackupKind};
use crate::model::{AdmissibleDirectives, CompiledSafetyModel, CoreError, OnUncertain};
use crate::monitors::Monitor;
use crate::shield::{narrow, ControlMode, Shield, ShieldConfig};

/// Which layer produced the certified action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// The proposal was certified unchanged.
    Primary,
    /// The CBF-QP shield perturbed the proposal.
    Shield,
    /// A verified backup controller took over.
    Backup,
}

/// What Guard did to the proposed action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intervention {
    None,
    Modified,
    Fallback,
}

/// Why the arbiter reached its verdict (the audit reason — guard.md §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    Certified,
    ShieldCorrected,
    ScalarViolated,
    MonitorFired,
    QpUncertifiable,
    BadInput,
    WatchdogExpired,
    /// The *kind* of action proposed cannot be certified at all: an actuator command in an
    /// unmodelled control mode, a setpoint the spatial plant cannot evaluate, or a discrete
    /// `MODE`/`TASK` directive that is not on the certified allowlist.
    NotCertifiable,
}

/// The proposal, classified by the one thing the TCB must know about it: what it commands.
#[derive(Debug, Clone, Copy)]
pub enum ProposedAction<'a> {
    /// Commanded acceleration (Core `ACTUATOR`/`EFFORT`), `spatial_dim` long.
    Effort(&'a [f64]),
    /// Commanded velocity (Core `ACTUATOR`/`VELOCITY`), `spatial_dim` long.
    Velocity(&'a [f64]),
    /// Commanded position target (Core `ACTUATOR`/`POSITION`), `spatial_dim` long.
    Position(&'a [f64]),
    /// An actuator command the TCB has no plant model for (`IMPEDANCE`/`TRAJECTORY`), or whose
    /// setpoint the spatial plant cannot evaluate. Never certifiable.
    Opaque,
    /// A discrete mode command, by name (Core `MODE`).
    Mode(&'a str),
    /// A task directive, by task kind (Core `TASK`).
    Task(&'a str),
}

impl<'a> ProposedAction<'a> {
    /// The control mode this proposal commands in, if the TCB models it.
    #[inline]
    pub fn control_mode(&self) -> Option<ControlMode> {
        match self {
            Self::Effort(_) => Some(ControlMode::Effort),
            Self::Velocity(_) => Some(ControlMode::Velocity),
            Self::Position(_) => Some(ControlMode::Position),
            Self::Opaque | Self::Mode(_) | Self::Task(_) => None,
        }
    }

    /// The numeric setpoint (empty for a directive or an opaque command).
    #[inline]
    pub fn setpoint(&self) -> &'a [f64] {
        match self {
            Self::Effort(x) | Self::Velocity(x) | Self::Position(x) => x,
            Self::Opaque | Self::Mode(_) | Self::Task(_) => &[],
        }
    }
}

/// The certified allowlist for **discrete** directives, plus the channel a rejected directive falls
/// back in (guard.md §3 "the arbiter certifies"; RM-P1-GUARD-03).
///
/// Empty by default: an unconfigured Guard certifies **no** `MODE` and **no** `TASK`.
///
/// As supplied by the caller this is the **configured** allowlist, and it may only ever *narrow* the
/// grant the reviewed `SafetySpec` authored (RFC-0004 Amendment 2). [`SafetyCore::from_model`]
/// intersects the two **once**, at construction (`shield::narrow`), and the core then gates against
/// that **effective** policy: `effective = spec ∩ config`. A configuration can therefore *tighten*
/// the reviewed contract but can never create a permission — and where the model authored **no**
/// grant, the effective allowlist is **empty**, so nothing is certifiable however permissive the
/// configuration is. That is the whole point: an admitted directive is re-emitted *untouched*, so
/// the grant is a safety decision and belongs in the content-addressed, signed contract, not in an
/// unsigned config dict.
///
/// `fallback_mode` stays pure configuration — it names the actuation *channel* a rejected proposal
/// is answered in (a plant property, not a permission), and no setting of it can widen what is
/// certifiable.
#[derive(Debug, Clone, Default)]
pub struct ActionPolicy {
    /// Pre-certified `ModeCommand.mode` names.
    pub certified_modes: Vec<String>,
    /// Pre-certified `TaskDirective.task_kind` values.
    pub certified_tasks: Vec<String>,
    /// The actuation channel a rejected *directive* (or opaque command) is answered in.
    pub fallback_mode: ControlMode,
}

impl ActionPolicy {
    /// Whether the gate admits `proposed` **as certified without projection**. Only a discrete
    /// directive on its allowlist can be; a modelled command must go through the shield, and an
    /// opaque command is never admitted (the fail-closed core of the gate — proved with Kani in
    /// `verify.rs`). Allocation-free: a linear scan over borrowed strings.
    ///
    /// On the hot path `self` is always the **effective** (already-narrowed) policy — see the type
    /// docs.
    #[inline]
    pub fn admits(&self, proposed: &ProposedAction<'_>) -> bool {
        match proposed {
            ProposedAction::Mode(name) => contains(&self.certified_modes, name),
            ProposedAction::Task(kind) => contains(&self.certified_tasks, kind),
            _ => false,
        }
    }

    /// The **effective** gate policy: this *configured* allowlist intersected with the grant the
    /// reviewed model authored (RFC-0004 Amendment 2).
    ///
    /// `authored == None` (the spec is silent) ⇒ **both allowlists are empty** — silence grants
    /// nothing. Computed once, at core construction; the hot path never allocates.
    fn narrowed_to(&self, authored: Option<&AdmissibleDirectives>) -> Self {
        Self {
            certified_modes: narrow(&self.certified_modes, authored.map(|a| a.modes.as_slice())),
            certified_tasks: narrow(&self.certified_tasks, authored.map(|a| a.tasks.as_slice())),
            fallback_mode: self.fallback_mode,
        }
    }
}

#[inline]
fn contains(list: &[String], needle: &str) -> bool {
    list.iter().any(|s| s == needle)
}

/// Per-tick input to the safety core.
#[derive(Debug, Clone, Copy)]
pub struct SafetyInput<'a> {
    /// One value per declared signal (`model.signals`), in signal-table order.
    pub signals: &'a [f64],
    /// Agent position in the keep-out frame (`spatial_dim` long).
    pub position: &'a [f64],
    /// Agent velocity.
    pub velocity: &'a [f64],
    /// The action the (untrusted) policy proposes, classified for the gate.
    pub proposed: ProposedAction<'a>,
}

/// The auditable output of one tick (guard.md §3 `SafetyVerdict`). Pre-allocated and reused.
#[derive(Debug, Clone)]
pub struct SafetyVerdict {
    pub certified_action: Vec<f64>,
    /// The control mode `certified_action` is expressed in. Meaningful whenever the verdict carries
    /// a numeric command; a certified *directive* carries no command (`certified_action` is empty)
    /// and the consumer re-emits the proposal untouched.
    pub certified_mode: ControlMode,
    pub layer: Layer,
    pub intervention: Intervention,
    pub reason: Reason,
    /// Indices into the fired-constraint catalog (resolve with [`SafetyCore::fired_name`]).
    pub fired: Vec<usize>,
    pub backup_kind: Option<BackupKind>,
    /// Smallest keep-out barrier margin this tick (the safety certificate); `+∞` if none.
    pub min_barrier_margin: f64,
}

impl SafetyVerdict {
    pub fn with_capacity(dim: usize, n_constraints: usize) -> Self {
        Self {
            certified_action: Vec::with_capacity(dim),
            certified_mode: ControlMode::Effort,
            layer: Layer::Backup,
            intervention: Intervention::Fallback,
            reason: Reason::BadInput,
            fired: Vec::with_capacity(n_constraints),
            backup_kind: Some(BackupKind::BrakeToStop),
            min_barrier_margin: f64::INFINITY,
        }
    }
}

fn backup_kind(on_uncertain: OnUncertain) -> BackupKind {
    match on_uncertain {
        OnUncertain::Fallback => BackupKind::BrakeToStop,
        OnUncertain::Hold => BackupKind::Hold,
        OnUncertain::SafeState => BackupKind::SafeState,
    }
}

/// Configuration for the safety core.
#[derive(Debug, Clone)]
pub struct CoreConfig {
    /// Lookahead (in samples) within which a *predicted* monitor violation fires the backup.
    pub predictive_horizon_samples: usize,
    /// CBF/QP tuning.
    pub shield: ShieldConfig,
    /// Brake gain of the simplex backup.
    pub k_brake: f64,
    /// Reject monitors whose history window exceeds this (static-bounds cap).
    pub max_history_cap: usize,
    /// Per-tick deadline for the real-time watchdog; `None` disables it (used by the
    /// determinism gate so the decision never depends on wall-clock timing).
    pub deadline: Option<Duration>,
    /// Tolerance below which the shielded action counts as "unchanged" from the proposal.
    pub unchanged_tol: f64,
    /// The certified-directive allowlist + the rejected-directive fallback channel.
    pub action_policy: ActionPolicy,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            predictive_horizon_samples: 5,
            shield: ShieldConfig::default(),
            k_brake: 4.0,
            max_history_cap: 1 << 20,
            deadline: None,
            unchanged_tol: 1e-6,
            action_policy: ActionPolicy::default(),
        }
    }
}

/// The trusted safety core: decode once, then certify one action per tick.
#[derive(Debug, Clone)]
pub struct SafetyCore {
    model: CompiledSafetyModel,
    cfg: CoreConfig,
    dim: usize,
    monitors: Vec<Monitor>,
    shield: Option<Shield>,
    backup: Backup,
    // Hot-path scratch (pre-allocated).
    pred_rob: Vec<f64>,
    // Fired-constraint catalog: scalar bounds first, then monitors.
    fired_catalog: Vec<String>,
    scalar_offset: usize,
    // Station-keep anchor for the Hold backup: the pose latched on the first tick a Hold episode
    // engages, reused across consecutive Hold ticks and cleared the moment any other outcome is
    // emitted. Pre-allocated (capacity `dim`) so latching never allocates on the hot path.
    hold_anchor: Vec<f64>,
    hold_active: bool,
}

impl SafetyCore {
    /// Build a core from a `CompiledSafetyModel` protobuf payload.
    pub fn from_wire(bytes: &[u8], cfg: CoreConfig) -> Result<Self, CoreError> {
        let model = CompiledSafetyModel::from_wire(bytes, cfg.max_history_cap)?;
        Self::from_model(model, cfg)
    }

    /// Build a core from an already-decoded model.
    ///
    /// The gate's **effective** directive allowlist is resolved here, once: the *configured*
    /// `cfg.action_policy` is intersected with the grant the *reviewed model* authored
    /// (`model.admissible_directives`), so configuration can only ever narrow the contract and a
    /// spec that authored nothing admits nothing (RFC-0004 Amendment 2). Doing it at construction
    /// rather than per tick is what keeps the hot path allocation-free (guard.md §2 principle 6).
    pub fn from_model(model: CompiledSafetyModel, mut cfg: CoreConfig) -> Result<Self, CoreError> {
        cfg.action_policy = cfg
            .action_policy
            .narrowed_to(model.admissible_directives.as_ref());

        let dim = model.spatial_dim.unwrap_or(0);
        let monitors = model.monitors.iter().map(Monitor::new).collect::<Vec<_>>();
        let shield = if dim > 0 && !model.keep_out_terms.is_empty() {
            Some(Shield::new(
                &model.keep_out_terms,
                dim,
                cfg.shield,
                model.sample_period_s,
                &model.action_limits,
            ))
        } else {
            None
        };
        // The backup shares the keep-out geometry (for the Hold/SafeState in-set guard), the
        // authored safe pose (the retreat target), and the *same* reviewed kinematic envelope the
        // shield projects onto — so a recover-layer command is bounded by exactly the ceilings the
        // correct layer enforces. Passing the *real* spatial dim (0 for non-spatial models) lets
        // Hold/SafeState degrade to braking when there is no spatial frame.
        let backup = Backup::new(
            dim,
            cfg.shield,
            cfg.k_brake,
            model.sample_period_s,
            model.safe_pose.clone(),
            &model.keep_out_terms,
            &model.action_limits,
        );

        let mut fired_catalog = Vec::new();
        for b in &model.scalar_bounds {
            fired_catalog.push(b.constraint_id.clone());
        }
        let scalar_offset = fired_catalog.len();
        for m in &model.monitors {
            fired_catalog.push(m.constraint_id.clone());
        }

        let pred_rob = vec![0.0; model.atoms.len()];
        Ok(Self {
            model,
            cfg,
            dim,
            monitors,
            shield,
            backup,
            pred_rob,
            fired_catalog,
            scalar_offset,
            hold_anchor: Vec::with_capacity(dim),
            hold_active: false,
        })
    }

    /// Resolve a `fired` index to its constraint id.
    pub fn fired_name(&self, idx: usize) -> &str {
        &self.fired_catalog[idx]
    }

    /// Number of catalogued constraints (scalar bounds + monitors).
    pub fn n_constraints(&self) -> usize {
        self.fired_catalog.len()
    }

    /// A pre-sized verdict buffer matched to this model (reuse it across ticks; no alloc).
    pub fn new_verdict(&self) -> SafetyVerdict {
        SafetyVerdict::with_capacity(self.dim, self.fired_catalog.len())
    }

    pub fn spatial_dim(&self) -> usize {
        self.dim
    }

    pub fn spec_id(&self) -> &str {
        &self.model.spec_id
    }

    pub fn spec_content_hash(&self) -> &str {
        &self.model.spec_content_hash
    }

    /// The enforced commanded-speed ceiling (the spec's `max_velocity_mps`, else the configured
    /// `v_max`); `None` for a non-spatial model, which has no shield.
    pub fn v_max(&self) -> Option<f64> {
        self.shield.as_ref().map(Shield::v_max)
    }

    /// Deterministic per-tick decision (watchdog disabled): fill `out`, no allocation. This is
    /// the golden/determinism path — the result depends only on the inputs and prior ticks.
    pub fn step_into(&mut self, input: &SafetyInput, out: &mut SafetyVerdict) {
        self.decide(input, false, out);
    }

    /// Convenience wrapper that returns a fresh verdict (allocates — not the hot path).
    pub fn step(&mut self, input: &SafetyInput) -> SafetyVerdict {
        let mut out = SafetyVerdict::with_capacity(self.dim, self.fired_catalog.len());
        self.step_into(input, &mut out);
        out
    }

    /// Real-time step with the watchdog armed: run the decision, and if it overran the
    /// configured per-tick budget, discard it for the verified safe fallback (guard.md §8,
    /// §10 — "the watchdog whose expiry *is* a safe-mode trigger").
    pub fn step_guarded(&mut self, input: &SafetyInput, out: &mut SafetyVerdict) {
        let start = Instant::now();
        self.decide(input, false, out);
        if let Some(budget) = self.cfg.deadline {
            if start.elapsed() > budget {
                self.emit_watchdog(input, out);
            }
        }
    }

    /// Force the watchdog-expiry path (deterministic; used to test the safe-mode trigger).
    pub fn force_watchdog(&mut self, input: &SafetyInput, out: &mut SafetyVerdict) {
        self.decide(input, true, out);
    }

    /// The actuation channel a fallback is emitted in: the proposal's own channel when it has one
    /// (so the safe command reaches the same actuator the policy was addressing), else the
    /// configured `fallback_mode`.
    #[inline]
    fn fallback_mode(&self, input: &SafetyInput) -> ControlMode {
        input
            .proposed
            .control_mode()
            .unwrap_or(self.cfg.action_policy.fallback_mode)
    }

    fn decide(&mut self, input: &SafetyInput, force_watchdog: bool, out: &mut SafetyVerdict) {
        out.fired.clear();
        out.min_barrier_margin = f64::INFINITY;

        // (1) Watchdog — highest precedence.
        if force_watchdog {
            self.emit_watchdog(input, out);
            return;
        }

        // (2) Input validation. Bad input is uncertifiable → backup. A directive (MODE/TASK) and an
        // opaque command carry no setpoint, so only the state is checked for them.
        let good_signals = input.signals.len() == self.model.signals.len()
            && input.signals.iter().all(|x| x.is_finite());
        let setpoint = input.proposed.setpoint();
        let numeric = input.proposed.control_mode().is_some();
        let good_state = self.dim == 0
            || (input.position.len() >= self.dim
                && input.velocity.len() >= self.dim
                && input.position[..self.dim].iter().all(|x| x.is_finite())
                && input.velocity[..self.dim].iter().all(|x| x.is_finite()));
        let good_command =
            !numeric || (setpoint.len() >= self.dim && setpoint.iter().all(|x| x.is_finite()));
        if !good_signals || !good_state || !good_command {
            self.emit_backup(input, BackupKind::BrakeToStop, Reason::BadInput, out);
            return;
        }

        // Compute predicate robustness for every atom, then advance every monitor (their
        // bounded history must stay gap-free regardless of what the arbiter decides).
        for (i, atom) in self.model.atoms.iter().enumerate() {
            let value = input.signals[atom.signal_index];
            self.pred_rob[i] = atom.op.robustness(value, atom.threshold);
        }
        let pred_rob = &self.pred_rob;
        let horizon = self.cfg.predictive_horizon_samples;
        let mut monitor_fired: Option<(usize, OnUncertain)> = None;
        for (mi, mon) in self.monitors.iter_mut().enumerate() {
            let verdict = mon.step(&|i| pred_rob[i], horizon);
            if verdict.fired() && monitor_fired.is_none() {
                monitor_fired = Some((mi, mon.on_uncertain));
            }
        }

        // (3) Scalar bounds (detect). First violation wins precedence. Capture the hit before
        // dropping the borrow of `self.model`, so `emit_backup` can take `&mut self`.
        let mut scalar_hit: Option<(usize, OnUncertain)> = None;
        for (bi, b) in self.model.scalar_bounds.iter().enumerate() {
            if self.pred_rob[b.atom_index] < 0.0 {
                scalar_hit = Some((bi, b.on_uncertain));
                break;
            }
        }
        if let Some((bi, on_uncertain)) = scalar_hit {
            out.fired.push(bi);
            self.emit_backup(
                input,
                backup_kind(on_uncertain),
                Reason::ScalarViolated,
                out,
            );
            return;
        }

        // (4) Monitors (detect).
        if let Some((mi, on_uncertain)) = monitor_fired {
            out.fired.push(self.scalar_offset + mi);
            self.emit_backup(input, backup_kind(on_uncertain), Reason::MonitorFired, out);
            return;
        }

        // (5) The action gate. A discrete directive is certified only by *enumeration*, against the
        // EFFECTIVE allowlist (reviewed spec ∩ configuration, resolved in `from_model`); an opaque
        // command is never certifiable. Everything else is a modelled command and falls through to
        // the shield.
        if !numeric {
            if self.cfg.action_policy.admits(&input.proposed) {
                // A pre-certified directive: re-emit it untouched. It carries no numeric command,
                // so `certified_action` stays empty and the marshal layer returns the proposal.
                out.certified_action.clear();
                out.certified_mode = self.cfg.action_policy.fallback_mode;
                out.layer = Layer::Primary;
                out.intervention = Intervention::None;
                out.reason = Reason::Certified;
                out.backup_kind = None;
                self.hold_active = false; // a certified tick ends any hold episode
            } else {
                self.emit_backup(input, BackupKind::BrakeToStop, Reason::NotCertifiable, out);
            }
            return;
        }

        // (6) Shield (correct). No keep-out terms ⇒ nothing spatial to correct: certify the
        // proposal directly (the detect layers already passed).
        let mode = input.proposed.control_mode().unwrap_or_default();
        let Some(shield) = self.shield.as_mut() else {
            out.certified_action.clear();
            out.certified_action.extend_from_slice(setpoint);
            out.certified_mode = mode;
            out.layer = Layer::Primary;
            out.intervention = Intervention::None;
            out.reason = Reason::Certified;
            out.backup_kind = None;
            self.hold_active = false; // a certified tick ends any hold episode
            return;
        };

        let (certified, min_h) = shield.solve_mode(
            mode,
            input.position,
            input.velocity,
            setpoint,
            &mut out.certified_action,
        );
        out.certified_mode = mode;
        out.min_barrier_margin = min_h;
        if !certified {
            self.emit_backup(input, BackupKind::BrakeToStop, Reason::QpUncertifiable, out);
            return;
        }

        // (7) Certified. Modified iff the shield perturbed the proposal.
        let tol = self.cfg.unchanged_tol;
        let modified = out.certified_action[..self.dim]
            .iter()
            .zip(&setpoint[..self.dim])
            .any(|(certified, proposed)| (certified - proposed).abs() > tol);
        out.layer = if modified {
            Layer::Shield
        } else {
            Layer::Primary
        };
        out.intervention = if modified {
            Intervention::Modified
        } else {
            Intervention::None
        };
        out.reason = if modified {
            Reason::ShieldCorrected
        } else {
            Reason::Certified
        };
        out.backup_kind = None;
        self.hold_active = false; // a certified/shielded tick ends any hold episode
    }

    fn emit_backup(
        &mut self,
        input: &SafetyInput,
        kind: BackupKind,
        reason: Reason,
        out: &mut SafetyVerdict,
    ) {
        // Manage the Hold station-keep anchor. It latches on the *first* Hold tick of an episode
        // (the pose to hold), persists across consecutive Hold ticks, and clears the instant any
        // other outcome is emitted — so a Hold genuinely returns to where it engaged rather than
        // tracking the drifting current pose (which would collapse to a plain brake). Latching only
        // reuses the pre-allocated buffer's capacity, so it never allocates on the hot path.
        if kind == BackupKind::Hold {
            if !self.hold_active && self.dim > 0 && input.position.len() >= self.dim {
                self.hold_anchor.clear();
                self.hold_anchor
                    .extend_from_slice(&input.position[..self.dim]);
                self.hold_active = true;
            }
        } else {
            self.hold_active = false;
        }
        let hold_target: Option<&[f64]> = if kind == BackupKind::Hold && self.hold_active {
            Some(&self.hold_anchor)
        } else {
            None
        };
        let mode = self.fallback_mode(input);
        self.backup.control(
            kind,
            mode,
            input.position,
            input.velocity,
            hold_target,
            &mut out.certified_action,
        );
        out.certified_mode = mode;
        out.layer = Layer::Backup;
        out.intervention = Intervention::Fallback;
        out.reason = reason;
        out.backup_kind = Some(kind);
    }

    fn emit_watchdog(&mut self, input: &SafetyInput, out: &mut SafetyVerdict) {
        out.fired.clear();
        self.hold_active = false;
        let mode = self.fallback_mode(input);
        // Velocity may be malformed on a watchdog trip; brake uses whatever is finite, else 0.
        self.backup.control(
            BackupKind::BrakeToStop,
            mode,
            input.position,
            input.velocity,
            None,
            &mut out.certified_action,
        );
        out.certified_mode = mode;
        out.layer = Layer::Backup;
        out.intervention = Intervention::Fallback;
        out.reason = Reason::WatchdogExpired;
        out.backup_kind = Some(BackupKind::BrakeToStop);
        out.min_barrier_margin = f64::NEG_INFINITY;
    }
}
