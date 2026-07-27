//! The compiled safety model: decode the canonical `CompiledSafetyModel` protobuf wire form
//! into the small, fully-resolved, pre-allocatable structs the TCB evaluates each tick.
//!
//! This is the *spec evaluator*'s front door (guard.md §3 — `spec` is part of the trusted
//! core). Decoding is **fail-closed**: any value outside the closed vocabulary
//! (`enums.py` — `PredicateOp`, `OnUncertain`, `GeometryKind`, `TemporalOp`), any shape whose
//! coefficients are malformed, or any construct the core cannot statically bound is rejected
//! here with a [`CoreError`], before a single control tick runs. A model that does not decode
//! never gets to enforce anything — and the caller's fallback is the only safe response
//! (guard.md §2 principle 4).

use crate::proto;

/// Comparison operator of an atomic predicate `signal <op> threshold` (mirrors `PredicateOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateOp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl PredicateOp {
    fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "lt" => Ok(Self::Lt),
            "le" => Ok(Self::Le),
            "gt" => Ok(Self::Gt),
            "ge" => Ok(Self::Ge),
            other => Err(CoreError::Vocabulary("PredicateOp", other.to_string())),
        }
    }

    /// Signed robustness of `value <op> threshold`: `>= 0` exactly when the predicate holds,
    /// and its magnitude is the margin to violation the monitors track (guard.md §9.2). A
    /// hard safety predicate is a one-sided bound, so `lt`/`le` and `gt`/`ge` share a
    /// robustness sign (strictness only matters at the measure-zero boundary).
    #[inline]
    pub fn robustness(self, value: f64, threshold: f64) -> f64 {
        match self {
            Self::Ge | Self::Gt => value - threshold,
            Self::Le | Self::Lt => threshold - value,
        }
    }
}

/// The verified safe action a constraint resolves to when it cannot be positively certified
/// (mirrors `OnUncertain`). There is deliberately no `passthrough` — fail-open is not
/// representable (guard.md §2 principle 4, §9.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnUncertain {
    Fallback,
    Hold,
    SafeState,
}

impl OnUncertain {
    fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "fallback" => Ok(Self::Fallback),
            "hold" => Ok(Self::Hold),
            "safe_state" => Ok(Self::SafeState),
            other => Err(CoreError::Vocabulary("OnUncertain", other.to_string())),
        }
    }
}

/// Keep-out geometry discriminant (mirrors `GeometryKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Box,
    Sphere,
    HalfSpace,
}

impl Shape {
    fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "box" => Ok(Self::Box),
            "sphere" => Ok(Self::Sphere),
            "half_space" => Ok(Self::HalfSpace),
            other => Err(CoreError::Vocabulary("GeometryKind", other.to_string())),
        }
    }
}

/// STL/MTL AST node kind (mirrors `TemporalOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalOp {
    Predicate,
    Not,
    And,
    Or,
    Always,
    Eventually,
    Until,
}

impl TemporalOp {
    fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "predicate" => Ok(Self::Predicate),
            "not" => Ok(Self::Not),
            "and" => Ok(Self::And),
            "or" => Ok(Self::Or),
            "always" => Ok(Self::Always),
            "eventually" => Ok(Self::Eventually),
            "until" => Ok(Self::Until),
            other => Err(CoreError::Vocabulary("TemporalOp", other.to_string())),
        }
    }
}

/// One atomic predicate `signals[signal_index] <op> threshold`.
#[derive(Debug, Clone)]
pub struct PredicateAtom {
    pub op: PredicateOp,
    pub signal_index: usize,
    pub threshold: f64,
}

/// A scalar floor/ceiling/torque/kinematic bound lowered to one predicate slot.
#[derive(Debug, Clone)]
pub struct ScalarBound {
    pub constraint_id: String,
    pub on_uncertain: OnUncertain,
    pub atom_index: usize,
}

/// A keep-out region with the geometry coefficients the CBF shield evaluates each tick.
/// `box`/`sphere` populate `center` (+ `half_extents` or `radius`); `half_space` populates a
/// unit `normal` + `offset` (safe set `normal·x + offset >= margin_m`).
#[derive(Debug, Clone)]
pub struct KeepOutTerm {
    pub constraint_id: String,
    pub on_uncertain: OnUncertain,
    pub shape: Shape,
    pub margin_m: f64,
    pub center: Vec<f64>,
    pub half_extents: Vec<f64>,
    pub radius: Option<f64>,
    pub normal: Vec<f64>,
    pub offset: Option<f64>,
}

/// One node of a fully-resolved, integer-keyed, bounded STL/MTL monitor tree.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub op: TemporalOp,
    pub predicate_index: Option<usize>,
    pub interval_lo_samples: Option<usize>,
    pub interval_hi_samples: Option<usize>,
    pub args: Vec<CompiledNode>,
}

/// A temporal clause lowered to a bounded-memory online monitor.
#[derive(Debug, Clone)]
pub struct MonitorAutomaton {
    pub constraint_id: String,
    pub on_uncertain: OnUncertain,
    pub root: CompiledNode,
    pub history_window_len: usize,
    pub node_count: usize,
    pub predicate_indices: Vec<usize>,
}

/// The reviewed **kinematic envelope** on the *commanded* action, lowered from the spec's
/// `kinematic_limit` constraints (the tightest bound wins).
///
/// The scalar bounds a `kinematic_limit` also lowers to police the *measured* signal (detect); this
/// is the same reviewed limit applied to the *command* (correct) — so a `POSITION`/`VELOCITY`
/// setpoint is projected onto the envelope the safety engineer signed off, not onto a runtime
/// configuration knob. Absent (`None`) means the spec authored no such limit and the core's
/// configured ceiling stands; an authored limit may only ever **tighten** it (`shield::tighten`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ActionLimits {
    /// Largest commanded speed `‖w‖` (m/s) — the `Velocity` ball, and (via `·dt`) the `Position`
    /// step ball.
    pub max_velocity_mps: Option<f64>,
    /// Largest commanded acceleration magnitude (m/s²) — tightens the `Effort` control box.
    pub max_accel_mps2: Option<f64>,
}

/// The reviewed **MODE/TASK allowlist** on the action gate, lowered from the spec's
/// `admissible_directives` (RFC-0004 Amendment 2; guard.md §3, §5).
///
/// A discrete directive carries no continuous quantity the shield could project, so the core can
/// only certify it by **enumeration** — and an admitted directive is re-emitted *untouched*. That
/// makes the grant itself the safety decision, which is why it rides in the reviewed,
/// content-addressed model rather than in the (unsigned) `CoreConfig`.
///
/// The core intersects this with the *configured* [`ActionPolicy`](crate::ActionPolicy) **once**, at
/// construction (`shield::narrow`), so the hot path stays allocation-free (guard.md §2 principle 6):
/// `effective = spec ∩ config`. Configuration may only ever **narrow** the reviewed grant.
///
/// **`None` on the model ⇒ the contract admits NO directive**, whatever the configuration grants.
/// Note the deliberate asymmetry with [`ActionLimits`], whose absent members leave the *configured*
/// ceiling standing: both merges are the greatest-lower-bound, but the identity of a ceiling's meet
/// is `+∞` while the identity of a permission set's meet is `∅`. Silence must never grant authority.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdmissibleDirectives {
    /// Pre-certified `ModeCommand.mode` names (an open vocabulary: SADF `loads_by_mode`).
    pub modes: Vec<String>,
    /// Pre-certified `TaskDirective.task_kind` values (Core `TaskKind`, as strings).
    pub tasks: Vec<String>,
}

/// The static worst-case counts a pre-allocating core needs (guard.md §2 principle 6).
#[derive(Debug, Clone, Default)]
pub struct ResourceBounds {
    pub predicate_slot_count: usize,
    pub scalar_bound_count: usize,
    pub keep_out_term_count: usize,
    pub monitor_count: usize,
    pub max_history_len: usize,
    pub worst_case_term_count: usize,
}

/// The decoded, validated compiled safety model — the input contract of the TCB.
#[derive(Debug, Clone)]
pub struct CompiledSafetyModel {
    pub compiled_version: String,
    pub spec_id: String,
    pub spec_content_hash: String,
    pub sample_period_s: f64,
    pub signals: Vec<String>,
    pub atoms: Vec<PredicateAtom>,
    pub scalar_bounds: Vec<ScalarBound>,
    pub keep_out_terms: Vec<KeepOutTerm>,
    pub monitors: Vec<MonitorAutomaton>,
    pub resource_bounds: ResourceBounds,
    /// The reviewed kinematic envelope on the *commanded* action (RM-P1-GUARD-03), lowered from the
    /// spec's `kinematic_limit` constraints. Default (both `None`) = the spec authored none.
    pub action_limits: ActionLimits,
    /// Spatial dimension of the keep-out geometry (the control/state dimension the shield
    /// operates in); `None` when the model declares no keep-out terms.
    pub spatial_dim: Option<usize>,
    /// The authored charging/safe pose (a bare position in the keep-out frame) the verified
    /// retreat (`safe_state`) backup steers toward (RM-P1-GUARD-04). `None` when the spec authored
    /// no safe pose — a `safe_state` fallback then degrades to brake-to-stop (fail-safe, never
    /// fail-open). Decoded fail-closed: a present-but-malformed pose (non-finite, or shorter than
    /// `spatial_dim`) rejects the whole model rather than enforcing against a bad target.
    pub safe_pose: Option<Vec<f64>>,
    /// The reviewed `MODE`/`TASK` allowlist the action gate certifies against (RFC-0004
    /// Amendment 2). `None` when the spec authored none — the contract then admits **no** directive,
    /// whatever the configured [`ActionPolicy`](crate::ActionPolicy) grants. Decoded fail-closed: a
    /// present-but-malformed grant (an empty directive name) rejects the whole model.
    pub admissible_directives: Option<AdmissibleDirectives>,
}

/// Every way loading or evaluating a safety model can go wrong. All are fail-closed: the
/// caller's only safe response to any of these is the verified fallback (guard.md §9.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// The protobuf payload did not decode.
    Decode(String),
    /// A closed-vocabulary field carried a value outside its enum (`field`, `value`).
    Vocabulary(&'static str, String),
    /// The model is structurally invalid (mismatched geometry arity, non-positive period, a
    /// monitor node missing a required field, etc.).
    Structure(String),
    /// A statically-bounded resource exceeds the configured cap — the core refuses rather
    /// than allocate unboundedly on the safety path.
    ResourceExceeded(String),
}

impl core::fmt::Display for CoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode(m) => write!(f, "decode error: {m}"),
            Self::Vocabulary(field, v) => write!(f, "value {v:?} is not a valid {field}"),
            Self::Structure(m) => write!(f, "structural error: {m}"),
            Self::ResourceExceeded(m) => write!(f, "resource bound exceeded: {m}"),
        }
    }
}

impl std::error::Error for CoreError {}

fn to_usize(v: i64, what: &str) -> Result<usize, CoreError> {
    usize::try_from(v).map_err(|_| CoreError::Structure(format!("{what} must be >= 0, got {v}")))
}

fn opt_usize(v: Option<i64>, what: &str) -> Result<Option<usize>, CoreError> {
    v.map(|x| to_usize(x, what)).transpose()
}

impl CompiledNode {
    fn from_proto(p: &proto::CompiledNode) -> Result<Self, CoreError> {
        let op = TemporalOp::parse(&p.op)?;
        let node = Self {
            op,
            predicate_index: opt_usize(p.predicate_index, "predicate_index")?,
            interval_lo_samples: opt_usize(p.interval_lo_samples, "interval_lo_samples")?,
            interval_hi_samples: opt_usize(p.interval_hi_samples, "interval_hi_samples")?,
            args: p
                .args
                .iter()
                .map(CompiledNode::from_proto)
                .collect::<Result<_, _>>()?,
        };
        node.validate()?;
        Ok(node)
    }

    /// Reject any node the evaluator cannot walk deterministically in bounded memory.
    fn validate(&self) -> Result<(), CoreError> {
        match self.op {
            TemporalOp::Predicate => {
                if self.predicate_index.is_none() {
                    return Err(CoreError::Structure("predicate node needs an index".into()));
                }
                if !self.args.is_empty() {
                    return Err(CoreError::Structure("predicate node takes no args".into()));
                }
            }
            TemporalOp::Not => {
                if self.args.len() != 1 {
                    return Err(CoreError::Structure("not takes exactly one arg".into()));
                }
            }
            TemporalOp::And | TemporalOp::Or => {
                if self.args.is_empty() {
                    return Err(CoreError::Structure("and/or need >= 1 arg".into()));
                }
            }
            TemporalOp::Always | TemporalOp::Eventually => {
                if self.args.len() != 1 {
                    return Err(CoreError::Structure(
                        "always/eventually take one arg".into(),
                    ));
                }
                self.require_interval()?;
            }
            TemporalOp::Until => {
                if self.args.len() != 2 {
                    return Err(CoreError::Structure("until takes two args".into()));
                }
                self.require_interval()?;
            }
        }
        Ok(())
    }

    fn require_interval(&self) -> Result<(), CoreError> {
        match (self.interval_lo_samples, self.interval_hi_samples) {
            (Some(lo), Some(hi)) if lo <= hi => Ok(()),
            _ => Err(CoreError::Structure(
                "temporal operator needs a finite [lo, hi] interval with lo <= hi".into(),
            )),
        }
    }

    /// Largest temporal horizon anywhere in the subtree, in samples — the ring-buffer length
    /// the monitor must retain to be evaluatable online with bounded memory.
    pub fn horizon(&self) -> usize {
        let here = self.interval_hi_samples.unwrap_or(0);
        let below = self
            .args
            .iter()
            .map(CompiledNode::horizon)
            .max()
            .unwrap_or(0);
        here.max(below)
    }
}

/// Resolve a typed `frame_ref` (RFC-0007) against the `require_frame` guard, fail-closed.
///
/// The typed frame is authoring metadata: the core reads only the geometry coefficients + the
/// safe-pose position for control, so this validates the frame but stores nothing (typing the
/// frame does not widen what the TCB reads). A present-but-malformed frame (whitespace/empty
/// name, non-`FrameClass` class) rejects the whole model rather than enforcing against a bad
/// frame — the same fail-closed discipline as every other decode error here. Absent is fine
/// (`frame_ref` is an additive optional sibling of the `string frame`).
fn validate_frame_ref(
    frame_ref: &Option<proto::core::units::v0::ReferenceFrame>,
    what: &str,
) -> Result<(), CoreError> {
    if let Some(rf) = frame_ref {
        crate::units::require_frame(&rf.name, &rf.frame_class, rf.center.as_deref())
            .map_err(|e| CoreError::Structure(format!("{what} frame_ref invalid: {e}")))?;
    }
    Ok(())
}

fn geometry_dim(t: &KeepOutTerm) -> Result<usize, CoreError> {
    let d = match t.shape {
        Shape::Sphere => {
            if t.radius.is_none() {
                return Err(CoreError::Structure(format!(
                    "sphere keep-out {} needs a radius",
                    t.constraint_id
                )));
            }
            t.center.len()
        }
        Shape::Box => {
            if t.center.len() != t.half_extents.len() {
                return Err(CoreError::Structure(format!(
                    "box keep-out {} center/half_extents arity mismatch",
                    t.constraint_id
                )));
            }
            t.center.len()
        }
        Shape::HalfSpace => {
            if t.offset.is_none() {
                return Err(CoreError::Structure(format!(
                    "half_space keep-out {} needs an offset",
                    t.constraint_id
                )));
            }
            t.normal.len()
        }
    };
    if d == 0 {
        return Err(CoreError::Structure(format!(
            "keep-out {} has zero spatial dimension",
            t.constraint_id
        )));
    }
    Ok(d)
}

impl CompiledSafetyModel {
    /// Decode and validate a `CompiledSafetyModel` protobuf payload, rejecting `max_history`
    /// windows larger than `max_history_cap` (fail-closed static-bounds enforcement).
    pub fn from_wire(bytes: &[u8], max_history_cap: usize) -> Result<Self, CoreError> {
        use prost::Message;
        let p = proto::CompiledSafetyModel::decode(bytes)
            .map_err(|e| CoreError::Decode(e.to_string()))?;
        Self::from_proto(p, max_history_cap)
    }

    fn from_proto(
        p: proto::CompiledSafetyModel,
        max_history_cap: usize,
    ) -> Result<Self, CoreError> {
        if !(p.sample_period_s.is_finite() && p.sample_period_s > 0.0) {
            return Err(CoreError::Structure(format!(
                "sample_period_s must be finite and positive, got {}",
                p.sample_period_s
            )));
        }

        let table = p
            .predicate_table
            .ok_or_else(|| CoreError::Structure("missing predicate_table".into()))?;
        let signal_count = table.signals.len();

        let atoms = table
            .atoms
            .iter()
            .map(|a| {
                let signal_index = to_usize(a.signal_index, "signal_index")?;
                if signal_index >= signal_count {
                    return Err(CoreError::Structure(format!(
                        "atom signal_index {signal_index} out of range ({signal_count} signals)"
                    )));
                }
                if !a.threshold.is_finite() {
                    return Err(CoreError::Structure("atom threshold must be finite".into()));
                }
                Ok(PredicateAtom {
                    op: PredicateOp::parse(&a.op)?,
                    signal_index,
                    threshold: a.threshold,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let check_atom = |i: usize| -> Result<usize, CoreError> {
            if i >= atoms.len() {
                Err(CoreError::Structure(format!(
                    "atom_index {i} out of range ({} atoms)",
                    atoms.len()
                )))
            } else {
                Ok(i)
            }
        };

        let scalar_bounds = p
            .scalar_bounds
            .iter()
            .map(|b| {
                Ok(ScalarBound {
                    constraint_id: b.constraint_id.clone(),
                    on_uncertain: OnUncertain::parse(&b.on_uncertain)?,
                    atom_index: check_atom(to_usize(b.atom_index, "atom_index")?)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let keep_out_terms = p
            .keep_out_terms
            .iter()
            .map(|t| {
                validate_frame_ref(&t.frame_ref, &format!("keep-out {}", t.constraint_id))?;
                let term = KeepOutTerm {
                    constraint_id: t.constraint_id.clone(),
                    on_uncertain: OnUncertain::parse(&t.on_uncertain)?,
                    shape: Shape::parse(&t.shape)?,
                    margin_m: t.margin_m,
                    center: t.center.clone(),
                    half_extents: t.half_extents.clone(),
                    radius: t.radius,
                    normal: t.normal.clone(),
                    offset: t.offset,
                };
                geometry_dim(&term)?; // validate arity
                Ok(term)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // All keep-out terms must live in one spatial frame/dimension (the state the shield
        // controls). A mismatch is a compile-time error surfaced fail-closed.
        let mut spatial_dim: Option<usize> = None;
        for t in &keep_out_terms {
            let d = geometry_dim(t)?;
            match spatial_dim {
                None => spatial_dim = Some(d),
                Some(prev) if prev != d => {
                    return Err(CoreError::Structure(format!(
                        "keep-out terms disagree on spatial dimension ({prev} vs {d})"
                    )));
                }
                _ => {}
            }
        }

        let monitors = p
            .monitors
            .iter()
            .map(|m| {
                let root = m
                    .root
                    .as_ref()
                    .ok_or_else(|| CoreError::Structure("monitor missing root".into()))?;
                let root = CompiledNode::from_proto(root)?;
                let history_window_len = to_usize(m.history_window_len, "history_window_len")?;
                if history_window_len > max_history_cap {
                    return Err(CoreError::ResourceExceeded(format!(
                        "monitor {} needs history {} > cap {}",
                        m.constraint_id, history_window_len, max_history_cap
                    )));
                }
                Ok(MonitorAutomaton {
                    constraint_id: m.constraint_id.clone(),
                    on_uncertain: OnUncertain::parse(&m.on_uncertain)?,
                    root,
                    history_window_len,
                    node_count: to_usize(m.node_count, "node_count")?,
                    predicate_indices: m
                        .predicate_indices
                        .iter()
                        .map(|&i| check_atom(to_usize(i, "predicate_index")?))
                        .collect::<Result<Vec<_>, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Safe pose (retreat target). Fail-closed: a present pose whose coordinates are non-finite,
        // or that is shorter than the keep-out spatial dimension, is rejected here — the retreat
        // backup must never steer toward a malformed target. Absence is fine (the safe_state backup
        // degrades to brake-to-stop at runtime).
        let safe_pose = match p.safe_pose {
            None => None,
            Some(sp) => {
                validate_frame_ref(&sp.frame_ref, "safe_pose")?;
                if sp.position.iter().any(|x| !x.is_finite()) {
                    return Err(CoreError::Structure(
                        "safe_pose position must be finite".into(),
                    ));
                }
                if let Some(d) = spatial_dim {
                    if sp.position.len() < d {
                        return Err(CoreError::Structure(format!(
                            "safe_pose has {} coordinates but the keep-out frame is {d}-dimensional",
                            sp.position.len()
                        )));
                    }
                }
                Some(sp.position)
            }
        };

        // The reviewed kinematic envelope on the commanded action. Fail-closed: a present-but-
        // malformed limit (non-finite, non-positive) rejects the whole model rather than silently
        // widening the envelope the shield projects onto.
        let action_limits = match p.action_limits {
            None => ActionLimits::default(),
            Some(al) => ActionLimits {
                max_velocity_mps: check_limit(al.max_velocity_mps, "max_velocity_mps")?,
                max_accel_mps2: check_limit(al.max_accel_mps2, "max_accel_mps2")?,
            },
        };

        // The reviewed MODE/TASK allowlist on the action gate (RFC-0004 Amendment 2). Absent stays
        // absent — and absent admits NOTHING, whatever the configuration grants (unlike an absent
        // `action_limits` ceiling, which leaves the configured one standing; see
        // `AdmissibleDirectives`). Fail-closed: an empty directive name is not a grant anyone could
        // have meant to review, and a `ModeCommand.mode` is never empty, so it rejects the whole
        // model rather than sitting in the allowlist as an unmatchable — or worse, matchable —
        // entry.
        let admissible_directives = match p.admissible_directives {
            None => None,
            Some(ad) => {
                for name in ad.modes.iter().chain(ad.tasks.iter()) {
                    if name.is_empty() {
                        return Err(CoreError::Structure(
                            "admissible_directives carries an empty directive name".into(),
                        ));
                    }
                }
                Some(AdmissibleDirectives {
                    modes: ad.modes,
                    tasks: ad.tasks,
                })
            }
        };

        let rb = p.resource_bounds.unwrap_or_default();
        let resource_bounds = ResourceBounds {
            predicate_slot_count: to_usize(rb.predicate_slot_count, "predicate_slot_count")?,
            scalar_bound_count: to_usize(rb.scalar_bound_count, "scalar_bound_count")?,
            keep_out_term_count: to_usize(rb.keep_out_term_count, "keep_out_term_count")?,
            monitor_count: to_usize(rb.monitor_count, "monitor_count")?,
            max_history_len: to_usize(rb.max_history_len, "max_history_len")?,
            worst_case_term_count: to_usize(rb.worst_case_term_count, "worst_case_term_count")?,
        };

        Ok(Self {
            compiled_version: p.compiled_version,
            spec_id: p.spec_id,
            spec_content_hash: p.spec_content_hash,
            sample_period_s: p.sample_period_s,
            signals: table.signals,
            atoms,
            scalar_bounds,
            keep_out_terms,
            monitors,
            resource_bounds,
            action_limits,
            spatial_dim,
            safe_pose,
            admissible_directives,
        })
    }
}

/// Validate an authored kinematic limit: absent is fine, present must be finite and positive
/// (a zero/negative/NaN ceiling is not a safety envelope — reject the model, never widen it).
fn check_limit(v: Option<f64>, what: &str) -> Result<Option<f64>, CoreError> {
    match v {
        None => Ok(None),
        Some(x) if x.is_finite() && x > 0.0 => Ok(Some(x)),
        Some(x) => Err(CoreError::Structure(format!(
            "{what} must be finite and positive, got {x}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto;
    use prost::Message;

    /// A valid typed Core ReferenceFrame (RFC-0007) for the `frame_ref` wire fields.
    fn moon_frame() -> proto::core::units::v0::ReferenceFrame {
        proto::core::units::v0::ReferenceFrame {
            name: "MOON_ME".into(),
            frame_class: "body_fixed".into(),
            center: Some("MOON".into()),
        }
    }

    fn good_proto() -> proto::CompiledSafetyModel {
        proto::CompiledSafetyModel {
            compiled_version: "0.1".into(),
            spec_id: "s".into(),
            spec_content_hash: "sha256:x".into(),
            sample_period_s: 0.5,
            predicate_table: Some(proto::PredicateTable {
                signals: vec!["soc".into()],
                atoms: vec![proto::PredicateAtom {
                    op: "ge".into(),
                    signal_index: 0,
                    threshold: 1.0,
                }],
            }),
            scalar_bounds: vec![proto::ScalarBound {
                constraint_id: "c".into(),
                on_uncertain: "fallback".into(),
                atom_index: 0,
            }],
            keep_out_terms: vec![proto::KeepOutTerm {
                constraint_id: "k".into(),
                on_uncertain: "hold".into(),
                shape: "sphere".into(),
                frame: "MOON_ME".into(),
                margin_m: 1.0,
                center: vec![0.0, 0.0, 0.0],
                half_extents: vec![],
                radius: Some(3.0),
                normal: vec![],
                offset: None,
                collision_pair: vec![],
                // The typed frame sibling (RFC-0007) — validated fail-closed by require_frame.
                frame_ref: Some(moon_frame()),
            }],
            monitors: vec![proto::MonitorAutomaton {
                constraint_id: "m".into(),
                on_uncertain: "safe_state".into(),
                root: Some(proto::CompiledNode {
                    op: "always".into(),
                    predicate_index: None,
                    interval_lo_samples: Some(0),
                    interval_hi_samples: Some(3),
                    args: vec![proto::CompiledNode {
                        op: "predicate".into(),
                        predicate_index: Some(0),
                        interval_lo_samples: None,
                        interval_hi_samples: None,
                        args: vec![],
                    }],
                }),
                history_window_len: 3,
                node_count: 2,
                predicate_indices: vec![0],
            }],
            resource_bounds: Some(proto::ResourceBounds::default()),
            safe_pose: None,
            action_limits: None,
            admissible_directives: None,
        }
    }

    #[test]
    fn wire_round_trip_decodes() {
        let bytes = good_proto().encode_to_vec();
        let m = CompiledSafetyModel::from_wire(&bytes, 1 << 20).unwrap();
        assert_eq!(m.signals, vec!["soc".to_string()]);
        assert_eq!(m.atoms.len(), 1);
        assert_eq!(m.spatial_dim, Some(3));
        assert_eq!(m.monitors[0].history_window_len, 3);
    }

    #[test]
    fn bad_predicate_op_is_rejected() {
        let mut p = good_proto();
        p.predicate_table.as_mut().unwrap().atoms[0].op = "eq".into();
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Vocabulary("PredicateOp", _)));
    }

    #[test]
    fn fail_open_is_not_representable() {
        // "passthrough" is deliberately absent from OnUncertain — a compiled model that tries
        // to smuggle it in is rejected, so the core can never be told to fail open.
        let mut p = good_proto();
        p.scalar_bounds[0].on_uncertain = "passthrough".into();
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Vocabulary("OnUncertain", _)));
    }

    #[test]
    fn history_window_over_cap_is_rejected() {
        let bytes = good_proto().encode_to_vec();
        let err = CompiledSafetyModel::from_wire(&bytes, 1).unwrap_err();
        assert!(matches!(err, CoreError::ResourceExceeded(_)));
    }

    /// A spec that authored no grant decodes to `None` — and `None` admits **nothing** (the gate
    /// intersects against it in `SafetyCore::from_model`). Contrast `action_limits`, whose absence
    /// leaves the *configured* ceiling standing: silence is `∅` for a permission set, `+∞` for a
    /// ceiling (RFC-0004 Amendment 2).
    #[test]
    fn absent_admissible_directives_decodes_to_none() {
        let m = CompiledSafetyModel::from_wire(&good_proto().encode_to_vec(), 1 << 20).unwrap();
        assert_eq!(m.admissible_directives, None);
    }

    #[test]
    fn an_authored_grant_decodes() {
        let mut p = good_proto();
        p.admissible_directives = Some(proto::CompiledAdmissibleDirectives {
            modes: vec!["safe_hold".into()],
            tasks: vec!["standby".into(), "charge".into()],
        });
        let m = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap();
        let grant = m.admissible_directives.unwrap();
        assert_eq!(grant.modes, vec!["safe_hold".to_string()]);
        assert_eq!(
            grant.tasks,
            vec!["standby".to_string(), "charge".to_string()]
        );
    }

    /// Fail-closed: an empty directive name is not a grant anyone meant to review, and a
    /// `ModeCommand.mode` is never empty — so it rejects the whole model rather than sitting in the
    /// allowlist as an entry that might one day match.
    #[test]
    fn an_empty_directive_name_is_rejected() {
        for (modes, tasks) in [(vec![String::new()], vec![]), (vec![], vec![String::new()])] {
            let mut p = good_proto();
            p.admissible_directives = Some(proto::CompiledAdmissibleDirectives { modes, tasks });
            let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
            assert!(matches!(err, CoreError::Structure(_)));
        }
    }

    #[test]
    fn absent_safe_pose_decodes_to_none() {
        let m = CompiledSafetyModel::from_wire(&good_proto().encode_to_vec(), 1 << 20).unwrap();
        assert_eq!(m.safe_pose, None);
    }

    #[test]
    fn valid_safe_pose_decodes() {
        let mut p = good_proto();
        p.safe_pose = Some(proto::CompiledSafePose {
            frame: "MOON_ME".into(),
            position: vec![1.0, 2.0, 3.0],
            frame_ref: Some(moon_frame()),
        });
        let m = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap();
        assert_eq!(m.safe_pose, Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn non_finite_safe_pose_is_rejected() {
        let mut p = good_proto();
        p.safe_pose = Some(proto::CompiledSafePose {
            frame: "MOON_ME".into(),
            position: vec![1.0, f64::NAN, 3.0],
            frame_ref: Some(moon_frame()),
        });
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Structure(_)));
    }

    #[test]
    fn safe_pose_shorter_than_spatial_dim_is_rejected() {
        // good_proto has a 3-D sphere keep-out ⇒ spatial_dim = 3; a 2-vector target is malformed.
        let mut p = good_proto();
        p.safe_pose = Some(proto::CompiledSafePose {
            frame: "MOON_ME".into(),
            position: vec![1.0, 2.0],
            frame_ref: Some(moon_frame()),
        });
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Structure(_)));
    }

    #[test]
    fn unbounded_temporal_operator_is_rejected() {
        let mut p = good_proto();
        // Strip the interval from the always node → not statically boundable.
        let root = p.monitors[0].root.as_mut().unwrap();
        root.interval_hi_samples = None;
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Structure(_)));
    }

    #[test]
    fn absent_frame_ref_still_decodes() {
        // frame_ref is an additive optional sibling of the `string frame`; a model that authored
        // only the string frame (no typed sibling) decodes exactly as before.
        let mut p = good_proto();
        p.keep_out_terms[0].frame_ref = None;
        let m = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap();
        assert_eq!(m.spatial_dim, Some(3));
    }

    #[test]
    fn whitespace_frame_ref_on_keep_out_is_rejected() {
        // A keep-out whose typed frame has a whitespace-bearing name is rejected fail-closed at
        // decode — the same require_frame guard the conformance vectors pin (RFC-0007).
        let mut p = good_proto();
        p.keep_out_terms[0].frame_ref = Some(proto::core::units::v0::ReferenceFrame {
            name: "MOON ME".into(),
            frame_class: "body_fixed".into(),
            center: None,
        });
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Structure(_)));
    }

    #[test]
    fn bad_frame_class_frame_ref_on_safe_pose_is_rejected() {
        let mut p = good_proto();
        p.safe_pose = Some(proto::CompiledSafePose {
            frame: "MOON_ME".into(),
            position: vec![1.0, 2.0, 3.0],
            frame_ref: Some(proto::core::units::v0::ReferenceFrame {
                name: "MOON_ME".into(),
                frame_class: "galactic".into(), // not a FrameClass member
                center: Some("MOON".into()),
            }),
        });
        let err = CompiledSafetyModel::from_wire(&p.encode_to_vec(), 1 << 20).unwrap_err();
        assert!(matches!(err, CoreError::Structure(_)));
    }
}
