//! Simplex backup — the *recover* layer (guard.md §9.2).
//!
//! When the arbiter cannot certify the primary/shielded action, it hands control to a **verified
//! backup controller** drawn from a small safe-state library. Each law is emitted **in the same
//! actuation channel the rejected proposal addressed** ([`ControlMode`]) — a `VELOCITY` proposal is
//! answered with a certified velocity command, not with an `EFFORT` brake a velocity-tracking
//! actuator would silently ignore. A "safe" command in a channel nobody reads is a fail-open in
//! disguise; the recover layer must speak the plant's language.
//!
//! For the double-integrator plant
//! (`ṗ = v`, `v̇ = u`, `|uᵢ| ≤ u_max`) the library has three *distinct, analyzable* laws, selected
//! by the fired constraint's `OnUncertain` (guard.md §9.2 "verified safe states"):
//!
//! - **BrakeToStop** — the canonical simplex fallback: maximal deceleration opposing the current
//!   velocity, `u = −clamp(k_brake·v, u_max)`.
//! - **Hold (hold-attitude / station-keep)** — return to and hold the pose latched when the hold
//!   engaged, damping velocity: a saturated PD law toward that anchor.
//! - **SafeState (retreat-to-charging-pose)** — steer toward the authored `safe_pose` (the lunar
//!   night-survival charging pose) with a saturated PD law.
//!
//! **Fail-safe composition (the load-bearing invariant).** Both Hold and SafeState are *guarded*:
//! the candidate PD action is emitted **only if** it is provably (a) within the control box,
//! (b) non-increasing in the target Lyapunov energy `V = ½·k_p·‖p − g‖² + ½·‖v‖²` (progress toward
//! the target for the *second-order* plant — a plain distance decrease is unachievable in one step
//! with adverse velocity, but the damped PD law provably decreases `V`), and (c) leaving the
//! one-step-predicted position inside the *same* certified safe set the shield enforces (the shared
//! [`Barrier`] geometry, with `margin_m` baked in). If any of those fails — no safe progressing
//! bounded action exists, the target is unreachable safely, the step would exit the safe set,
//! `safe_pose` is absent, the model is non-spatial, or any input is non-finite — the law
//! **degrades to BrakeToStop**. There is deliberately no path that emits the untrusted proposal:
//! fail-safe, never fail-open (guard.md §2 principle 4, §9.1).
//!
//! **Invariant arguments.**
//! - *BrakeToStop*: `u` is anti-parallel to `v`, so `d/dt ‖v‖² = 2 vᵀu ≤ 0` — speed is
//!   monotonically non-increasing and the system comes to rest; it never adds velocity toward a
//!   keep-out, so invoked with stopping margin (the predictive monitors + CBF give that lead time)
//!   the barrier stays `h ≥ 0` through to rest.
//! - *Hold / SafeState*: `V = ½·k_p·‖p − g‖² + ½·‖v‖²` is a Lyapunov function of the closed loop —
//!   under the unsaturated damped PD law `u = k_p(g − p) − k_d·v` one has `V̇ = −k_d·‖v‖² ≤ 0`, so
//!   `V` (hence, asymptotically, distance-to-target) strictly decreases until the plant rests at
//!   `g`. The guard admits an action only when the one-step-predicted `V` is non-increasing, so the
//!   state stays inside the sublevel set `{V ≤ V₀}` — a bounded neighbourhood of `g`. With the
//!   per-step in-set check on the predicted position and BrakeToStop as the ever-present floor, the
//!   recover layer can never *increase* exposure to a keep-out. The one-step prediction is the exact
//!   constant-acceleration (zero-order-hold) step `p' = p + v·dt + ½u·dt²`, `v' = v + u·dt`; between
//!   samples the margin `margin_m` absorbs discretization exactly as it does for the shield (a
//!   documented, conservative first slice — an exact reach-avoid retreat filter is deferred to
//!   GUARD-05 / a follow-up RFC).
//!
//! **The kinematic channels.** In `Velocity` the single-integrator plant makes the safe floor
//! *exact*: **zero commanded velocity** is a stop, and Hold/SafeState emit a speed-capped pull
//! toward the target admitted only when the one-step-predicted pose `p + w·dt` stays inside the
//! same certified safe set. In `Position` the safe floor is **command the current pose** (do not
//! move), and Hold/SafeState emit a step-capped target admitted only when the target itself is in
//! the safe set. Both degrade to their floor whenever the guard fails — never to the proposal.

use crate::model::{ActionLimits, KeepOutTerm};
use crate::shield::{ball_scale, Barrier, ControlMode, ShieldConfig};

/// The target Lyapunov energy must be *non-increasing* to admit a Hold/SafeState step; this
/// relative slack absorbs floating-point round-off without admitting a genuine increase.
const DIST_TOL: f64 = 1e-9;
/// A predicted position counts as in-set while its barrier value stays at or above `−SET_TOL`
/// (matches the shield's certification tolerance scale).
const SET_TOL: f64 = 1e-7;

/// Which verified backup behaviour the arbiter selected (mirrors `OnUncertain`). The verdict
/// reports the *selected* behaviour for traceability even when it degrades to braking internally
/// (e.g. a `safe_state` fallback with no authored `safe_pose`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupKind {
    /// Full simplex fallback: brake to a stop.
    BrakeToStop,
    /// Station-keep: return to and hold the pose latched when the hold engaged.
    Hold,
    /// Retreat toward the authored charging pose (`safe_pose`).
    SafeState,
}

/// The verified backup controller — the small safe-state library.
#[derive(Debug, Clone)]
pub struct Backup {
    /// Spatial dimension of the plant/keep-out frame; `0` for a non-spatial model (no keep-out
    /// geometry), where Hold/SafeState have no meaning and always degrade to braking.
    spatial_dim: usize,
    /// Length of an emitted action vector (`spatial_dim.max(1)`, matching the pre-`§9.2` behaviour
    /// for non-spatial models).
    ctrl_dim: usize,
    u_max: f64,
    /// The reviewed commanded-speed ceiling — the same one the shield projects onto, so a recover
    /// command can never exceed the envelope the correct layer enforces.
    v_max: f64,
    /// The largest commanded position step (`v_max · dt`).
    step_max: f64,
    k_brake: f64,
    /// Control/prediction step (SI seconds) — the authored `sample_period_s`. Only the guarded
    /// Hold/SafeState one-step prediction depends on it; BrakeToStop is dt-free.
    dt: f64,
    /// Proportional (pull-to-target) and derivative (velocity-damping) gains of the guarded PD law.
    /// Critically damped w.r.t. `k_brake`; safety does not depend on the gains (the runtime guard
    /// enforces it) — they only shape how briskly a *certified* retreat/hold approaches its target.
    k_p: f64,
    k_d: f64,
    /// The authored retreat target (`safe_state`), or `None` (then SafeState degrades to braking).
    safe_pose: Option<Vec<f64>>,
    /// The keep-out geometry, shared with the shield's safe set, for the per-step in-set check.
    barriers: Vec<Barrier>,
}

fn all_finite(xs: &[f64]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

/// Sanitise a state component: a non-finite coordinate (the very fault that may have triggered the
/// fallback) reads as zero, so a recover command is always finite.
#[inline]
fn finite_at(xs: &[f64], i: usize) -> f64 {
    xs.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0)
}

fn norm(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Bounded deceleration opposing one velocity component: `u = clamp(−k_brake·v, ±u_max)`.
///
/// The scalar kernel of BrakeToStop — the ever-present safe floor. Its two invariants (`|u| ≤ u_max`
/// and `v·u ≤ 0`, i.e. braking never *adds* speed) are the load-bearing property of the recover
/// layer and are **machine-checked with Kani** in `verify.rs`.
#[inline]
pub(crate) fn brake_axis(v: f64, k_brake: f64, u_max: f64) -> f64 {
    (-k_brake * v).clamp(-u_max, u_max)
}

impl Backup {
    pub fn new(
        spatial_dim: usize,
        shield_cfg: ShieldConfig,
        k_brake: f64,
        dt: f64,
        safe_pose: Option<Vec<f64>>,
        terms: &[KeepOutTerm],
        limits: &ActionLimits,
    ) -> Self {
        let barriers = terms
            .iter()
            .map(|t| Barrier::from_term(t, spatial_dim))
            .collect::<Vec<_>>();
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 1.0 };
        let tighten = |configured: f64, authored: Option<f64>| match authored {
            Some(limit) if limit.is_finite() && limit > 0.0 => configured.min(limit),
            _ => configured,
        };
        let v_max = tighten(shield_cfg.v_max, limits.max_velocity_mps);
        Self {
            spatial_dim,
            ctrl_dim: spatial_dim.max(1),
            u_max: tighten(shield_cfg.u_max, limits.max_accel_mps2),
            v_max,
            step_max: v_max * dt,
            k_brake,
            dt,
            k_p: k_brake * k_brake / 4.0,
            k_d: k_brake,
            safe_pose,
            barriers,
        }
    }

    /// Compute the certified backup command for `kind`, **in control mode `mode`**, and write it
    /// into `action`. `p`/`v` are the current position/velocity; `hold_target` is the latched
    /// station-keep anchor (only meaningful for `Hold`). Allocation-free once the barriers are
    /// built; the result is always finite and inside the mode's kinematic set.
    pub fn control(
        &self,
        kind: BackupKind,
        mode: ControlMode,
        p: &[f64],
        v: &[f64],
        hold_target: Option<&[f64]>,
        action: &mut Vec<f64>,
    ) {
        let target = match kind {
            BackupKind::BrakeToStop => None,
            BackupKind::Hold => hold_target,
            BackupKind::SafeState => self.safe_pose.as_deref(),
        };
        match (mode, target) {
            (ControlMode::Effort, None) => self.brake(v, action),
            (ControlMode::Effort, Some(g)) => self.guarded_move(g, p, v, action),
            (ControlMode::Velocity, None) => self.stop_velocity(action),
            (ControlMode::Velocity, Some(g)) => self.guarded_velocity(g, p, action),
            (ControlMode::Position, None) => self.hold_position(p, action),
            (ControlMode::Position, Some(g)) => self.guarded_position(g, p, action),
        }
    }

    /// Brake-to-stop: bounded deceleration opposing `v`. A malformed velocity component (the very
    /// fault that may have triggered the fallback) sanitises to zero, so the command is always
    /// finite. This is the ever-present safe floor every other law degrades to.
    fn brake(&self, v: &[f64], action: &mut Vec<f64>) {
        action.clear();
        for i in 0..self.ctrl_dim {
            action.push(brake_axis(finite_at(v, i), self.k_brake, self.u_max));
        }
    }

    /// The `Velocity` safe floor: **command zero velocity**. For the single-integrator plant this
    /// *is* the stop (exactly, in one step) — the kinematic analogue of brake-to-stop.
    fn stop_velocity(&self, action: &mut Vec<f64>) {
        action.clear();
        action.resize(self.ctrl_dim, 0.0);
    }

    /// The `Position` safe floor: **command the current pose** (do not move). The plant is already
    /// there, so the step is the zero step; a non-finite coordinate sanitises to zero.
    fn hold_position(&self, p: &[f64], action: &mut Vec<f64>) {
        action.clear();
        for i in 0..self.ctrl_dim {
            action.push(finite_at(p, i));
        }
    }

    /// Guarded move-toward-target in `Velocity`: a speed-capped pull toward `g`, emitted **iff** the
    /// one-step-predicted pose `p + w·dt` stays inside the certified safe set; otherwise the zero
    /// (stop) command. Never emits the untrusted proposal.
    fn guarded_velocity(&self, g: &[f64], p: &[f64], action: &mut Vec<f64>) {
        let dim = self.spatial_dim;
        if dim == 0
            || p.len() < dim
            || g.len() < dim
            || !all_finite(&p[..dim])
            || !all_finite(&g[..dim])
        {
            return self.stop_velocity(action);
        }
        // Candidate: proportional pull toward the target, projected onto the speed ball.
        action.clear();
        for i in 0..dim {
            action.push(self.k_p * (g[i] - p[i]));
        }
        let scale = ball_scale(norm(&action[..dim]), self.v_max);
        for a in action.iter_mut().take(dim) {
            *a *= scale;
        }
        let in_set = self
            .barriers
            .iter()
            .all(|b| b.static_h(dim, |i| p[i] + action[i] * self.dt) >= -SET_TOL);
        if !(in_set && all_finite(&action[..dim])) {
            self.stop_velocity(action);
        }
    }

    /// Guarded move-toward-target in `Position`: a step-capped target toward `g`, emitted **iff**
    /// the target itself lies inside the certified safe set; otherwise the current pose (stay put).
    fn guarded_position(&self, g: &[f64], p: &[f64], action: &mut Vec<f64>) {
        let dim = self.spatial_dim;
        if dim == 0
            || p.len() < dim
            || g.len() < dim
            || !all_finite(&p[..dim])
            || !all_finite(&g[..dim])
        {
            return self.hold_position(p, action);
        }
        action.clear();
        for i in 0..dim {
            action.push(g[i] - p[i]);
        }
        let scale = ball_scale(norm(&action[..dim]), self.step_max);
        for i in 0..dim {
            action[i] = p[i] + action[i] * scale;
        }
        let in_set = self
            .barriers
            .iter()
            .all(|b| b.static_h(dim, |i| action[i]) >= -SET_TOL);
        if !(in_set && all_finite(&action[..dim])) {
            self.hold_position(p, action);
        }
    }

    /// Guarded move-toward-target (the Hold/SafeState kernel). Emits a saturated PD command toward
    /// `g` **iff** it is provably distance-non-increasing and keeps the one-step-predicted position
    /// inside the certified safe set; otherwise brakes. Never emits a non-finite or out-of-box
    /// command, and never emits the (untrusted) proposal.
    fn guarded_move(&self, g: &[f64], p: &[f64], v: &[f64], action: &mut Vec<f64>) {
        let dim = self.spatial_dim;
        // Preconditions. A non-spatial model, a short/garbled state, or a malformed target has no
        // safe move-toward — fall back to the dt-free brake (fail-safe).
        if dim == 0
            || p.len() < dim
            || v.len() < dim
            || g.len() < dim
            || !all_finite(&p[..dim])
            || !all_finite(&v[..dim])
            || !all_finite(&g[..dim])
        {
            return self.brake(v, action);
        }

        // Candidate: saturated PD toward the target. The box-clamp guarantees `|uᵢ| ≤ u_max`.
        action.clear();
        for i in 0..dim {
            let u = (self.k_p * (g[i] - p[i]) - self.k_d * v[i]).clamp(-self.u_max, self.u_max);
            action.push(u);
        }

        // Guard (a): the target Lyapunov energy V = ½·k_p·‖p−g‖² + ½·‖v‖² must not increase under
        // the exact constant-acceleration one-step prediction p' = p + v·dt + ½u·dt², v' = v + u·dt.
        // (Distance alone can rise for one step while an adverse velocity is being reversed; the
        // energy — which the damped PD law provably dissipates — is the right monotone quantity.)
        let dt = self.dt;
        let half_dt2 = 0.5 * dt * dt;
        let mut v_cur = 0.0;
        let mut v_next = 0.0;
        for i in 0..dim {
            let ec = p[i] - g[i];
            let vc = v[i];
            v_cur += 0.5 * self.k_p * ec * ec + 0.5 * vc * vc;
            let pn = p[i] + vc * dt + half_dt2 * action[i];
            let vn = vc + action[i] * dt;
            let en = pn - g[i];
            v_next += 0.5 * self.k_p * en * en + 0.5 * vn * vn;
        }
        let progress_ok = v_next <= v_cur * (1.0 + DIST_TOL) + DIST_TOL;

        // Guard (b): the predicted position must stay inside every keep-out's safe set (the shared
        // shield geometry, with margin). `coord` recomputes p'ᵢ on the fly — no scratch buffer.
        let mut in_set = true;
        for b in &self.barriers {
            let h = b.static_h(dim, |i| p[i] + v[i] * dt + half_dt2 * action[i]);
            if h < -SET_TOL {
                in_set = false;
                break;
            }
        }

        // Guard (c): defence-in-depth — the emitted command must be finite (the clamp already
        // bounds it). If any guard fails, degrade to the safe floor.
        if !(progress_ok && in_set && all_finite(&action[..dim])) {
            self.brake(v, action);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{KeepOutTerm, Shape};

    /// The historical constructor arity, kept in the tests so each case reads as before: an
    /// unlimited kinematic envelope (no spec-authored limit) with the given `u_max`.
    fn backup(
        dim: usize,
        u_max: f64,
        k_brake: f64,
        dt: f64,
        safe_pose: Option<Vec<f64>>,
        terms: &[KeepOutTerm],
    ) -> Backup {
        let cfg = ShieldConfig {
            u_max,
            ..Default::default()
        };
        Backup::new(
            dim,
            cfg,
            k_brake,
            dt,
            safe_pose,
            terms,
            &ActionLimits::default(),
        )
    }

    fn sphere_term(center: Vec<f64>, radius: f64, margin: f64) -> KeepOutTerm {
        KeepOutTerm {
            constraint_id: "c".into(),
            on_uncertain: crate::model::OnUncertain::Fallback,
            shape: Shape::Sphere,
            margin_m: margin,
            center,
            half_extents: vec![],
            radius: Some(radius),
            normal: vec![],
            offset: None,
        }
    }

    /// The exact constant-acceleration (ZOH) step the guard predicts against — used to advance a
    /// plant in the tests so the realised trajectory matches the guard's one-step prediction.
    fn step(p: &mut [f64], v: &mut [f64], u: &[f64], dt: f64) {
        for i in 0..p.len() {
            p[i] += v[i] * dt + 0.5 * u[i] * dt * dt;
            v[i] += u[i] * dt;
        }
    }

    fn bounded(action: &[f64], u_max: f64) {
        for &a in action {
            assert!(a.is_finite(), "action {a} is not finite");
            assert!(a.abs() <= u_max + 1e-9, "action {a} exceeds u_max {u_max}");
        }
    }

    fn dist2(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// The target Lyapunov energy the guarded law keeps non-increasing (k_p = k_brake²/4 = 4 for
    /// these fixtures): V = ½·k_p·‖p−g‖² + ½·‖v‖².
    fn energy(p: &[f64], v: &[f64], g: &[f64]) -> f64 {
        let k_p = 4.0;
        0.5 * k_p * dist2(p, g) + 0.5 * v.iter().map(|x| x * x).sum::<f64>()
    }

    // --- BrakeToStop --------------------------------------------------------------------------

    #[test]
    fn brake_is_anti_parallel_and_bounded() {
        let b = backup(3, 20.0, 4.0, 0.1, None, &[]);
        let mut a = Vec::new();
        let v = [2.0, -3.0, 0.5];
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Effort,
            &[0.0, 0.0, 0.0],
            &v,
            None,
            &mut a,
        );
        // Anti-parallel: vᵀu ≤ 0 (never adds speed).
        let dot: f64 = v.iter().zip(&a).map(|(x, y)| x * y).sum();
        assert!(dot <= 0.0, "brake must not increase speed: vᵀu = {dot}");
        bounded(&a, 20.0);
    }

    #[test]
    fn brake_sanitises_non_finite_velocity() {
        let b = backup(3, 20.0, 4.0, 0.1, None, &[]);
        let mut a = Vec::new();
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Effort,
            &[0.0, 0.0, 0.0],
            &[f64::NAN, 1.0, f64::INFINITY],
            None,
            &mut a,
        );
        bounded(&a, 20.0);
        assert_eq!(a[0], 0.0, "NaN velocity must sanitise to a zero command");
    }

    #[test]
    fn brake_monotonically_stops_the_plant() {
        let b = backup(2, 20.0, 4.0, 0.02, None, &[]);
        let mut p = [5.0f64, 5.0];
        let mut v = [1.5f64, -2.0];
        let mut a = Vec::new();
        let mut speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
        for _ in 0..2000 {
            b.control(
                BackupKind::BrakeToStop,
                ControlMode::Effort,
                &p,
                &v,
                None,
                &mut a,
            );
            bounded(&a, 20.0);
            step(&mut p, &mut v, &a, 0.02);
            let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!(s <= speed + 1e-9, "speed increased: {s} > {speed}");
            speed = s;
        }
        assert!(
            speed < 1e-3,
            "brake failed to bring the plant to rest: {speed}"
        );
    }

    // --- SafeState (retreat-to-charging-pose) -------------------------------------------------

    #[test]
    fn retreat_monotonically_reduces_distance_and_stays_in_set() {
        // Charging pose at (40, 0); a lander keep-out sphere (r=10, margin=2) at the origin. Start
        // out past the pose; the retreat must close on it every step while never entering the set.
        let g = vec![40.0, 0.0];
        let term = sphere_term(vec![0.0, 0.0], 10.0, 2.0);
        let b = backup(
            2,
            20.0,
            4.0,
            0.05,
            Some(g.clone()),
            std::slice::from_ref(&term),
        );
        let barrier = Barrier::from_term(&term, 2);

        let mut p = [70.0f64, 0.0];
        let mut v = [0.0f64, 0.0];
        let mut a = Vec::new();
        let mut e = energy(&p, &v, &g);
        let mut moved = false;
        for _ in 0..4000 {
            b.control(
                BackupKind::SafeState,
                ControlMode::Effort,
                &p,
                &v,
                None,
                &mut a,
            );
            bounded(&a, 20.0);
            // Never inside the keep-out.
            let h = barrier.static_h(2, |i| p[i]);
            assert!(h >= -SET_TOL, "retreat entered the keep-out: h = {h}");
            step(&mut p, &mut v, &a, 0.05);
            let en = energy(&p, &v, &g);
            assert!(
                en <= e + 1e-6,
                "target Lyapunov energy increased: {en} > {e}"
            );
            if en < e - 1e-9 {
                moved = true;
            }
            e = en;
        }
        assert!(
            moved,
            "retreat never made progress toward the charging pose"
        );
        assert!(
            dist2(&p, &g) < 1.0,
            "retreat did not converge to the charging pose: d² = {}",
            dist2(&p, &g)
        );
    }

    #[test]
    fn retreat_falls_back_to_brake_when_target_is_behind_a_keepout() {
        // Target on the *far* side of a keep-out sphere from the rover: any PD command that reduces
        // distance-to-target drives the predicted position into the set, so the guard must reject
        // it and brake. Rover at (12,0) (just outside r=10+margin=2 at origin), target at (-40,0).
        let g = vec![-40.0, 0.0];
        let term = sphere_term(vec![0.0, 0.0], 10.0, 2.0);
        let b = backup(2, 20.0, 4.0, 0.05, Some(g), std::slice::from_ref(&term));
        let mut a = Vec::new();
        // Moving inward toward the target/centre.
        b.control(
            BackupKind::SafeState,
            ControlMode::Effort,
            &[12.0, 0.0],
            &[-1.0, 0.0],
            None,
            &mut a,
        );
        bounded(&a, 20.0);
        // Brake fallback opposes the inward velocity: +x command (decelerating), not -x (inward PD).
        assert!(
            a[0] > 0.0,
            "expected a braking (outward) command, got {:?}",
            a
        );
    }

    #[test]
    fn safe_state_without_pose_degrades_to_brake() {
        let b = backup(2, 20.0, 4.0, 0.05, None, &[]);
        let mut a = Vec::new();
        let v = [1.0, -1.0];
        b.control(
            BackupKind::SafeState,
            ControlMode::Effort,
            &[5.0, 5.0],
            &v,
            None,
            &mut a,
        );
        // Same as an explicit brake.
        let mut brake = Vec::new();
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Effort,
            &[5.0, 5.0],
            &v,
            None,
            &mut brake,
        );
        assert_eq!(a, brake, "missing safe_pose must degrade to brake-to-stop");
    }

    #[test]
    fn safe_state_non_finite_state_degrades_to_brake() {
        let g = vec![40.0, 0.0];
        let b = backup(2, 20.0, 4.0, 0.05, Some(g), &[]);
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Effort,
            &[f64::NAN, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        bounded(&a, 20.0);
        // A non-finite position must never yield a non-finite or unbounded command.
        assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn non_spatial_model_degrades_to_brake() {
        // spatial_dim = 0 (no keep-out geometry): retreat/hold are meaningless → brake over v.
        let b = backup(0, 20.0, 4.0, 0.05, Some(vec![1.0]), &[]);
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Effort,
            &[],
            &[3.0],
            None,
            &mut a,
        );
        assert_eq!(a.len(), 1);
        bounded(&a, 20.0);
        assert!(a[0] <= 0.0, "brake over a +v component must be ≤ 0");
    }

    // --- Hold (station-keep) ------------------------------------------------------------------

    #[test]
    fn hold_returns_to_anchor_and_nulls_drift() {
        // Latched anchor at (10,10); the rover has drifted to (12,9) with residual velocity. Hold
        // must pull it back toward the anchor and damp velocity — distinct from a pure brake, which
        // would ignore the position error.
        let anchor = [10.0f64, 10.0];
        let term = sphere_term(vec![0.0, 0.0], 3.0, 1.0); // far away; never binds here
        let b = backup(2, 20.0, 4.0, 0.05, None, std::slice::from_ref(&term));
        let mut p = [12.0f64, 9.0];
        let mut v = [0.3f64, -0.2]; // adverse drift: initially moving *away* from the anchor
        let mut a = Vec::new();
        let mut e = energy(&p, &v, &anchor);
        for _ in 0..4000 {
            b.control(
                BackupKind::Hold,
                ControlMode::Effort,
                &p,
                &v,
                Some(&anchor),
                &mut a,
            );
            bounded(&a, 20.0);
            step(&mut p, &mut v, &a, 0.05);
            let en = energy(&p, &v, &anchor);
            assert!(
                en <= e + 1e-6,
                "hold let the target energy grow: {en} > {e}"
            );
            e = en;
        }
        let d2 = dist2(&p, &anchor);
        let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!(d2 < 1e-3, "hold did not return to the anchor: d² = {d2}");
        assert!(
            speed < 1e-3,
            "hold did not null the drift velocity: {speed}"
        );
    }

    #[test]
    fn hold_without_target_degrades_to_brake() {
        let b = backup(2, 20.0, 4.0, 0.05, None, &[]);
        let mut a = Vec::new();
        let v = [1.0, 2.0];
        b.control(
            BackupKind::Hold,
            ControlMode::Effort,
            &[1.0, 1.0],
            &v,
            None,
            &mut a,
        );
        let mut brake = Vec::new();
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Effort,
            &[1.0, 1.0],
            &v,
            None,
            &mut brake,
        );
        assert_eq!(
            a, brake,
            "hold with no anchor must degrade to brake-to-stop"
        );
    }

    // --- the kinematic channels (RM-P1-GUARD-03) ----------------------------------------------

    /// A backup with a spec-authored 0.5 m/s speed ceiling, dt = 1 s, and a lander keep-out.
    fn kinematic_backup(safe_pose: Option<Vec<f64>>) -> Backup {
        let term = sphere_term(vec![0.0, 0.0], 10.0, 2.0);
        Backup::new(
            2,
            ShieldConfig::default(),
            4.0,
            1.0,
            safe_pose,
            std::slice::from_ref(&term),
            &ActionLimits {
                max_velocity_mps: Some(0.5),
                max_accel_mps2: None,
            },
        )
    }

    #[test]
    fn velocity_brake_commands_zero_velocity() {
        // The single-integrator safe floor is exactly "stop": a zero velocity command, NOT an
        // effort brake a velocity-tracking actuator would ignore.
        let b = kinematic_backup(None);
        let mut a = Vec::new();
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Velocity,
            &[30.0, 0.0],
            &[5.0, -5.0],
            None,
            &mut a,
        );
        assert_eq!(a, vec![0.0, 0.0]);
    }

    #[test]
    fn position_brake_commands_the_current_pose() {
        // The pose-tracking safe floor is "do not move": command where you already are.
        let b = kinematic_backup(None);
        let mut a = Vec::new();
        b.control(
            BackupKind::BrakeToStop,
            ControlMode::Position,
            &[30.0, -4.0],
            &[5.0, -5.0],
            None,
            &mut a,
        );
        assert_eq!(a, vec![30.0, -4.0]);
    }

    #[test]
    fn velocity_retreat_respects_the_reviewed_speed_ceiling_and_the_safe_set() {
        // Charging pose at (40,0), rover at (30,0): the retreat pulls toward it, capped at the
        // reviewed 0.5 m/s ceiling, and the predicted pose stays outside the keep-out.
        let b = kinematic_backup(Some(vec![40.0, 0.0]));
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Velocity,
            &[30.0, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        assert!(a[0] > 0.0, "retreat must pull toward the pose: {a:?}");
        assert!(
            norm(&a) <= 0.5 + 1e-9,
            "retreat exceeded the reviewed speed ceiling: {a:?}"
        );
    }

    #[test]
    fn velocity_retreat_through_a_keepout_degrades_to_stop() {
        // Target on the far side of the keep-out sphere: any pull toward it drives the predicted
        // pose into the set, so the guard must reject and command zero velocity.
        let b = kinematic_backup(Some(vec![-40.0, 0.0]));
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Velocity,
            &[12.1, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        assert_eq!(a, vec![0.0, 0.0], "unsafe retreat must degrade to a stop");
    }

    #[test]
    fn position_retreat_is_step_capped_and_in_set() {
        let b = kinematic_backup(Some(vec![40.0, 0.0]));
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Position,
            &[30.0, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        // step_max = v_max·dt = 0.5 m ⇒ the commanded target is 0.5 m toward the pose.
        assert!((a[0] - 30.5).abs() < 1e-9, "expected a 0.5 m step: {a:?}");
        // …and the target itself is outside the keep-out.
        let barrier = Barrier::from_term(&sphere_term(vec![0.0, 0.0], 10.0, 2.0), 2);
        assert!(barrier.static_h(2, |i| a[i]) >= -SET_TOL);
    }

    #[test]
    fn position_retreat_into_a_keepout_degrades_to_stay_put() {
        let b = kinematic_backup(Some(vec![0.0, 0.0])); // the pose IS the keep-out centre
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Position,
            &[12.1, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        assert_eq!(a, vec![12.1, 0.0], "unsafe target must degrade to stay-put");
    }

    #[test]
    fn kinematic_backups_sanitise_a_non_finite_pose() {
        let b = kinematic_backup(Some(vec![40.0, 0.0]));
        let mut a = Vec::new();
        b.control(
            BackupKind::SafeState,
            ControlMode::Velocity,
            &[f64::NAN, 0.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        assert_eq!(a, vec![0.0, 0.0]);
        b.control(
            BackupKind::SafeState,
            ControlMode::Position,
            &[f64::NAN, 3.0],
            &[0.0, 0.0],
            None,
            &mut a,
        );
        assert!(a.iter().all(|x| x.is_finite()), "non-finite command: {a:?}");
    }
}
