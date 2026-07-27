//! CBF-QP shield — the *correct* layer (guard.md §9.2).
//!
//! Minimally perturbs the policy's proposed command so the state provably stays inside the
//! keep-out safe set. The shield understands **three commanded quantities** — the three Core
//! `ControlMode`s that have a plant model in the TCB (RM-P1-GUARD-03):
//!
//! | [`ControlMode`] | Plant | Commanded variable | Barrier relative degree |
//! |---|---|---|---|
//! | [`Effort`](ControlMode::Effort) | double integrator `ṗ = v`, `v̇ = u` | acceleration `u` | 2 (HOCBF) |
//! | [`Velocity`](ControlMode::Velocity) | single integrator `ṗ = w` | velocity `w` | 1 |
//! | [`Position`](ControlMode::Position) | pose-tracking `p⁺ = q` | target `q` | 0 (the target itself) |
//!
//! Every mode reduces each keep-out term to **linear inequalities `aᵢ·x ≥ bᵢ`** in the commanded
//! variable `x`, plus one **kinematic-limit** set:
//!
//! ```text
//!     minimise ‖x − x₀‖²   s.t.  aᵢ·x ≥ bᵢ  (each keep-out) ,  x ∈ K
//!
//!     Effort    K = { |uᵢ| ≤ u_max }                 aᵢ = ∇h ,  bᵢ = −(∇²h[v,v] + k1·ḣ + k0·h)
//!     Velocity  K = { ‖w‖ ≤ v_max }                  aᵢ = ∇h ,  bᵢ = −h / dt
//!     Position  K = { ‖q − p‖ ≤ v_max·dt }           aᵢ = ∇h ,  bᵢ = aᵢ·p − h
//! ```
//!
//! solved with a **fixed-capacity, allocation-free** Dykstra projection onto the intersection of
//! the constraint half-spaces and the kinematic set — no heap traffic on the hot path (guard.md §2
//! principle 6). The returned command is then **hard-certified** against every constraint; if it
//! cannot be certified (empty intersection, a degenerate barrier, non-finite input) the shield
//! reports `certified = false` and the arbiter falls back — never fail-open (§9.1).
//!
//! **The kinematic-mode certificate is a one-step *set* check, not just a linear one.** For
//! `Velocity`/`Position` the barrier rows above are the supporting half-space of the safe set at
//! the current pose, so satisfying them implies the *next* pose is outside the keep-out; the shield
//! nevertheless re-checks the realised next pose (`p + w·dt`, resp. `q`) against the exact barrier
//! `h ≥ −tol` before certifying. That makes the kinematic guarantee a genuine discrete-time
//! forward-invariance certificate rather than a linearisation, and an infeasible tick (e.g. the
//! pose is already in the set) is *detected* and falls back rather than being papered over.
//!
//! Box keep-out is handled by its **enclosing sphere** (radius `‖half_extents‖`): staying
//! outside the enclosing sphere is strictly *more* conservative than staying outside the box,
//! so the guarantee is sound (over-restrictive at the box corners — a documented first-slice
//! simplification; an exact face-selecting box CBF is deferred).
//!
//! **Solver.** The QP is solved by a bespoke, allocation-free Dykstra alternating projection rather
//! than by OSQP/Clarabel (`guard.md §11`). That is a deliberate, recorded deviation — see
//! `docs/adr/0001-cbf-qp-solver.md`; `rust/tests/clarabel_crosscheck.rs` keeps Clarabel as the
//! independent optimality oracle, and `rust/src/verify.rs` carries the Kani proofs of the
//! projection kernels.

// The numeric kernels below step several parallel fixed-length arrays (`x`, `y`, `corrections`,
// `amat` rows) in lockstep by index — a range loop is the clearest, allocation-free form.
#![allow(clippy::needless_range_loop)]

use crate::model::{ActionLimits, KeepOutTerm, Shape};

/// The commanded quantity an actuator setpoint carries — the action space the shield projects in.
///
/// A 1:1 mirror of the three Core `ControlMode` members the TCB has a plant model for. The Core
/// modes it does **not** model (`IMPEDANCE`, `TRAJECTORY`) are deliberately absent: an action the
/// core cannot express as one of these is uncertifiable by construction and the arbiter substitutes
/// a verified safe command (fail-closed; there is no "unmodelled ⇒ pass through" path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlMode {
    /// Commanded acceleration (Core `EFFORT`) — force/torque per unit mass.
    #[default]
    Effort,
    /// Commanded velocity setpoint (Core `VELOCITY`).
    Velocity,
    /// Commanded position/target setpoint (Core `POSITION`).
    Position,
}

/// Tunable control-barrier parameters (the class-𝒦 gains of the HOCBF and the QP tolerances).
#[derive(Debug, Clone, Copy)]
pub struct ShieldConfig {
    /// `k0`, `k1` in `ḧ + k1·ḣ + k0·h ≥ 0`. Defaults are critically damped (`k1² = 4k0`).
    pub k0: f64,
    pub k1: f64,
    /// Symmetric per-axis control authority `|uᵢ| ≤ u_max` (the `Effort` kinematic set). Tightened
    /// by the compiled model's `max_accel_mps2` when the reviewed SafetySpec authors one.
    pub u_max: f64,
    /// Commanded-speed ceiling `‖w‖ ≤ v_max` (the `Velocity`/`Position` kinematic set) used when
    /// the reviewed SafetySpec authors no `kinematic_limit.max_velocity_mps`. A spec-authored limit
    /// always **tightens** this — configuration may never loosen the reviewed contract.
    pub v_max: f64,
    /// Feasibility / certification tolerance.
    pub tol: f64,
    /// Dykstra iteration cap (bounds worst-case work — a static latency guarantee).
    pub max_iter: usize,
}

impl Default for ShieldConfig {
    fn default() -> Self {
        Self {
            k0: 9.0,
            k1: 6.0,
            u_max: 20.0,
            v_max: 2.0,
            tol: 1e-7,
            // The kinematic modes made the QP harder than the effort mode ever was: their feasible
            // set is a *ball* intersected with the barrier half-spaces (rather than a roomy box that
            // is usually inactive), so a proposal far outside a tight speed ceiling starts far from
            // the optimum and Dykstra's tail crawls. 64 cycles was enough for the box; a
            // multi-keep-out kinematic solve needs a few hundred. This is a *fixed* budget — the
            // worst-case-work / static-latency guarantee (guard.md §2 principle 6, §8) is unchanged,
            // and an easy solve still breaks out on the first stall. Overrunning it is not unsafe:
            // an uncertified iterate falls back (§9.1).
            max_iter: 512,
        }
    }
}

/// A keep-out term reduced to the coefficients the per-tick barrier evaluation needs.
///
/// `pub(crate)` so the simplex backup (`backup.rs`) can reuse the *same* geometry to check that a
/// candidate retreat/hold command stays inside the certified safe set — one source of truth for
/// "the safe set", shared between the correct (shield) and recover (backup) layers.
#[derive(Debug, Clone)]
pub(crate) enum Barrier {
    /// Norm barrier `h = ‖p − c‖ − R` (sphere, and box via its enclosing sphere).
    Norm { center: Vec<f64>, radius: f64 },
    /// Half-space barrier `h = n·p − rhs` with unit `n` (safe set `n·p + offset ≥ margin`).
    Half { normal: Vec<f64>, rhs: f64 },
}

impl Barrier {
    /// Build the barrier coefficients for one keep-out term (the effective radius/rhs already bake
    /// in `margin_m`, so `static_h ≥ 0` means "outside the keep-out by the safety margin").
    pub(crate) fn from_term(t: &KeepOutTerm, dim: usize) -> Self {
        match t.shape {
            Shape::Sphere => Barrier::Norm {
                center: t.center.clone(),
                radius: t.radius.unwrap_or(0.0) + t.margin_m,
            },
            Shape::Box => Barrier::Norm {
                center: t.center.clone(),
                radius: norm(&t.half_extents) + t.margin_m,
            },
            Shape::HalfSpace => {
                let n = norm(&t.normal).max(f64::MIN_POSITIVE);
                let rhs = (t.margin_m - t.offset.unwrap_or(0.0)) / n;
                let mut unit: Vec<f64> = t.normal.iter().map(|x| x / n).collect();
                unit.resize(dim, 0.0);
                Barrier::Half { normal: unit, rhs }
            }
        }
    }

    /// The *static* (position-only) barrier value `h(x)` at coordinates supplied by `coord`,
    /// without materialising the point — `coord(i)` yields the i-th coordinate. `h ≥ 0` ⇔ the
    /// point is in the safe set (outside the keep-out by the margin). Allocation-free; used by the
    /// backup to certify a candidate retreat/hold position. A degenerate norm barrier (exactly at
    /// the centre) returns `−radius < 0`, i.e. "inside" — the conservative, fail-safe verdict.
    pub(crate) fn static_h(&self, dim: usize, coord: impl Fn(usize) -> f64) -> f64 {
        match self {
            Barrier::Norm { center, radius } => {
                let mut d2 = 0.0;
                for i in 0..dim {
                    let ri = coord(i) - center.get(i).copied().unwrap_or(0.0);
                    d2 += ri * ri;
                }
                d2.sqrt() - radius
            }
            Barrier::Half { normal, rhs } => {
                let mut s = 0.0;
                for i in 0..dim {
                    s += normal[i] * coord(i);
                }
                s - rhs
            }
        }
    }
}

/// The CBF-QP safety filter.
#[derive(Debug, Clone)]
pub struct Shield {
    dim: usize,
    barriers: Vec<Barrier>,
    cfg: ShieldConfig,
    /// The compiled sample period — the one-step horizon of the kinematic (velocity/position) modes.
    dt: f64,
    /// Effective acceleration box: `cfg.u_max` tightened by the spec's `max_accel_mps2`.
    u_max: f64,
    /// Effective speed ceiling: `cfg.v_max` tightened by the spec's `max_velocity_mps`.
    v_max: f64,
    /// Largest commanded position step: `v_max · dt`.
    step_max: f64,
    // Pre-allocated hot-path workspace (never grows after construction).
    amat: Vec<f64>,        // m × dim constraint normals
    bvec: Vec<f64>,        // m constraint rhs
    corrections: Vec<f64>, // (m + 1) × dim Dykstra corrections
    x: Vec<f64>,
    y: Vec<f64>,
    xprev: Vec<f64>,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Clamp a scalar into the symmetric box `[−limit, limit]` — the `Effort` kinematic projection,
/// applied per axis. Free (and `pub(crate)`) so `verify.rs` can prove its invariants with Kani.
#[inline]
pub(crate) fn project_box_axis(x: f64, limit: f64) -> f64 {
    x.clamp(-limit, limit)
}

/// The scale factor that projects a vector of length `len` onto the ball of radius `radius`
/// (`1.0` when it is already inside) — the `Velocity` speed ceiling and the `Position` step cap are
/// the same kernel. Free (and `pub(crate)`) so `verify.rs` can prove its invariants with Kani.
///
/// **Rounding is checked, not assumed.** The naive `radius / len` is *not* always sound in IEEE-754:
/// when the quotient underflows into the subnormal range (a tiny radius against a huge length) it
/// carries enormous relative error, and `len · scale` can then land measurably **outside** the ball
/// — Kani's counterexample (`len ≈ 1.5e172`, `radius ≈ 7.6e-152`) overshoots by 0.4%. Neither the
/// 400-QP Clarabel cross-check nor the proptest suite ever sampled that pair; the model checker
/// states it as fact.
///
/// So the kernel **verifies its own result** rather than trusting the float unit: the candidate
/// scale must be in `[0, 1]` *and* must actually keep `len · scale` inside the ball, or it collapses
/// to `0.0` — the zero command, which is inside every ball and is the safe floor in every mode (a
/// zero velocity is a stop; a zero step is stay-put). The `4·ε` slack is exactly the one-ulp residue
/// of a rounded division followed by a rounded multiplication — orders of magnitude below
/// `ShieldConfig::tol`, so an honest projection is never collapsed by it.
///
/// This is the ADR-0001 discipline applied at the kernel level: **certify the output, don't trust
/// the solver** — here the "solver" is the FPU's divider.
#[inline]
pub(crate) fn ball_scale(len: f64, radius: f64) -> f64 {
    // NaN-safe by construction: a non-finite length or radius means the *command* is already
    // malformed, so no scaling is meaningful — leave it and let `certify` reject it.
    let clamp_needed = len.is_finite() && radius.is_finite() && len > radius && len > 0.0;
    if !clamp_needed {
        return 1.0;
    }
    let candidate = radius / len;
    if ball_scale_is_sound(len, radius, candidate) {
        candidate
    } else {
        0.0 // the FPU betrayed us — collapse to the zero command (inside every ball).
    }
}

/// The **post-condition** [`ball_scale`] checks its own division against: is `scale` a factor that
/// provably keeps a vector of length `len` inside the ball of radius `radius`?
///
/// Factored out as a free function because it is the load-bearing half of the kernel, and because it
/// is *decidable* — [`verify.rs`](crate::verify) proves with Kani that **any** scale this accepts
/// keeps the command in the ball, over *every* `f64` triple, with no assumption whatsoever about
/// where the candidate came from. That is what makes [`ball_scale`] sound for all inputs including
/// the subnormal-underflow corner: whatever the divider produces, only a validated factor is ever
/// returned. (Proving the *divider itself* over the unbounded `f64` range is an SMT bit-blasting
/// blow-up, and would prove strictly less.)
///
/// The `4·ε` slack is exactly the one-ulp residue of a rounded division followed by a rounded
/// multiplication (`len · fl(radius/len) = radius·(1+δ₁)(1+δ₂)`, `|δᵢ| ≤ ε/2`) — orders of magnitude
/// below `ShieldConfig::tol`, so an honest projection is never rejected by it.
#[inline]
pub(crate) fn ball_scale_is_sound(len: f64, radius: f64, scale: f64) -> bool {
    scale.is_finite()
        && (0.0..=1.0).contains(&scale)
        && len * scale <= radius * (1.0 + 4.0 * f64::EPSILON)
}

/// Evaluate one barrier at `(p, v)` for control mode `mode`: write its constraint normal into `a`
/// and return `(rhs b, barrier value h, ok)`. `ok = false` marks a degenerate barrier (e.g. exactly
/// at a sphere centre, where the normal is undefined) that cannot be certified this tick. A free
/// function so the caller can hold disjoint borrows of `barriers` and `amat` — no allocation.
#[allow(clippy::too_many_arguments)]
fn eval_barrier(
    barrier: &Barrier,
    mode: ControlMode,
    dim: usize,
    k0: f64,
    k1: f64,
    dt: f64,
    p: &[f64],
    v: &[f64],
    a: &mut [f64],
) -> (f64, f64, bool) {
    // The gradient `a = ∇h(p)` and the barrier value `h(p)` are shared by every mode; only the
    // right-hand side differs (the plant's relative degree).
    let (h, ok) = match barrier {
        Barrier::Norm { center, radius } => {
            // r = p − c ; d = ‖r‖ ; n̂ = r/d
            let mut d2 = 0.0;
            for i in 0..dim {
                let ri = p[i] - center.get(i).copied().unwrap_or(0.0);
                a[i] = ri;
                d2 += ri * ri;
            }
            let d = d2.sqrt();
            if d < 1e-9 {
                return (0.0, -*radius, false); // at the centre — undefined normal
            }
            for ai in a.iter_mut().take(dim) {
                *ai /= d;
            }
            (d - radius, true)
        }
        Barrier::Half { normal, rhs } => {
            a[..dim].copy_from_slice(&normal[..dim]);
            (dot(normal, p) - rhs, true)
        }
    };

    let b = match mode {
        // Relative degree 2 (double integrator): n̂·u ≥ −(∇²h[v,v] + k1·ḣ + k0·h).
        ControlMode::Effort => {
            let hdot = dot(&a[..dim], &v[..dim]);
            let curvature = match barrier {
                Barrier::Norm { center, .. } => {
                    // v^T ∇²h v = (‖v‖² − (n̂·v)²)/d  — zero for the flat half-space barrier.
                    let mut d2 = 0.0;
                    for i in 0..dim {
                        let ri = p[i] - center.get(i).copied().unwrap_or(0.0);
                        d2 += ri * ri;
                    }
                    let vv = dot(&v[..dim], &v[..dim]);
                    (vv - hdot * hdot) / d2.sqrt()
                }
                Barrier::Half { .. } => 0.0,
            };
            -(curvature + k1 * hdot + k0 * h)
        }
        // Relative degree 1 (single integrator): the *next* pose `p + w·dt` must stay on the safe
        // side of the supporting half-space at `p`  ⇔  n̂·w ≥ −h/dt.
        ControlMode::Velocity => -h / dt,
        // Relative degree 0: the commanded target `q` must itself be on the safe side of the
        // supporting half-space at `p`  ⇔  n̂·q ≥ n̂·p − h.
        ControlMode::Position => dot(&a[..dim], &p[..dim]) - h,
    };
    (b, h, ok)
}

impl Shield {
    /// Build a shield over `terms` in a `dim`-dimensional frame.
    ///
    /// `dt` is the compiled `sample_period_s` (the one-step horizon of the kinematic modes) and
    /// `limits` the reviewed SafetySpec's `kinematic_limit` envelope: a spec-authored limit always
    /// **tightens** the corresponding `cfg` ceiling and can never loosen it, so the enforced
    /// envelope is `min(config, reviewed contract)` — configuration cannot widen what was reviewed.
    pub fn new(
        terms: &[KeepOutTerm],
        dim: usize,
        cfg: ShieldConfig,
        dt: f64,
        limits: &ActionLimits,
    ) -> Self {
        let barriers = terms
            .iter()
            .map(|t| Barrier::from_term(t, dim))
            .collect::<Vec<_>>();
        let m = barriers.len();
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 1.0 };
        let u_max = tighten(cfg.u_max, limits.max_accel_mps2);
        let v_max = tighten(cfg.v_max, limits.max_velocity_mps);
        Self {
            dim,
            barriers,
            cfg,
            dt,
            u_max,
            v_max,
            step_max: v_max * dt,
            amat: vec![0.0; m * dim],
            bvec: vec![0.0; m],
            corrections: vec![0.0; (m + 1) * dim],
            x: vec![0.0; dim],
            y: vec![0.0; dim],
            xprev: vec![0.0; dim],
        }
    }

    #[inline]
    pub fn num_barriers(&self) -> usize {
        self.barriers.len()
    }

    /// The enforced acceleration box (`cfg.u_max` tightened by the spec's kinematic limit).
    #[inline]
    pub fn u_max(&self) -> f64 {
        self.u_max
    }

    /// The enforced commanded-speed ceiling (`cfg.v_max` tightened by the spec's kinematic limit).
    #[inline]
    pub fn v_max(&self) -> f64 {
        self.v_max
    }

    /// The largest commanded position step (`v_max · dt`).
    #[inline]
    pub fn step_max(&self) -> f64 {
        self.step_max
    }

    /// Run the shield in `Effort` mode (the commanded-acceleration path). Kept as the historical
    /// entry point; see [`Shield::solve_mode`] for the general form.
    pub fn solve(
        &mut self,
        p: &[f64],
        v: &[f64],
        u0: &[f64],
        action: &mut Vec<f64>,
    ) -> (bool, f64) {
        self.solve_mode(ControlMode::Effort, p, v, u0, action)
    }

    /// Run the shield for control mode `mode`. Fills `action` with the certified minimally-perturbed
    /// command and returns `(certified, min_barrier_margin)`. Allocation-free.
    pub fn solve_mode(
        &mut self,
        mode: ControlMode,
        p: &[f64],
        v: &[f64],
        x0: &[f64],
        action: &mut Vec<f64>,
    ) -> (bool, f64) {
        let dim = self.dim;
        let m = self.barriers.len();

        action.clear();

        // Non-finite or dimensionally-wrong input is uncertifiable by construction.
        if p.len() < dim
            || v.len() < dim
            || x0.len() < dim
            || p[..dim]
                .iter()
                .chain(&v[..dim])
                .chain(&x0[..dim])
                .any(|x| !x.is_finite())
        {
            action.extend_from_slice(&x0[..x0.len().min(dim)]);
            return (false, f64::NEG_INFINITY);
        }
        action.extend_from_slice(&x0[..dim]);

        // Build the constraint set and the barrier certificate. Disjoint field borrows
        // (`barriers` read, `amat` written) keep this allocation-free.
        let mut min_h = f64::INFINITY;
        let mut degenerate = false;
        for i in 0..m {
            let row = &mut self.amat[i * dim..i * dim + dim];
            let (b, h, ok) = eval_barrier(
                &self.barriers[i],
                mode,
                dim,
                self.cfg.k0,
                self.cfg.k1,
                self.dt,
                p,
                v,
                row,
            );
            self.bvec[i] = b;
            min_h = min_h.min(h);
            degenerate |= !ok;
        }
        if degenerate {
            return (false, min_h);
        }

        // Dykstra projection of x0 onto  (∩ half-spaces) ∩ K(mode).
        self.dykstra(mode, p, x0);
        action.clear();
        action.extend_from_slice(&self.x[..dim]);

        (self.certify(mode, p), min_h)
    }

    /// Hard certification: the returned command must satisfy every barrier row, lie inside the
    /// mode's kinematic set, **and** (for the kinematic modes) place the realised next pose inside
    /// the exact safe set. No positive certificate ⇒ the arbiter falls back (guard.md §9.1).
    fn certify(&self, mode: ControlMode, p: &[f64]) -> bool {
        let dim = self.dim;
        let tol = self.cfg.tol;

        for i in 0..self.barriers.len() {
            let row = &self.amat[i * dim..i * dim + dim];
            if dot(row, &self.x[..dim]) < self.bvec[i] - tol {
                return false;
            }
        }

        // The kinematic set K(mode).
        match mode {
            ControlMode::Effort => {
                for &xi in &self.x[..dim] {
                    if !xi.is_finite() || xi.abs() > self.u_max + tol {
                        return false;
                    }
                }
            }
            ControlMode::Velocity => {
                if !norm(&self.x[..dim]).is_finite() || norm(&self.x[..dim]) > self.v_max + tol {
                    return false;
                }
            }
            ControlMode::Position => {
                let mut d2 = 0.0;
                for i in 0..dim {
                    if !self.x[i].is_finite() {
                        return false;
                    }
                    d2 += (self.x[i] - p[i]).powi(2);
                }
                if d2.sqrt() > self.step_max + tol {
                    return false;
                }
            }
        }

        // The exact one-step safe-set check (the kinematic modes' forward-invariance certificate).
        // `Effort`'s certificate is the HOCBF row itself (relative degree 2 — the next *position*
        // is not a function of the commanded acceleration alone over one step).
        let next = |i: usize| match mode {
            ControlMode::Velocity => p[i] + self.x[i] * self.dt,
            ControlMode::Position => self.x[i],
            ControlMode::Effort => p[i],
        };
        if mode != ControlMode::Effort {
            for barrier in &self.barriers {
                if barrier.static_h(dim, next) < -tol {
                    return false;
                }
            }
        }
        true
    }

    /// Dykstra's alternating-projection algorithm: projects `x0` onto the intersection of the
    /// mode's kinematic set and the `m` constraint half-spaces. Converges to the exact projection
    /// when the intersection is non-empty; leaves a detectable residual when it is empty (→ the
    /// certification step rejects and the arbiter falls back). Fixed iteration budget.
    fn dykstra(&mut self, mode: ControlMode, p: &[f64], x0: &[f64]) {
        let dim = self.dim;
        let m = self.barriers.len();

        self.x[..dim].copy_from_slice(&x0[..dim]);
        for c in self.corrections.iter_mut() {
            *c = 0.0;
        }

        for _ in 0..self.cfg.max_iter {
            self.xprev[..dim].copy_from_slice(&self.x[..dim]);

            // Set 0: the mode's kinematic set K (box for Effort, ball for Velocity/Position).
            let box_corr = 0;
            for i in 0..dim {
                self.y[i] = self.x[i] + self.corrections[box_corr * dim + i];
            }
            self.project_kinematic(mode, p);
            for i in 0..dim {
                self.corrections[box_corr * dim + i] = self.y[i] - self.x[i];
            }

            // Sets 1..=m: the constraint half-spaces a·x ≥ b.
            for j in 0..m {
                let corr = j + 1;
                for i in 0..dim {
                    self.y[i] = self.x[i] + self.corrections[corr * dim + i];
                }
                let row = &self.amat[j * dim..j * dim + dim];
                let ay = dot(row, &self.y[..dim]);
                let nn = dot(row, row);
                if ay < self.bvec[j] && nn > 0.0 {
                    let t = (self.bvec[j] - ay) / nn;
                    for i in 0..dim {
                        let proj = self.y[i] + t * row[i];
                        self.corrections[corr * dim + i] = self.y[i] - proj;
                        self.x[i] = proj;
                    }
                } else {
                    for i in 0..dim {
                        self.corrections[corr * dim + i] = 0.0;
                        self.x[i] = self.y[i];
                    }
                }
            }

            // Convergence: stop once the iterate is **feasible** *and* has stopped moving.
            //
            // "Stopped moving" alone is not convergence — Dykstra's tail can crawl, and the
            // iterate then sits *infeasible* with a per-cycle step below any stall threshold.
            // Breaking there silently hands the certifier an infeasible point, which it (correctly)
            // rejects, so the shield falls back on a problem it could in fact have solved: the
            // policy gets vetoed instead of *minimally corrected*, and the shield is needlessly
            // conservative rather than needlessly permissive. Requiring feasibility to break costs
            // nothing on the fast path (a feasible iterate breaks on the first stall) and spends the
            // remaining fixed budget only when the solve is actually hard. The budget itself is
            // unchanged, so worst-case work — the static latency guarantee — is untouched.
            let mut delta = 0.0;
            for i in 0..dim {
                delta += (self.x[i] - self.xprev[i]).powi(2);
            }
            if delta < self.cfg.tol * self.cfg.tol && self.rows_feasible() {
                break;
            }
        }
    }

    /// Whether the current iterate satisfies every barrier row (within `tol`). The cheap half of
    /// [`Shield::certify`] — the kinematic set is a fixed point of the cycle's first projection, so
    /// only the rows can be violated at a stall.
    fn rows_feasible(&self) -> bool {
        let dim = self.dim;
        (0..self.barriers.len()).all(|i| {
            let row = &self.amat[i * dim..i * dim + dim];
            dot(row, &self.x[..dim]) >= self.bvec[i] - self.cfg.tol
        })
    }

    /// Project the working point `y` onto the mode's kinematic set, writing the result into `x`.
    /// `Effort` clamps each axis into the acceleration box; `Velocity` projects onto the speed ball
    /// `‖w‖ ≤ v_max`; `Position` projects onto the step ball `‖q − p‖ ≤ v_max·dt` about the pose.
    fn project_kinematic(&mut self, mode: ControlMode, p: &[f64]) {
        let dim = self.dim;
        match mode {
            ControlMode::Effort => {
                for i in 0..dim {
                    self.x[i] = project_box_axis(self.y[i], self.u_max);
                }
            }
            ControlMode::Velocity => {
                let scale = ball_scale(norm(&self.y[..dim]), self.v_max);
                for i in 0..dim {
                    self.x[i] = self.y[i] * scale;
                }
            }
            ControlMode::Position => {
                let mut d2 = 0.0;
                for i in 0..dim {
                    d2 += (self.y[i] - p[i]).powi(2);
                }
                let scale = ball_scale(d2.sqrt(), self.step_max);
                for i in 0..dim {
                    self.x[i] = p[i] + (self.y[i] - p[i]) * scale;
                }
            }
        }
    }
}

/// `min(configured, reviewed)` — a spec-authored limit tightens the configured ceiling and can
/// never loosen it. A non-finite or non-positive authored limit is ignored (the loader/compiler
/// reject those upstream; this is defence in depth).
fn tighten(configured: f64, authored: Option<f64>) -> f64 {
    match authored {
        Some(limit) if limit.is_finite() && limit > 0.0 => configured.min(limit),
        _ => configured,
    }
}

/// `configured ∩ reviewed` — the configured directive allowlist may only ever **narrow** the grant
/// the reviewed `SafetySpec` authored, and can never create a permission (RFC-0004 Amendment 2).
///
/// The permission-set sibling of [`tighten`], and it merges the **other way round on silence**.
/// Both are the greatest-lower-bound of `(configured, authored)` — "configuration may only tighten
/// the reviewed contract" is *one* rule — but the **identity element of the meet** differs:
///
/// - a scalar **ceiling** with no opinion is `+∞`, so `tighten(config, None) == config`;
/// - a **permission set** with no opinion is `∅`, so `narrow(config, None) == ∅`.
///
/// So an unauthored ceiling leaves the configured ceiling standing, while an unauthored grant admits
/// **nothing**. Reading `None` here as "the configuration stands" would be fail-open-by-silence:
/// every spec written before Amendment 2 is silent, so it would preserve exactly the unreviewed
/// config-only grants the amendment exists to revoke, and would make *adding* a permission
/// achievable by **deleting** a line from the safety contract.
///
/// Called **once**, at core construction — never on the hot path, which stays allocation-free
/// (guard.md §2 principle 6). The Kani harness `configuration_cannot_widen_the_authored_grant`
/// (`verify.rs`) discharges the invariant over a *symbolic* directive name.
pub(crate) fn narrow(configured: &[String], authored: Option<&[String]>) -> Vec<String> {
    match authored {
        // The model authored no grant: nothing is admissible, however permissive the config is.
        None => Vec::new(),
        Some(grant) => configured
            .iter()
            .filter(|name| grant.iter().any(|g| g == *name))
            .cloned()
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Regression: the subnormal-underflow counterexample Kani found.**
    ///
    /// The naive `radius / len` quotient underflows into the subnormal range for this pair, carries
    /// ~0.4 % relative error, and lets the "projected" command escape the ball — a soundness hole in
    /// the `VELOCITY` speed ceiling and the `POSITION` step cap that 400 randomized Clarabel
    /// cross-checks and the whole proptest suite both missed (`rust/src/verify.rs`).
    ///
    /// `ball_scale` must now detect it and collapse to the zero command, which is inside every ball
    /// and is the safe floor in every mode.
    #[test]
    fn ball_scale_rejects_the_subnormal_underflow() {
        let len = 1.5458150092069032e172_f64;
        let radius = 7.60750754582421e-152_f64;

        // The naive quotient really is unsound — this is the bug, reproduced.
        let naive = radius / len;
        assert!(
            len * naive > radius * (1.0 + 4.0 * f64::EPSILON),
            "the counterexample no longer escapes the ball — has the FPU changed?"
        );

        // The shipped kernel validates its own quotient and refuses it.
        let scale = ball_scale(len, radius);
        assert_eq!(
            scale, 0.0,
            "an unsound scale must collapse to the zero command"
        );
        assert!(
            len * scale <= radius,
            "the fallback command escaped the ball"
        );
    }

    /// An honest quotient is *not* collapsed by the validator's `4·ε` slack: an ordinary projection
    /// still lands exactly on the ball, so the fail-closed check costs no precision on the hot path.
    #[test]
    fn ball_scale_keeps_an_honest_quotient() {
        let scale = ball_scale(15.0, 0.5);
        assert!((15.0 * scale - 0.5).abs() < 1e-12, "scale = {scale}");
        // …and a feasible command is untouched (the "minimally perturb" property).
        assert_eq!(ball_scale(0.25, 0.5), 1.0);
    }

    fn v(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    /// **Silence grants nothing.** The load-bearing half of RFC-0004 Amendment 2, and the deliberate
    /// departure from `tighten`: an unauthored *ceiling* leaves the configured one standing, but an
    /// unauthored *permission set* admits nothing at all.
    #[test]
    fn narrow_admits_nothing_when_the_model_authored_nothing() {
        assert!(narrow(&v(&["a", "b"]), None).is_empty());
        // …whereas the ceiling's identity really is `+inf` — the configured value survives.
        assert_eq!(tighten(2.0, None), 2.0);
    }

    #[test]
    fn narrow_is_the_intersection() {
        assert_eq!(narrow(&v(&["b", "c"]), Some(&v(&["a", "b"]))), v(&["b"]));
        // Configured-but-unauthored (`c`) is dropped: configuration cannot create a permission.
        // Authored-but-unconfigured (`a`) is dropped too: a deployment may run stricter.
        assert!(narrow(&v(&["c"]), Some(&v(&["a"]))).is_empty());
        assert!(narrow(&[], Some(&v(&["a"]))).is_empty());
        assert!(narrow(&v(&["a"]), Some(&[])).is_empty());
    }

    /// The effective set is a subset of *both* inputs — the tighten-only invariant, as a unit test
    /// (Kani discharges it over every directive name in `verify.rs`).
    #[test]
    fn narrow_never_exceeds_either_input() {
        let configured = v(&["a", "b", "c"]);
        let authored = v(&["b", "c", "d"]);
        let effective = narrow(&configured, Some(&authored));
        assert_eq!(effective, v(&["b", "c"]));
        assert!(effective
            .iter()
            .all(|n| configured.contains(n) && authored.contains(n)));
    }
}
