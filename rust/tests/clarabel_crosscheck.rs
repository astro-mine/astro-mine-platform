// SPDX-License-Identifier: Apache-2.0
//! Cross-validate the TCB's bespoke allocation-free CBF filter against **Clarabel** — the
//! Rust-native QP solver guard.md §11 recommends. For a half-space keep-out the shield QP is
//! `min ‖u − u₀‖² s.t. a·u ≥ b, |uᵢ| ≤ u_max`; we solve the same program with Clarabel and
//! confirm the tiny in-TCB solver reaches the same optimum. This lets the safety path keep a
//! minimal, no-allocation solver while proving it agrees with a reference optimizer.

mod common;

use astro_mine_guard_core::model::{ActionLimits, KeepOutTerm, OnUncertain, Shape};

/// The compiled sample period the kinematic modes use as their one-step horizon.
const DT: f64 = 0.05;
use astro_mine_guard_core::shield::{Shield, ShieldConfig};

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT::NonnegativeConeT,
};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn unit(mut v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// Solve `min ½‖u−u₀‖² s.t. a·u ≥ b, |uᵢ| ≤ u_max` with Clarabel; returns `Some(u*)` if solved.
fn clarabel_optimum(a: &[f64; 3], b: f64, u0: &[f64; 3], u_max: f64) -> Option<[f64; 3]> {
    // P = I, q = −u0.
    let p = CscMatrix::<f64>::identity(3);
    let q = [-u0[0], -u0[1], -u0[2]];

    // Constraints (all NonnegativeCone, i.e. Ax + s = bvec, s ≥ 0  ⇔  Ax ≤ bvec):
    //   row 0:  −a·u ≤ −b        (a·u ≥ b)
    //   rows 1..3:  uᵢ ≤ u_max
    //   rows 4..6: −uᵢ ≤ u_max
    // CSC, column-major; each column j has nonzeros in rows {0, 1+j, 4+j}.
    let colptr = vec![0, 3, 6, 9];
    let rowval = vec![0, 1, 4, 0, 2, 5, 0, 3, 6];
    let nzval = vec![
        -a[0], 1.0, -1.0, //
        -a[1], 1.0, -1.0, //
        -a[2], 1.0, -1.0,
    ];
    let amat = CscMatrix::new(7, 3, colptr, rowval, nzval);
    let bvec = [-b, u_max, u_max, u_max, u_max, u_max, u_max];
    let cones = [NonnegativeConeT(7)];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .tol_gap_abs(1e-10)
        .tol_gap_rel(1e-10)
        .build()
        .unwrap();
    let mut solver = DefaultSolver::new(&p, &q, &amat, &bvec, &cones, settings);
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let x = &solver.solution.x;
            Some([x[0], x[1], x[2]])
        }
        _ => None,
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

#[test]
fn shield_matches_clarabel_optimum() {
    let cfg = ShieldConfig {
        max_iter: 4000,
        tol: 1e-13,
        ..Default::default()
    };
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mut compared = 0;

    for _ in 0..400 {
        let n = unit([rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)]);
        let offset = rng.r(-3.0, 3.0);
        let margin = rng.r(0.0, 3.0);
        let p = [rng.r(-8.0, 8.0), rng.r(-8.0, 8.0), rng.r(-8.0, 8.0)];
        let v = [rng.r(-4.0, 4.0), rng.r(-4.0, 4.0), rng.r(-4.0, 4.0)];
        let u0 = [rng.r(-40.0, 40.0), rng.r(-40.0, 40.0), rng.r(-40.0, 40.0)];

        let term = KeepOutTerm {
            constraint_id: "c".into(),
            on_uncertain: OnUncertain::Fallback,
            shape: Shape::HalfSpace,
            margin_m: margin,
            center: vec![],
            half_extents: vec![],
            radius: None,
            normal: n.to_vec(),
            offset: Some(offset),
        };
        let mut shield = Shield::new(&[term], 3, cfg, DT, &ActionLimits::default());
        let mut action = Vec::new();
        let (certified, _h) = shield.solve(&p, &v, &u0, &mut action);
        if !certified {
            continue;
        }

        // Reconstruct a·u ≥ b (unit normal).
        let rhs = margin - offset;
        let h = dot(&n, &p) - rhs;
        let b = -(cfg.k1 * dot(&n, &v) + cfg.k0 * h);

        let Some(opt) = clarabel_optimum(&n, b, &u0, cfg.u_max) else {
            continue;
        };
        for i in 0..3 {
            assert!(
                (action[i] - opt[i]).abs() < 1e-3,
                "shield {:?} disagrees with Clarabel {:?} on axis {i}",
                action,
                opt
            );
        }
        compared += 1;
    }
    assert!(
        compared > 50,
        "too few feasible comparisons ({compared}) — test is not exercising the solver"
    );
}
