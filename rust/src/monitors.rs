//! STL/MTL runtime monitors — the *detect* layer (guard.md §9.2).
//!
//! Each temporal clause of the [`CompiledSafetyModel`](crate::model::CompiledSafetyModel) is
//! evaluated online with **robust semantics** (a signed margin to violation, not a bare
//! bool), **incrementally** (one tick at a time), and in **bounded memory** (fixed ring
//! buffers sized at load time from the compiled `history_window_len`). Temporal operators use
//! the online-computable **bounded past-time** robust semantics (the `historically`/`once`/
//! `since` mirror of `always`/`eventually`/`until`) — the standard choice for online runtime
//! verification, because future-time robustness is not computable from the samples seen so
//! far. On top of that, each monitor reports a **predictive time-to-violation**: extrapolating
//! the robustness trend flags a property that is *about to* be violated in time for the
//! arbiter to act under latency (guard.md §9.2 "predictive monitoring").
//!
//! Nothing here allocates after construction: the evaluation plan is a flat array walked in
//! post-order, and every ring buffer is pre-sized.

use crate::model::{CompiledNode, MonitorAutomaton, OnUncertain, TemporalOp};

/// A fixed-capacity ring buffer of past robustness values. `push` overwrites the oldest slot
/// and never reallocates; `ago(k)` reads the value `k` ticks ago (`ago(0)` = most recent).
#[derive(Debug, Clone)]
struct RingBuf {
    data: Vec<f64>,
    cap: usize,
    head: usize,
    len: usize,
}

impl RingBuf {
    fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            data: vec![0.0; cap],
            cap,
            head: 0,
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, v: f64) {
        self.data[self.head] = v;
        self.head = (self.head + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }

    #[inline]
    fn ago(&self, k: usize) -> Option<f64> {
        if k >= self.len {
            None
        } else {
            let idx = (self.head + self.cap - 1 - k) % self.cap;
            Some(self.data[idx])
        }
    }
}

/// One node of the flattened, post-ordered evaluation plan.
#[derive(Debug, Clone)]
struct NodeEval {
    op: TemporalOp,
    children: Vec<usize>,
    predicate_index: Option<usize>,
    lo: usize,
    hi: usize,
    buf_a: RingBuf,
    buf_b: RingBuf,
    cur: f64,
}

/// The online state of a single compiled monitor.
#[derive(Debug, Clone)]
pub struct Monitor {
    pub constraint_id: String,
    pub on_uncertain: OnUncertain,
    nodes: Vec<NodeEval>,
    root: usize,
    prev_root: Option<f64>,
}

/// The per-tick verdict of a single monitor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorVerdict {
    /// Signed robustness of the whole clause (`>= 0` ⇒ currently satisfied).
    pub robustness: f64,
    /// The clause is currently violated (`robustness < 0`).
    pub violated: bool,
    /// The clause still holds but is trending to violation within the predictive horizon.
    pub predicted: bool,
    /// Extrapolated ticks until robustness crosses zero, if it is decreasing.
    pub time_to_violation_samples: Option<f64>,
}

impl MonitorVerdict {
    /// The monitor has *fired* — violated now, or predicted to be imminently. Either forces
    /// the arbiter down the constraint's fail-safe path (guard.md §9.2 precedence).
    #[inline]
    pub fn fired(&self) -> bool {
        self.violated || self.predicted
    }
}

fn flatten(node: &CompiledNode, out: &mut Vec<NodeEval>) -> usize {
    // Post-order: children are pushed before the parent, so a single forward pass over
    // `out` evaluates every child before its parent.
    let children: Vec<usize> = node.args.iter().map(|a| flatten(a, out)).collect();
    let hi = node.interval_hi_samples.unwrap_or(0);
    let lo = node.interval_lo_samples.unwrap_or(0);
    // A window operator needs to retain the value from `hi` ticks ago, i.e. `hi + 1` slots.
    let cap = hi.saturating_add(1);
    let idx = out.len();
    out.push(NodeEval {
        op: node.op,
        children,
        predicate_index: node.predicate_index,
        lo,
        hi,
        buf_a: RingBuf::new(cap),
        buf_b: RingBuf::new(cap),
        cur: 0.0,
    });
    idx
}

impl Monitor {
    pub fn new(spec: &MonitorAutomaton) -> Self {
        let mut nodes = Vec::new();
        let root = flatten(&spec.root, &mut nodes);
        Self {
            constraint_id: spec.constraint_id.clone(),
            on_uncertain: spec.on_uncertain,
            nodes,
            root,
            prev_root: None,
        }
    }

    /// Advance the monitor by one tick. `predicate_robustness(atom_index)` supplies the signed
    /// robustness of each atomic predicate this tick. Bounded, allocation-free work.
    pub fn step(
        &mut self,
        predicate_robustness: &dyn Fn(usize) -> f64,
        predictive_horizon: usize,
    ) -> MonitorVerdict {
        for i in 0..self.nodes.len() {
            let cur = self.eval_node(i, predicate_robustness);
            self.nodes[i].cur = cur;
        }
        let robustness = self.nodes[self.root].cur;
        let violated = robustness < 0.0;

        // Predictive time-to-violation: linear extrapolation of the robustness trend. Only
        // meaningful while the clause still holds and its margin is shrinking.
        let mut ttv = None;
        let mut predicted = false;
        if let Some(prev) = self.prev_root {
            let slope = robustness - prev; // change per tick
            if !violated && slope < 0.0 {
                let samples = robustness / -slope;
                ttv = Some(samples);
                if samples <= predictive_horizon as f64 {
                    predicted = true;
                }
            }
        }
        self.prev_root = Some(robustness);

        MonitorVerdict {
            robustness,
            violated,
            predicted,
            time_to_violation_samples: ttv,
        }
    }

    fn eval_node(&mut self, i: usize, pred: &dyn Fn(usize) -> f64) -> f64 {
        let op = self.nodes[i].op;
        match op {
            TemporalOp::Predicate => pred(self.nodes[i].predicate_index.expect("validated")),
            TemporalOp::Not => -self.child_cur(i, 0),
            TemporalOp::And => self.reduce_children(i, f64::INFINITY, f64::min),
            TemporalOp::Or => self.reduce_children(i, f64::NEG_INFINITY, f64::max),
            TemporalOp::Always => {
                let v = self.child_cur(i, 0);
                self.nodes[i].buf_a.push(v);
                self.window_reduce(i, true)
            }
            TemporalOp::Eventually => {
                let v = self.child_cur(i, 0);
                self.nodes[i].buf_a.push(v);
                self.window_reduce(i, false)
            }
            TemporalOp::Until => {
                let phi = self.child_cur(i, 0);
                let psi = self.child_cur(i, 1);
                self.nodes[i].buf_a.push(phi);
                self.nodes[i].buf_b.push(psi);
                self.until_reduce(i)
            }
        }
    }

    #[inline]
    fn child_cur(&self, i: usize, which: usize) -> f64 {
        self.nodes[self.nodes[i].children[which]].cur
    }

    fn reduce_children(&self, i: usize, init: f64, f: fn(f64, f64) -> f64) -> f64 {
        self.nodes[i]
            .children
            .iter()
            .fold(init, |acc, &c| f(acc, self.nodes[c].cur))
    }

    /// Bounded past-time `always` (min) / `eventually` (max) over the window `[lo, hi]`.
    /// An empty window is vacuous: `always` ⇒ +∞ (satisfied), `eventually` ⇒ −∞.
    fn window_reduce(&self, i: usize, is_min: bool) -> f64 {
        let (lo, hi) = (self.nodes[i].lo, self.nodes[i].hi);
        let mut acc = if is_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let mut any = false;
        for k in lo..=hi {
            if let Some(v) = self.nodes[i].buf_a.ago(k) {
                any = true;
                acc = if is_min { acc.min(v) } else { acc.max(v) };
            }
        }
        if any {
            acc
        } else if is_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Bounded past-time `until`: `max_{k in [lo,hi]} min( ρ_ψ(t-k), min_{0<=j<k} ρ_φ(t-j) )`
    /// — "ψ became true within the window and φ held from then until now."
    fn until_reduce(&self, i: usize) -> f64 {
        let (lo, hi) = (self.nodes[i].lo, self.nodes[i].hi);
        let node = &self.nodes[i];
        let mut best = f64::NEG_INFINITY;
        let mut phi_run = f64::INFINITY; // min of φ over [t-(k-1), t]
        for k in 0..=hi {
            // phi_run covers ρ_φ from now back to k-1 samples ago.
            if k >= lo {
                if let Some(psi_k) = node.buf_b.ago(k) {
                    best = best.max(psi_k.min(phi_run));
                }
            }
            if let Some(phi_k) = node.buf_a.ago(k) {
                phi_run = phi_run.min(phi_k);
            }
        }
        best
    }
}
