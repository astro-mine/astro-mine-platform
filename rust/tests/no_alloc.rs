//! Acceptance: **no hot-path allocation** + **bounded latency** (guard.md §2 principle 6, §8).
//!
//! A counting global allocator proves `step_into` performs **zero** heap allocations per tick
//! once the reused buffers are warm, and the mean per-tick latency of the CBF-QP + monitor
//! path stays well under the ≤ ~1 ms single-agent design target.
//!
//! This is the **only** test in its binary on purpose so nothing else drives the core here.
//! The allocation counter is **thread-scoped** (a `#[thread_local]` `Cell`, initialised with a
//! `const` so reading it never itself allocates): it counts allocations on the measuring thread
//! only, so incidental heap traffic from the test harness's own bookkeeping threads on a busy CI
//! runner can never pollute the count. That isolation is what lets the assertion stay exactly
//! `== 0` — the guarantee we actually want — instead of a fuzzy tolerance.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use astro_mine_guard_core::{CoreConfig, ProposedAction, SafetyCore, SafetyInput};
use common::*;

thread_local! {
    // `const` init ⇒ a plain `#[thread_local]` static: reading/updating it is a direct memory
    // access with no lazy registration, so the allocator hook below never re-enters the allocator.
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn thread_allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        System.dealloc(ptr, l)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn hot_path_is_allocation_free_and_fast() {
    let mut core = SafetyCore::from_model(combined_model(), CoreConfig::default()).unwrap();
    let mut out = core.new_verdict();

    // Mutable stack buffers reused every tick — no allocation from the driver either.
    let mut signals = [18.0f64, 130.0];
    let mut position = [12.0f64, 0.0, 0.0];
    let mut velocity = [0.1f64, 0.0, 0.0];
    let mut proposed = [-3.0f64, 0.5, 0.0];

    let iters = 20_000usize;

    // Warm-up pass: run the *full* varying trajectory once, uncounted, so every branch the
    // measured pass will reach (shield correction, backup entry, monitor windows filling) has
    // already done any one-time lazy sizing. A constant-input hold would leave those paths cold
    // and let their first-touch allocation land inside the counted window.
    for k in 0..iters {
        let t = k as f64 * 1e-3;
        signals[0] = 18.0 + t.sin();
        signals[1] = 130.0 + (0.5 * t).cos();
        position[0] = 12.0 + 0.4 * (0.3 * t).cos();
        velocity[0] = -0.3 * (0.3 * t).sin();
        proposed[0] = -3.0 - t.cos();
        core.step_into(
            &SafetyInput {
                signals: &signals,
                position: &position,
                velocity: &velocity,
                proposed: ProposedAction::Effort(&proposed),
            },
            &mut out,
        );
    }

    // Measured pass: the same trajectory again, now counting this thread's allocations only.
    let before = thread_allocs();
    let start = Instant::now();
    for k in 0..iters {
        let t = k as f64 * 1e-3;
        signals[0] = 18.0 + t.sin();
        signals[1] = 130.0 + (0.5 * t).cos();
        position[0] = 12.0 + 0.4 * (0.3 * t).cos();
        velocity[0] = -0.3 * (0.3 * t).sin();
        proposed[0] = -3.0 - t.cos();
        core.step_into(
            &SafetyInput {
                signals: &signals,
                position: &position,
                velocity: &velocity,
                proposed: ProposedAction::Effort(&proposed),
            },
            &mut out,
        );
    }
    let elapsed = start.elapsed();
    let allocs = thread_allocs() - before;

    assert_eq!(
        allocs, 0,
        "hot path allocated {allocs} times in {iters} ticks"
    );

    let per_tick = elapsed / iters as u32;
    println!("mean per-tick latency: {per_tick:?} over {iters} ticks (debug build)");
    assert!(
        per_tick < Duration::from_millis(1),
        "mean per-tick latency {per_tick:?} exceeds the 1 ms single-agent target"
    );
}
