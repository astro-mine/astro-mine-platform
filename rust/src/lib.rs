// SPDX-License-Identifier: Apache-2.0
//! Astro-Mine-Guard trusted safety core (the TCB) — RM-P1-GUARD-02.
//!
//! The small, deterministic, allocation-free (on the hot path) Rust core that actually
//! enforces the safety guarantee. It composes defense-in-depth (guard.md §9.2):
//!
//! - [`monitors`] — STL/MTL runtime monitors (**detect**), online + predictive;
//! - [`shield`] — a CBF-QP safety filter (**correct**);
//! - [`backup`] — verified simplex backup controllers (**recover**);
//! - [`arbiter`] — the decision core that certifies exactly one action per tick under strict
//!   precedence, fail-safe and never fail-open;
//! - [`model`] — the spec evaluator: decode + validate the compiled `CompiledSafetyModel`.
//!
//! Everything outside this crate is untrusted. The core runs standalone on the edge (no
//! Python); the optional `python` feature adds the PyO3 binding that Python orchestration
//! calls in-process (guard.md §3, §4).

pub mod arbiter;
pub mod backup;
pub mod model;
pub mod monitors;
pub mod shield;
pub mod units;

/// Kani proof harnesses for the TCB's load-bearing kernels (guard.md §9.3). Compiled only under
/// `cargo kani`, so the shipped core carries none of it.
#[cfg(kani)]
mod verify;

/// Prost-generated bindings for the canonical `CompiledSafetyModel` wire form. Generated at
/// build time from the same `.proto` the Python side uses (see `build.rs`).
///
/// `compiled_safety_model.proto` imports Core's shared `units.proto` (RFC-0007), so prost emits
/// two packages: the Guard `astro_mine.guard.spec.v0` types and the Core
/// `astro_mine.core.units.v0` vocabulary they reference. The generated Guard code refers to the
/// units types by `super::super::super::core::units::v0::…`, so the module tree is reconstructed
/// here; the Guard `v0` types are then re-exported at the `proto` root so the rest of the crate
/// keeps addressing them as `proto::CompiledSafetyModel` etc.
pub(crate) mod proto {
    pub mod core {
        pub mod units {
            // Core's shared units vocabulary (RFC-0007). Guard's compiled model references only
            // `ReferenceFrame` (via `frame_ref`); the sibling `Epoch`/`EpochWindow`/`PlanetaryCRS`
            // types come along with the imported file but are unused here, so silence dead-code.
            #[allow(dead_code)]
            pub mod v0 {
                include!(concat!(env!("OUT_DIR"), "/astro_mine.core.units.v0.rs"));
            }
        }
    }
    pub mod guard {
        pub mod spec {
            pub mod v0 {
                include!(concat!(env!("OUT_DIR"), "/astro_mine.guard.spec.v0.rs"));
            }
        }
    }
    pub use guard::spec::v0::*;
}

pub use arbiter::{
    ActionPolicy, CoreConfig, Intervention, Layer, ProposedAction, Reason, SafetyCore, SafetyInput,
    SafetyVerdict,
};
pub use backup::BackupKind;
pub use model::{ActionLimits, AdmissibleDirectives, CoreError};
pub use shield::{ControlMode, ShieldConfig};

#[cfg(feature = "python")]
mod py;
