//! Rust client for the Astro-Mine-Core message catalog (RM-P0-CORE-07).
//!
//! Pulls in the prost-generated code for each Core interface. The generated files
//! (codegen/rust/generated, gitignored) are produced by `scripts/gen_proto.sh --langs`.
//! The modules are nested to mirror the proto package hierarchy
//! (`astro_mine.core.<comp>.v0`) so cross-package references — e.g. messages / mission
//! into the shared `units` vocabulary (RFC-0007) — resolve via prost's generated
//! `super::super::units::v0::...` paths. `include!` paths are relative to this file
//! (`src/`), so the module nesting does not change them. This crate exists to prove the
//! schemas compile against the prost runtime; it is not published.

pub mod astro_mine {
    pub mod core {
        pub mod sadf {
            pub mod v0 {
                include!("../generated/astro_mine/core/sadf/v0/astro_mine.core.sadf.v0.rs");
            }
        }
        pub mod objective {
            pub mod v0 {
                include!(
                    "../generated/astro_mine/core/objective/v0/astro_mine.core.objective.v0.rs"
                );
            }
        }
        pub mod units {
            pub mod v0 {
                include!("../generated/astro_mine/core/units/v0/astro_mine.core.units.v0.rs");
            }
        }
        pub mod messages {
            pub mod v0 {
                include!("../generated/astro_mine/core/messages/v0/astro_mine.core.messages.v0.rs");
            }
        }
        pub mod mission {
            pub mod v0 {
                include!("../generated/astro_mine/core/mission/v0/astro_mine.core.mission.v0.rs");
            }
        }
    }
}
