// SPDX-License-Identifier: Apache-2.0
//! Build script: generate the prost bindings for the CompiledSafetyModel wire form.
//!
//! The `.proto` sources are the *same* canonical files the Python side generates its
//! bindings from (`schemas/proto/...`, RM-P1-GUARD-01) — so the Rust safety core and the
//! Python compiler agree on the wire form byte-for-byte. Compilation goes through `protox`
//! (a pure-Rust protobuf compiler) rather than a system `protoc`, keeping the build hermetic
//! and reproducible (conventions.md §11). Generated code lands in `OUT_DIR` and is pulled in
//! by `model.rs`; it is never checked in.

use std::path::PathBuf;

fn main() {
    // astro-mine-platform: guard's proto sources live under schemas/guard/
    // (the platform root's schemas/proto belongs to core).
    let proto_root = PathBuf::from("../schemas/guard/proto");
    let proto = proto_root.join("astro_mine/guard/spec/_proto/compiled_safety_model.proto");
    // compiled_safety_model.proto imports Core's shared units.proto (RFC-0007), vendored under the
    // same proto root so protox resolves the cross-file import hermetically (no system protoc).
    let units_proto = proto_root.join("astro_mine/core/units/_proto/units.proto");

    println!("cargo:rerun-if-changed={}", proto.display());
    println!("cargo:rerun-if-changed={}", units_proto.display());
    println!("cargo:rerun-if-changed=build.rs");

    let file_descriptors =
        protox::compile([&proto], [&proto_root]).expect("protox failed to compile the proto");

    prost_build::Config::new()
        .compile_fds(file_descriptors)
        .expect("prost-build failed to generate bindings");
}
