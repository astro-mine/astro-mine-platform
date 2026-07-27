# Protobuf schema sources

Canonical Protobuf (proto3) sources for the Core message catalog and the SADF wire
form. Multi-language bindings (Python, C++, Rust, TypeScript) are generated from
these with [`buf`](https://buf.build/) via `scripts/gen_proto.sh` (RM-P0-CORE-07):

- **Python** bindings are checked in under `src/astro_mine/core/*/_proto/` and kept
  in sync by a CI freshness gate (regenerate-and-diff).
- **C++/Rust/TypeScript** clients are build-only: generated into `codegen/` (gitignored)
  and compiled in CI / the conda dev env to prove the schemas build in every target
  language. See [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

CI also runs `buf lint` and a `buf breaking` gate against `main`. Per-tick hot-path
messages use Cap'n Proto instead (see `docs/architecture/conventions.md`).

These `.proto` sources are bundled with the JSON/Cap'n Proto schemas into the
content-addressed schema bundle published to private GHCR (RM-P0-CORE-08); see
[`../json/README.md`](../json/README.md) and `docs/VERSIONING.md` §5.
