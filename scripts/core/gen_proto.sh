#!/usr/bin/env bash
# Regenerate the Core protobuf bindings from the canonical .proto sources with buf
# (RM-P0-CORE-07). Replaces the grpcio-tools/protoc stopgap.
#
#   ./scripts/core/gen_proto.sh            # regenerate checked-in Python bindings (src/)
#   ./scripts/core/gen_proto.sh --langs    # also generate native C++/Rust/TS clients (codegen/)
#
# buf runs via npx (no system buf/protoc needed); the CLI and all remote plugins
# are version-pinned for reproducibility (CX-REPRO). See buf.gen.yaml / buf.gen.langs.yaml.
set -euo pipefail
# `../..`, not `..`: consolidation moved this from `scripts/` to `scripts/core/` and left the
# hop count behind, so it landed in `scripts/` -- a directory with no `buf.yaml`. buf then fell
# back to a default module at `.` and reported `Module "path: "." " had no .proto files`, which
# names the symptom and not one word of the cause. Its only caller is `publish-schemas.yml`,
# which had never run, so nothing exercised it until 2026-08-18.
cd "$(dirname "$0")/../.."

BUF="npx -y @bufbuild/buf@1.71.0"

echo "buf lint ..."
$BUF lint

echo "Generating Python bindings (src/) ..."
$BUF generate --template buf.gen.yaml

if [[ "${1:-}" == "--langs" ]]; then
  echo "Generating native C++/Rust/TS clients (codegen/) ..."
  $BUF generate --template buf.gen.langs.yaml
  echo "Generated: src/**/_proto (Python) + codegen/{cpp,rust,ts}/generated"
else
  echo "Generated: src/**/_proto (Python). Pass --langs for C++/Rust/TS."
fi
