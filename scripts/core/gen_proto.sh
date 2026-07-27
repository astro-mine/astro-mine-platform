#!/usr/bin/env bash
# Regenerate the Core protobuf bindings from the canonical .proto sources with buf
# (RM-P0-CORE-07). Replaces the grpcio-tools/protoc stopgap.
#
#   ./scripts/gen_proto.sh            # regenerate checked-in Python bindings (src/)
#   ./scripts/gen_proto.sh --langs    # also generate native C++/Rust/TS clients (codegen/)
#
# buf runs via npx (no system buf/protoc needed); the CLI and all remote plugins
# are version-pinned for reproducibility (CX-REPRO). See buf.gen.yaml / buf.gen.langs.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

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
