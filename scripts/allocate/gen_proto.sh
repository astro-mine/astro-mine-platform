#!/usr/bin/env bash
# Regenerate the Allocation IR protobuf bindings from the canonical .proto sources with buf
# (RM-P1-ALLOC-01). Mirrors astro-mine-surrogate/scripts/gen_proto.sh.
#
#   ./scripts/gen_proto.sh            # regenerate checked-in Python bindings (src/)
#
# buf runs via npx (no system buf/protoc needed); the CLI and all remote plugins are
# version-pinned for reproducibility (CX-REPRO). See buf.gen.yaml / buf.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

BUF="npx -y @bufbuild/buf@1.71.0"

echo "buf lint ..."
$BUF lint

echo "Generating Python bindings (src/) ..."
$BUF generate --template buf.gen.yaml

echo "Generated: src/astro_mine/allocate/model/ir/_proto/allocation_ir_pb2.py(+.pyi)."
