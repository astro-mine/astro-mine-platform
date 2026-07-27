#!/usr/bin/env bash
# Regenerate the field-service protobuf + gRPC bindings from the canonical .proto sources with buf
# (RM-P1-PROSPECT-11). Mirrors astro-mine-allocate/scripts/gen_proto.sh.
#
#   ./scripts/gen_proto.sh            # regenerate checked-in Python bindings (src/)
#
# buf runs via npx (no system buf/protoc needed); the CLI and all remote plugins are
# version-pinned for reproducibility (CX-REPRO). See buf.gen.yaml / buf.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

BUF="npx -y @bufbuild/buf@1.71.0"

# field_service.proto imports Core's units.proto (RFC-0007), vendored under schemas/proto for import
# resolution only. Generation is scoped to field_service.proto with --path so buf resolves the units
# import for compilation but does NOT emit a second `units_pb2` — the generated binding imports
# Core's installed astro_mine.core.units._proto.units_pb2 instead, keeping one descriptor in the pool.
FIELD_SERVICE="schemas/proto/astro_mine/prospect/service/_proto/field_service.proto"

echo "buf lint ..."
$BUF lint

echo "Generating Python bindings + gRPC stubs (src/) ..."
$BUF generate --template buf.gen.yaml --path "$FIELD_SERVICE"

echo "Generated: src/astro_mine/prospect/service/_proto/field_service_pb2.py(+.pyi, +_pb2_grpc.py)."
