#!/usr/bin/env bash
# Regenerate the Guard protobuf bindings from the canonical .proto sources with buf
# (RM-P1-GUARD-01). Mirrors astro-mine-core / astro-mine-surrogate scripts/gen_proto.sh.
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

# safety_spec.proto / compiled_safety_model.proto import the shared Core units.proto (RFC-0007).
# buf's Python plugin also emits bindings for that imported file; Guard reuses the installed
# astro-mine-core copy at runtime (the generated Guard bindings `import astro_mine.core.units._proto.units_pb2`),
# so drop the duplicate here — committing it would double-register units.proto in the descriptor pool.
rm -rf src/astro_mine/core

echo "Generated: src/astro_mine/guard/spec/_proto/{safety_spec,compiled_safety_model}_pb2.py(+.pyi)."
