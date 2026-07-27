#!/usr/bin/env python
"""Regenerate the EnvironmentService Protobuf/gRPC stubs (sim.md §3; conventions.md §3).

The generated modules are **committed** (so the package installs with no ``protoc`` in the
toolchain), and ``tests/test_service.py`` gates them against drift from the ``.proto``. Run this
after editing ``environment.proto``:

    uv run python scripts/gen_proto.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = _REPO_ROOT / "src" / "astro_mine" / "sim" / "service" / "_proto"


def main() -> int:
    proto = PROTO_DIR / "environment.proto"
    argv = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={PROTO_DIR}",
        f"--pyi_out={PROTO_DIR}",
        f"--grpc_python_out={PROTO_DIR}",
        str(proto),
    ]
    subprocess.run(argv, check=True)
    # protoc emits `import environment_pb2` (a bare module name); rewrite it to the package-relative
    # form so the generated gRPC stub imports correctly from inside astro_mine.sim.service._proto.
    grpc_module = PROTO_DIR / "environment_pb2_grpc.py"
    text = grpc_module.read_text(encoding="utf-8")
    grpc_module.write_text(
        text.replace(
            "import environment_pb2 as environment__pb2",
            "from astro_mine.sim.service._proto import environment_pb2 as environment__pb2",
        ),
        encoding="utf-8",
    )
    print(f"generated stubs in {PROTO_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
