#!/usr/bin/env python3
"""Run the platform test suite the way each source repo ran its own.

Every component repo had its own default pytest selection (addopts). A single
rootdir has a single [tool.pytest.ini_options], so this runner re-applies each
component's original defaults when running that component's suite:

    python scripts/test.py                 # all components, original defaults each
    python scripts/test.py sim worlds      # a subset
    python scripts/test.py sim -- -k dem   # extra pytest args after --

CI calls this per component; `pytest tests/<comp>` directly is equivalent for
every component except the ones listed in COMPONENT_ADDOPTS below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMPONENTS = [
    "core", "spice", "seal", "worlds", "prospect", "link", "fleet", "sim",
    "bench", "cloud", "surrogate", "mind", "learn", "allocate", "guard",
    "hub", "studio",
    # Not a component: the platform-wide layering suite (conventions.md §11), which asserts
    # the tier and inversion rules no single component's directory could own.
    "platform",
]

# Verbatim from each source repo's [tool.pytest.ini_options] addopts (beyond "-ra"), plus the
# deselections a source repo kept in its CI workflow rather than its addopts. Keep this table and
# the `marks:` column of .github/workflows/ci.yml's matrix in step: they are the same selection
# expressed twice, and when they drifted the opt-in `container` test ran in CI and failed.
COMPONENT_ADDOPTS: dict[str, list[str]] = {
    "sim": ["-m", "not gpu and not ray and not docker"],
    "allocate": ["-m", "not scale"],
    # bench#30's real-container round trip needs a locally built evaluation-runner image; with a
    # Docker daemon on PATH its skipif does not fire, so it must be deselected, not left to skip.
    "bench": ["-m", "not cluster and not gpu and not container"],
}


def main() -> int:
    args = sys.argv[1:]
    extra: list[str] = []
    if "--" in args:
        i = args.index("--")
        args, extra = args[:i], args[i + 1:]
    comps = args or COMPONENTS
    unknown = [c for c in comps if c not in COMPONENTS]
    if unknown:
        sys.exit(f"unknown component(s): {', '.join(unknown)}")

    failed: list[str] = []
    for comp in comps:
        tests = ROOT / "tests" / comp
        if not tests.is_dir():
            continue
        cmd = [sys.executable, "-m", "pytest", str(tests), *COMPONENT_ADDOPTS.get(comp, []), *extra]
        print(f"\n=== {comp}: {' '.join(cmd[2:])}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(comp)

    if failed:
        print(f"\nFAILED components: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll selected component suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
