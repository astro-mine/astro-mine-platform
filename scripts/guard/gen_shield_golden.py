#!/usr/bin/env python
"""Regenerate the PolicyShield determinism golden (RM-P1-GUARD-03/-06).

Drives the shared deterministic fixture (``tests/shield_fixture.py``) through a ``PolicyShield`` and
pins the per-tick :class:`~astro_mine.guard.audit.model.SafetyVerdict` **provenance** sequence — the
certified actions, layers, reasons, invoked constraint ids, and content-hash provenance, *excluding*
the wall-clock ``shield_latency_us`` — plus its aggregate content hash, in
``tests/golden/shield_run.json``.

This is a **reviewed safety artifact**: the shield's certify/correct/fall-back behaviour and its
verdict provenance are reproducible byte-for-byte (the seeded golden gate, guard.md §9.3, §10). Only
regenerate it for an intentional change and review the diff.

Usage: ``uv run python scripts/gen_shield_golden.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from astro_mine.core.hashing import canonical_json, content_hash_json

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "shield_run.json"


def main() -> None:
    # The fixture lives under tests/ (a reviewed artifact, not shipped in the package).
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.shield_fixture import run_shield

    provenance = [v.provenance() for v in run_shield()]
    document = {"run": provenance, "digest": content_hash_json(provenance)}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_bytes(canonical_json(document) + b"\n")
    print(
        f"wrote {GOLDEN.relative_to(ROOT)} "
        f"({GOLDEN.stat().st_size} bytes, {len(provenance)} verdicts)"
    )


if __name__ == "__main__":
    main()
