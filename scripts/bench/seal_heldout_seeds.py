#!/usr/bin/env python3
"""Seal / verify an embargoed held-out seed set for a zoo scenario (RM-P0-BENCH-02).

The held-out seeds are the anti-gaming reserve (bench.md §9): committed to this *private* repo but
excluded from the packaged zoo artifact (the wheel ships only ``src/astro_mine``), and bound into
the public ``ScenarioSpec`` by a ``heldout_commit`` — a sha256 over the sealed ``{salt, seeds}``
payload — so the seeds influence the spec hash without being disclosed.

    python scripts/seal_heldout_seeds.py <scenario-id>
    python scripts/seal_heldout_seeds.py <scenario-id> --verify sha256:<hex>

The commitment formula mirrors ``astro_mine.bench.scenario._hash.content_hash`` (canonical,
key-sorted, compact JSON + sha256); it is replicated here with the standard library only, so the
script runs without the project environment installed.

SECURITY: rotate these seeds before the repo flips public (see ``embargo/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# `parents[2]`, not `parents[1]`: this script sits at `scripts/bench/`, so the repo root — where
# `embargo/` lives, deliberately outside `src/` so the wheel cannot ship it — is two levels up.
EMBARGO_ROOT = Path(__file__).resolve().parents[2] / "embargo"


def content_hash(payload: object) -> str:
    """A deterministic ``sha256:<hex>`` over the canonical JSON of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seal(scenario_id: str) -> str:
    """The ``heldout_commit`` over the sealed seed payload for ``scenario_id``."""
    path = EMBARGO_ROOT / scenario_id / "heldout_seeds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return content_hash(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal/verify an embargoed held-out seed set.")
    parser.add_argument("scenario_id", help="zoo scenario id, e.g. lunar-polar-ice-prospecting-v1")
    parser.add_argument(
        "--verify", metavar="COMMIT", help="assert the commitment equals this value"
    )
    args = parser.parse_args(argv)

    commit = seal(args.scenario_id)
    if args.verify is None:
        print(commit)
        return 0

    if commit == args.verify:
        print(f"OK: {args.scenario_id} heldout_commit == {commit}")
        return 0
    print(f"MISMATCH: computed {commit} != expected {args.verify}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
