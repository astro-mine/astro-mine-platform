#!/usr/bin/env python3
"""Seal / verify an embargoed held-out seed set for a zoo scenario (RM-P0-BENCH-02).

The held-out seeds are the anti-gaming reserve (bench.md §9): held in the private
``astro-mine/embargo`` repository, never in this tree, and bound into the public ``ScenarioSpec``
by a ``heldout_commit`` — a sha256 over the sealed ``{salt, seeds}`` payload — so the seeds
influence the spec hash without being disclosed.

    python scripts/seal_heldout_seeds.py <scenario-id>
    python scripts/seal_heldout_seeds.py <scenario-id> --verify sha256:<hex>

The commitment formula mirrors ``astro_mine.bench.scenario._hash.content_hash`` (canonical,
key-sorted, compact JSON + sha256); it is replicated here with the standard library only, so the
script runs without the project environment installed.

SECURITY: the seeds live in the private ``astro-mine/embargo`` repository, not here. Point
``$ASTRO_MINE_BENCH_EMBARGO_ROOT`` at a checkout of it (see ``embargo/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# The store moved out of this repository entirely (astro-mine-platform#37): rotating in place would
# have republished the same seeds one commit later, because the flip publishes every commit rather
# than `HEAD`. `$ASTRO_MINE_BENCH_EMBARGO_ROOT` is the same seam
# `astro_mine.bench.leaderboard.resolve_embargo_root` reads, so the script and the evaluator agree
# on where the seeds are by construction rather than by convention.
#
# The repo-relative fallback is kept, and it now resolves to a directory holding only a README. That
# is deliberate: a reader who runs this without the variable set gets a `FileNotFoundError` naming a
# path whose README says where the seeds went, which is a better failure than a silent default.
EMBARGO_ROOT_ENV = "ASTRO_MINE_BENCH_EMBARGO_ROOT"


def embargo_root() -> Path:
    """Where the sealed sets live: ``$ASTRO_MINE_BENCH_EMBARGO_ROOT``, else the repo-relative
    path."""
    configured = os.environ.get(EMBARGO_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    # `parents[2]`, not `parents[1]`: this script sits at `scripts/bench/`, so the repo root is two
    # levels up.
    return Path(__file__).resolve().parents[2] / "embargo"


def content_hash(payload: object) -> str:
    """A deterministic ``sha256:<hex>`` over the canonical JSON of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seal(scenario_id: str) -> str:
    """The ``heldout_commit`` over the sealed seed payload for ``scenario_id``."""
    path = embargo_root() / scenario_id / "heldout_seeds.json"
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
