#!/usr/bin/env python3
"""Bench determinism gate — the platform reproducibility oracle (RM-P0-BENCH-04; bench.md §10).

Runs a zoo scenario (the anchor by default) through the reproducibility harness and asserts a
byte-identical, content-addressed Result across runs. Exits non-zero on any non-reproducibility, so
CI — and the container image — fail on drift. Runs locally with no account (CX-LOCAL); no container
required.

When the ``[cloud]`` extra is installed it *also* asserts the **scale-out** path reproduces
(RM-P1-BENCH-11): a Cloud-dispatched evaluation batch (via the no-cluster ``DryRunClient``) collects
a scorecard byte-identical to the in-process workstation run — "a cluster run reproduces the
workstation run for the same inputs + seed" (bench.md §7). It is skipped (not failed) when the extra
is absent, so the base-tier gate still runs offline.

``--runner`` selects which runner the oracle exercises (default ``fixture``, the deterministic
stand-in). ``--runner sim`` gates the real physics engine (G2.16 — the repro oracle otherwise never
exercises physics); it needs ``astro-mine-sim[bench]`` and fetched content. The runner is discovered
through the ``astro_mine.bench.runners`` entry-point group, so this gate never imports Sim
(conventions.md §1.1). Note this threads the harness ``Runner`` protocol — a *distinct* protocol
from the ``EpisodeRunner`` the ``score`` CLI selects; ``astro-mine-sim[bench]`` supplies both.

    python scripts/determinism_gate.py [scenario-id] [--runner fixture|sim]
"""

from __future__ import annotations

import argparse
import sys

from astro_mine.bench.baseline import RunnerNotAvailableError, load_runner_provider
from astro_mine.bench.harness import DeterminismError, assert_reproducible
from astro_mine.bench.metrics import scored_metric_values
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario

#: A local, importable baseline policy the scale-out gate fans out under submit-policy-we-run.
_GATE_POLICY_REF = "astro_mine.bench.baseline:BaselinePolicy"


def _eval_batch_gate(scenario_id: str) -> int:
    """Assert the Cloud eval batch reproduces the local run; skip without the ``[cloud]`` extra."""
    try:
        from astro_mine.bench.eval import assert_batch_reproducible
    except ImportError:  # pragma: no cover - exercised only without the [cloud] extra
        pass
    else:
        try:
            import astro_mine.cloud  # noqa: F401  (the [cloud] extra backs the dispatcher)
        except ImportError:  # pragma: no cover - exercised only without the [cloud] extra
            print("scale-out eval gate SKIPPED: install the [cloud] extra to run it")
            return 0
        spec = load_scenario(scenario_id)
        card_hash = assert_batch_reproducible(
            spec, policy_ref=_GATE_POLICY_REF, seeds=spec.seeds.public
        )
        print(f"scale-out eval gate OK: {scenario_id} -> {card_hash}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="determinism_gate", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "scenario_id",
        nargs="?",
        default=ANCHOR_SCENARIO_ID,
        help=f"zoo scenario id to gate (default: {ANCHOR_SCENARIO_ID})",
    )
    parser.add_argument(
        "--runner",
        default="fixture",
        metavar="NAME",
        help="runner to gate (default: fixture; 'sim' needs astro-mine-sim[bench] + content)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        provider = load_runner_provider(args.runner)
    except RunnerNotAvailableError as exc:
        print(f"determinism gate error: {exc}", file=sys.stderr)
        return 2
    try:
        result = assert_reproducible(
            load_scenario(args.scenario_id),
            # The gate is a composition point: it resolves the runner by name and supplies
            # Bench's own scorer, because resolving a scenario's metric references is Bench's
            # job and not the runner's (astro-mine-platform#5).
            provider.harness_runner(scorer=scored_metric_values),
            runner_id=provider.runner_id,
        )
        rc = _eval_batch_gate(args.scenario_id)
    except DeterminismError as exc:
        print(f"DETERMINISM GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"determinism gate OK: {args.scenario_id} [{provider.runner_id}] -> {result.result_hash}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
