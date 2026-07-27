#!/usr/bin/env python3
"""Build and publish the anchor scenario's relay + DSN ContactPlan (RM-P0-LINK-04).

The **maintainer path** that mints the artifact a Bench ``ScenarioSpec`` pins as its ``link``
content (bench#28). It wires the three real inputs Link consumes but does not own, then hands the
resulting Core ContactPlan to :func:`~astro_mine.link.registry.publish_contact_plan`:

1. **SPICE kernels** (``--metakernel``) — DE440 + the lunar PCK/FK set. Earth's body-fixed position
   and the DSN stations' elevation of a target come from :mod:`astro_mine.spice` (RFC-0002); Link
   never re-derives them.
2. **The relay orbiter's SPK** (``--relay-spk``) — the notional relay's trajectory, produced by a
   flight-dynamics tool (GMAT's ``EphemerisFile``, as in the RM-P0-LINK-05 oracle regression) and
   furnished as an ordinary kernel. Its NAIF id must match
   :data:`~astro_mine.link.anchor.ANCHOR_RELAY_TARGET`. **Link does not propagate orbits**
   (link.md §2.2) — that is precisely why this is an input, not a computation.
3. **The terrain world** (``--world-registry`` / ``--world-ref``) — the published Worlds bundle for
   the Shackleton-de Gerlache ridge, pulled from a local OCI-layout Hub registry **by content hash**
   and rebuilt through the ``astro_mine.providers`` → ``world_provider`` entry point. Link therefore
   consumes terrain through the Core ``WorldProvider`` contract and takes **no**
   ``astro-mine-worlds`` import (conventions.md §1.1) — but the entry point must be *installed* in
   whatever environment runs this script, so run it from an environment that has both packages, e.g.

       uv run --with /path/to/astro-mine-link python scripts/build_anchor_contact_plan.py ...

   from the astro-mine-worlds project. This is a build-time tool, not part of Link's runtime or CI.

Every pinned input is content-hashed into the artifact's provenance (the LINK-05 ``CacheKey``), so
the published digest identifies exactly which kernels, which terrain, and which config produced the
plan (link.md §5, §9; conventions.md §1.5).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from astro_mine import spice
from astro_mine.core.registry import PluginManifest
from astro_mine.link.anchor import (
    ANCHOR_ARTIFACT_NAME,
    ANCHOR_ARTIFACT_VERSION,
    ANCHOR_EPOCH_WINDOW,
    ANCHOR_RELAY_TARGET,
    ANCHOR_SCENARIO_ID,
    anchor_config,
    anchor_node_ids,
    anchor_scenario,
    build_anchor_contact_plan,
)
from astro_mine.link.cache import build_cache_key, plan_digest
from astro_mine.link.geometry import SpiceEphemeris
from astro_mine.link.registry import publish_contact_plan
from astro_mine.link.windows import SpiceTopocentric


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metakernel", required=True, type=Path, help="SPICE meta-kernel (.tm).")
    parser.add_argument(
        "--relay-spk",
        required=True,
        type=Path,
        help=f"SPK for the notional relay (NAIF id {ANCHOR_RELAY_TARGET}).",
    )
    parser.add_argument(
        "--world-registry", required=True, type=Path, help="Local OCI-layout Hub registry."
    )
    parser.add_argument(
        "--world-ref",
        required=True,
        help="World artifact reference or digest (e.g. shackleton-de-gerlache-v1:0.1.0).",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Registry to publish the plan into — a local OCI-layout path or a remote registry URL "
        "like ghcr.io/astro-mine (default: --world-registry). Kept a string, not a Path, so a URL "
        "scheme survives.",
    )
    parser.add_argument("--key", type=Path, default=None, help="Cosign private key (PEM) to sign.")
    parser.add_argument("--name", default=ANCHOR_ARTIFACT_NAME)
    parser.add_argument("--version", default=ANCHOR_ARTIFACT_VERSION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the plan and print its digest without publishing.",
    )
    args = parser.parse_args(argv)

    # The plan build is a single-core grind of tens of minutes (the 30-day mission window searched
    # at a 60 s step and refined to 5 s across every node pair - ~27 min measured, ~30x the 24 h
    # window's ~1 min) and it used to print nothing until it was done, which is indistinguishable
    # from a hang. Turn the library's per-pair progress on, with timestamps.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    started = time.monotonic()
    logging.info("loading world %s from %s ...", args.world_ref, args.world_registry)
    world, world_digest = _world_provider(args.world_registry, args.world_ref)
    logging.info("world loaded in %.1fs (%s)", time.monotonic() - started, world_digest)

    with spice.kernel_pool(args.metakernel, args.relay_spk):
        scenario = anchor_scenario(
            world=world,
            ephemeris=SpiceEphemeris(),
            topocentric=SpiceTopocentric(),
            window=ANCHOR_EPOCH_WINDOW,
        )
        plan = build_anchor_contact_plan(scenario)

    key = build_cache_key(
        kernels=[args.metakernel, args.relay_spk],
        nodes=list(anchor_node_ids()),
        epoch=ANCHOR_EPOCH_WINDOW,
        config=anchor_config(),
    )
    input_hashes = {
        "kernels": key.kernels,
        "terrain": world_digest,
        "nodes": key.nodes,
        "epoch": key.epoch,
        "config": key.config,
    }

    print(f"plan: {len(plan.nodes)} nodes, {len(plan.intervals)} intervals")
    print(f"plan_digest: sha256:{plan_digest(plan)}")
    if args.dry_run:
        print(json.dumps(input_hashes, indent=2, sort_keys=True))
        return 0

    artifact = publish_contact_plan(
        plan,
        registry_path=args.registry or args.world_registry,
        name=args.name,
        version=args.version,
        scenario_id=ANCHOR_SCENARIO_ID,
        input_hashes=input_hashes,
        private_key_pem=args.key.read_bytes() if args.key is not None else None,
    )
    print(f"published {artifact.reference} -> {artifact.digest}")
    return 0


def _world_provider(registry_path: Path, reference: str) -> tuple[Any, str]:
    """Pull the anchor world by content hash and rebuild it through the Core entry point.

    Returns the live ``WorldProvider`` and the world artifact's digest (the ``terrain`` pinned input
    of the plan's provenance). No ``astro_mine.worlds`` import happens here — only the
    ``astro_mine.providers`` → ``world_provider`` factory the Worlds package registers.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Registry

    registry = Registry(registry_path)
    digest = registry.resolve(reference).digest
    manifest = PluginManifest.model_validate_json(HubClient(registry).pull(digest))
    layers: Mapping[str, bytes] = {
        layer["mediaType"]: registry.pull_blob(layer["digest"])
        for layer in registry.read_manifest(digest)["layers"]
    }
    entry_points = importlib.metadata.entry_points(group="astro_mine.providers")
    if "world_provider" not in entry_points.names:
        raise SystemExit(
            "no `astro_mine.providers` -> `world_provider` entry point is installed; run this "
            "script from an environment that has astro-mine-worlds (see the module docstring)"
        )
    return entry_points["world_provider"].load()(manifest, layers), digest


if __name__ == "__main__":
    sys.exit(main())
