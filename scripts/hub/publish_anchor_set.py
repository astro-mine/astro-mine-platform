#!/usr/bin/env python3
"""Publish the anchor content set into a registry under the **org signing key**.

`docs/hub/publishing-the-anchor-content-set.md` describes this as a manual, by-component procedure.
It was manual, and that is how the org registry came to hold content signed by one key while the
workspace store held the same content signed by another: nothing checked which key, because nothing
did the publishing in one place.

**Why re-publishing works rather than re-signing.** A cosign signature rides as an OCI *referrer*,
so it does not enter the artifact's own digest — but ``verify`` requires that **every** attached
signature verifies against the trusted key (``hub.md`` §2.3, fail-closed). Appending an org-key
signature beside a dev-key one therefore makes verification worse, not better. Publishing into a
*clean* store instead gives each artifact exactly one signature, and because every producer here is
deterministic the digests come out identical — so the zoo's pins keep resolving and no scenario
needs re-versioning.

That determinism is asserted, not assumed: ``--expect`` checks each digest against what the zoo pins
and refuses on the first mismatch, because a digest that moved would silently invalidate every
scenario that pins it.

    python scripts/hub/publish_anchor_set.py \\
        --registry ./anchor-store \\
        --key /secure/cosign.key \\
        --world-bundle ../../files/data/shackleton-0.4.0/bundle \\
        --trust-anchor anchor-signing.pub --expect

The contact plan is **not** published here — it has to be rebuilt to be re-signed, and that is
``scripts/link/build_anchor_contact_plan.py``'s job (now memoized, so only the first run pays for
it). This script prints the invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
ZOO = REPO / "src" / "astro_mine" / "bench" / "zoo"


def zoo_pins() -> dict[str, str]:
    """``content id -> content_hash`` across every zoo scenario — the digests that must not move."""
    pinned: dict[str, str] = {}
    for path in sorted(ZOO.rglob("scenario.json")):
        content = json.loads(path.read_text(encoding="utf-8")).get("content") or {}
        refs: list[dict[str, Any]] = []
        if isinstance(content.get("world"), dict):
            refs.append(content["world"])
        for key in ("fleet", "prospect"):
            refs.extend(content.get(key) or [])
        if isinstance(content.get("link"), dict):
            refs.append(content["link"])
        for ref in refs:
            pinned[str(ref["id"])] = str(ref["content_hash"])
    return pinned


def _check(name: str, digest: str, pins: dict[str, str], expect: bool) -> None:
    want = pins.get(name)
    if not expect or want is None:
        return
    if digest != want:
        raise SystemExit(
            f"{name}: published {digest} but the zoo pins {want}. A producer is no longer "
            f"deterministic, or an input changed — publishing this set would break every scenario "
            f"that pins it. Nothing further has been published."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=pathlib.Path)
    parser.add_argument("--key", required=True, type=pathlib.Path, help="Org signing key (PEM).")
    parser.add_argument("--world-bundle", required=True, type=pathlib.Path)
    parser.add_argument("--world-version", default="0.4.0")
    parser.add_argument("--trust-anchor", type=pathlib.Path, default=REPO / "anchor-signing.pub")
    parser.add_argument(
        "--expect", action="store_true", help="Assert each digest matches what the zoo pins."
    )
    args = parser.parse_args(argv)

    if "://" in str(args.registry) or "ghcr.io" in str(args.registry):
        raise SystemExit(
            "publish locally, then mirror — see docs/hub/publishing-the-anchor-content-set.md"
        )
    if not os.environ.get("ASTRO_MINE_PROSPECT_CONDITIONING"):
        raise SystemExit(
            "set $ASTRO_MINE_PROSPECT_CONDITIONING to the materialized PDS conditioning bundle; "
            "the PDS prior cannot be refitted without it and publishing a partial set is the "
            "half-migrated state this exists to avoid"
        )

    from astro_mine.core.sadf import load_sadf
    from astro_mine.fleet.packaging.hub import publish_asset
    from astro_mine.hub.registry import Registry
    from astro_mine.hub.supply_chain import verify
    from astro_mine.prospect.priors import artifact_name_for, load_prior
    from astro_mine.prospect.publish import publish_prior
    from astro_mine.worlds.spec import WorldBundle, publish_world_bundle

    registry = Registry(args.registry)
    key = args.key.read_bytes()
    anchor = args.trust_anchor.read_bytes()
    pins = zoo_pins()
    published: dict[str, str] = {}

    print("fleet:")
    for path in sorted((REPO / "src" / "astro_mine" / "fleet" / "library").rglob("*.sadf.yaml")):
        doc = load_sadf(path.read_text(encoding="utf-8"))
        result = publish_asset(doc, registry, sign_key=key, base_dir=path.parent)
        name = doc.asset.identity.id
        _check(name, result.digest, pins, args.expect)
        published[name] = result.digest
        print(f"  {name}:{doc.asset.identity.version} -> {result.digest[:23]}")

    print("world:")
    bundle = WorldBundle.load(args.world_bundle)
    result = publish_world_bundle(
        bundle,
        registry,
        private_key_pem=key,
        name="shackleton-de-gerlache",
        version=args.world_version,
    )
    _check("shackleton-de-gerlache", result.digest, pins, args.expect)
    published["shackleton-de-gerlache"] = result.digest
    print(f"  shackleton-de-gerlache:{args.world_version} -> {result.digest[:23]}")

    print("priors:")
    for recipe in ("shackleton_water_ice_v1", "shackleton_water_ice_pds_v1"):
        name = artifact_name_for(recipe)
        result = publish_prior(
            load_prior(recipe), registry=registry, private_key_pem=key, name=name, version="1.0.0"
        )
        _check(name, result.digest, pins, args.expect)
        published[name] = result.digest
        print(f"  {name}:1.0.0 -> {result.digest[:23]}")

    print("\nverifying every published artifact against the trust anchor ...")
    for name, digest in sorted(published.items()):
        verify(registry, digest, trusted_public_key_pem=anchor)
        print(f"  {name}: OK")

    print(
        "\nthe contact plan is not published here -- it must be rebuilt to be re-signed:\n"
        "  python scripts/link/build_anchor_contact_plan.py \\\n"
        "      --metakernel <spice/metakernel.tm> --relay-spk <spice/relay_orbiter.bsp> \\\n"
        f"      --world-registry {args.registry} --world-ref shackleton-de-gerlache:0.4.0 \\\n"
        f"      --name lunar-polar-relay-dsn --version 0.3.0 --key {args.key}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
