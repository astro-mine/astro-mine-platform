#!/usr/bin/env python3
"""Re-publish the anchor content set under conforming artifact names (``conventions.md`` §13).

§13 requires the migration to run as **one sweep** rather than piecemeal, so the registry never
carries a half-migrated set. That is the reason this is a script and not a runbook: a sequence of
hand-typed publishes is piecemeal by construction, and the one thing §13 asks for is that it not be.

**A rename is a re-publish, not a rename.** Registry names are immutable, so every artifact here
gets new digests, the zoo is re-pinned to them, and every previously published scorecard keeps
resolving *by digest*. Nothing that was published stops existing.

The worklist is ``registry-inventory.json`` — each non-conforming entry carries the conforming name
it becomes as ``migrates_to`` — so this script and the record cannot disagree about what is left.

Order is not arbitrary, and the script enforces it:

1. **fleet** — the SADF ``identity.id`` *is* the registry name (``publish_asset`` passes
   ``name=identity.id``), so the ids are edited in the tree and the content genuinely changes.
2. **world** — republished from the already-built bundle. ``publish_world_bundle`` takes ``name``
   as a parameter and the payload layer is a deterministic tar of the bundle directory, so this
   costs nothing: no DEM re-projection, no PSR re-derivation, no hour-long build.
3. **priors** — likewise ``name``-parameterised. The recipe *key* stays snake_case; what changes is
   the published name it maps to (``astro_mine.prospect.priors.artifact_name_for``).
4. **link** — *not* done here. The contact plan embeds both the world's content hash and the Fleet
   node ids, so it must be rebuilt after 1 and 2, and that build is a ~27-minute single-core grind.
   ``scripts/link/build_anchor_contact_plan.py`` owns it; this script prints the exact invocation.

Publishing is **local only**. It writes to whatever ``--registry`` names, which is the workspace
OCI-layout store; mirroring to ``ghcr.io`` is a separate, deliberate act
(``docs/hub/publishing-the-anchor-content-set.md``).

    python scripts/hub/migrate_artifact_names.py \\
        --registry ../../files/hub-registry \\
        --key ../../files/hub-registry/keys/anchor-dev.key.pem \\
        --world-bundle ../../files/data/shackleton-0.4.0/bundle \\
        --out migration-digests.json

``--dry-run`` resolves and validates everything, and publishes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "registry-inventory.json"


def _worklist() -> dict[str, str]:
    """``old name -> new name`` for every published artifact still to migrate."""
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    work: dict[str, str] = {}
    for key, entry in inventory["artifacts"].items():
        target = entry.get("migrates_to")
        if target is None:
            continue
        old = key.rsplit(":", 1)[0]
        if old in work and work[old] != target:
            raise SystemExit(f"inventory disagrees with itself about {old!r}")
        work[old] = str(target)
    return work


def _require_conforming(name: str) -> str:
    from astro_mine.hub.registry import validate_artifact_name

    return str(validate_artifact_name(name))


def _fleet(registry: Any, key: bytes, work: dict[str, str], dry: bool) -> dict[str, Any]:
    """Re-publish the shipped Fleet library. The ids must already be migrated in the tree."""
    from astro_mine.core.sadf import load_sadf
    from astro_mine.fleet.packaging.hub import publish_asset

    out: dict[str, Any] = {}
    library = REPO / "src" / "astro_mine" / "fleet" / "library"
    for path in sorted(library.rglob("*.sadf.yaml")):
        doc = load_sadf(path.read_text(encoding="utf-8"))
        new = _require_conforming(doc.asset.identity.id)
        old = next((o for o, n in work.items() if n == new), None)
        if old is None:
            print(f"  skip  {new}: not on the worklist")
            continue
        print(f"  fleet {old} -> {new}:{doc.asset.identity.version}")
        if dry:
            out[old] = {"new_name": new, "version": doc.asset.identity.version, "dry_run": True}
            continue
        published = publish_asset(doc, registry, sign_key=key, base_dir=path.parent)
        out[old] = {
            "new_name": new,
            "version": doc.asset.identity.version,
            "reference": published.reference,
            "manifest_digest": published.digest,
            "bundle_digest": published.asset_digest,
        }
    return out


def _world(registry: Any, key: bytes, bundle_path: Path, version: str, dry: bool) -> dict[str, Any]:
    from astro_mine.worlds.spec import WorldBundle, publish_world_bundle

    bundle = WorldBundle.load(bundle_path)
    new = _require_conforming("shackleton-de-gerlache")
    print(f"  world shackleton-de-gerlache-v1 -> {new}:{version}  (from {bundle_path})")
    if dry:
        return {"shackleton-de-gerlache-v1": {"new_name": new, "version": version, "dry_run": True}}
    published = publish_world_bundle(
        bundle, registry, private_key_pem=key, name=new, version=version
    )
    return {
        "shackleton-de-gerlache-v1": {
            "new_name": new,
            "version": version,
            "reference": published.reference,
            "manifest_digest": published.digest,
        }
    }


def _priors(registry: Any, key: bytes, versions: dict[str, str], dry: bool) -> dict[str, Any]:
    from astro_mine.prospect.priors import artifact_name_for, load_prior
    from astro_mine.prospect.publish import publish_prior

    out: dict[str, Any] = {}
    for recipe_key, version in versions.items():
        new = _require_conforming(artifact_name_for(recipe_key))
        print(f"  prior {recipe_key} -> {new}:{version}")
        if dry:
            out[recipe_key] = {"new_name": new, "version": version, "dry_run": True}
            continue
        prior = load_prior(recipe_key)
        published = publish_prior(
            prior, registry=registry, private_key_pem=key, name=new, version=version
        )
        out[recipe_key] = {
            "new_name": new,
            "version": version,
            "reference": published.reference,
            "manifest_digest": published.digest,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path, help="Local OCI-layout store.")
    parser.add_argument("--key", required=True, type=Path, help="Cosign private key (PEM).")
    parser.add_argument("--world-bundle", type=Path, help="Built world bundle directory.")
    parser.add_argument("--world-version", default="0.4.0")
    parser.add_argument("--out", type=Path, help="Write the old->new digest map here.")
    parser.add_argument(
        "--only",
        default="fleet,world,priors",
        help="Comma-separated phases to run (fleet, world, priors).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if "ghcr.io" in str(args.registry) or "://" in str(args.registry):
        raise SystemExit(
            "this script publishes locally only; mirroring to a remote registry is a separate, "
            "deliberate step (docs/hub/publishing-the-anchor-content-set.md)"
        )

    from astro_mine.hub.registry import Registry

    registry = Registry(args.registry)
    key = args.key.read_bytes()
    work = _worklist()
    phases = {p.strip() for p in args.only.split(",") if p.strip()}
    results: dict[str, Any] = {}

    if "fleet" in phases:
        print("fleet:")
        results.update(_fleet(registry, key, work, args.dry_run))
    if "world" in phases:
        if args.world_bundle is None:
            raise SystemExit("--world-bundle is required for the world phase")
        print("world:")
        results.update(_world(registry, key, args.world_bundle, args.world_version, args.dry_run))
    if "priors" in phases:
        print("priors:")
        # The PDS-conditioned prior fits from a materialized conditioning bundle rather than from
        # code, and `load_prior` finds it through this variable (priors/RECIPE.md). Checked up front
        # because the failure otherwise lands mid-sweep, after the fleet assets have published.
        if not os.environ.get("ASTRO_MINE_PROSPECT_CONDITIONING"):
            raise SystemExit(
                "set $ASTRO_MINE_PROSPECT_CONDITIONING to the materialized PDS conditioning bundle "
                "(see src/astro_mine/prospect/priors/RECIPE.md) — the PDS prior cannot be refitted "
                "without it, and publishing the other prior alone would half-migrate the set"
            )
        results.update(
            _priors(
                registry,
                key,
                {"shackleton_water_ice_v1": "1.0.0", "shackleton_water_ice_pds_v1": "1.0.0"},
                args.dry_run,
            )
        )

    remaining = sorted(set(work) - {k for k in work if work[k] in _published_names(results)})
    print(f"\n{len(results)} artifact(s) handled; still to migrate: {remaining or 'none'}")
    print(
        "\nlink is not handled here — rebuild it after the world lands, because the plan embeds\n"
        "the world content hash and the Fleet node ids:\n"
        "  python scripts/link/build_anchor_contact_plan.py \\\n"
        "      --metakernel <files/data/spice/metakernel.tm> \\\n"
        "      --relay-spk <files/data/spice/relay_orbiter.bsp> \\\n"
        f"      --world-registry {args.registry} \\\n"
        "      --world-ref shackleton-de-gerlache:0.4.0 \\\n"
        "      --name lunar-polar-relay-dsn --version 0.3.0 --key <key>"
    )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _published_names(results: dict[str, Any]) -> set[str]:
    return {str(v["new_name"]) for v in results.values()}


if __name__ == "__main__":
    sys.exit(main())
