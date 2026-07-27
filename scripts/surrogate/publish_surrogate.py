#!/usr/bin/env python
"""Train, export, sign and publish the excavation surrogate tier (RM-P1-SURR-04).

The one command that turns the frozen DEM fixture into a **published, content-addressed, signed**
ONNX tier that Sim can resolve by digest and substitute for the DEM solver:

    uv run --extra serve --extra publish python scripts/publish_surrogate.py \
        --registry /path/to/hub-registry \
        --key /path/to/signing.key.pem \
        --version 0.2.0

train (torch, CPU) -> calibrate (split-conformal) -> export (self-contained ONNX) -> sign (ECDSA
P-256 over the bundle's content hash) -> publish (OCI artifact) -> **pull it back and verify
fail-closed**, so the script never reports a success it has not proved.

The published manifest pins the tier's whole provenance: the bundle digest it is signed over, the
ErrorReport digest (the calibrated bound the tier commits to), the training-dataset hash, and — the
piece surrogate#17 was missing — the **sampling-policy hash**. A surrogate's trust region is derived
from the configs its fixture swept, so the sampling policy is what actually decides the domain Sim
will admit the tier under. Pinning it by hash makes that domain traceable to a declaration instead
of to a set of module constants nobody was versioning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _published_bundle_digest(registry, name: str, version: str) -> str:
    """The bundle content hash the registry's ``name:version`` is signed over."""
    from astro_mine.core.registry import PluginManifest

    descriptor = registry.resolve(f"{name}:{version}")
    manifest = PluginManifest.model_validate_json(registry.read_config(descriptor.digest))
    return "" if manifest.provenance is None else manifest.provenance.digest


def _describe_published(registry, name: str, version: str, bundle):
    """The :class:`PublishedSurrogate` describing an artifact already in the registry."""
    from astro_mine.surrogate.serve.publish import PublishedSurrogate

    descriptor = registry.resolve(f"{name}:{version}")
    return PublishedSurrogate(
        name=name,
        version=version,
        reference=f"{name}:{version}",
        manifest_digest=descriptor.digest,
        artifact_digest=bundle.content_hash(),
        error_report_digest=bundle.error_report.content_hash(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", required=True, type=Path, help="path to a local OCI-layout Hub registry"
    )
    parser.add_argument(
        "--key", required=True, type=Path, help="ECDSA P-256 private-key PEM to sign the tier with"
    )
    parser.add_argument(
        "--pub", type=Path, help="trusted public-key PEM; pins the round-trip verifier's trust"
    )
    parser.add_argument("--name", default="excavation-gns", help="published artifact name")
    parser.add_argument("--version", default="0.2.0", help="published artifact version")
    parser.add_argument("--seed", type=int, default=0, help="training seed (determinism)")
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    args = parser.parse_args(argv)

    from astro_mine.hub.registry import ArtifactExistsError, Registry
    from astro_mine.hub.supply_chain import make_verifier
    from astro_mine.surrogate import __version__ as surrogate_version
    from astro_mine.surrogate.models import build_excavation_surrogate
    from astro_mine.surrogate.models.dataset import load_dem_dataset
    from astro_mine.surrogate.serve import (
        export_excavation_surrogate,
        publish_served_surrogate,
        resolve_and_load,
    )

    dataset = load_dem_dataset()
    train_dataset_hash = dataset.content_hash()
    print(
        f"dataset  : {dataset.n_configs} configs x {dataset.n_steps} steps x "
        f"{dataset.n_particles} particles @ dt={dataset.dt_s:g}s\n"
        f"           bed {dataset.bed_width_m:g} m, blade {dataset.tool_height_m:g} m\n"
        f"           content_hash   {train_dataset_hash}\n"
        f"           sampling_policy {dataset.sampling_policy_hash}",
        flush=True,
    )

    t0 = time.perf_counter()
    surrogate = build_excavation_surrogate(seed=args.seed, name=args.name, version=args.version)
    train_s = time.perf_counter() - t0

    # The trust region as *published*: the box the ErrorReport declares, which is what Sim reads off
    # the manifest to decide whether a query is in-domain. (It is derived from the swept configs by
    # `ExcavationTrustRegion.from_configs`, and with the GRID design that comes out equal to the
    # sampling policy's declared bounds.)
    report = surrogate.error_report
    region = report.trust_region.bounds
    print(f"\ntrained in {train_s:.1f}s. trust region (declared by the ErrorReport):", flush=True)
    for name, bound in region.items():
        print(f"  {name:12s} [{bound.low:.4f}, {bound.high:.4f}]")

    budget = report.substitution_policy.recommended_error_budget
    print("\ncalibrated error budget (the bound the tier declares it holds):")
    for channel, value in sorted(budget.items()):
        print(f"  {channel:12s} {value:.6g}")

    bundle = export_excavation_surrogate(surrogate)
    registry = Registry(args.registry)
    try:
        published = publish_served_surrogate(
            bundle,
            registry,
            name=args.name,
            version=args.version,
            private_key_pem=args.key.read_bytes(),
            code_version=surrogate_version,
            toolchain_version=f"python-{sys.version_info.major}.{sys.version_info.minor}",
            train_dataset_hash=train_dataset_hash,
            sampling_policy_hash=dataset.sampling_policy_hash,
        )
    except ArtifactExistsError:
        # A registry digest is immutable, so re-publishing a version is refused — but this script is
        # deterministic (fixed seed, byte-reproducible bundle), so a *re-run* legitimately rebuilds
        # the identical tier. Treat that as the no-op it is, and reserve the failure for what the
        # immutability rule actually exists to catch: a **different** tier claiming a version that
        # is already taken.
        existing = _published_bundle_digest(registry, args.name, args.version)
        rebuilt = bundle.content_hash()
        if existing != rebuilt:
            print(
                f"\nrefusing to republish {args.name}:{args.version}: the registry already holds a "
                f"tier at that version whose bundle digest is {existing}, but this build produced "
                f"{rebuilt}. Registry digests are immutable — publish a new version.",
                file=sys.stderr,
            )
            return 1
        print(
            f"\n{args.name}:{args.version} is already published with this exact bundle "
            f"({rebuilt}); nothing to write. Verifying the published artifact.",
            flush=True,
        )
        published = _describe_published(registry, args.name, args.version, bundle)

    # Never report a publish we have not proved: pull the artifact back out of the registry by the
    # reference we just wrote and run the full fail-closed load gate (signature -> bundle hash vs
    # signed provenance.digest -> embedded ErrorReport hash vs the manifest's declared digest).
    verifier = make_verifier(trusted_public_key_pem=args.pub.read_bytes()) if args.pub else None
    served = resolve_and_load(registry, published.reference, verifier=verifier)
    assert served.error_report.content_hash() == published.error_report_digest

    result = {
        "reference": published.reference,
        "manifest_digest": published.manifest_digest,
        "artifact_digest": published.artifact_digest,
        "error_report_digest": published.error_report_digest,
        "train_dataset_hash": train_dataset_hash,
        "sampling_policy_hash": dataset.sampling_policy_hash,
        "trust_region": {name: [bound.low, bound.high] for name, bound in region.items()},
        "recommended_error_budget": dict(budget),
        "train_seconds": round(train_s, 1),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"\npublished {published.reference}"
            f"\n  manifest digest : {published.manifest_digest}"
            f"\n  artifact digest : {published.artifact_digest}"
            f"\n  error report    : {published.error_report_digest}"
            f"\n  round-trip pull + fail-closed verify: OK"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
