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

**Republishing a name the tree has already used is guarded, and the guard does not consult the
registry** — ``registry-inventory.json`` is the record, because a pruned registry cannot distinguish
"never published" from "published and deleted". See :func:`_check_republish`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: The committed record of what this tree has published. See `registry-inventory.md`.
ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "registry-inventory.json"


def _recorded_artifact(name: str, version: str) -> dict | None:
    """The committed record for ``name:version``, or ``None`` if the tree never published it.

    **Why this does not ask the registry.** The guard below exists to catch a republish under a name
    whose bytes are gone, and a pruned registry answers "nothing here" — which is indistinguishable
    from "never published". A guard founded on the registry is therefore silent in exactly the case
    it exists to refuse; see the block comment in :func:`_check_republish` for the incident.

    Missing inventory is not treated as "nothing is recorded". The file is committed, so its absence
    means the checkout is broken rather than that the artifact is new, and inventing an empty record
    would re-open the hole. Callers fail closed on the raised error.
    """
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"{INVENTORY_PATH} is missing. It is the committed record of what this tree has "
            f"published, and this script cannot decide whether a publish is a rebuild without it."
        )
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    return inventory.get("artifacts", {}).get(f"{name}:{version}")


def _held_by_registry(registry, name: str, version: str) -> bool:
    """Whether the registry resolves ``name:version`` — a fact about the store, not about truth."""
    from astro_mine.hub.registry import ArtifactNotFound

    try:
        registry.resolve(f"{name}:{version}")
    except (ArtifactNotFound, KeyError):
        return False
    return True


def _check_republish(
    registry, name: str, version: str, expect_digest: str | None, rebuilt: str | None
) -> str | None:
    """Refuse a republish the operator cannot substantiate. Returns an error message, or ``None``.

    **The hole this closes.** The original guard compared a rebuild against the digest the registry
    already held, inside an ``except ArtifactExistsError`` branch — so it only ran when the registry
    still had the artifact. If the name had been pruned there was no ``ArtifactExistsError``,
    nothing to compare against, and the publish proceeded. Because ``--version`` is a *label*
    rather than a checkout (it is passed straight to ``build_excavation_surrogate``, and the tier's
    error budget is calibrated from whatever the code says at build time), the result was today's
    model published under an old tier's name, into the empty slot, with the guard unable to object.

    That is not theoretical: ``excavation-gns:0.2.0``-``0.5.0`` were pruned on 2026-08-08, and
    ``0.4.0`` is the tier ``bench/zoo/lunar_polar_ice_excavation_fidelity_v1/CROSSOVER.md``
    publishes a cost curve against. Anyone reading that document and reaching for the obvious
    rebuild command was one step from minting a counterfeit of it. Provenance pins ``code_version``,
    so it would be *detectable* afterwards — but detection after publication is not refusal, and by
    then anything resolving the name gets the counterfeit.

    So the record moved out of the registry (``registry-inventory.json``) and the check moved out of
    the exception handler. Three refusals, in the order a reader meets them:

    1. **Recorded, absent from the store, no digest on record.** Nothing can verify a rebuild.
       Refused outright — there is no flag for this, because there is no evidence to offer.
    2. **Recorded, absent from the store, digest on record, no ``--expect-digest``.** Refused until
       the operator states which artifact they believe they are reproducing. This turns "rebuild it"
       from an assertion into a claim the script checks.
    3. **A stated or recorded digest the rebuild does not reproduce.** Refused. ``--expect-digest``
       is a claim, never an override.
    """
    record = _recorded_artifact(name, version)
    reference = f"{name}:{version}"

    if record is None:
        # Never published by this tree. A digest the operator states anyway is still checked.
        if expect_digest and rebuilt and expect_digest != rebuilt:
            return (
                f"refusing to publish {reference}: you passed --expect-digest {expect_digest}, but "
                f"this build produced {rebuilt}. The tree has no record of this name:version, so "
                f"nothing corroborates either value — resolve the discrepancy before publishing."
            )
        return None

    recorded_digest = record.get("bundle_digest")
    disposition = record.get("disposition", "published")
    held = _held_by_registry(registry, name, version)

    if not held:
        # The pruned-name case — the whole reason this function exists.
        if recorded_digest is None:
            return (
                f"refusing to publish {reference}: it is recorded in registry-inventory.json as "
                f"'{disposition}' with no bundle digest, and the registry no longer holds it. "
                f"There is nothing to verify a rebuild against, so this publish would put "
                f"unverifiable bytes under a name that already means something. Publish a new "
                f"version instead."
            )
        if expect_digest is None:
            return (
                f"refusing to publish {reference}: it is recorded in registry-inventory.json as "
                f"'{disposition}' (bundle {recorded_digest}) but the registry no longer holds it, "
                f"so the immutability check has nothing to compare against.\n"
                f"  --version is a label, not a checkout: this build reconstructs today's model "
                f"under that name, and its error budget is calibrated from today's code.\n"
                f"  If you believe you are reproducing the recorded artifact, say so explicitly:\n"
                f"      --expect-digest {recorded_digest}\n"
                f"  and the publish proceeds only if the rebuild actually reproduces it."
            )
        if expect_digest != recorded_digest:
            return (
                f"refusing to publish {reference}: you expect bundle {expect_digest}, but the "
                f"committed record says {recorded_digest}. registry-inventory.json is the record "
                f"of what was published; a disagreement means one of you is wrong about which "
                f"artifact this is."
            )

    if rebuilt is not None:
        target = recorded_digest if recorded_digest is not None else expect_digest
        if target is not None and rebuilt != target:
            return (
                f"refusing to publish {reference}: the bundle this build produced ({rebuilt}) is "
                f"not the one on record ({target}). Registry digests are immutable — publish a new "
                f"version rather than re-pointing an existing one."
            )
    return None


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
    parser.add_argument(
        "--expect-digest",
        help="the bundle digest you believe this build reproduces. Required to republish a "
        "name:version the tree has already published but the registry no longer holds — a claim "
        "the script checks, never an override.",
    )
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

    # Pre-flight, before the training run: a refusal the operator cannot argue their way out of is
    # one they should meet in a second rather than after a CPU-bound train/calibrate/export cycle.
    # The bundle does not exist yet, so this catches the "no digest on record" and "no
    # --expect-digest" refusals; the byte comparison repeats below once there is something to hash.
    registry = Registry(args.registry)
    refusal = _check_republish(registry, args.name, args.version, args.expect_digest, rebuilt=None)
    if refusal is not None:
        print(f"\n{refusal}", file=sys.stderr)
        return 1

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

    # Now there are bytes to check the claim against. This is the refusal that actually stops a
    # counterfeit: the pre-flight above proves the operator *made* a claim, this proves it is true.
    refusal = _check_republish(
        registry, args.name, args.version, args.expect_digest, rebuilt=bundle.content_hash()
    )
    if refusal is not None:
        print(f"\n{refusal}", file=sys.stderr)
        return 1

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
        #
        # This stays as the second line of defence, for the case `_check_republish` cannot see: an
        # artifact the registry holds and `registry-inventory.json` does not record. The first line
        # is the inventory check above, which is what covers the inverse — recorded but pruned.
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
