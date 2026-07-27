"""Publish a belief prior to a local Hub registry, and the ``prospect publish`` CLI.

The publish half of RM-P1-PROSPECT-13: serialize a prior to the content-addressed bundle, build the
Core ``resource_field_backend`` manifest, and store + sign it in a **local OCI-layout registry**
through the ``astro-mine-hub`` client — the tier-1 offline path, no hosted Hub (hub.md principle 7;
``LUNAR-TR-004``). ``astro-mine-hub`` is a **publish-time** dependency (the ``publish`` extra),
imported lazily here so loading a bundle via
:func:`~astro_mine.prospect.publish._bundle.from_bundle` never needs it.

Backlog: RM-P1-PROSPECT-13 — https://github.com/astro-mine/astro-mine-prospect/issues/23
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.publish._bundle import (
    BUNDLE_MEDIA_TYPE,
    bundle_digest,
    serialize_bundle,
)
from astro_mine.prospect.publish._manifest import build_field_manifest

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact

__all__ = ["main", "publish_prior"]

_DEFAULT_RECIPE = "shackleton_water_ice_v1"


def publish_prior(
    prior: Prior,
    *,
    registry_path: str | Path,
    private_key_pem: bytes,
    name: str | None = None,
    version: str | None = None,
    publisher: str = "local",
    namespace: str = "open",
    zarr: bool = False,
) -> PublishedArtifact:
    """Serialize, manifest, sign, and publish *prior* to the local OCI registry at the given path.

    Returns the :class:`~astro_mine.hub.registry.PublishedArtifact` (its immutable ``name:version``
    reference and content digest). With a ``private_key_pem`` the artifact is signed and gets its
    cosign signature / SLSA provenance / SBOM attestations (verified fail-closed at pull); without
    one it is stored for integrity-only verification. ``name``/``version`` default to the prior's
    recipe name and version. ``namespace`` is the Hub namespace (default ``open``; community
    contributions publish under ``community`` — see
    :func:`~astro_mine.prospect.publish._community.publish_community_prior`).

    ``zarr=True`` additionally ships the prior's **Zarr** store (parametric encoding) as a second
    layer (prospect.md §5). Both layers describe the same field and resolve through the same
    ``from_bundle`` entry point, which prefers the Zarr one — so a consumer with the ``zarr`` extra
    reads the architecture's field format while one without it still resolves the dependency-light
    ``.npy`` bundle (``LUNAR-TR-004``). It needs the ``zarr`` extra at *publish* time.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob, open_registry

    bundle = serialize_bundle(prior)
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(bundle))
    layers = [Blob(BUNDLE_MEDIA_TYPE, bundle)]
    if zarr:
        from astro_mine.prospect.publish._zarr import (
            ZARR_MEDIA_TYPE,
            FieldArchive,
            serialize_zarr,
        )

        layers.append(Blob(ZARR_MEDIA_TYPE, serialize_zarr(FieldArchive.parametric(prior))))
    client = HubClient(open_registry(registry_path))
    return client.publish(
        name=name or prior.provenance.recipe,
        version=version or prior.provenance.recipe_version,
        kind="plugin",
        manifest=manifest,
        layers=layers,
        private_key_pem=private_key_pem,
        namespace=namespace,
        publisher=publisher,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The ``prospect`` CLI entry point — ``prospect publish`` publishes a prior bundle."""
    parser = argparse.ArgumentParser(prog="astro-mine-prospect", description="Astro-Mine-Prospect tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser(
        "publish", help="Publish a belief-prior bundle to a local Hub registry."
    )
    publish.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Local OCI-layout registry path, or a remote registry URL (e.g. ghcr.io/astro-mine).",
    )
    publish.add_argument(
        "--name", default=_DEFAULT_RECIPE, help="Prior recipe to publish (default: the anchor)."
    )
    publish.add_argument(
        "--version", default=None, help="Artifact version (default: the recipe version)."
    )
    publish.add_argument(
        "--private-key",
        type=Path,
        default=None,
        help="ECDSA P-256 signing key (PEM); a fresh keypair is generated if omitted.",
    )
    publish.add_argument(
        "--public-key-out",
        type=Path,
        default=None,
        help="Write the generated public key (PEM) here (only when a key is generated).",
    )
    args = parser.parse_args(argv)
    return _cmd_publish(args)


def _cmd_publish(args: argparse.Namespace) -> int:
    from astro_mine.hub.supply_chain import generate_keypair

    prior = load_prior(args.name)
    if args.private_key is not None:
        private_pem = args.private_key.read_bytes()
    else:
        private_pem, public_pem = generate_keypair()
        if args.public_key_out is not None:
            args.public_key_out.write_bytes(public_pem)
    artifact = publish_prior(
        prior,
        registry_path=args.registry,
        private_key_pem=private_pem,
        version=args.version,
    )
    print(f"published {artifact.reference} -> {artifact.digest}")
    return 0


def deprecated_alias(argv: Sequence[str] | None = None) -> int:
    """The pre-RFC-0011 ``prospect`` name — kept for one deprecation cycle.

    ``prospect`` was a generic binary planted on every user's ``PATH``; the platform now names a
    component's command after its package (``conventions.md §13``, normative). The old name keeps
    working unchanged, prints one line naming its replacement, and is **removed at the first
    public-benchmark milestone** — i.e. before the platform is public, so no outside user ever
    learns the transitional name.

    The notice goes to **stderr**, never stdout: `--json`-style output has to stay
    machine-readable, and a warning on stdout would corrupt exactly the pipelines most likely
    to be using the old name in a script.
    """
    print(
        "warning: `prospect` is deprecated and will be removed at the first public-benchmark "
        "milestone; use `astro-mine-prospect` instead (RFC-0011 §5).",
        file=sys.stderr,
    )
    return main(argv)
