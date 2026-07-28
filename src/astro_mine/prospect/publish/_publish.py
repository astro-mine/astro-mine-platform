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

from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.publish._bundle import (
    BUNDLE_MEDIA_TYPE,
    bundle_digest,
    serialize_bundle,
)
from astro_mine.prospect.publish._manifest import build_field_manifest

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact

__all__ = ["publish_prior"]

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






