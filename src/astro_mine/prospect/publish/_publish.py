"""Publish a belief prior to a local Hub registry, and the ``prospect publish`` CLI.

The publish half of RM-P1-PROSPECT-13: serialize a prior to the content-addressed bundle, build the
Core ``resource_field_backend`` manifest, and store + sign it in a **local OCI-layout registry**
through the ``astro-mine-hub`` client — the tier-1 offline path, no hosted Hub (hub.md principle 7;
``LUNAR-TR-004``). ``astro-mine-hub`` is a **publish-time** dependency (the ``publish`` extra),
imported lazily here so loading a bundle via
:func:`~astro_mine.prospect.publish._bundle.from_bundle` never needs it.

Backlog: RM-P1-PROSPECT-13 — astro-mine-prospect#23
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.prospect.priors.recipe import Prior, artifact_name_for
from astro_mine.prospect.publish._bundle import (
    BUNDLE_MEDIA_TYPE,
    bundle_digest,
    serialize_bundle,
)
from astro_mine.prospect.publish._manifest import build_field_manifest

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact, RegistryClient

__all__ = ["publish_prior"]

_DEFAULT_RECIPE = "shackleton_water_ice_v1"


def publish_prior(
    prior: Prior,
    *,
    registry: RegistryClient,
    private_key_pem: bytes,
    name: str | None = None,
    version: str | None = None,
    publisher: str = "local",
    namespace: str = "open",
    zarr: bool = False,
) -> PublishedArtifact:
    """Serialize, manifest, sign, and publish *prior* to the OCI ``registry`` the caller supplies.

    The registry is **injected** rather than opened from a path here: local OCI layout and
    remote OCI Distribution are two implementations of one protocol, and picking one is the
    caller's decision (conventions.md §3.3).

    Returns the :class:`~astro_mine.hub.registry.PublishedArtifact` (its immutable ``name:version``
    reference and content digest). With a ``private_key_pem`` the artifact is signed and gets its
    cosign signature / SLSA provenance / SBOM attestations (verified fail-closed at pull); without
    one it is stored for integrity-only verification. ``version`` defaults to the prior's version.
    ``name`` defaults to the *published artifact name* the recipe maps to
    (:func:`~astro_mine.prospect.priors.artifact_name_for`) — **not** the recipe key, which is a
    Python-side identifier and stays snake_case. Defaulting to the key is what put
    ``shackleton_water_ice_pds_v1`` in the registry (conventions.md §13).
    ``namespace`` is the Hub namespace (default ``open``; community
    contributions publish under ``community`` — see
    :func:`~astro_mine.prospect.publish._community.publish_community_prior`).

    ``zarr=True`` additionally ships the prior's **Zarr** store (parametric encoding) as a second
    layer (prospect.md §5). Both layers describe the same field and resolve through the same
    ``from_bundle`` entry point, which prefers the Zarr one — so a consumer with the ``zarr`` extra
    reads the architecture's field format while one without it still resolves the dependency-light
    ``.npy`` bundle (``LUNAR-TR-004``). It needs the ``zarr`` extra at *publish* time.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob

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
    client = HubClient(registry)
    return client.publish(
        name=name or artifact_name_for(prior.provenance.recipe),
        version=version or prior.provenance.recipe_version,
        kind="plugin",
        manifest=manifest,
        layers=layers,
        private_key_pem=private_key_pem,
        namespace=namespace,
        publisher=publisher,
    )






