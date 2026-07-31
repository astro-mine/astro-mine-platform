"""Hub-publish the belief prior + reopen it as a live ResourceField (RM-P1-PROSPECT-13).

The publish/resolve half of Prospect (prospect.md §3, §4, §6; hub.md §3, §9): take the offline
parametric belief prior the anchor scenario pins (``shackleton_water_ice_v1``), serialize it to a
content-addressed :data:`~astro_mine.prospect.publish._bundle.BUNDLE_MEDIA_TYPE` bundle, emit a
Core ``resource_field_backend`` :class:`~astro_mine.core.registry.PluginManifest`, and publish it —
signed and content-addressed — to a **local OCI-layout registry** through the ``astro-mine-hub``
client (no hosted Hub; ``LUNAR-TR-004``). A consumer (Sim, Bench) then resolves the field **by
digest** and rebuilds a live :func:`~astro_mine.prospect.publish._bundle.from_bundle` field through
the ``astro_mine.providers`` entry point — never importing :mod:`astro_mine.prospect` by name.

Security invariant (``RM-P0-PROSPECT-05``): only the **public belief prior** is published; the
sealed :class:`~astro_mine.prospect.belief.ground_truth.GroundTruthField` is never serialized.

:mod:`astro_mine.hub` is a **publish-time** collaborator: :func:`publish_prior` imports it lazily,
so :func:`from_bundle` — the field-loading path — needs only Core + numpy. Consolidation put Hub
in the same distribution, so there is nothing extra to install;
``astro-mine-platform[prospect-publish]`` survives as an empty alias for the install line that used
to matter.
"""

from __future__ import annotations

from astro_mine.prospect.publish._bundle import (
    BUNDLE_MEDIA_TYPE,
    MEAN_MEMBER,
    METADATA_MEMBER,
    PROVENANCE_MEMBER,
    PROVIDER_ENTRY_POINT,
    RESOURCE_FIELD_INTERFACE,
    RESOURCE_FIELD_INTERFACE_VERSION,
    VARIANCE_MEMBER,
    bundle_digest,
    from_bundle,
    prior_from_bundle,
    serialize_bundle,
)
from astro_mine.prospect.publish._community import (
    RECIPE_SPEC_MEDIA_TYPE,
    DiscoveredArtifact,
    PriorRecipeSpec,
    build_recipe_manifest,
    discover_priors,
    publish_community_prior,
    publish_recipe,
    recipe_reference_name,
    recipe_spec_from_bytes,
    resolve_recipe,
    serialize_recipe_spec,
)
from astro_mine.prospect.publish._manifest import build_field_manifest
from astro_mine.prospect.publish._publish import publish_prior
from astro_mine.prospect.publish._zarr import (
    DEFAULT_LEVELS,
    ENCODINGS,
    ZARR_MEDIA_TYPE,
    Encoding,
    FieldArchive,
    archive_from_zarr_bytes,
    quantile_grids,
    read_zarr,
    serialize_zarr,
    write_zarr,
)

__all__ = [
    "BUNDLE_MEDIA_TYPE",
    "DEFAULT_LEVELS",
    "ENCODINGS",
    "MEAN_MEMBER",
    "METADATA_MEMBER",
    "PROVENANCE_MEMBER",
    "PROVIDER_ENTRY_POINT",
    "RECIPE_SPEC_MEDIA_TYPE",
    "RESOURCE_FIELD_INTERFACE",
    "RESOURCE_FIELD_INTERFACE_VERSION",
    "VARIANCE_MEMBER",
    "ZARR_MEDIA_TYPE",
    "DiscoveredArtifact",
    "Encoding",
    "FieldArchive",
    "PriorRecipeSpec",
    "archive_from_zarr_bytes",
    "build_field_manifest",
    "build_recipe_manifest",
    "bundle_digest",
    "discover_priors",
    "from_bundle",
    "prior_from_bundle",
    "publish_community_prior",
    "publish_prior",
    "publish_recipe",
    "quantile_grids",
    "read_zarr",
    "recipe_reference_name",
    "recipe_spec_from_bytes",
    "resolve_recipe",
    "serialize_bundle",
    "serialize_recipe_spec",
    "serialize_zarr",
    "write_zarr",
]
