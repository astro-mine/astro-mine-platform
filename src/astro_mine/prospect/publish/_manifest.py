# SPDX-License-Identifier: Apache-2.0
"""Build the Core ``resource_field_backend`` plugin manifest for a published prior.

Prospect **consumes** Core's manifest schema, it does not invent one (prospect.md §3; hub.md §3).
This maps a :class:`~astro_mine.prospect.priors.recipe.Prior` onto a
:class:`~astro_mine.core.registry.PluginManifest` (``kind=resource_field_backend``) whose
``core_interfaces`` declares the ``resource_field`` version it implements, whose
``provenance.digest`` is the prior's content address (its identity), and whose ``attributes`` carry
open, kind-specific facets a resolver needs (the bundle media type, the entry-point name, the
species/unit). It declares **no capability tags** — the published belief prior is open-commons; the
gated ``GROUND_TRUTH_ACCESS`` tag never appears because the sealed field is never published
(``RM-P0-PROSPECT-05``).

Backlog: RM-P1-PROSPECT-13 — astro-mine-prospect#23
"""

from __future__ import annotations

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.prospect import __version__ as _PROSPECT_VERSION
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.publish._bundle import (
    BUNDLE_MEDIA_TYPE,
    PROVIDER_ENTRY_POINT,
    RESOURCE_FIELD_INTERFACE,
    RESOURCE_FIELD_INTERFACE_VERSION,
)

__all__ = ["build_field_manifest"]


def build_field_manifest(prior: Prior, *, bundle_sha256: str) -> PluginManifest:
    """Build the ``kind=resource_field_backend`` manifest for *prior* (unsigned; caller attaches).

    ``bundle_sha256`` is the ``sha256:<hex>`` content address of the serialized bundle layer, folded
    into ``attributes`` so a catalog can cross-check the payload. The manifest carries the prior's
    own content hash as ``provenance.digest`` — the field's content-addressed identity — so a Bench
    scenario can pin the field by digest.
    """
    metadata = prior.metadata
    prov = prior.provenance
    # Real raster source hashes only exist for the #11 ingest recipe; parametric priors cite
    # published characterizations (source_hash is None), so this is empty for the anchor prior.
    input_hashes = [c.source_hash for c in prov.citations if c.source_hash is not None]
    return PluginManifest(
        name=prov.recipe,
        version=prov.recipe_version,
        kind=PluginKind.RESOURCE_FIELD_BACKEND,
        core_interfaces={RESOURCE_FIELD_INTERFACE: RESOURCE_FIELD_INTERFACE_VERSION},
        license="Apache-2.0",
        description=(
            f"Public belief prior {prov.recipe!r} for {metadata.species} ({metadata.unit}) — "
            "reopens as a GridField ResourceField from the bundle without re-running the recipe."
        ),
        provenance=Provenance(
            digest=f"sha256:{prior.content_hash}",
            code_version=prov.recipe_version,
            toolchain_version=f"astro-mine-prospect {_PROSPECT_VERSION}",
            input_hashes=input_hashes,
        ),
        attributes={
            "species": metadata.species,
            "unit": metadata.unit,
            "recipe": prov.recipe,
            "recipe_version": prov.recipe_version,
            "bundle_media_type": BUNDLE_MEDIA_TYPE,
            "bundle_sha256": bundle_sha256,
            "provider_entry_point": PROVIDER_ENTRY_POINT,
        },
    )
