"""Build a Core plugin manifest for a SADF asset (RM-P0-FLEET-06).

Fleet **consumes** Core's manifest schema -- it does not invent one (``fleet.md`` §1,
§6). This maps a validated :class:`~astro_mine.core.sadf.SadfDocument` onto a
:class:`~astro_mine.core.registry.PluginManifest` (``kind=asset``) so each packaged
asset is a discoverable, signature-gated plugin whose ``provenance.digest`` is the
asset's content identity and whose ``capability_tags`` are the *same* Core-owned
vocabulary the SADF asset declares. The signature is attached to the OCI artifact
separately (see :mod:`astro_mine.core.registry` and :mod:`astro_mine.fleet.packaging.oci`),
so the manifest embedded as the OCI config stays byte-stable.

Backlog: RM-P0-FLEET-06 -- astro-mine-fleet#6
"""

from __future__ import annotations

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.core.sadf import SadfDocument
from astro_mine.fleet import __version__ as _FLEET_VERSION
from astro_mine.fleet._core import CORE_INTERFACES

__all__ = ["build_plugin_manifest"]


def build_plugin_manifest(
    doc: SadfDocument,
    *,
    asset_digest: str,
    source_content_hashes: dict[str, str] | None = None,
) -> PluginManifest:
    """Build the ``kind=asset`` plugin manifest for *doc* (unsigned; caller attaches it)."""
    asset = doc.asset
    identity = asset.identity
    provenance = Provenance(
        digest=asset_digest,
        toolchain_version=_FLEET_VERSION,
        source_content_hashes=dict(source_content_hashes or {}),
    )
    return PluginManifest(
        name=identity.id,
        version=identity.version,
        kind=PluginKind.ASSET,
        core_interfaces=dict(asset.core_interface_versions) or dict(CORE_INTERFACES),
        capability_tags=list(asset.capabilities),
        description=identity.description,
        provenance=provenance,
        attributes={"asset_kind": identity.kind, "asset_name": identity.name},
    )
