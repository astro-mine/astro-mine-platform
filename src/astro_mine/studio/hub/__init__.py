"""The →Hub / ←Hub seam (RM-P1-STUDIO-06; studio.md §6).

Studio reads assets, worlds, and policies from Hub **by content hash**, and writes validated designs
and campaigns back as **content-addressed, signed** artifacts indexed by the Core plugin manifest.
Both directions are typed Protocols with an injected implementation, like the rest of Studio's
sibling seams — so the API layer never imports Hub, and a test binds a temporary registry.

This subpackage lives behind the ``[hub]`` extra. The base wheel imports only Core (+ FastAPI); a
sibling-package import in the base wheel would break the narrow-waist constraint (studio.md §2).
"""

from __future__ import annotations

from .catalog import (
    AssetCatalog,
    AssetPreviewMaterializer,
    HubAssetCatalog,
    HubAssetPreviewMaterializer,
    MaterializedPreview,
    MenuEntry,
    PreviewError,
    StudioCatalog,
    WorldCatalog,
    WorldEntry,
)
from .materialize import (
    HubWorldMaterializer,
    MaterializedWorld,
    MaterializeError,
    WorldMaterializer,
)
from .publish import (
    CAMPAIGN_ARTIFACT_KIND,
    CAMPAIGN_LAYER_MEDIA_TYPE,
    DESIGN_ARTIFACT_KIND,
    DESIGN_LAYER_MEDIA_TYPE,
    ArtifactPublisher,
    CapabilityResolver,
    HubArtifactPublisher,
    HubCapabilityResolver,
    PublishedArtifactRef,
    PublishError,
    build_campaign_manifest,
    build_design_manifest,
)

__all__ = [
    "CAMPAIGN_ARTIFACT_KIND",
    "CAMPAIGN_LAYER_MEDIA_TYPE",
    "DESIGN_ARTIFACT_KIND",
    "DESIGN_LAYER_MEDIA_TYPE",
    "ArtifactPublisher",
    "AssetCatalog",
    "AssetPreviewMaterializer",
    "CapabilityResolver",
    "HubArtifactPublisher",
    "HubAssetCatalog",
    "HubAssetPreviewMaterializer",
    "HubCapabilityResolver",
    "HubWorldMaterializer",
    "MaterializeError",
    "MaterializedPreview",
    "MaterializedWorld",
    "MenuEntry",
    "PreviewError",
    "PublishError",
    "PublishedArtifactRef",
    "StudioCatalog",
    "WorldCatalog",
    "WorldEntry",
    "WorldMaterializer",
    "build_campaign_manifest",
    "build_design_manifest",
]
