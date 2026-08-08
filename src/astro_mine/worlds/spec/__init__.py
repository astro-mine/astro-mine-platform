"""WorldSpec — content-addressed world bundle, STAC catalog, 3D-Tiles export.

The declarative :class:`WorldSpec` (authored as YAML, validated by JSON Schema) plus
:func:`build_world_bundle`, which composes the content-addressed layer products of
RM-P0-WORLDS-01..06 into a :class:`WorldBundle`: the layer COGs, a STAC catalog, a 3D-Tiles
terrain export, and a ``world.json`` manifest whose ``world_hash`` is reproducible from the spec +
pinned toolchain, so Bench can pin a world by hash and View can render it (worlds.md §5, §12).

Backlog: RM-P0-WORLDS-07 — astro-mine-worlds#7
"""

from __future__ import annotations

from astro_mine.worlds.spec._bundle import WorldBundle, build_world_bundle
from astro_mine.worlds.spec._example import EXAMPLE_RESOURCE, example_world_spec_text
from astro_mine.worlds.spec._json_schema import (
    SCHEMA_RESOURCE,
    published_json_schema,
    published_json_schema_text,
)
from astro_mine.worlds.spec._model import (
    JSON_SCHEMA_DIALECT,
    WORLDSPEC_SCHEMA_ID,
    LayerSpec,
    Region,
    SourceRef,
    WorldSpec,
)
from astro_mine.worlds.spec._publish import (
    BUNDLE_LAYER_MEDIA_TYPE,
    WORLD_ARTIFACT_KIND,
    build_world_manifest,
    publish_world_bundle,
)

__all__ = [
    "BUNDLE_LAYER_MEDIA_TYPE",
    "EXAMPLE_RESOURCE",
    "JSON_SCHEMA_DIALECT",
    "SCHEMA_RESOURCE",
    "WORLDSPEC_SCHEMA_ID",
    "WORLD_ARTIFACT_KIND",
    "LayerSpec",
    "Region",
    "SourceRef",
    "WorldBundle",
    "WorldSpec",
    "build_world_bundle",
    "build_world_manifest",
    "example_world_spec_text",
    "publish_world_bundle",
    "published_json_schema",
    "published_json_schema_text",
]
