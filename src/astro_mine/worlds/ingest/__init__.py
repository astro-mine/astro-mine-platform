"""Source adapters for external world field layers (worlds.md §3, §5).

New dataset providers/instruments arrive here as **source adapters**, not core changes: each
reads a published raster, reprojects it onto an existing world's CRS/grid, and registers it as
an additional field layer with explicit CRS + per-product provenance, catalogued in STAC. The
first adapters ingest the RM-P1-WORLDS-14 conditioning layers — Diviner measured temperature,
LEND epithermal-neutron water-equivalent hydrogen, and M³ surficial OH/H₂O — that
:mod:`~astro_mine.prospect` conditions real priors on (RM-P1-PROSPECT-12).
"""

from __future__ import annotations

from astro_mine.worlds.ingest._conditioning import (
    CONDITIONING_SPECS,
    DIVINER_TEMPERATURE,
    LEND_WEH,
    M3_WATER,
    ConditioningField,
    ConditioningLayer,
    ConditioningLayerSet,
    ConditioningSpec,
    ingest_conditioning_layers,
    lend_epithermal_to_weh,
    m3_band_depth_to_water,
)

__all__ = [
    "CONDITIONING_SPECS",
    "DIVINER_TEMPERATURE",
    "LEND_WEH",
    "M3_WATER",
    "ConditioningField",
    "ConditioningLayer",
    "ConditioningLayerSet",
    "ConditioningSpec",
    "ingest_conditioning_layers",
    "lend_epithermal_to_weh",
    "m3_band_depth_to_water",
]
