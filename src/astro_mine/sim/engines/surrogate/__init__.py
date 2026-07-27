"""The learned-surrogate granular fidelity tier (RM-P1-SIM-03).

Sim loads a content-addressed [Surrogate](surrogate.md) ONNX artifact through Core contracts
(:func:`load_surrogate_tier`, fail-closed on the signed manifest) and runs it as an
:class:`AdaptiveGranularEngine` — the cheap tier the scheduler substitutes for the DEM ground truth
within budget, escalating back mid-episode on drift/OOD. Sim never imports ``astro_mine.surrogate``;
the tier arrives as ONNX + a Core ``PluginManifest``. numpy/onnxruntime are the ``[surrogate]``
extra, imported lazily so the base engine set stays free of them.
"""

from __future__ import annotations

from astro_mine.sim.engines.surrogate._descriptor import SURROGATE_GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.surrogate._engine import (
    AdaptiveGranularEngine,
    build_scheduled_granular_engine,
    build_surrogate_granular_engine,
)
from astro_mine.sim.engines.surrogate._loader import (
    LoadedSurrogate,
    SurrogateIntegrityError,
    load_surrogate_tier,
)

__all__ = [
    "SURROGATE_GRANULAR_ENGINE_DESCRIPTOR",
    "AdaptiveGranularEngine",
    "LoadedSurrogate",
    "SurrogateIntegrityError",
    "build_scheduled_granular_engine",
    "build_surrogate_granular_engine",
    "load_surrogate_tier",
]
