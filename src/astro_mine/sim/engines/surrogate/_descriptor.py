# SPDX-License-Identifier: Apache-2.0
"""The learned-surrogate granular tier descriptor (RM-P1-SIM-03).

A ``SURROGATE``-tier :class:`~astro_mine.sim.engines.adapter.EngineDescriptor` for the excavation
regime — the cheap, calibrated tier the scheduler substitutes for the DEM ground truth within
budget. ``TOLERANCE`` determinism: ONNX Runtime is single-threaded/seeded here, but float results
are not bit-portable across builds (like the DEM engine), so CI gates by tolerance.
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import (
    DeterminismClass,
    FidelityTier,
    Regime,
    SurrogatePhysicsDomain,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines.adapter import EngineDescriptor, FidelityDescriptor

__all__ = ["SURROGATE_GRANULAR_ENGINE_DESCRIPTOR"]

SURROGATE_GRANULAR_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.surrogate.granular",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(
        tier=FidelityTier.SURROGATE,
        surrogate_domain=SurrogatePhysicsDomain.GRANULAR_EXCAVATION,
    ),
)
