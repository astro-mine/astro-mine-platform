# SPDX-License-Identifier: Apache-2.0
"""The DEM engine's static self-declaration — numpy-free (RM-P1-SIM-06).

Kept in its own module so :mod:`astro_mine.sim.engines.dem` (and therefore
``engines/builtins.py``) can expose the descriptor and register the engine **without
importing numpy** — the numpy solver is pulled in only when the factory actually builds an
engine (the ``[dem]`` extra). A pure :class:`~astro_mine.sim.engines.adapter.EngineDescriptor`
of Core enums, this imports nothing heavy.
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines.adapter import EngineDescriptor, FidelityDescriptor

__all__ = ["DEM_GRANULAR_ENGINE_DESCRIPTOR"]

#: The DEM granular engine's self-declaration: a surface tool-soil contact tier in the lunar
#: body-fixed frame. Determinism is ``TOLERANCE`` — the O(N²) float contact sums are not
#: bit-portable across builds, so it is gated by the analytic terramechanics oracle and
#: in-process reproducibility (sim.md §11), not a golden hash. It reuses the highest existing
#: non-surrogate rung, ``ARTICULATED`` (the tier the Fleet excavator SADF declares for "bucket
#: joint + wheel/soil contact"); a dedicated ground-truth tier would be an append-only Core RFC.
DEM_GRANULAR_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.dem_granular",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.ARTICULATED),
)
