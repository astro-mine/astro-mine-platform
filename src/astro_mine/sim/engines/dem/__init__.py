"""High-fidelity DEM granular-excavation engine (RM-P1-SIM-06) — the surrogate's oracle.

The ground-truth tool-soil contact tier behind the reduced-order ``GranularEngine`` seam: a
2D soft-sphere discrete-element bed a blade excavates, producing the ``(state, action) ->
next_state`` particle dynamics a [Surrogate](surrogate.md) learns (sim.md §4, §11). Project
Chrono / Taichi-MPM class methods at production scale; this CPU reference tier is the
always-works local realization.

**numpy-free package surface by design.** This module exposes only the engine's
:data:`DEM_GRANULAR_ENGINE_DESCRIPTOR` (registered in ``engines/builtins.py``) and a factory
whose body imports the numpy solver *lazily* — so importing the engine set, and registering
the DEM engine's manifest, need no numpy. numpy arrives with the ``[dem]`` extra; a scenario
that actually selects the DEM tier calls the factory, which then requires it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.sim.engines.dem._descriptor import DEM_GRANULAR_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from astro_mine.sim.engines.adapter import RegimeEngine
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["DEM_GRANULAR_ENGINE_DESCRIPTOR", "dem_granular_engine_factory"]


def dem_granular_engine_factory(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
    """Build the DEM engine for a scenario's ``dem_granular`` agents (needs the ``[dem]`` extra).

    Lazy-imports the numpy kernel so the engine set stays importable — and the manifest
    registrable — without numpy. Raises a clear :class:`ModuleNotFoundError` (from the import)
    only if a scenario actually selects the DEM tier without ``astro-mine-platform[sim-dem]``
    installed.
    """
    from astro_mine.sim.engines.dem._engine import build_dem_engine

    return build_dem_engine(scenario, rng)
