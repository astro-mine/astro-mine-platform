"""The MuJoCo mobility engine's static self-declaration — MuJoCo-free (RM-P0-SIM-03).

In its own module so :mod:`astro_mine.sim.engines.mujoco` (and ``engines/builtins.py``) can expose
the descriptor and register the engine's manifest **without importing MuJoCo** — the contact solver
arrives only when the factory builds an engine (the ``[mujoco]`` extra).
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import (
    CapabilityTag,
    DeterminismClass,
    FidelityTier,
    Regime,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines.adapter import EngineDescriptor, FidelityDescriptor

__all__ = ["MUJOCO_MOBILITY_ENGINE_DESCRIPTOR"]

#: The MuJoCo mobility engine's self-declaration: the **articulated wheel-soil contact** tier
#: (RM-P0-SIM-03 "mobility/contact (MuJoCo/Brax, CPU-capable) for rovers"; sim.md §4, §11).
#:
#: It is the ``ARTICULATED`` rung — a genuine step up the fidelity ladder from the reduced-order
#: :class:`~astro_mine.sim.engines.mobility.MobilityEngine`, which sits at ``KINEMATIC``. The
#: difference is not a tuning constant: the reduced-order tier *is* a closed-form formula (ramp the
#: velocity toward a setpoint under an ``a = F/m`` cap), whereas this tier simulates a chassis and
#: four torque-driven wheels in frictional contact with the ground. The rover here can slip, sink,
#: pitch under acceleration, and fail to climb — none of which the kinematic tier can represent, and
#: all of which are what "traversability" actually means on regolith.
#:
#: Determinism is ``TOLERANCE``: MuJoCo's solver is deterministic for a fixed seed and build, but
#: its contact iterations are not bit-portable across builds/platforms. Like the other TOLERANCE
#: tiers it is admitted against an explicit error budget — the analytic drawbar-pull oracle — rather
#: than a golden hash (sim.md §11; conventions.md §11).
MUJOCO_MOBILITY_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.mujoco_mobility",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.ARTICULATED),
    capability_tags=(CapabilityTag.MOBILITY_WHEELED,),
)
