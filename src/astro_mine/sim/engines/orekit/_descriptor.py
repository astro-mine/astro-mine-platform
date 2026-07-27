"""The Orekit orbital engine's static self-declaration — JVM-free (RM-P0-SIM-03).

Kept in its own module so :mod:`astro_mine.sim.engines.orekit` (and therefore
``engines/builtins.py``) can expose the descriptor and **register the engine's manifest without
booting a JVM** — the Orekit binding is pulled in only when the factory actually builds an engine
(the ``[orekit]`` extra). A pure :class:`~astro_mine.sim.engines.adapter.EngineDescriptor` of Core
enums; this imports nothing heavy.
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines.adapter import EngineDescriptor, FidelityDescriptor

__all__ = ["OREKIT_ORBITAL_ENGINE_DESCRIPTOR"]

#: The Orekit orbital engine's self-declaration: the **higher-fidelity** orbital tier (sim.md §4,
#: §11 "Basilisk + Orekit ... GMAT/STK as oracles only"), sitting behind the same orbital adapter as
#: the reduced-order :class:`~astro_mine.sim.engines.orbital.OrbitalEngine`.
#:
#: Where the reduced-order tier is a fixed-step RK4 integration of pure two-body motion, this tier
#: is Orekit's **adaptive Dormand-Prince 8(5,3)** numerical propagator carrying a real force model —
#: Newtonian central gravity **plus the body's J2 oblateness term**. So it is higher-fidelity in two
#: independent senses: a tighter, error-controlled integration *and* a richer dynamical model (a
#: perturbation the two-body tier cannot represent at all).
#:
#: It stays at the ``MASSMODEL`` rung of the fidelity ladder — an orbiter is a point mass either
#: way; the ladder rung describes the *body* model, not the force model — so the multi-fidelity
#: scheduler treats the two orbital tiers as interchangeable, which is exactly what makes the
#: reduced-order tier a valid CX-LOCAL fallback for it.
#:
#: Determinism is ``TOLERANCE``, not ``BIT_EXACT``: an adaptive-step integrator's step sequence, and
#: the JVM's floating-point reductions, are not bit-portable across builds. Like the other TOLERANCE
#: tiers (the RK4 orbital engine, the ``sqrt``-based mobility engine) it is admitted against an
#: explicit **error budget** — the closed-form Keplerian oracle, with J2 disabled — rather than a
#: golden hash (sim.md §11; conventions.md §11).
OREKIT_ORBITAL_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.orekit_orbital",
    version="0.1.0",
    regimes=(Regime.PROXIMITY_ORBIT,),
    frames=(INERTIAL_J2000,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.MASSMODEL),
)
