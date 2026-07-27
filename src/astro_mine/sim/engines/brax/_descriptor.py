"""The Brax/MJX engine's static self-declaration — JAX-free (RM-P1-SIM-04).

Kept in its own module so :mod:`astro_mine.sim.engines.brax` (and therefore
``engines/builtins.py``) can expose the descriptor and register the engine **without
importing JAX** — the JAX/Brax/MJX kernel is pulled in only when the factory actually
builds an engine (the ``[brax]`` extra). A pure
:class:`~astro_mine.sim.engines.adapter.EngineDescriptor` of Core enums, this imports
nothing heavy.
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

__all__ = ["BRAX_CONTACT_ENGINE_DESCRIPTOR"]

#: The Brax/MJX contact engine's self-declaration: a GPU-vectorizable surface mobility/contact
#: tier in the lunar body-fixed frame, the fast-contact **training** engine Learn's swarm-scale
#: rollouts consume (sim.md §8, §11 "Brax for differentiable/JAX-native massively parallel
#: rollouts"). It sits at the ``KINEMATIC`` rung of the surface fidelity ladder — the same rung
#: the reduced-order :class:`~astro_mine.sim.engines.mobility.MobilityEngine` declares, so the
#: analytic mobility oracle cross-checks it — and advertises ``MOBILITY_WHEELED`` for
#: task↔asset matching (the tag is un-gated; open assets may declare it).
#:
#: Determinism is ``TOLERANCE``, **not** ``BIT_EXACT``, and deliberately gated by a documented
#: tolerance rather than a golden hash (sim.md §11): JAX/XLA is deterministic per process on a
#: fixed CPU build (same seed ⇒ same trace), but its floating-point reductions are
#: **non-associative** and the reduction/fusion order is **not bit-portable** across builds —
#: CPU↔GPU or across XLA versions the last bits differ. A bit-exact golden trace would therefore
#: be a false gate; the engine is instead admitted against an explicit error budget (the analytic
#: drawbar-pull oracle) plus in-process reproducibility, exactly as the other ``TOLERANCE`` tiers
#: (orbital RK4, mobility ``sqrt``) are.
BRAX_CONTACT_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.brax_contact",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.KINEMATIC),
    capability_tags=(CapabilityTag.MOBILITY_WHEELED,),
)
