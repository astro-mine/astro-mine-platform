"""The built-in engine set for the anchor scenario (RM-P0-SIM-03).

One place that names the concrete regime engines Phase-0 ships — the reference kinematic engine plus
the orbital, mobility, manipulation, and granular engines — and registers them through Core's gated
:class:`~astro_mine.sim.engines.registry.EngineRegistry`. A host (the stepping core, Bench) gets the
whole set with :func:`default_engine_registry`, then resolves each engine by name behind the Core
waist. New engines join the tuple append-only.
"""

from __future__ import annotations

from astro_mine.sim.engines.adapter import EngineDescriptor
from astro_mine.sim.engines.brax import (
    BRAX_CONTACT_ENGINE_DESCRIPTOR,
    MJX_CONTACT_ENGINE_DESCRIPTOR,
    brax_contact_engine_factory,
    mjx_contact_engine_factory,
)
from astro_mine.sim.engines.dem import (
    DEM_GRANULAR_ENGINE_DESCRIPTOR,
    dem_granular_engine_factory,
)
from astro_mine.sim.engines.granular import (
    GRANULAR_ENGINE_DESCRIPTOR,
    granular_engine_factory,
)
from astro_mine.sim.engines.manipulation import (
    MANIPULATION_ENGINE_DESCRIPTOR,
    manipulation_engine_factory,
)
from astro_mine.sim.engines.mobility import (
    MOBILITY_ENGINE_DESCRIPTOR,
    mobility_engine_factory,
)
from astro_mine.sim.engines.mujoco import (
    MUJOCO_MOBILITY_ENGINE_DESCRIPTOR,
    mujoco_mobility_engine_factory,
)
from astro_mine.sim.engines.orbital import (
    ORBITAL_ENGINE_DESCRIPTOR,
    orbital_engine_factory,
)
from astro_mine.sim.engines.orekit import (
    OREKIT_ORBITAL_ENGINE_DESCRIPTOR,
    orekit_orbital_engine_factory,
)
from astro_mine.sim.engines.reference import (
    KINEMATIC_ENGINE_DESCRIPTOR,
    kinematic_engine_factory,
)
from astro_mine.sim.engines.registry import EngineFactory, EngineRegistry

__all__ = [
    "BUILTIN_ENGINES",
    "default_engine_registry",
    "register_builtin_engines",
]

#: The engine set as ``(descriptor, factory)`` pairs, in registration order: the reference kinematic
#: engine, the four Phase-0 reduced-order anchor-scenario engines (RM-P0-SIM-03), the high-fidelity
#: DEM granular tier (RM-P1-SIM-06), the Brax JAX kernel + the MJX contact tier (RM-P1-SIM-04), and
#: the two **real-backend** tiers RM-P0-SIM-03 names — Orekit (orbital) and MuJoCo (articulated
#: wheel-soil contact) — each sitting behind the same waist as its reduced-order counterpart, which
#: remains the always-works local fallback (CX-LOCAL).
#:
#: Every heavy solver (numpy, JAX, MuJoCo, the Orekit JVM) loads **lazily** inside its factory, so
#: registering the whole set here needs none of the ``[dem]`` / ``[brax]`` / ``[mujoco]`` /
#: ``[orekit]`` extras — a manifest is registrable without the backend it describes. New engines
#: join append-only.
BUILTIN_ENGINES: tuple[tuple[EngineDescriptor, EngineFactory], ...] = (
    (KINEMATIC_ENGINE_DESCRIPTOR, kinematic_engine_factory),
    (ORBITAL_ENGINE_DESCRIPTOR, orbital_engine_factory),
    (MOBILITY_ENGINE_DESCRIPTOR, mobility_engine_factory),
    (MANIPULATION_ENGINE_DESCRIPTOR, manipulation_engine_factory),
    (GRANULAR_ENGINE_DESCRIPTOR, granular_engine_factory),
    (DEM_GRANULAR_ENGINE_DESCRIPTOR, dem_granular_engine_factory),
    (BRAX_CONTACT_ENGINE_DESCRIPTOR, brax_contact_engine_factory),
    (MJX_CONTACT_ENGINE_DESCRIPTOR, mjx_contact_engine_factory),
    (OREKIT_ORBITAL_ENGINE_DESCRIPTOR, orekit_orbital_engine_factory),
    (MUJOCO_MOBILITY_ENGINE_DESCRIPTOR, mujoco_mobility_engine_factory),
)


def register_builtin_engines(registry: EngineRegistry) -> EngineRegistry:
    """Register every built-in engine into ``registry`` (gated through Core), and return it."""
    for descriptor, factory in BUILTIN_ENGINES:
        registry.register(descriptor, factory)
    return registry


def default_engine_registry() -> EngineRegistry:
    """A fresh :class:`~astro_mine.sim.engines.registry.EngineRegistry` with every built-in
    engine registered. Signing is off (the Phase-0 local tier runs with no key material,
    CX-LOCAL); a hardened host builds its own registry with a verifier."""
    return register_builtin_engines(EngineRegistry())
