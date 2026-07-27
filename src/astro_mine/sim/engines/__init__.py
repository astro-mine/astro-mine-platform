"""Engine-adapter framework — the ``RegimeEngine`` plugin (RM-P0-SIM-02).

The plugin seam that routes physics engines behind the Core Environment waist: the
:class:`RegimeEngine` adapter (the coupling triad ``advance`` / ``export_coupling_state``
/ ``import_coupling_state`` plus a ``retire`` hook), the introspectable
:class:`EngineDescriptor` (frames + determinism class + :class:`FidelityDescriptor`) that
renders to a Core plugin manifest, the :class:`CouplingState` boundary payload, and the
:class:`EngineRegistry` that gates engine loads through Core's plugin registry
(RM-P0-CORE-05).

The reference :class:`KinematicEngine` is the trivial engine that drives the stepping core
today and the always-works local tier; the concrete anchor-scenario engines —
:class:`~astro_mine.sim.engines.orbital.OrbitalEngine`,
:class:`~astro_mine.sim.engines.mobility.MobilityEngine`,
:class:`~astro_mine.sim.engines.manipulation.ManipulationEngine`, and
:class:`~astro_mine.sim.engines.granular.GranularEngine` — plug in behind this same contract
(RM-P0-SIM-03) and are bundled by :func:`~astro_mine.sim.engines.builtins.default_engine_registry`.

Backlog: RM-P0-SIM-02, RM-P0-SIM-03
https://github.com/astro-mine/astro-mine-sim/issues/2
"""

from __future__ import annotations

from astro_mine.sim.engines.actuation import actions_by_agent
from astro_mine.sim.engines.adapter import (
    ENGINE_CORE_INTERFACES,
    CouplingState,
    EngineDescriptor,
    FidelityDescriptor,
    RegimeEngine,
)
from astro_mine.sim.engines.brax import (
    BRAX_CONTACT_ENGINE_DESCRIPTOR,
    MJX_CONTACT_ENGINE_DESCRIPTOR,
    brax_contact_engine_factory,
    mjx_contact_engine_factory,
)
from astro_mine.sim.engines.builtins import (
    BUILTIN_ENGINES,
    default_engine_registry,
    register_builtin_engines,
)
from astro_mine.sim.engines.dem import (
    DEM_GRANULAR_ENGINE_DESCRIPTOR,
    dem_granular_engine_factory,
)
from astro_mine.sim.engines.granular import (
    GRANULAR_ENGINE_DESCRIPTOR,
    GranularEngine,
    granular_engine_factory,
)
from astro_mine.sim.engines.manipulation import (
    MANIPULATION_ENGINE_DESCRIPTOR,
    ManipulationEngine,
    manipulation_engine_factory,
)
from astro_mine.sim.engines.mobility import (
    MOBILITY_ENGINE_DESCRIPTOR,
    MobilityEngine,
    mobility_engine_factory,
)
from astro_mine.sim.engines.mujoco import (
    MUJOCO_MOBILITY_ENGINE_DESCRIPTOR,
    mujoco_mobility_engine_factory,
)
from astro_mine.sim.engines.orbital import (
    ORBITAL_ENGINE_DESCRIPTOR,
    OrbitalEngine,
    orbital_engine_factory,
)
from astro_mine.sim.engines.orekit import (
    OREKIT_ORBITAL_ENGINE_DESCRIPTOR,
    orekit_orbital_engine_factory,
)
from astro_mine.sim.engines.reference import (
    KINEMATIC_ENGINE_DESCRIPTOR,
    KinematicEngine,
    kinematic_engine_factory,
)
from astro_mine.sim.engines.registry import EngineFactory, EngineRegistry

__all__ = [
    "BRAX_CONTACT_ENGINE_DESCRIPTOR",
    "BUILTIN_ENGINES",
    "DEM_GRANULAR_ENGINE_DESCRIPTOR",
    "ENGINE_CORE_INTERFACES",
    "GRANULAR_ENGINE_DESCRIPTOR",
    "KINEMATIC_ENGINE_DESCRIPTOR",
    "MANIPULATION_ENGINE_DESCRIPTOR",
    "MJX_CONTACT_ENGINE_DESCRIPTOR",
    "MOBILITY_ENGINE_DESCRIPTOR",
    "MUJOCO_MOBILITY_ENGINE_DESCRIPTOR",
    "ORBITAL_ENGINE_DESCRIPTOR",
    "OREKIT_ORBITAL_ENGINE_DESCRIPTOR",
    "CouplingState",
    "EngineDescriptor",
    "EngineFactory",
    "EngineRegistry",
    "FidelityDescriptor",
    "GranularEngine",
    "KinematicEngine",
    "ManipulationEngine",
    "MobilityEngine",
    "OrbitalEngine",
    "RegimeEngine",
    "actions_by_agent",
    "brax_contact_engine_factory",
    "default_engine_registry",
    "dem_granular_engine_factory",
    "granular_engine_factory",
    "kinematic_engine_factory",
    "manipulation_engine_factory",
    "mjx_contact_engine_factory",
    "mobility_engine_factory",
    "mujoco_mobility_engine_factory",
    "orbital_engine_factory",
    "orekit_orbital_engine_factory",
    "register_builtin_engines",
]
