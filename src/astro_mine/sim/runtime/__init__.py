"""Deterministic stepping core (RM-P0-SIM-01).

The scenario loader (:class:`Scenario` / :func:`load_scenario`), a SPICE-time
:class:`SimClock`, the seeded :class:`RngStreams` manager, and the :class:`Simulator`
``reset`` / ``step`` episode loop that implements the Core Environment API — plus
:func:`run_episode` / :class:`Trace`, the canonical reproducibility artifact two same-seed
runs reproduce byte-for-byte.

Backlog: RM-P0-SIM-01 -- astro-mine-sim#1
"""

from __future__ import annotations

from astro_mine.sim.runtime._hub_adapter import HubBundleStore, open_bundle_store
from astro_mine.sim.runtime.clock import SimClock
from astro_mine.sim.runtime.content import (
    PROVIDER_ENTRY_POINT_GROUP,
    BundleStore,
    ContentPin,
    ContentResolver,
    ProviderFactory,
    ResolvedAsset,
    ResolvedContent,
    ScenarioContent,
    SiteConditions,
    agent_spec_from_asset,
    dem_granular_dynamics_from_content,
    granular_dynamics_from_content,
    mjx_dynamics_from_content,
    mobility_dynamics_from_content,
    mujoco_dynamics_from_content,
    site_conditions,
)
from astro_mine.sim.runtime.episode import CORE_INTERFACES, Simulator, Trace, run_episode
from astro_mine.sim.runtime.rng import RngStreams
from astro_mine.sim.runtime.scenario import (
    AgentSpec,
    BraxContactDynamics,
    DemGranularDynamics,
    Dynamics,
    GranularDynamics,
    JointSpec,
    KinematicDynamics,
    ManipulationDynamics,
    MjxContactDynamics,
    MobilityDynamics,
    MujocoMobilityDynamics,
    OrbitalDynamics,
    OrekitOrbitalDynamics,
    Scenario,
    load_scenario,
)
from astro_mine.sim.runtime.timing import (
    EngineTiming,
    TimedEngine,
    TimingRecorder,
    timed_engine_factory,
)

__all__ = [
    "CORE_INTERFACES",
    "PROVIDER_ENTRY_POINT_GROUP",
    "AgentSpec",
    "BraxContactDynamics",
    "BundleStore",
    "ContentPin",
    "ContentResolver",
    "DemGranularDynamics",
    "Dynamics",
    "EngineTiming",
    "GranularDynamics",
    "HubBundleStore",
    "JointSpec",
    "KinematicDynamics",
    "ManipulationDynamics",
    "MjxContactDynamics",
    "MobilityDynamics",
    "MujocoMobilityDynamics",
    "OrbitalDynamics",
    "OrekitOrbitalDynamics",
    "ProviderFactory",
    "ResolvedAsset",
    "ResolvedContent",
    "RngStreams",
    "Scenario",
    "ScenarioContent",
    "SimClock",
    "Simulator",
    "SiteConditions",
    "TimedEngine",
    "TimingRecorder",
    "Trace",
    "agent_spec_from_asset",
    "dem_granular_dynamics_from_content",
    "granular_dynamics_from_content",
    "load_scenario",
    "mjx_dynamics_from_content",
    "mobility_dynamics_from_content",
    "mujoco_dynamics_from_content",
    "open_bundle_store",
    "run_episode",
    "site_conditions",
    "timed_engine_factory",
]
