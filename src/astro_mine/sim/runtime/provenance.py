"""Run-provenance helpers — input/engine identity + content hashing (RM-P0-SIM-09).

The reproduction-grade provenance an :class:`~astro_mine.sim.runtime.episode.Trace` stamps and the
MCAP recording carries needs three deterministic facts beyond the seed: the **content hash of every
input**, the **version of every engine** the run routed through, and a canonical **content digest**
to compute them with. This module owns that small, dependency-light vocabulary.

The digest is deliberately a thin wrapper over ``sha256`` of the same canonical JSON form
:meth:`~astro_mine.sim.runtime.episode.Trace.to_canonical_json` already uses (``sort_keys=True``,
compact separators), so when Core ships its shared content-hash helper (core#19) and run-provenance
schema (core#18) this swaps to them with no digest change. The provenance keys mirror
Core's build-time provenance vocabulary (``source_content_hashes``, ``engine_versions``, ``seed``)
so a Sim run-provenance round-trips cleanly into the future Core ``RunProvenance``.

Engine identity comes off the **constructed** engine's ``EngineDescriptor`` (#65) — it used to be
read from the static descriptor each ``dynamics.kind`` routes to
(RM-P0-SIM-03), so a provenance never drifts from the engines the scenario actually exercises.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from astro_mine.sim.engines.brax import (
    BRAX_CONTACT_ENGINE_DESCRIPTOR,
    MJX_CONTACT_ENGINE_DESCRIPTOR,
)
from astro_mine.sim.engines.dem import DEM_GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.granular import GRANULAR_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.manipulation import MANIPULATION_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.mobility import MOBILITY_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.mujoco import MUJOCO_MOBILITY_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.orbital import ORBITAL_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.orekit import OREKIT_ORBITAL_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.reference import KINEMATIC_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from astro_mine.sim.engines import RegimeEngine
    from astro_mine.sim.engines.adapter import EngineDescriptor
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["content_digest", "engine_versions", "engines_that_ran", "scenario_digest"]

#: ``dynamics.kind`` → the engine descriptor that regime routes to (RM-P0-SIM-03). Read straight off
#: the engines so a provenance's engine versions never drift from the built-in set.
_KIND_DESCRIPTOR: dict[str, EngineDescriptor] = {
    "kinematic": KINEMATIC_ENGINE_DESCRIPTOR,
    "orbital": ORBITAL_ENGINE_DESCRIPTOR,
    "mobility": MOBILITY_ENGINE_DESCRIPTOR,
    "manipulation": MANIPULATION_ENGINE_DESCRIPTOR,
    "granular": GRANULAR_ENGINE_DESCRIPTOR,
    "dem_granular": DEM_GRANULAR_ENGINE_DESCRIPTOR,
    "brax_contact": BRAX_CONTACT_ENGINE_DESCRIPTOR,
    "mjx_contact": MJX_CONTACT_ENGINE_DESCRIPTOR,
    "orekit_orbital": OREKIT_ORBITAL_ENGINE_DESCRIPTOR,
    "mujoco_mobility": MUJOCO_MOBILITY_ENGINE_DESCRIPTOR,
}


def content_digest(data: Any) -> str:
    """The SHA-256 of ``data``'s canonical JSON form — the one content-hash primitive.

    Canonical means ``sort_keys=True`` with compact separators (matching
    :meth:`~astro_mine.sim.runtime.episode.Trace.to_canonical_json`), so the digest is stable across
    runs and key orderings. A thin wrapper to swap for Core's shared helper (core#19) when it ships.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def scenario_digest(scenario: Scenario) -> str:
    """The content hash of the full validated ``scenario`` — its input-identity in provenance.

    Hashes the whole canonical model dump (not just the name), so any change to an agent, a dynamics
    parameter, the seed, or the horizon changes the digest — the input-hash a Bench scenario pins.
    """
    return content_digest(scenario.model_dump(mode="json"))


def engines_that_ran(engine: RegimeEngine) -> tuple[EngineDescriptor, ...]:
    """The descriptors of the engines that actually stepped ``engine``'s agents (#65).

    Duck-typed on two public accessors rather than on concrete types, so the stepping core does not
    have to know which wrappers are in play: ``TimedEngine.inner`` unwraps the timing instrument,
    and ``CoupledEngine.engines`` yields the per-kind sub-engines. Anything else is a single engine
    and reports itself.
    """
    inner = getattr(engine, "inner", engine)
    sub_engines = getattr(inner, "engines", None)
    if sub_engines:
        return tuple(sub.descriptor for sub in sub_engines.values())
    return (inner.descriptor,)


def engine_versions(engine: RegimeEngine) -> dict[str, str]:
    """``engine name → version`` for the regime engines that **actually ran**.

    The provenance's record of *which physics, at what version* produced the run. Read off the
    constructed engines, not off ``AgentSpec.dynamics.kind`` — the declaration and the execution
    were two different things until #65, and a recorded MCAP could name an orbital engine for a run
    in which only the kinematic reference engine ever stepped. Reading the built engine is what
    makes this a measurement rather than a claim.
    """
    return {d.name: d.version for d in engines_that_ran(engine)}
