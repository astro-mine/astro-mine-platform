# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo articulated mobility/contact tier (RM-P0-SIM-03) — real wheel-soil contact.

The contact-rich surface backend RM-P0-SIM-03 names, behind the *same* ``RegimeEngine`` waist as the
reduced-order kinematic mobility engine, so selecting it is **configuration** (a scenario's
``dynamics.kind``) and no MuJoCo type leaks through the Core Environment API. The reduced-order tier
stays the always-works local fallback (CX-LOCAL); this is the tier you select when wheel slip,
sinkage, and the friction cone actually matter — none of which a closed-form velocity ramp can
represent.

**MuJoCo-free package surface by design.** This module exposes only the engine's
:data:`MUJOCO_MOBILITY_ENGINE_DESCRIPTOR` (registered in ``engines/builtins.py``) and a factory
whose body imports the contact solver *lazily* — so importing the engine set, and registering the
manifest, need no MuJoCo. It arrives with the ``[mujoco]`` extra; a scenario that actually selects
the tier calls the factory, which then requires it and otherwise raises a clean
:class:`ModuleNotFoundError` naming the extra.

The rover's physical description is shared with the Brax/MJX GPU tier
(:mod:`astro_mine.sim.engines._rover_mjcf`), so the CPU contact tier and the batched GPU contact
tier step *the same machine* — they differ only in execution substrate (sim.md §11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.sim.engines.mujoco._descriptor import MUJOCO_MOBILITY_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from astro_mine.sim.engines.adapter import RegimeEngine
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = [
    "MUJOCO_MOBILITY_ENGINE_DESCRIPTOR",
    "mujoco_mobility_engine_factory",
]

_MUJOCO_HINT = (
    "the MuJoCo articulated mobility tier requires the MuJoCo contact solver (mujoco, numpy); "
    "install it with: pip install 'astro-mine-platform[sim-mujoco]'"
)


def mujoco_mobility_engine_factory(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
    """Build the MuJoCo mobility engine for a scenario's ``mujoco_mobility`` agents (``[mujoco]``).

    Lazy-imports the contact solver so the engine set stays importable — and the manifest
    registrable — without MuJoCo. Raises a clear :class:`ModuleNotFoundError` naming
    ``astro-mine-platform[sim-mujoco]``
    only when a scenario actually selects the tier without it."""
    try:
        from astro_mine.sim.engines.mujoco._engine import build_mujoco_mobility_engine
    except ModuleNotFoundError as exc:  # mujoco/numpy absent
        raise ModuleNotFoundError(_MUJOCO_HINT) from exc
    return build_mujoco_mobility_engine(scenario, rng)
