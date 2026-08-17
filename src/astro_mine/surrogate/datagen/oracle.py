# SPDX-License-Identifier: Apache-2.0
"""The high-fidelity oracle seam ``datagen`` labels against (RM-P1-SURR-03; surrogate.md §3).

``datagen`` needs *labeled* rollouts — a ``config -> particle trajectory`` map produced by a
high-fidelity source. The real source is astro-mine-sim's SIM-06 DEM engine, but the surrogate
package **never imports Sim** (the narrow waist; conventions.md §1.1): the dependency is inverted
behind a **Core-typed seam** — a :class:`RolloutOracle` Protocol — exactly as
:mod:`astro_mine.bench` injects a Sim ``EpisodeRunner``. A caller with the ``[datagen]`` extra
supplies a Sim-backed adapter (a guarded script, never imported at package import); everyone else —
CI included — uses :func:`reference_rollout_oracle`, a numpy-only synthetic granular proxy that is
always available and fully deterministic.

This module imports **only** numpy — never ``astro_mine.sim``, never torch, never scipy — so the
seam stays in the pure-numpy datagen layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "REFERENCE_BED_WIDTH_M",
    "REFERENCE_DT_S",
    "REFERENCE_FEATURE_NAMES",
    "REFERENCE_N_PARTICLES",
    "REFERENCE_STEPS",
    "REFERENCE_TOOL_HEIGHT_M",
    "REFERENCE_TOOL_X0_M",
    "RolloutOracle",
    "RolloutSample",
    "reference_rollout_oracle",
]

FloatArray = npt.NDArray[np.float64]

#: The reference proxy's fixed rig — matches the DEM fixture's convention (dataset.py) so a
#: dataset it generates featurizes identically under the GNS node/edge features (bed width, blade).
REFERENCE_N_PARTICLES = 24
REFERENCE_STEPS = 24
REFERENCE_DT_S = 0.02
REFERENCE_BED_WIDTH_M = 0.4
REFERENCE_TOOL_HEIGHT_M = 0.1
REFERENCE_TOOL_X0_M = 0.04
REFERENCE_FEATURE_NAMES = ("pos_x", "pos_z", "vel_x", "vel_z")

# Proxy dynamics constants — small, bounded motions in the bed (finite residuals are all the
# conformal layer needs; the physics need only be smooth and config-dependent, not realistic).
_PARTICLE_RADIUS_M = 0.02
_GRID_SPACING_M = 0.035
_GRID_COLS = 8
_GRID_X0_M = 0.06
_GRID_Z0_M = 0.03
_GRAVITY = 0.2  # effective settling acceleration (m/s^2), scaled for a short bounded rollout
_TOOL_REACH_M = 0.08
_PUSH_GAIN = 0.6
_INIT_JITTER_M = 1.0e-3
_REFERENCE_DENSITY = 1400.0  # inverse-mass reference: heavier soil accelerates less


def _initial_grid() -> FloatArray:
    """A packed particle lattice within the bed — spaced so the radius graph connects."""
    index = np.arange(REFERENCE_N_PARTICLES)
    x = _GRID_X0_M + _GRID_SPACING_M * (index % _GRID_COLS)
    z = _GRID_Z0_M + _GRID_SPACING_M * (index // _GRID_COLS)
    return np.column_stack([x, z]).astype(np.float64)


@dataclass(frozen=True)
class RolloutSample:
    """One labeled high-fidelity rollout for a single excavation config.

    ``states`` is ``(T+1, N, 4)`` — ``T+1`` timesteps of ``N`` particles' ``(pos_x, pos_z, vel_x,
    vel_z)``; ``tool_x`` is the blade position ``(T+1,)``; ``params`` is the ``(P,)`` config that
    produced it. Stacked over configs, these are exactly a
    :class:`~astro_mine.surrogate.models.dataset.DemDataset`.
    """

    states: FloatArray
    tool_x: FloatArray
    params: FloatArray


@runtime_checkable
class RolloutOracle(Protocol):
    """The high-fidelity producer ``datagen`` labels against — pure and deterministic.

    ``__call__(config, seed)`` returns the :class:`RolloutSample` for one excavation ``config``
    ``(P,)``; ``seed`` seeds any stochastic settling so the same ``(config, seed)`` always yields
    the same rollout (the reproducibility contract, surrogate.md §5). Being ``runtime_checkable``,
    a Sim-backed adapter passes :func:`isinstance` structurally — the seam is behavioural, not a
    base class to inherit (mirroring :class:`~astro_mine.surrogate.model.SurrogateModel`).
    """

    def __call__(self, config: FloatArray, seed: int) -> RolloutSample:
        """Produce the labeled rollout for ``config`` seeded by ``seed``."""
        ...


def reference_rollout_oracle(config: FloatArray, seed: int) -> RolloutSample:
    """A numpy-only synthetic granular proxy — the always-available, CI-safe :class:`RolloutOracle`.

    Rolls ``N`` particles under a simple, bounded, **config-dependent** dynamics: soil density sets
    the inverse mass, the tool pushes particles just ahead of the blade at ``tool_speed``, friction
    damps velocity, gravity settles the bed, and floor/wall contacts reflect with ``restitution``.
    Deterministic given ``(config, seed)`` (``seed`` perturbs only the initial packing). It is a
    *proxy*, not physics — enough to exercise the full datagen → train → calibrate → gate path with
    finite residuals; the real oracle is the DEM engine behind the ``[datagen]`` extra.
    """
    cfg = np.asarray(config, dtype=np.float64).reshape(-1)
    density, friction, restitution, tool_speed = (
        float(cfg[0]),
        float(cfg[1]),
        float(cfg[2]),
        float(cfg[3]),
    )
    inv_mass = _REFERENCE_DENSITY / max(density, 1.0)
    damping = 1.0 - min(0.9, friction * 0.3)
    rng = np.random.default_rng(seed)

    pos = _initial_grid() + rng.normal(0.0, _INIT_JITTER_M, size=(REFERENCE_N_PARTICLES, 2))
    vel = np.zeros((REFERENCE_N_PARTICLES, 2), dtype=np.float64)
    tool_x = REFERENCE_TOOL_X0_M

    states = np.zeros((REFERENCE_STEPS + 1, REFERENCE_N_PARTICLES, 4), dtype=np.float64)
    tool_hist = np.zeros(REFERENCE_STEPS + 1, dtype=np.float64)
    states[0, :, :2], states[0, :, 2:] = pos, vel
    tool_hist[0] = tool_x

    for t in range(1, REFERENCE_STEPS + 1):
        ahead = pos[:, 0] - tool_x  # >0 for particles to the right of the blade
        push = _PUSH_GAIN * tool_speed * np.exp(-np.clip(ahead, 0.0, None) / _TOOL_REACH_M)
        push = np.where(ahead > -_PARTICLE_RADIUS_M, push, 0.0)
        accel = np.column_stack(
            [push * inv_mass, np.full(REFERENCE_N_PARTICLES, -_GRAVITY * inv_mass)]
        )
        vel = (vel + accel * REFERENCE_DT_S) * damping
        pos = pos + vel * REFERENCE_DT_S
        # Floor contact at z=0: reflect and lose (1 - restitution) of the normal velocity.
        below = pos[:, 1] < 0.0
        pos[below, 1] = -pos[below, 1]
        vel[below, 1] = -restitution * vel[below, 1]
        # Side walls at x in [0, bed_width]: reflect with restitution.
        left = pos[:, 0] < 0.0
        pos[left, 0] = -pos[left, 0]
        vel[left, 0] = -restitution * vel[left, 0]
        right = pos[:, 0] > REFERENCE_BED_WIDTH_M
        pos[right, 0] = 2.0 * REFERENCE_BED_WIDTH_M - pos[right, 0]
        vel[right, 0] = -restitution * vel[right, 0]
        tool_x += tool_speed * REFERENCE_DT_S
        states[t, :, :2], states[t, :, 2:] = pos, vel
        tool_hist[t] = tool_x

    return RolloutSample(states=states, tool_x=tool_hist, params=cfg.copy())
