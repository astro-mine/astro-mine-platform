# SPDX-License-Identifier: Apache-2.0
"""The MJX contact engine's static self-declaration — JAX-free (RM-P1-SIM-04).

In its own module so :mod:`astro_mine.sim.engines.brax` (and ``engines/builtins.py``) can expose the
descriptor and register the engine's manifest **without importing JAX or MuJoCo** — the MJX kernel
arrives only when the factory builds an engine (the ``[brax]`` extra).
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

__all__ = ["MJX_CONTACT_ENGINE_DESCRIPTOR"]

#: The MJX contact engine's self-declaration: **real Brax/MJX contact physics**, GPU-vectorized
#: (RM-P1-SIM-04; sim.md §8, §11 "MuJoCo/MJX default, Brax for differentiable/JAX-native massively
#: parallel rollouts").
#:
#: This is the tier that closes the gap the sibling
#: :data:`~astro_mine.sim.engines.brax.BRAX_CONTACT_ENGINE_DESCRIPTOR` left open. That engine's
#: kernel is algebraically the reduced-order kinematic mobility model re-expressed in ``jax.numpy``
#: — fast and vmappable, but not *contact*. This one compiles the **same articulated wheel-soil
#: rover** the MuJoCo CPU tier steps (:mod:`astro_mine.sim.engines._rover_mjcf`) through **MuJoCo
#: MJX** and ``jax.vmap``s it across parallel envs: real frictional contact, real wheel slip, at
#: training scale.
#:
#: It therefore sits at the ``ARTICULATED`` rung (like the MuJoCo CPU tier), where the reduced-order
#: JAX kernel sits at ``KINEMATIC`` — so the multi-fidelity scheduler can see, and choose between,
#: "cheap batched kinematics" and "batched contact", which is exactly the trade a swarm-scale
#: training run needs to make.
#:
#: **Determinism class: ``TOLERANCE``, with an explicitly documented GPU caveat** (conventions.md
#: §11; sim.md §11). Two guarantees, and one deliberate non-guarantee:
#:
#: - *In-process, same-device, same-seed runs reproduce* — every stochastic input flows through a
#:   seeded ``jax.random`` key folded from the ``RngStreams`` root, and MJX's solve is a pure
#: function
#:   of its inputs. This is what the determinism gate checks.
#: - *Across devices or XLA versions the last bits differ.* XLA's floating-point reductions are
#:   **non-associative** and its fusion/reduction order is not bit-portable, so a CPU rollout and a
#:   GPU rollout of the identical batch will diverge in the low bits — and a *contact* solve
#: amplifies
#:   that: contact is a stiff, near-discontinuous problem, so a last-bit difference in a normal
#: force
#:   can flip a make/break contact decision and grow. A bit-exact golden hash would therefore be a
#:   **false gate**, and this tier deliberately does not carry one.
#: - Instead the tier is admitted against an **explicit error budget** — its steady-state
#: drawbar-pull
#:   velocity is regressed against the analytic terramechanics oracle, the same way the other
#:   ``TOLERANCE`` tiers (orbital RK4, ``sqrt``-based mobility, MuJoCo CPU contact) are.
MJX_CONTACT_ENGINE_DESCRIPTOR = EngineDescriptor(
    name="astro-mine.sim.mjx_contact",
    version="0.1.0",
    regimes=(Regime.SURFACE,),
    frames=(MOON_BODY_FIXED,),
    determinism_class=DeterminismClass.TOLERANCE,
    fidelity=FidelityDescriptor(tier=FidelityTier.ARTICULATED),
    capability_tags=(CapabilityTag.MOBILITY_WHEELED,),
)
