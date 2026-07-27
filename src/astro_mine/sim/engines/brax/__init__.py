"""Brax/MJX GPU-vectorized swarm-scale rollout path (RM-P1-SIM-04) — the fast-contact training tier.

A JAX-native low-fidelity surface-mobility/contact engine behind the ``RegimeEngine`` waist, plus
the ``jax.vmap`` batched rollout ([Learn](learn.md) consumes thousands of parallel low-fidelity envs
per GPU, sim.md §8) and the Ray fan-out for horizontal throughput on [Cloud](cloud.md) (sim.md §6,
§7). Routing this engine is *configuration*, not a Sim code change: no engine leaks through the Core
Environment API (sim.md §2 principle 1); Learn drives it via the Gymnasium/PettingZoo views over the
same Core message types.

**JAX-free package surface by design.** This module exposes only the engine's
:data:`BRAX_CONTACT_ENGINE_DESCRIPTOR` (registered in ``engines/builtins.py``) and factories whose
bodies import the JAX kernels *lazily* — so importing the engine set, and registering the Brax
engine's manifest, need no JAX. JAX/Brax/MJX arrive with the ``[brax]`` extra and Ray with
``[ray]``; a scenario that actually selects the Brax tier (or fans out) calls a factory, which then
requires them and otherwise raises a clean :class:`ModuleNotFoundError` naming the extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.sim.engines.brax._descriptor import BRAX_CONTACT_ENGINE_DESCRIPTOR
from astro_mine.sim.engines.brax._mjx_descriptor import MJX_CONTACT_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.core.messages.model import ActionBatch
    from astro_mine.sim.engines.adapter import RegimeEngine
    from astro_mine.sim.engines.brax._batch import MjxVectorizedRollout, VectorizedRollout
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = [
    "BRAX_CONTACT_ENGINE_DESCRIPTOR",
    "MJX_CONTACT_ENGINE_DESCRIPTOR",
    "brax_contact_engine_factory",
    "build_mjx_vectorized_rollout",
    "build_vectorized_rollout",
    "fan_out",
    "mjx_contact_engine_factory",
]

_BRAX_HINT = (
    "the Brax/MJX contact engine requires the JAX stack (jax, brax, mujoco); "
    "install it with: pip install 'astro-mine-sim[brax]'"
)


def brax_contact_engine_factory(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
    """Build the reduced-order JAX mobility engine for a scenario's ``brax_contact`` agents.

    The **cheaper** of the two JAX tiers: its kernel is algebraically the kinematic mobility model
    in ``jax.numpy`` — no contact — kept for very large sweeps where batched kinematics is enough.
    The real contact tier is :func:`mjx_contact_engine_factory`.

    Lazy-imports the JAX kernel so the engine set stays importable — and the manifest registrable —
    without JAX. Raises a clear :class:`ModuleNotFoundError` naming ``astro-mine-sim[brax]`` only
    when a scenario actually selects the tier without the JAX stack installed."""
    try:
        from astro_mine.sim.engines.brax._engine import build_brax_contact_engine
    except ModuleNotFoundError as exc:  # jax/brax/mujoco absent
        raise ModuleNotFoundError(_BRAX_HINT) from exc
    return build_brax_contact_engine(scenario, rng)


def mjx_contact_engine_factory(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
    """Build the **MJX contact** engine for a scenario's ``mjx_contact`` agents (``[brax]``).

    Real Brax/MJX wheel-soil contact physics — ``mjx.step`` (MuJoCo's constraint solver in JAX) over
    the same articulated rover the MuJoCo CPU tier steps — ``jax.vmap``-batched across parallel envs
    (RM-P1-SIM-04). This is the GPU-vectorized *contact* tier; :func:`brax_contact_engine_factory`
    is the cheaper reduced-order JAX kernel kept alongside it.

    Lazy-imports the MJX kernel, so registering the manifest needs no JAX; raises a clear
    :class:`ModuleNotFoundError` naming ``astro-mine-sim[brax]`` when the stack is absent."""
    try:
        from astro_mine.sim.engines.brax._mjx import build_mjx_contact_engine
    except ModuleNotFoundError as exc:  # jax/brax/mujoco absent
        raise ModuleNotFoundError(_BRAX_HINT) from exc
    return build_mjx_contact_engine(scenario, rng)


def build_mjx_vectorized_rollout(
    scenario: Scenario,
    rng: RngStreams,
    *,
    n_envs: int | None = None,
    env_indices: Sequence[int] | None = None,
) -> MjxVectorizedRollout:
    """Build the GPU-batched N-env **contact** rollout over the scenario's ``mjx_contact`` agents.

    The same :class:`~astro_mine.sim.engines.brax._batch.VectorizedRollout` surface the
    reduced-order JAX tier exposes — so the in-process Ray fan-out (:func:`fan_out`) and its
    aggregation oracle
    drive it unchanged — but each env runs a real MJX contact solve. Requires ``[brax]``."""
    try:
        from astro_mine.sim.engines.brax._batch import (
            build_mjx_vectorized_rollout as _build,
        )
    except ModuleNotFoundError as exc:  # jax/brax/mujoco absent
        raise ModuleNotFoundError(_BRAX_HINT) from exc
    return _build(scenario, rng, n_envs=n_envs, env_indices=env_indices)


def build_vectorized_rollout(
    scenario: Scenario,
    rng: RngStreams,
    *,
    n_envs: int | None = None,
    env_indices: Sequence[int] | None = None,
) -> VectorizedRollout:
    """Build the GPU-batched N-env rollout over the scenario's ``brax_contact`` agents (``[brax]``).

    Lazy-imports the batched JAX path; raises a clear :class:`ModuleNotFoundError` naming
    ``astro-mine-sim[brax]`` when the JAX stack is absent. See
    :func:`~astro_mine.sim.engines.brax._batch.build_vectorized_rollout`."""
    try:
        from astro_mine.sim.engines.brax._batch import (
            build_vectorized_rollout as _build,
        )
    except ModuleNotFoundError as exc:  # jax/brax/mujoco absent
        raise ModuleNotFoundError(_BRAX_HINT) from exc
    return _build(scenario, rng, n_envs=n_envs, env_indices=env_indices)


def fan_out(
    scenario: Scenario,
    rng: RngStreams,
    *,
    actions: ActionBatch,
    steps: int,
    n_envs: int | None = None,
    num_shards: int = 2,
) -> list[list[list[float]]]:
    """Ray-shard the batched rollout across actors and aggregate (``[brax]`` + ``[ray]`` extras).

    Lazy-imports the Ray fan-out (which needs the JAX stack for the rollout and Ray for the actors);
    raises a clear :class:`ModuleNotFoundError` naming the missing extra. See
    :func:`~astro_mine.sim.engines.brax._ray.fan_out`. KubeRay/GPU-Operator scheduling of the actors
    is [Cloud](cloud.md)'s job (RM-P1-CLOUD-01), not Sim's."""
    try:
        from astro_mine.sim.engines.brax._ray import fan_out as _fan_out
    except ModuleNotFoundError as exc:  # jax/brax/mujoco absent (Ray is imported inside _fan_out)
        raise ModuleNotFoundError(_BRAX_HINT) from exc
    return _fan_out(  # pragma: no cover  (dispatch needs a live Ray cluster; deselected in CI)
        scenario, rng, actions=actions, steps=steps, n_envs=n_envs, num_shards=num_shards
    )
