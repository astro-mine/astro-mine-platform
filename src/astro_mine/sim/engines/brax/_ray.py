# SPDX-License-Identifier: Apache-2.0
"""Ray fan-out — horizontal throughput for the batched rollout on [Cloud](cloud.md) (RM-P1-SIM-04).

A thin Ray shim over :class:`~astro_mine.sim.engines.brax._batch.VectorizedRollout`: shard the N
parallel envs across :class:`ray.remote` actors and aggregate the shards back into one batch (sim.md
§6, §7 "horizontal fan-out of stateless rollout actors via Ray/KubeRay"). Per-env seeding is by
**global** env index, so a shard computes the exact rows it owns in the whole-batch run — the shards
concatenate to the identical result the single-process rollout produces.

Deliberately minimal: only the fan-out/aggregation lives here. The *scheduling* of those actors —
KubeRay ``RayJob``/``RayCluster``, the NVIDIA GPU Operator, MIG sharing — is [Cloud](cloud.md)'s job
(RM-P1-CLOUD-01), not Sim's. Ray is imported lazily inside :func:`fan_out`, so importing this module
needs no Ray (the ``[ray]`` extra); the base wheel stays Ray-free. The sharder / actor / aggregation
are ordinary Python — unit-tested in-process without a cluster; only the actual ``ray.remote``
dispatch (which needs live worker processes) runs under Ray.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

from astro_mine.core.messages.model import ActionBatch
from astro_mine.sim.engines.brax._batch import (
    RolloutBatch,
    build_mjx_vectorized_rollout,
    build_vectorized_rollout,
)
from astro_mine.sim.runtime.rng import RngStreams
from astro_mine.sim.runtime.scenario import Scenario

if TYPE_CHECKING:
    from astro_mine.sim.runtime.rng import RngStreams as _RngStreams  # noqa: F401 (doc ref)

__all__ = ["ShardResult", "build_rollout", "fan_out", "run_in_process"]

#: The two JAX tiers the fan-out can shard, keyed by ``dynamics.kind``. Both builders return a
#: :class:`~astro_mine.sim.engines.brax._batch.RolloutBatch`, so **everything below this line is
#: tier-agnostic** — the sharder, the actor, and the aggregation oracle work unchanged for the real
#: MJX contact tier exactly as they did for the reduced-order kernel (RM-P1-SIM-04).
_ROLLOUT_BUILDERS: dict[str, _RolloutBuilder] = {
    "brax_contact": build_vectorized_rollout,
    "mjx_contact": build_mjx_vectorized_rollout,
}


def build_rollout(
    scenario: Scenario,
    rng: RngStreams,
    *,
    n_envs: int | None = None,
    env_indices: Sequence[int] | None = None,
) -> RolloutBatch:
    """Build the batched rollout for whichever JAX tier the scenario selects.

    Dispatches on ``dynamics.kind``: ``mjx_contact`` gives the real MJX contact batch,
    ``brax_contact`` the cheaper reduced-order kernel. Raises ``ValueError`` if the scenario
    declares
    neither."""
    for kind, builder in _ROLLOUT_BUILDERS.items():
        if any(s.dynamics.kind == kind for s in scenario.agents):
            return builder(scenario, rng, n_envs=n_envs, env_indices=env_indices)
    raise ValueError("a vectorized rollout needs at least one brax_contact or mjx_contact agent")


#: One shard's contribution: the per-(env, agent) final positions and the global env ids they
#: correspond to — a plain, transport-friendly payload (no JAX/engine type crosses the boundary).
ShardResult = tuple[list[list[list[float]]], list[int]]


class _RolloutBuilder(Protocol):
    """A tier's batched-rollout builder — both JAX tiers' builders share this shape."""

    def __call__(
        self,
        scenario: Scenario,
        rng: RngStreams,
        *,
        n_envs: int | None = ...,
        env_indices: Sequence[int] | None = ...,
    ) -> RolloutBatch: ...


def _positions_as_list(rollout: RolloutBatch) -> list[list[list[float]]]:
    """The rollout's live positions as a nested ``[env][agent][xyz]`` Python list."""
    pos = rollout.positions
    return [
        [[float(pos[e, a, k]) for k in range(3)] for a in range(rollout.n_agents)]
        for e in range(rollout.n_envs)
    ]


def _roll(rollout: RolloutBatch, actions: ActionBatch, steps: int) -> None:
    """Reset and step ``rollout`` ``steps`` times under one broadcast action batch."""
    rollout.reset()
    for _ in range(steps):
        rollout.step(actions)


def run_in_process(
    scenario: Scenario,
    rng: RngStreams,
    *,
    actions: ActionBatch,
    steps: int,
    n_envs: int | None = None,
) -> list[list[list[float]]]:
    """The single-process reference: roll the whole N-env batch here and return its positions.

    The oracle the Ray fan-out is checked against — :func:`fan_out` sharded across actors must
    aggregate to exactly this (same global env seeding, same kernel). Tier-agnostic: it is the
    oracle
    for the MJX contact batch exactly as for the reduced-order kernel."""
    rollout = build_rollout(scenario, rng, n_envs=n_envs)
    _roll(rollout, actions, steps)
    return _positions_as_list(rollout)


def _shard_ranges(total: int, num_shards: int) -> list[list[int]]:
    """Split ``range(total)`` into ``num_shards`` contiguous, near-equal index shards."""
    num_shards = max(1, min(num_shards, total))
    base, extra = divmod(total, num_shards)
    shards: list[list[int]] = []
    start = 0
    for s in range(num_shards):
        size = base + (1 if s < extra else 0)
        shards.append(list(range(start, start + size)))
        start += size
    return shards


def _aggregate(results: Sequence[ShardResult], total: int) -> list[list[list[float]]]:
    """Reassemble shard results into one ``[env][agent][xyz]`` batch, ordered by global env id."""
    by_index: dict[int, list[list[float]]] = {}
    for positions, indices in results:
        for row, idx in zip(positions, indices, strict=True):
            by_index[idx] = row
    return [by_index[i] for i in range(total)]


class _RolloutActor:
    """The per-shard worker wrapping a :class:`VectorizedRollout` over one range of env indices.

    Rebuilds the rollout from the (JSON-serialized) scenario + root seed, so only Core payloads
    cross the Ray boundary — no JAX array or engine object is shipped. A plain class (``ray.remote``
    is applied at dispatch), so it is exercised in-process by the unit tests too."""

    def __init__(self, scenario_json: str, root_seed: int, env_indices: list[int]) -> None:
        scenario = Scenario.from_json(scenario_json)
        self._rollout = build_rollout(scenario, RngStreams(root_seed), env_indices=env_indices)

    def rollout(self, actions_json: str, steps: int) -> ShardResult:
        """Roll this shard ``steps`` steps under ``actions`` and return its positions + env ids."""
        actions = ActionBatch.model_validate_json(actions_json)
        _roll(self._rollout, actions, steps)
        return _positions_as_list(self._rollout), list(self._rollout.env_indices)


def _fan_out_on_cluster(  # pragma: no cover  (needs live Ray workers; deselected in CI)
    scenario: Scenario,
    rng: RngStreams,
    *,
    actions: ActionBatch,
    steps: int,
    total: int,
    num_shards: int,
) -> list[list[list[float]]]:
    """Dispatch the shards across ``ray.remote`` actors and aggregate — the live-cluster path.

    Requires real Ray worker processes, so it is deselected in CI (GitHub runners can't spawn them,
    like the GPU test) and verified locally; on Cloud the actors land on KubeRay GPU workers
    (RM-P1-CLOUD-01). The sharder / actor / aggregation it composes are unit-tested in-process."""
    import ray

    scenario_json = scenario.model_dump_json()
    actions_json = actions.model_dump_json()
    # Ray types `ray.remote(Cls)` fine; what it does not model is that calling `.remote(...)` on
    # it yields an **ActorHandle** rather than an instance of the class. Left unannotated, mypy
    # reads `a.rollout` as the plain method and reports that a `Callable` has no `.remote` -- at
    # the *use* site, which is why #40's reading (a cast at construction) does not fix it: the
    # cast is redundant there, and mypy says so. `.remote()` is also overloaded, so it yields
    # `ActorHandle[_RolloutActor] | type[_RolloutActor]`; naming the handles resolves that too.
    # (astro-mine-platform#40.)
    remote_actor = ray.remote(_RolloutActor)
    actors = cast(
        "list[ray.actor.ActorHandle[_RolloutActor]]",
        [
            remote_actor.remote(scenario_json, rng.root_seed, shard)
            for shard in _shard_ranges(total, num_shards)
            if shard
        ],
    )
    results: list[ShardResult] = ray.get([a.rollout.remote(actions_json, steps) for a in actors])
    return _aggregate(results, total)


def fan_out(
    scenario: Scenario,
    rng: RngStreams,
    *,
    actions: ActionBatch,
    steps: int,
    n_envs: int | None = None,
    num_shards: int = 2,
) -> list[list[list[float]]]:
    """Shard the N-env rollout across Ray actors and aggregate to one ``[env][agent][xyz]`` result.

    Splits the envs into ``num_shards`` contiguous shards, one :class:`_RolloutActor` each, rolls
    them under the same broadcast ``actions``, and reassembles the rows in global env-index order.
    Because seeding is by global env index, the aggregate equals :func:`run_in_process` (the CPU
    equivalence gate; on Cloud the actors land on KubeRay GPU workers, RM-P1-CLOUD-01). Works
    unchanged for **either** JAX tier — the reduced-order kernel or the real MJX contact batch.
    Raises ``ValueError`` if the scenario declares neither."""
    specs = [s for s in scenario.agents if s.dynamics.kind in _ROLLOUT_BUILDERS]
    if not specs:
        raise ValueError("Ray fan-out needs at least one brax_contact or mjx_contact agent")
    first = specs[0].dynamics  # pragma: no cover  (valid-scenario path needs a live Ray cluster)
    total = n_envs if n_envs is not None else first.batch_size  # type: ignore[union-attr]  # pragma: no cover
    return _fan_out_on_cluster(  # pragma: no cover
        scenario, rng, actions=actions, steps=steps, total=total, num_shards=num_shards
    )
