# SPDX-License-Identifier: Apache-2.0
"""The generic Environment-as-Ray-actor wrapper — Cloud-level fan-out (sim.md §3, §6, §7).

The second half of the "service skin": sim.md §6 describes Sim being consumed *"as a gRPC
``EnvironmentService`` (server-streaming ``step``) **wrapped in a Ray actor** for distributed
rollouts on Cloud"*. This is that wrapper.

**It is not the Brax fan-out.** :mod:`astro_mine.sim.engines.brax._ray` parallelizes *inside one
engine* — it shards a vectorized rollout's env axis across actors, and only the Brax/MJX tiers can
be driven that way. :class:`EnvironmentActor` is the orthogonal thing: it exposes **any** Core
:class:`~astro_mine.core.env.Environment` — the kinematic reference engine, a coupled multi-engine
scenario, an Orekit relay, a MuJoCo rover — as a distributable actor, so Cloud can fan *whole
environments* out across a cluster (a seed sweep, a policy sweep, a Studio design-loop batch).

Only **Core payloads** cross the actor boundary: the actor is constructed from a JSON scenario and a
seed, and returns JSON observations. No engine object, JAX array, or gRPC channel is ever pickled —
the same discipline the Brax fan-out follows, for the same reason (an engine is not serializable,
and a Ray boundary is a serialization boundary).

Ray is imported **lazily**, inside :func:`fan_out_episodes`, so the base wheel stays Ray-free (the
``[ray]`` extra). :class:`EnvironmentActor` is a plain class — ``ray.remote`` is applied at dispatch
— so it is unit-tested in-process without a cluster, and only the live dispatch needs Ray workers.
The *scheduling* of those actors (KubeRay, the GPU Operator) is Cloud's job (RM-P1-CLOUD-01), not
Sim's.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from astro_mine.core.messages.model import ActionBatch
from astro_mine.sim.runtime.episode import Simulator
from astro_mine.sim.runtime.scenario import Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.core.env import Environment
    from astro_mine.sim.engines import EngineFactory

__all__ = ["EnvironmentActor", "EpisodeResult", "fan_out_episodes", "run_episode_in_process"]

#: One episode's transport-friendly result: the seed it ran under and the per-tick observation
#: frames as canonical JSON. Plain data — nothing engine-typed crosses the Ray boundary.
EpisodeResult = tuple[int, list[dict[str, Any]]]


class EnvironmentActor:
    """Wraps a Core :class:`~astro_mine.core.env.Environment` as a distributable actor.

    Built from a **JSON scenario + seed** rather than from a live environment, because a Ray actor
    is constructed on a remote worker: the scenario is the serializable description, and the
    environment is rebuilt there. ``engine_factory`` selects the regime engine(s) — a sweep can
    therefore fan out any tier, not just the vectorizable ones.

    A plain class by design: ``ray.remote`` is applied at dispatch (:func:`fan_out_episodes`), so
    the actor's whole behaviour is exercised in-process by the ordinary tests, and only the dispatch
    itself needs a live cluster."""

    def __init__(
        self,
        scenario_json: str,
        seed: int,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._scenario = Scenario.from_json(scenario_json)
        self._seed = seed
        self._env: Environment = Simulator(self._scenario, engine_factory=engine_factory)

    def run(self, *, steps: int | None = None) -> EpisodeResult:
        """Run one episode to its horizon and return ``(seed, frames)``.

        Frames are canonical JSON observation maps — the same shape the in-process
        :func:`~astro_mine.sim.runtime.run_episode` records, so a fanned-out episode and a local one
        are directly comparable (which is what makes the fan-out checkable against an oracle)."""
        horizon = self._scenario.horizon_steps if steps is None else steps
        reset = self._env.reset(seed=self._seed)
        frames: list[dict[str, Any]] = [_dump(reset.observations)]
        for _ in range(horizon):
            result = self._env.step(ActionBatch())
            frames.append(_dump(result.observations))
            if not self._env.agents:  # every agent terminated
                break
        return self._seed, frames

    def run_json(self, steps: int | None = None) -> str:
        """:meth:`run`, serialized — the method the Ray actor is actually driven through.

        Ray serializes return values, and a canonical JSON string is the narrowest possible payload
        (no pydantic models, no numpy, no engine types cross the boundary)."""
        seed, frames = self.run(steps=steps)
        return json.dumps({"seed": seed, "frames": frames}, sort_keys=True, separators=(",", ":"))


def _dump(observations: Any) -> dict[str, Any]:
    return {aid: obs.model_dump(mode="json") for aid, obs in observations.items()}


def run_episode_in_process(
    scenario: Scenario,
    seeds: Sequence[int],
    *,
    engine_factory: EngineFactory | None = None,
    steps: int | None = None,
) -> list[EpisodeResult]:
    """The single-process reference: run every seed here, in seed order.

    The oracle :func:`fan_out_episodes` is checked against — a sweep fanned across actors must
    return
    exactly this, because each episode is a pure function of ``(scenario, seed)``."""
    scenario_json = scenario.model_dump_json()
    return [
        EnvironmentActor(scenario_json, seed, engine_factory=engine_factory).run(steps=steps)
        for seed in seeds
    ]


def fan_out_episodes(
    scenario: Scenario,
    seeds: Sequence[int],
    *,
    engine_factory: EngineFactory | None = None,
    steps: int | None = None,
    actor_options: dict[str, Any] | None = None,
) -> list[EpisodeResult]:
    """Fan a seed sweep out across Ray actors — one :class:`EnvironmentActor` per seed (``[ray]``).

    Each actor rebuilds the environment from the JSON scenario on its worker and returns its
    episode's frames; results are reassembled in **seed order**, so the aggregate equals
    :func:`run_episode_in_process` (the equivalence gate). On Cloud these actors land on KubeRay
    workers (RM-P1-CLOUD-01); scheduling them is Cloud's job, not Sim's.

    ``actor_options`` is passed to ``ray.remote(...)`` (e.g. ``{"num_cpus": 2}``). Requires a live
    Ray cluster, so it is deselected in CI (GitHub runners cannot spawn Ray workers) — the actor it
    dispatches, and the seed-order reassembly, are unit-tested in-process against
    :func:`run_episode_in_process`.
    """
    return _dispatch_on_cluster(
        scenario, seeds, engine_factory=engine_factory, steps=steps, actor_options=actor_options
    )


def _dispatch_on_cluster(  # pragma: no cover  (needs live Ray workers; deselected in CI)
    scenario: Scenario,
    seeds: Sequence[int],
    *,
    engine_factory: EngineFactory | None,
    steps: int | None,
    actor_options: dict[str, Any] | None,
) -> list[EpisodeResult]:
    """Dispatch one actor per seed and reassemble in seed order — the live-cluster path."""
    import ray

    scenario_json = scenario.model_dump_json()
    # Ray types `ray.remote(Cls)` fine; what it does not model is that calling `.remote(...)` on
    # it yields an **ActorHandle** rather than an instance of the class. Left unannotated, mypy
    # reads `a.rollout` as the plain method and reports that a `Callable` has no `.remote` -- at
    # the *use* site, which is why #40's reading (a cast at construction) does not fix it: the
    # cast is redundant there, and mypy says so. `.remote()` is also overloaded, so it yields
    # `ActorHandle[EnvironmentActor] | type[EnvironmentActor]`; naming the handles resolves that
    # too.
    # (astro-mine-platform#40.)
    remote_actor = (
        ray.remote(**actor_options)(EnvironmentActor)
        if actor_options
        else ray.remote(EnvironmentActor)
    )
    actors = cast(
        "list[ray.actor.ActorHandle[EnvironmentActor]]",
        [remote_actor.remote(scenario_json, seed, engine_factory=engine_factory) for seed in seeds],
    )
    payloads: list[str] = ray.get([actor.run_json.remote(steps) for actor in actors])
    by_seed = {r["seed"]: r["frames"] for r in (json.loads(p) for p in payloads)}
    return [(seed, by_seed[seed]) for seed in seeds]
