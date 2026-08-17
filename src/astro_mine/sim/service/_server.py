# SPDX-License-Identifier: Apache-2.0
"""The gRPC ``EnvironmentService`` — Sim's Core Environment API, served (sim.md §3, §6).

The "service skin" sim.md §3 names in the package layout and §6 names in the integration view: *"as
a gRPC ``EnvironmentService`` (server-streaming ``step``) wrapped in a Ray actor for distributed
rollouts on Cloud"*. It is **additive** — the in-process library path (a plain
:class:`~astro_mine.sim.runtime.Simulator`) is untouched, and every existing consumer keeps using
it.

The service is a thin adapter over *any* Core :class:`~astro_mine.core.env.Environment`, not over
``Simulator`` specifically: it is constructed with an environment **factory**, so what gets served
is the caller's choice (a scenario-backed Simulator, a coupled multi-engine environment, a
Guard-shielded wrapper). Nothing engine-typed crosses the wire — the client sees Core messages,
exactly as the in-process consumer does (sim.md §2 principle 1).

**Two planes, per conventions.md §3 / sim.md §11.** The RPC envelopes are Protobuf (the control
plane: one message per ``reset``/``step``, not per agent per tick); the per-tick observations inside
them are a **FlatBuffers** frame (:mod:`._payload`) — the high-rate telemetry payload, read by
pointer offset rather than parsed.

``Step`` is **server-streaming**: a client sends one action batch and a tick count, and the server
streams one ``StepResponse`` per tick. Each frame is pushed through the environment's ``on_frame``
seam as it is produced, so a served run can fan live frames out to a consumer (View, per sim.md §6
"Streamed to View") without forking the stepping loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import grpc

from astro_mine.core.messages.model import ActionBatch
from astro_mine.sim.runtime.episode import CORE_INTERFACES
from astro_mine.sim.service._payload import encode_frame
from astro_mine.sim.service._proto import environment_pb2 as pb
from astro_mine.sim.service._proto import environment_pb2_grpc as pb_grpc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from astro_mine.core.env import Environment
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["EnvironmentServicer", "FrameSink", "serve"]

#: A live-frame consumer: called with each ``(tick, observations-frame-bytes)`` as the server
#: produces it. The ``on_frame`` seam of the served path — the same idea
#: :func:`~astro_mine.sim.runtime.run_episode` uses to keep headless and interactive one runtime
#: (sim.md §2.6), here spanning the network boundary so a View can watch a served rollout.
FrameSink = "Callable[[int, bytes], None]"


class EnvironmentServicer(pb_grpc.EnvironmentServiceServicer):
    """Serves any Core :class:`~astro_mine.core.env.Environment` over gRPC.

    ``environment_factory`` builds a fresh environment per servicer (a served instance owns one
    episode's environment; Cloud fans *instances* out, rather than multiplexing one). ``scenario``
    is carried only for the ``Describe`` response's provenance fields. ``on_frame``, when given,
    receives
    every frame the server emits — the live-streaming seam."""

    def __init__(
        self,
        environment_factory: Callable[[], Environment],
        *,
        scenario: Scenario | None = None,
        on_frame: Callable[[int, bytes], None] | None = None,
    ) -> None:
        self._env = environment_factory()
        self._scenario = scenario
        self._on_frame = on_frame

    # --- the served Environment API ---------------------------------------------

    def Describe(self, request: pb.DescribeRequest, context: Any) -> pb.DescribeResponse:
        """The environment's static description — everything that does not change per tick.

        The frame name and Core interface versions live here rather than in every high-rate step
        frame: that separation is the whole point of the two-plane split."""
        frame = self._scenario.frame if self._scenario is not None else None
        return pb.DescribeResponse(
            possible_agents=list(self._env.possible_agents),
            core_interfaces=dict(CORE_INTERFACES),
            frame_name=frame.name if frame is not None else "",
            scenario=self._scenario.name if self._scenario is not None else "",
            dt_s=self._scenario.dt_s if self._scenario is not None else 0.0,
        )

    def Reset(self, request: pb.ResetRequest, context: Any) -> pb.ResetResponse:
        """Reset the episode and return the initial observations as a FlatBuffers frame."""
        seed = request.seed if request.HasField("seed") else None
        result = self._env.reset(seed=seed)
        payload = encode_frame(result.observations)
        self._emit(0, payload)
        return pb.ResetResponse(observations=payload, agents=list(self._env.agents))

    def Step(self, request: pb.StepRequest, context: Any) -> Iterator[pb.StepResponse]:
        """Advance the environment and **stream one response per tick** (sim.md §3, §6).

        The client's :class:`~astro_mine.core.messages.model.ActionBatch` is applied to each tick of
        the requested run, so a whole horizon costs one round trip. Streaming stops early once every
        agent has terminated (Core's attrition contract: the active set only shrinks)."""
        try:
            actions = ActionBatch.model_validate_json(request.action_batch_json or "{}")
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid ActionBatch: {exc}")
            raise  # pragma: no cover  (abort raises; this satisfies the type-checker)

        for _ in range(max(1, request.steps)):
            result = self._env.step(actions)
            payload = encode_frame(result.observations)
            tick = _tick_of(result.observations)
            self._emit(tick, payload)
            yield pb.StepResponse(
                observations=payload,
                tick=tick,
                sim_time_s=result.sim_time_s,
                dt_s=result.dt_s,
                terminations=dict(result.terminations),
                truncations=dict(result.truncations),
                agents=list(self._env.agents),
            )
            if not self._env.agents:  # every agent terminated — nothing left to step
                return

    def _emit(self, tick: int, payload: bytes) -> None:
        """Push a frame to the live consumer, if one is attached (the ``on_frame`` seam)."""
        if self._on_frame is not None:
            self._on_frame(tick, payload)


def _tick_of(observations: Any) -> int:
    """The tick a step's observations are stamped with (0 when the active set is empty)."""
    for observation in observations.values():
        return int(observation.tick)
    return 0


def serve(
    environment_factory: Callable[[], Environment],
    *,
    address: str = "localhost:0",
    scenario: Scenario | None = None,
    on_frame: Callable[[int, bytes], None] | None = None,
    max_workers: int = 4,
) -> tuple[grpc.Server, str]:
    """Start an ``EnvironmentService`` server and return ``(server, bound_address)``.

    ``address`` defaults to an ephemeral port (``localhost:0``), so a caller — a test, a Cloud
    sidecar — gets a free port back rather than guessing one. The caller owns the server's lifetime
    (``server.stop(grace)``)."""
    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_EnvironmentServiceServicer_to_server(  # type: ignore[no-untyped-call]
        EnvironmentServicer(environment_factory, scenario=scenario, on_frame=on_frame), server
    )
    port = server.add_insecure_port(address)
    server.start()
    host = address.rsplit(":", 1)[0]
    return server, f"{host}:{port}"
