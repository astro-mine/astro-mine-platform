"""The gRPC ``EnvironmentService`` client — a *served* Core Environment (sim.md §3, §6).

:class:`RemoteEnvironment` drives the service and **is itself a Core**
:class:`~astro_mine.core.env.Environment`: same ``possible_agents`` / ``agents`` / ``reset`` /
``step`` surface, same Core message types. That is the load-bearing property of the whole service
skin — a consumer (Learn's Gymnasium view, Bench's runner, Studio's design loop) cannot tell whether
it holds an in-process ``Simulator`` or a remote one, so moving a rollout onto Cloud is
*deployment*, not a code change.

It is proved, not asserted: the contract test runs Core's own
:func:`~astro_mine.core.env.check_environment` against a live served environment, so the served path
honours the declared Environment API version exactly as the library path does (conventions.md §11's
consumer-driven contract tests).

The client decodes the FlatBuffers observation frames the server streams (:mod:`._payload`) and
buffers the server-streamed ticks, so a caller's ``step`` still looks like one tick — the streaming
is a transport optimization (one round trip per horizon), not a change to the Environment contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import grpc

from astro_mine.core.env import ResetResult, StepResult
from astro_mine.core.units import MOON_BODY_FIXED, FrameClass, ReferenceFrame
from astro_mine.sim.service._payload import decode_frame
from astro_mine.sim.service._proto import environment_pb2 as pb
from astro_mine.sim.service._proto import environment_pb2_grpc as pb_grpc

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astro_mine.core.env import Environment
    from astro_mine.core.messages.model import ActionBatch

__all__ = ["RemoteEnvironment", "connect"]


class RemoteEnvironment:
    """A Core :class:`~astro_mine.core.env.Environment` backed by a remote ``EnvironmentService``.

    Constructed from a live gRPC channel. On construction it calls ``Describe`` once to learn the
    environment's static facts (agent roster, reference frame, Core interface versions) — the things
    the high-rate step stream deliberately does not repeat.

    ``stream_steps`` is the number of ticks the server streams per ``Step`` RPC. It is purely a
    transport knob: the client buffers them and hands the caller one tick per ``step()``, so the
    Environment contract is unchanged whatever it is set to. Raise it to amortize the round trip
    over
    a long horizon."""

    def __init__(self, channel: grpc.Channel, *, stream_steps: int = 1) -> None:
        if stream_steps < 1:
            raise ValueError(f"stream_steps must be >= 1, got {stream_steps}")
        self._stub = pb_grpc.EnvironmentServiceStub(channel)
        self._stream_steps = stream_steps
        description = self._stub.Describe(pb.DescribeRequest())
        self._possible_agents: tuple[str, ...] = tuple(description.possible_agents)
        self._core_interfaces: dict[str, str] = dict(description.core_interfaces)
        self._frame = _frame_from_name(description.frame_name)
        self._scenario = description.scenario
        self._active: tuple[str, ...] = ()
        self._pending: list[pb.StepResponse] = []

    @property
    def possible_agents(self) -> tuple[str, ...]:
        """Every agent id that may appear in this environment."""
        return self._possible_agents

    @property
    def agents(self) -> tuple[str, ...]:
        """Agent ids currently active — shrinks as agents terminate (Core's attrition contract)."""
        return self._active

    @property
    def core_interfaces(self) -> Mapping[str, str]:
        """The Core interface versions the *served* environment declares (RM-P0-CORE-07)."""
        return dict(self._core_interfaces)

    @property
    def frame(self) -> ReferenceFrame:
        """The reference frame the served observations' poses are expressed in."""
        return self._frame

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        """Reset the remote episode and decode its initial observations."""
        request = pb.ResetRequest()
        if seed is not None:
            request.seed = seed
        response = self._stub.Reset(request)
        self._pending.clear()
        self._active = tuple(response.agents)
        return ResetResult(observations=decode_frame(response.observations, frame=self._frame))

    def step(self, actions: ActionBatch) -> StepResult:
        """Advance one tick, refilling from the server's stream when the buffer is empty.

        The server streams ``stream_steps`` ticks per RPC; this hands the caller one at a time, so
        the Environment contract sees exactly the same per-tick surface as the in-process path."""
        if not self._pending:
            request = pb.StepRequest(
                action_batch_json=actions.model_dump_json(), steps=self._stream_steps
            )
            self._pending = list(self._stub.Step(request))
            if not self._pending:  # pragma: no cover  (the server always yields at least one tick)
                raise RuntimeError("the environment service returned no step frames")
        response = self._pending.pop(0)
        self._active = tuple(response.agents)
        return StepResult(
            observations=decode_frame(response.observations, frame=self._frame),
            sim_time_s=response.sim_time_s,
            terminations=dict(response.terminations),
            truncations=dict(response.truncations),
            dt_s=response.dt_s,
        )


def _frame_from_name(name: str) -> ReferenceFrame:
    """Rebuild the served environment's reference frame from its SPICE name.

    Core's frames are identified by their SPICE name, so the name is enough to reconstruct the frame
    on the client; an empty name (a server that carries no scenario) falls back to the lunar
    body-fixed default the anchor scenario uses."""
    if not name or name == MOON_BODY_FIXED.name:
        return MOON_BODY_FIXED
    return ReferenceFrame(name=name, frame_class=FrameClass.BODY_FIXED, center="MOON")


def connect(address: str, *, stream_steps: int = 1) -> tuple[grpc.Channel, RemoteEnvironment]:
    """Open a channel to an ``EnvironmentService`` at ``address`` and return ``(channel, env)``.

    The caller owns the channel's lifetime (``channel.close()``)."""
    channel = grpc.insecure_channel(address)
    return channel, RemoteEnvironment(channel, stream_steps=stream_steps)


if TYPE_CHECKING:

    def _assert_environment(env: RemoteEnvironment) -> Environment:
        # mypy fails here if the served client drifts from the Core Environment Protocol — the same
        # static proof the in-process Simulator carries.
        return env
