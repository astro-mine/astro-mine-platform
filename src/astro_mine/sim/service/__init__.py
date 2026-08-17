# SPDX-License-Identifier: Apache-2.0
"""The service skin — gRPC ``EnvironmentService`` + a generic Ray-actor wrapper (sim.md §3, §6, §7).

sim.md §3's package layout names a ``service/`` module (*"gRPC EnvironmentService + Ray-actor
wrapper (the 'service skin')"*) and §6 names its shape: Sim is consumed **in-process as a library**
for local dev and for embedding in Learn/Bench, and **as a gRPC ``EnvironmentService``
(server-streaming ``step``) wrapped in a Ray actor** for distributed rollouts on Cloud. This package
is the second half of that; the first half is unchanged.

**Additive, never a replacement.** The in-process path — construct a
:class:`~astro_mine.sim.runtime.Simulator`, call ``reset``/``step`` — is untouched, and remains the
always-works local tier (CX-LOCAL). Nothing in ``astro_mine.sim``'s runtime imports this package;
gRPC arrives only with the ``astro-mine-platform[sim-service]`` extra.

Four pieces:

- :class:`EnvironmentServicer` / :func:`serve` — the gRPC service over **any** Core
  :class:`~astro_mine.core.env.Environment` (it takes an environment *factory*, so what is served is
  the caller's choice). ``Step`` is **server-streaming**: one action batch in, one response per tick
  out, so a horizon costs one round trip.
- :class:`RemoteEnvironment` / :func:`connect` — the client, which **is itself a Core Environment**.
  A consumer cannot tell a served environment from an in-process one, which is what makes moving a
  rollout onto Cloud a *deployment* decision rather than a code change. Proven, not asserted: the
  contract test runs Core's own ``check_environment`` against the served path.
- :func:`encode_frame` / :func:`decode_frame` — the per-tick **FlatBuffers** observation payload.
  The control plane is Protobuf, the high-rate telemetry is FlatBuffers (conventions.md §3;
  sim.md §11).
- :class:`EnvironmentActor` / :func:`fan_out_episodes` — the **generic** Environment-as-Ray-actor
  wrapper, for Cloud-level fan-out of *whole environments*. Distinct from
  :mod:`astro_mine.sim.engines.brax._ray`, which parallelizes inside one (vectorizable) engine.

The ``on_frame`` seam spans the network boundary too: a served run pushes each frame to a live
consumer as it is produced (sim.md §6 "live frames stream to View"), the same way the in-process
recorder does — one stepping loop, different sinks (sim.md §2.6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the public names, for type-checkers and IDEs (no gRPC import at type time)
    from astro_mine.sim.service._client import RemoteEnvironment, connect
    from astro_mine.sim.service._payload import decode_frame, encode_frame
    from astro_mine.sim.service._ray import (
        EnvironmentActor,
        EpisodeResult,
        fan_out_episodes,
        run_episode_in_process,
    )
    from astro_mine.sim.service._server import EnvironmentServicer, serve

__all__ = [
    "EnvironmentActor",
    "EnvironmentServicer",
    "EpisodeResult",
    "RemoteEnvironment",
    "connect",
    "decode_frame",
    "encode_frame",
    "fan_out_episodes",
    "run_episode_in_process",
    "serve",
]

_SERVICE_HINT = (
    "the Sim service skin requires gRPC + FlatBuffers (grpcio, protobuf, flatbuffers); "
    "install it with: pip install 'astro-mine-platform[sim-service]'"
)

#: Which private module each public name lives in — the lazy-import map. Keeping the gRPC-dependent
#: modules off the package's import path is what lets ``import astro_mine.sim`` stay service-free:
#: the base wheel carries no gRPC, and only a caller that actually *serves* pays for it.
_EXPORTS = {
    "EnvironmentActor": "._ray",
    "EnvironmentServicer": "._server",
    "EpisodeResult": "._ray",
    "RemoteEnvironment": "._client",
    "connect": "._client",
    "decode_frame": "._payload",
    "encode_frame": "._payload",
    "fan_out_episodes": "._ray",
    "run_episode_in_process": "._ray",
    "serve": "._server",
}


def __getattr__(name: str) -> object:
    """Resolve a public name by importing its module on first access (PEP 562).

    Raises a clear :class:`ModuleNotFoundError` naming ``astro-mine-platform[sim-service]``
    when the gRPC /
    FlatBuffers stack is absent — the same "the error tells you which extra to install" contract the
    engine factories follow."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(f"{__name__}{module_name}")
    except ModuleNotFoundError as exc:  # grpc / flatbuffers absent
        raise ModuleNotFoundError(_SERVICE_HINT) from exc
    return getattr(module, name)
