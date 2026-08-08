"""The distributed field service — the single writer that keeps one consistent posterior.

A :class:`FieldServicer` holds, per shared field, one authoritative :class:`BeliefField`. Producers
append observations through :meth:`FieldServicer.SubmitObservations`; the servicer conditions the
*one* posterior under a per-field lock and broadcasts a :class:`BeliefUpdate` — the appended
observations plus the resulting content hash — to every subscriber of
:meth:`FieldServicer.StreamBeliefUpdates`. A late subscriber first catches up from its requested
revision, then follows live. Because the server is the single writer and every update carries the
ordered observations, a distributed swarm sim reconstructs a **byte-identical** posterior by replay
(prospect.md §3, §5, §7) and can fail closed the instant its reconstructed hash diverges.

**Every RPC is authenticated and authorized** (:mod:`astro_mine.prospect.service._auth`): the server
is served over TLS, callers present an OIDC-issued bearer token, and the *ground-truth-adjacent*
``SubmitObservations`` — the RPC that writes into the one shared posterior, whose observations are
drawn from the sealed truth — additionally requires the Core ``GROUND_TRUTH_ACCESS`` capability
grant (prospect.md §9; ``LUNAR-SR-001``, ``LUNAR-SR-005``). :func:`serve` has **no unauthenticated
default**: the cleartext local-dev path must be asked for by name, by environment variable, and on
loopback.

``grpcio`` is an optional (``service`` extra) dependency, imported here — the importable library
(fields + belief + infogain) never needs it.

Backlog: RM-P1-PROSPECT-11; prospect.md §9 —
astro-mine-prospect#21
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent import futures
from contextlib import contextmanager

import grpc

from astro_mine.core.units.wire import planetary_crs_to_proto, reference_frame_to_proto
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.publish._bundle import serialize_bundle
from astro_mine.prospect.service._auth import (
    INSECURE_DEV_ENV_VAR,
    AuthInterceptor,
    InsecureDevAuth,
    ServiceAuth,
    insecure_dev_enabled,
    is_loopback,
)
from astro_mine.prospect.service._convert import (
    from_proto_observations,
    to_proto_observations,
)
from astro_mine.prospect.service._proto import field_service_pb2 as pb
from astro_mine.prospect.service._proto import field_service_pb2_grpc as pbg

__all__ = ["FieldServicer", "serve"]

#: How often a streaming RPC wakes to re-check the client is still connected (seconds).
_STREAM_POLL_S = 0.25


class _SharedField:
    """One shared field's authoritative state: prior + belief + an append-only event log."""

    def __init__(self, field_id: str, prior: Prior) -> None:
        self.field_id = field_id
        self.prior_bundle = serialize_bundle(prior)
        # The field's georeference travels with every snapshot so a subscriber places the belief
        # field without re-deriving the vocabulary (RFC-0007). Encoded once from the prior's
        # FieldMetadata — the frame/CRS are fixed for a field's lifetime.
        self._frame = reference_frame_to_proto(prior.metadata.frame)
        self._crs = planetary_crs_to_proto(prior.metadata.crs)
        self._belief = BeliefField.from_prior(prior)
        self._events: list[pb.BeliefUpdate] = []  # index i == revision i+1
        self._condition = threading.Condition()

    @property
    def revision(self) -> int:
        with self._condition:
            return len(self._events)

    def snapshot(self) -> pb.FieldSnapshot:
        """The current replayable state — prior bundle + the whole ordered log + content hash."""
        with self._condition:
            belief = self._belief
            return pb.FieldSnapshot(
                field_id=self.field_id,
                prior_bundle=self.prior_bundle,
                observations=to_proto_observations(belief.log),
                content_hash=belief.content_hash,
                revision=len(self._events),
                frame=self._frame,
                crs=self._crs,
            )

    def submit(self, observations: Sequence[pb.Observation]) -> pb.UpdateAck:
        """Append ``observations`` to the one posterior and broadcast the increment (serialized)."""
        decoded = from_proto_observations(observations)
        with self._condition:
            self._belief = self._belief.update(decoded)
            revision = len(self._events) + 1
            update = pb.BeliefUpdate(
                field_id=self.field_id,
                observations=list(observations),
                content_hash=self._belief.content_hash,
                revision=revision,
            )
            self._events.append(update)
            self._condition.notify_all()
            return pb.UpdateAck(
                field_id=self.field_id,
                content_hash=self._belief.content_hash,
                revision=revision,
                applied=len(decoded),
            )

    def follow(
        self, from_revision: int, is_active: Callable[[], bool]
    ) -> Iterator[pb.BeliefUpdate]:
        """Yield events from ``from_revision`` on (catch-up then live) while the client connects.

        ``is_active`` is the gRPC ``context.is_active`` callable; the stream wakes every
        :data:`_STREAM_POLL_S` to honour cancellation even when no update is pending.
        """
        index = max(0, from_revision)
        while is_active():
            with self._condition:
                while index >= len(self._events):
                    if not is_active():
                        return
                    self._condition.wait(timeout=_STREAM_POLL_S)
                event = self._events[index]
                index += 1
            yield event


class FieldServicer(pbg.FieldServiceServicer):
    """gRPC servicer serving a set of shared fields, each registered from a seed :class:`Prior`.

    Register fields up front with :meth:`register_field` (or the ``fields`` mapping passed to
    :func:`serve`); the servicer then serves snapshots, applies observation batches to the one
    posterior per field, and streams belief updates to distributed subscribers.
    """

    def __init__(self, fields: Mapping[str, Prior] | None = None) -> None:
        self._fields: dict[str, _SharedField] = {}
        self._registry_lock = threading.Lock()
        for field_id, prior in (fields or {}).items():
            self.register_field(field_id, prior)

    def register_field(self, field_id: str, prior: Prior) -> None:
        """Register a shared field ``field_id`` seeded by ``prior`` (fails on a duplicate id)."""
        with self._registry_lock:
            if field_id in self._fields:
                raise ValueError(f"field {field_id!r} is already registered")
            self._fields[field_id] = _SharedField(field_id, prior)

    def _get(self, field_id: str, context: grpc.ServicerContext) -> _SharedField:
        with self._registry_lock:
            shared = self._fields.get(field_id)
        if shared is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown field {field_id!r}")
            raise AssertionError("unreachable: context.abort raises")  # for the type checker
        return shared

    # --- gRPC surface -------------------------------------------------------------------------

    def GetField(self, request: pb.FieldRequest, context: grpc.ServicerContext) -> pb.FieldSnapshot:
        return self._get(request.field_id, context).snapshot()

    def SubmitObservations(
        self, request: pb.ObservationBatch, context: grpc.ServicerContext
    ) -> pb.UpdateAck:
        return self._get(request.field_id, context).submit(request.observations)

    def StreamBeliefUpdates(
        self, request: pb.SubscribeRequest, context: grpc.ServicerContext
    ) -> Iterator[pb.BeliefUpdate]:
        shared = self._get(request.field_id, context)
        yield from shared.follow(request.from_revision, context.is_active)


@contextmanager
def serve(
    fields: Mapping[str, Prior],
    *,
    auth: ServiceAuth | InsecureDevAuth,
    address: str = "127.0.0.1:0",
    max_workers: int = 8,
) -> Iterator[tuple[grpc.Server, str]]:
    """Run a :class:`FieldServicer` for ``fields`` as a context manager; yields ``(server, addr)``.

    ``auth`` is **required** — there is no unauthenticated default (prospect.md §9;
    ``LUNAR-SR-001``). Pass a :class:`~astro_mine.prospect.service._auth.ServiceAuth` (TLS +
    OIDC-token authentication + per-method, capability-gated authorization) for any real deployment:
    every RPC is then authorized before the servicer is reached, and the ground-truth-adjacent
    ``SubmitObservations`` additionally requires the Core ``GROUND_TRUTH_ACCESS`` grant.

    :class:`~astro_mine.prospect.service._auth.InsecureDevAuth` is the documented local-dev escape
    hatch, and it is **triply** opt-in: the caller must pass it explicitly, the
    ``ASTRO_MINE_PROSPECT_INSECURE_DEV`` environment variable must be set, and ``address`` must bind
    loopback. Any of those missing is a ``ValueError``, not a downgrade — so a production config
    cannot slip into cleartext by omission.

    ``address`` defaults to an ephemeral loopback port (``:0``); the bound address is returned so a
    client can connect. The server is stopped on exit.
    """
    interceptor, credentials = _configure_auth(auth, address)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), interceptors=[interceptor]
    )
    pbg.add_FieldServiceServicer_to_server(FieldServicer(fields), server)  # type: ignore[no-untyped-call]
    if credentials is None:
        port = server.add_insecure_port(address)  # loopback-only local dev (guarded above)
    else:
        port = server.add_secure_port(address, credentials)
    host = address.rsplit(":", 1)[0]
    server.start()
    try:
        yield server, f"{host}:{port}"
    finally:
        server.stop(grace=None).wait()


def _configure_auth(
    auth: ServiceAuth | InsecureDevAuth, address: str
) -> tuple[AuthInterceptor, grpc.ServerCredentials | None]:
    """Resolve the auth posture into ``(interceptor, transport credentials)`` — or refuse.

    The single place the insecure downgrade is gated, and deliberately a hard refusal rather than a
    warning: a service that *warns* about serving the belief posterior in the clear, and then serves
    it, has not protected anything (conventions.md §9).
    """
    if isinstance(auth, InsecureDevAuth):
        if not insecure_dev_enabled():
            raise ValueError(
                "insecure local-dev mode additionally requires the environment variable "
                f"{INSECURE_DEV_ENV_VAR}=1 (prospect.md §9: TLS + token auth is the default; the "
                "cleartext path is opt-in and local-only)"
            )
        if not is_loopback(address):
            raise ValueError(
                f"insecure local-dev mode may bind loopback only, not {address!r} — a cleartext "
                "field service must never be reachable off-host (conventions.md §9)"
            )
        return AuthInterceptor(policy=auth.policy, static_principal=auth.principal), None
    return AuthInterceptor(policy=auth.policy, verifier=auth.verifier), auth.tls.credentials()
