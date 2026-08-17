# SPDX-License-Identifier: Apache-2.0
"""The field-service client — reconstruct and track the one shared posterior over gRPC.

A :class:`FieldServiceClient` fetches a field's replayable state, submits observations, and follows
the belief-update stream. Reconstruction is **fail-closed**: after replaying the prior + the ordered
log (or applying a streamed increment), the client checks its own :attr:`BeliefField.content_hash`
against the server's; a mismatch means the client and server diverged and raises
:class:`PosteriorConsistencyError` rather than serving a silently-wrong posterior (prospect.md §5).

Backlog: RM-P1-PROSPECT-11 — astro-mine-prospect#21
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import grpc

from astro_mine.core.units import PlanetaryCRS, ReferenceFrame
from astro_mine.core.units.wire import planetary_crs_from_proto, reference_frame_from_proto
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.publish._bundle import prior_from_bundle
from astro_mine.prospect.service._auth import bearer_metadata
from astro_mine.prospect.service._convert import (
    from_proto_observations,
    to_proto_observations,
)
from astro_mine.prospect.service._proto import field_service_pb2 as pb
from astro_mine.prospect.service._proto import field_service_pb2_grpc as pbg

__all__ = ["FieldServiceClient", "FieldState", "PosteriorConsistencyError"]


class PosteriorConsistencyError(RuntimeError):
    """A reconstructed posterior's content hash diverged from the server's — the fail-closed
    guard."""


@dataclass(frozen=True)
class FieldState:
    """A client-side view of the one shared posterior: the belief, its revision, and its hash.

    :attr:`belief` is a live :class:`BeliefField` (uncertainty-first — ``mean``/``variance``/
    ``posterior``); :attr:`revision` counts the observation batches folded in; :attr:`content_hash`
    is the server-consistent content address (equal to ``belief.content_hash`` — verified).
    :attr:`frame` and :attr:`crs` are the field's georeference, round-tripped from the served
    snapshot (RFC-0007): the Core :class:`~astro_mine.core.units.ReferenceFrame` queried positions
    resolve in and the explicit :class:`~astro_mine.core.units.PlanetaryCRS` for reprojection, so a
    distributed subscriber places the belief field without re-deriving the vocabulary.
    """

    belief: BeliefField
    revision: int
    content_hash: str
    frame: ReferenceFrame
    crs: PlanetaryCRS


class FieldServiceClient:
    """A thin client over the :class:`~astro_mine.prospect.service.server.FieldServicer` gRPC
    surface.

    Construct from a channel and the caller's bearer token — ``FieldServiceClient(secure_channel(
    addr, root_certificates=ca), token=access_token)``. The token is the OIDC credential the service
    authenticates and authorizes every RPC against (prospect.md §9), and it rides as
    ``authorization`` call metadata on **every** request this client makes. Omit it only against a
    local-dev server running the documented insecure mode.

    Reconstructing a :class:`BeliefField` needs the prior, which the server ships in the snapshot's
    ``prior_bundle`` — so the client rebuilds the exact posterior without importing the recipe or
    re-fitting.
    """

    def __init__(self, channel: grpc.Channel, *, token: str | None = None) -> None:
        self._stub = pbg.FieldServiceStub(channel)  # type: ignore[no-untyped-call]
        self._metadata = () if token is None else bearer_metadata(token)

    def get_field(self, field_id: str) -> FieldState:
        """Fetch and reconstruct the current shared posterior for ``field_id`` (fail-closed)."""
        snapshot = self._stub.GetField(pb.FieldRequest(field_id=field_id), metadata=self._metadata)
        prior = prior_from_bundle(snapshot.prior_bundle)
        log = from_proto_observations(snapshot.observations)
        belief = BeliefField.from_prior(prior).update(log)
        frame = reference_frame_from_proto(snapshot.frame)
        crs = planetary_crs_from_proto(snapshot.crs)
        return self._verified(
            belief, snapshot.revision, snapshot.content_hash, frame=frame, crs=crs
        )

    def submit_observations(
        self, field_id: str, observations: Sequence[FieldObservation]
    ) -> pb.UpdateAck:
        """Append ``observations`` to the field's log; returns the server's :class:`UpdateAck`."""
        batch = pb.ObservationBatch(
            field_id=field_id, observations=to_proto_observations(observations)
        )
        ack: pb.UpdateAck = self._stub.SubmitObservations(batch, metadata=self._metadata)
        return ack

    def stream_belief_updates(
        self, field_id: str, *, from_revision: int = 0
    ) -> Iterator[pb.BeliefUpdate]:
        """Stream raw belief-update increments from ``from_revision`` (catch-up then live)."""
        request = pb.SubscribeRequest(field_id=field_id, from_revision=from_revision)
        yield from self._stub.StreamBeliefUpdates(request, metadata=self._metadata)

    def follow(self, field_id: str) -> Iterator[FieldState]:
        """Track the one shared posterior: yield a fresh, hash-verified :class:`FieldState` per
        update.

        Fetches the current state, then applies each streamed increment in order — reconstructing
        the
        posterior the same way the server holds it. Each yield is checked against the server's
        content
        hash, so a distributed sim following this stream shares the server's exact posterior or
        fails
        closed. The generator ends when the stream is cancelled / the server stops.
        """
        state = self.get_field(field_id)
        yield state
        belief = state.belief
        # The georeference is fixed for a field's lifetime and rides on the initial snapshot; the
        # belief-update stream carries only the observation increments, so carry frame/CRS forward.
        for update in self.stream_belief_updates(field_id, from_revision=state.revision):
            belief = belief.update(from_proto_observations(update.observations))
            state = self._verified(
                belief, update.revision, update.content_hash, frame=state.frame, crs=state.crs
            )
            yield state

    @staticmethod
    def _verified(
        belief: BeliefField,
        revision: int,
        expected_hash: str,
        *,
        frame: ReferenceFrame,
        crs: PlanetaryCRS,
    ) -> FieldState:
        if belief.content_hash != expected_hash:
            raise PosteriorConsistencyError(
                f"reconstructed posterior hash {belief.content_hash} != server {expected_hash} "
                f"at revision {revision}: client and server diverged"
            )
        return FieldState(
            belief=belief, revision=revision, content_hash=expected_hash, frame=frame, crs=crs
        )
