"""RM-P1-PROSPECT-11 — the distributed field service (prospect.md §3, §5, §7).

Proves the acceptance criterion "the distributed field service streams a single consistent posterior
to a distributed sim; belief updates stay replayable":

- **Serve + reconstruct** — a client fetches the shared field (prior bundle + ordered log) and
  rebuilds a live, uncertainty-first ``BeliefField`` whose content hash matches the server's.
- **Single consistent posterior** — two independent subscribers following the same field converge on
  the *same* content hash the server holds, after the same ordered updates.
- **Replayable** — reconstructing from the streamed observation log reproduces a byte-identical
  posterior (the content hash is exactly what conditioning the log locally yields).
- **Fail-closed** — a divergent reconstruction is rejected, not silently served.

The server runs in-process on an ephemeral loopback port (no external broker).
"""

from __future__ import annotations

import grpc
import pytest

from astro_mine.prospect.belief import BeliefField, FieldObservation
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.service import (
    INSECURE_DEV_ENV_VAR,
    FieldServiceClient,
    InsecureDevAuth,
    PosteriorConsistencyError,
    serve,
)

_FIELD = "shackleton_water_ice"

#: These tests exercise the *transport + replay* contract, not the auth contract (that is
#: ``test_field_service_auth.py``), so they run the server in its documented, loopback-only,
#: explicitly-opt-in local-dev mode — which grants every caller the full scope + capability grant.
#: There is no unauthenticated default to fall back on: ``serve`` requires an ``auth`` posture.
_DEV_AUTH = InsecureDevAuth()


@pytest.fixture(autouse=True)
def _enable_insecure_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt in to the cleartext local-dev mode (prospect.md §9): explicit env var + loopback."""
    monkeypatch.setenv(INSECURE_DEV_ENV_VAR, "1")


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _obs(value: float, x: float = 0.0, y: float = 0.0) -> FieldObservation:
    return FieldObservation(x_m=x, y_m=y, value=value, noise_sigma=0.01, sensor="neutron")


def test_get_field_reconstructs_the_uncertainty_first_posterior() -> None:
    prior = load_prior(grid=_grid())
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        state = client.get_field(_FIELD)

    # The reconstructed belief equals a from-prior belief, and is uncertainty-first (variance > 0).
    reference = BeliefField.from_prior(prior)
    assert state.revision == 0
    assert state.content_hash == reference.content_hash
    assert state.belief.variance((0.0, 0.0, 0.0)) > 0.0


def test_submitted_observations_condition_the_one_posterior() -> None:
    prior = load_prior(grid=_grid())
    observations = [_obs(0.05, 100.0, 100.0), _obs(0.04, -200.0, 50.0)]
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        ack = client.submit_observations(_FIELD, observations)
        state = client.get_field(_FIELD)

    # The server-side posterior is exactly "prior conditioned on the ordered log" — replayable.
    expected = BeliefField.from_prior(prior).update(observations)
    assert ack.revision == 1 and ack.applied == 2
    assert ack.content_hash == expected.content_hash
    assert state.content_hash == expected.content_hash


def test_two_subscribers_share_one_consistent_posterior() -> None:
    prior = load_prior(grid=_grid())
    batches = [[_obs(0.05, 100.0, 100.0)], [_obs(0.03, -300.0, 200.0)], [_obs(0.06, 0.0, -400.0)]]
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        writer = FieldServiceClient(grpc.insecure_channel(address))
        sub_a = FieldServiceClient(grpc.insecure_channel(address))
        sub_b = FieldServiceClient(grpc.insecure_channel(address))

        # Both subscribers catch up from the beginning; drain exactly the 3 increments each.
        stream_a = sub_a.stream_belief_updates(_FIELD, from_revision=0)
        stream_b = sub_b.stream_belief_updates(_FIELD, from_revision=0)
        for batch in batches:
            writer.submit_observations(_FIELD, batch)

        seen_a = [next(stream_a) for _ in batches]
        seen_b = [next(stream_b) for _ in batches]
        server_hash = writer.get_field(_FIELD).content_hash

    # Same ordered updates ⇒ same content hash on both subscribers and the server (one posterior).
    assert [u.content_hash for u in seen_a] == [u.content_hash for u in seen_b]
    assert seen_a[-1].content_hash == server_hash
    assert [u.revision for u in seen_a] == [1, 2, 3]


def test_follow_tracks_the_posterior_and_matches_a_local_replay() -> None:
    prior = load_prior(grid=_grid())
    batches = [[_obs(0.05, 100.0, 100.0)], [_obs(0.03, -300.0, 200.0)]]
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        writer = FieldServiceClient(grpc.insecure_channel(address))
        follower = FieldServiceClient(grpc.insecure_channel(address))
        stream = follower.follow(_FIELD)  # yields the initial state, then one per update

        initial = next(stream)
        states = []
        for batch in batches:
            writer.submit_observations(_FIELD, batch)
            states.append(next(stream))

    # A local replay of the same ordered log reproduces the followed posterior byte-for-byte.
    local = BeliefField.from_prior(prior)
    assert initial.content_hash == local.content_hash
    for batch, state in zip(batches, states, strict=True):
        local = local.update(batch)
        assert state.content_hash == local.content_hash  # single consistent, replayable posterior


def test_reconstruction_fails_closed_on_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = load_prior(grid=_grid())
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        client.submit_observations(_FIELD, [_obs(0.05, 10.0, 10.0)])  # server log has one obs

        # Simulate a lossy/tampered decode: the client drops the streamed log, so its reconstructed
        # posterior (prior only) no longer matches the server's snapshot hash — it must fail closed.
        import astro_mine.prospect.service.client as client_mod

        monkeypatch.setattr(client_mod, "from_proto_observations", lambda observations: ())
        with pytest.raises(PosteriorConsistencyError, match="diverged"):
            client.get_field(_FIELD)


def test_snapshot_round_trips_the_planetary_crs_and_reference_frame() -> None:
    # RM-P1-PROSPECT-14 / RFC-0007: a ResourceField served over field_service.proto carries its
    # georeference. The client decodes the typed ReferenceFrame + PlanetaryCRS back into the exact
    # Core value types FieldMetadata bound, so a distributed subscriber places the field correctly.
    prior = load_prior(grid=_grid())
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        state = client.get_field(_FIELD)

    assert state.frame == prior.metadata.frame
    assert state.crs == prior.metadata.crs
    # The CRS is an explicit planetary CRS (never an implicit/Earth default), and the frame agrees
    # with the CRS body — the invariant FieldMetadata enforces survives the wire round-trip.
    assert state.crs.body == prior.metadata.crs.body
    assert state.frame.center is None or state.frame.center == state.crs.body


def test_follow_carries_the_georeference_across_streamed_updates() -> None:
    prior = load_prior(grid=_grid())
    with serve({_FIELD: prior}, auth=_DEV_AUTH) as (_server, address):
        writer = FieldServiceClient(grpc.insecure_channel(address))
        follower = FieldServiceClient(grpc.insecure_channel(address))
        stream = follower.follow(_FIELD)

        initial = next(stream)
        writer.submit_observations(_FIELD, [_obs(0.05, 100.0, 100.0)])
        after = next(stream)

    # BeliefUpdate increments carry only observations; the georeference rides the initial snapshot
    # and is carried forward, so every followed state exposes the same frame/CRS.
    assert initial.crs == prior.metadata.crs
    assert after.crs == prior.metadata.crs
    assert after.frame == prior.metadata.frame


def test_unknown_field_is_a_not_found_error() -> None:
    with serve({_FIELD: load_prior(grid=_grid())}, auth=_DEV_AUTH) as (_server, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        with pytest.raises(grpc.RpcError) as excinfo:
            client.get_field("no-such-field")
        assert excinfo.value.code() is grpc.StatusCode.NOT_FOUND
