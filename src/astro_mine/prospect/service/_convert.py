"""Convert between the Prospect :class:`FieldObservation` and its wire :class:`Observation` proto.

The single place the field service crosses the library ↔ wire boundary, so the belief log a client
replays is exactly the log the server conditioned (the replay property, prospect.md §5). That
includes the **sensor-likelihood tag**: a subscriber that dropped it would replay the log under a
different instrument model, reach a different posterior, and be rejected by its own fail-closed hash
check — so the tag is as load-bearing on the wire as the value itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.service._proto import field_service_pb2 as pb

__all__ = ["from_proto_observations", "to_proto_observation", "to_proto_observations"]


def to_proto_observation(observation: FieldObservation) -> pb.Observation:
    """Encode a :class:`FieldObservation` as its wire :class:`Observation`."""
    return pb.Observation(
        x_m=observation.x_m,
        y_m=observation.y_m,
        z_m=observation.z_m,
        value=observation.value,
        noise_sigma=observation.noise_sigma,
        time_s=observation.time_s,
        sensor=observation.sensor or "",
        likelihood=observation.likelihood or "",
    )


def to_proto_observations(observations: Iterable[FieldObservation]) -> list[pb.Observation]:
    """Encode an ordered log of :class:`FieldObservation` as wire observations (order preserved)."""
    return [to_proto_observation(o) for o in observations]


def from_proto_observations(observations: Sequence[pb.Observation]) -> tuple[FieldObservation, ...]:
    """Decode wire observations into an ordered :class:`FieldObservation` log (order preserved).

    ``sensor == ""`` / ``likelihood == ""`` decode to ``None`` (the wire has no null string),
    matching how :func:`to_proto_observation` encodes an absent tag.
    """
    return tuple(
        FieldObservation(
            x_m=o.x_m,
            y_m=o.y_m,
            z_m=o.z_m,
            value=o.value,
            noise_sigma=o.noise_sigma,
            time_s=o.time_s,
            sensor=o.sensor or None,
            likelihood=o.likelihood or None,
        )
        for o in observations
    )
