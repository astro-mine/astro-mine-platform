# SPDX-License-Identifier: Apache-2.0
"""The typed sensor return that drives belief updating (prospect.md §key-abstractions, §5).

A :class:`FieldObservation` is one atomic entry of a belief field's **ordered observation log**:
a located, noisy reading of the resource field — a value plus its sensor likelihood — never a
point ground-truth guess. It is the Prospect-side observation record, named distinctly from the
Core per-tick :class:`~astro_mine.core.messages.model.Observation` (which bundles an agent's whole
perception for one step); :meth:`FieldObservation.from_sensor_reading` adapts a Core
:class:`~astro_mine.core.messages.model.SensorReading` into one, so Sim and Ops feed the belief the
*same* records they emit on the wire (prospect.md §6).

:func:`load_observations` reads an ordered log from CSV — the ``(location, sensor reading)`` feed of
RM-P0-PROSPECT-04's acceptance criterion. File order is log order: the log is replayed exactly as
read, so a posterior is reproducible from its observation log (prospect.md §5).

Backlog: RM-P0-PROSPECT-04 — astro-mine-prospect#4
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.messages.model import SensorReading
from astro_mine.core.resource import Position
from astro_mine.prospect.sensors import is_registered

__all__ = ["FieldObservation", "load_observations"]

#: The CSV columns :func:`load_observations` reads. ``x_m``/``y_m``/``value``/``noise_sigma`` are
#: required; ``z_m``/``time_s``/``sensor``/``likelihood`` are optional (planar fields default
#: ``z=0``; an untagged reading conditions under the default point-Gaussian likelihood).
_REQUIRED_COLUMNS = ("x_m", "y_m", "value", "noise_sigma")


class FieldObservation(BaseModel):
    """One located, noisy reading of a resource field — an entry of the ordered belief log.

    ``value`` is the sensor's reading (in the field's unit) at ``position``; ``noise_sigma`` is the
    reading's likelihood standard deviation (the measurement-noise model — strictly positive, since
    a noiseless reading is ground truth, not an observation). ``time_s`` is the ordering timestamp
    and ``sensor`` an optional provenance tag. Frozen and content-addressable, so a log hashes
    reproducibly (prospect.md §5).

    ``likelihood`` names the **registered sensor likelihood** the reading was rendered by
    (:mod:`astro_mine.prospect.sensors`) — ``"neutron_spectrometer"``, ``"nir_reflectance"``,
    ``"gpr"``, ``"drill_assay"``, … . It is the instrument tag
    :meth:`~astro_mine.prospect.belief.field.BeliefField.update` resolves, so the belief conditions
    with the same **footprint** and **depth response** the reading was synthesized under, rather
    than treating every instrument as one scalar-sigma point measurement (prospect.md §3, §6).
    ``None`` — the default — conditions under the zero-footprint, unit-gain point model, i.e.
    exactly the pre-instrument behavior.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float
    y_m: float
    z_m: float = 0.0
    value: float
    noise_sigma: float = Field(gt=0.0)
    time_s: float = 0.0
    sensor: str | None = None
    likelihood: str | None = None

    @property
    def position(self) -> Position:
        """The reading's location as a Core :data:`~astro_mine.core.resource.Position`."""
        return (self.x_m, self.y_m, self.z_m)

    @classmethod
    def from_sensor_reading(
        cls,
        reading: SensorReading,
        *,
        position: Position,
        time_s: float = 0.0,
        index: int = 0,
        likelihood: str | None = None,
    ) -> FieldObservation:
        """Adapt a Core :class:`SensorReading` at ``position`` into a belief-log observation.

        Maps the ``index``-th scalar of the reading's ``values`` (a reading may carry several) and
        its realized ``noise_sigma`` likelihood — the seam that lets Sim/Ops drive the belief with
        the very records they emit on the wire. A reading with no ``noise_sigma`` cannot be a belief
        observation (a likelihood is mandatory) and is rejected.

        The instrument model is resolved automatically: when ``likelihood`` is not given and the
        reading's ``sensor`` tag names a **registered** sensor likelihood, the observation is tagged
        with it. That is the seam that closes the Sim↔Prospect loop — Sim renders a reading through
        :meth:`~astro_mine.prospect.sensors.SensorLikelihood.sense` (whose ``SensorReading.sensor``
        defaults to the likelihood's name) and the belief conditions it back under the *same* model,
        with no per-instrument code on either side.
        """
        if reading.noise_sigma is None:
            raise ValueError(
                f"SensorReading {reading.sensor!r} has no noise_sigma; a belief observation "
                "requires a sensor likelihood"
            )
        try:
            value = reading.values[index]
        except IndexError:
            raise ValueError(
                f"SensorReading {reading.sensor!r} has no value at index {index} "
                f"(values has length {len(reading.values)})"
            ) from None
        if likelihood is None and is_registered(reading.sensor):
            likelihood = reading.sensor
        return cls(
            x_m=position[0],
            y_m=position[1],
            z_m=position[2],
            value=value,
            noise_sigma=reading.noise_sigma,
            time_s=time_s,
            sensor=reading.sensor,
            likelihood=likelihood,
        )


def load_observations(source: str | PathLike[str] | Iterable[str]) -> tuple[FieldObservation, ...]:
    """Read an ordered observation log from CSV (a path, or an iterable of CSV lines).

    The CSV MUST have a header naming at least ``x_m``, ``y_m``, ``value``, ``noise_sigma``;
    ``z_m`` (default ``0``), ``time_s`` (default ``0``), ``sensor``, and ``likelihood`` (the
    registered instrument tag) are optional. Rows are returned in file order — the order in which
    the belief replays them. A missing required column, a non-numeric field, an unknown
    ``likelihood``, or a violated constraint (e.g. ``noise_sigma <= 0``) raises ``ValueError``.
    """
    if isinstance(source, str | PathLike):
        with Path(source).open(newline="") as handle:
            return _parse_rows(handle)
    return _parse_rows(source)


def _parse_rows(lines: Iterable[str]) -> tuple[FieldObservation, ...]:
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError("observation CSV is empty (no header row)")
    missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"observation CSV is missing required column(s): {', '.join(missing)}")
    observations: list[FieldObservation] = []
    for line_no, row in enumerate(reader, start=2):  # line 1 is the header
        observations.append(_row_to_observation(row, line_no))
    return tuple(observations)


def _row_to_observation(row: dict[str, str | None], line_no: int) -> FieldObservation:
    def num(column: str, default: float | None = None) -> float:
        raw = row.get(column)
        if raw is None or raw == "":
            if default is not None:
                return default
            raise ValueError(f"observation CSV line {line_no}: empty required field {column!r}")
        try:
            return float(raw)
        except ValueError:
            raise ValueError(
                f"observation CSV line {line_no}: field {column!r} is not a number: {raw!r}"
            ) from None

    sensor = row.get("sensor")
    likelihood = row.get("likelihood")
    if likelihood and not is_registered(likelihood):
        raise ValueError(
            f"observation CSV line {line_no}: unknown sensor likelihood {likelihood!r} "
            "(see astro_mine.prospect.sensors.list_likelihoods)"
        )
    try:
        return FieldObservation(
            x_m=num("x_m"),
            y_m=num("y_m"),
            z_m=num("z_m", 0.0),
            value=num("value"),
            noise_sigma=num("noise_sigma"),
            time_s=num("time_s", 0.0),
            sensor=sensor if sensor else None,
            likelihood=likelihood if likelihood else None,
        )
    except ValueError as exc:  # pydantic constraint (e.g. noise_sigma <= 0)
        raise ValueError(f"observation CSV line {line_no}: {exc}") from exc
