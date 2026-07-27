"""The conditionable-belief seam a simulator drives — Core types in, Core types out.

A belief is Prospect's (prospect.md §6): a simulator renders sensor observations *of* a sealed
field and deliberately never maintains a posterior over it, so the Bayes has to live here. But
the consumer that *has* the observations is [Sim](sim.md), and the dependency between them is
one-way — Sim never imports Prospect (conventions.md §1.1). Sim reaches this module the way it
reaches every other producer: through the ``astro_mine.providers`` entry-point group, calling a
factory it resolved by :class:`~astro_mine.core.registry.PluginKind` and holding the result
behind a protocol of its own shape. Link's ``ConnectivitySampler`` is the same arrangement.

That constraint decides this module's whole surface. Everything crossing the boundary is a
**Core** type — :class:`~astro_mine.core.messages.model.SensorReading` in,
:class:`~astro_mine.core.resource.FieldDistribution` out — because the caller cannot name a
Prospect type. In particular the caller cannot construct a
:class:`~astro_mine.prospect.belief.observation.FieldObservation`: adapting a reading into a
belief-log entry needs the instrument-likelihood registry, which is Prospect's. So
:meth:`GriddedBelief.observe` takes raw readings and adapts them here.

**Cell identity is Prospect's too**, for the same reason: the grid is Prospect's, the projection
is Prospect's, and a consumer scoring belief-quality metrics needs stable keys it can compare
across a prior and a posterior without knowing either. See :data:`CELL_ID_FORMAT`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from astro_mine.core.resource import FieldDistribution
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.belief.observation import FieldObservation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from astro_mine.core.messages.model import SensorReading
    from astro_mine.core.registry import PluginManifest
    from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata

__all__ = ["CELL_ID_FORMAT", "GriddedBelief", "belief_from_bundle", "cell_id"]

#: How a grid cell is named. Row-major zero-based indices into the field's own
#: :class:`~astro_mine.prospect.field.metadata.FieldGrid`, zero-padded so ids sort lexicographically
#: in grid order: ``r0119c0120``. Zero-padding is to **four** digits, which covers a 9999x9999 grid;
#: a larger grid widens the field and changes the ids, so it is a new field version, not a silent
#: re-key.
#:
#: The ids are meaningful **only against the grid that minted them** — they are indices, not
#: coordinates. Two fields on different grids may use the same id for different ground. A consumer
#: comparing a prior to a posterior is safe because both come from one :class:`GriddedBelief` and
#: therefore one grid; a consumer comparing across *fields* must not assume the ids align.
CELL_ID_FORMAT = "r{row:04d}c{col:04d}"


def cell_id(row: int, col: int) -> str:
    """The stable id of the cell at zero-based ``(row, col)`` — see :data:`CELL_ID_FORMAT`."""
    return CELL_ID_FORMAT.format(row=row, col=col)


def _cell_centre_m(grid: FieldGrid, row: int, col: int) -> tuple[float, float]:
    """The projected-metre centre of a cell, matching the backends' own convention."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    return (grid.min_x_m + (col + 0.5) * dx, grid.min_y_m + (row + 0.5) * dy)


def _to_lat_lon(x_m: float, y_m: float, radius_m: float) -> tuple[float, float]:
    """Inverse south-polar stereographic: projected metres -> planetocentric degrees.

    The closed-form spherical inverse for ``+proj=stere +lat_0=-90 +R=<radius>``, which is the
    projection every Prospect grid declares (``priors.catalog.SHACKLETON_CRS``). Spherical because
    the CRS models the body as a sphere (PROJ ``+R``), so there is no ellipsoidal series to sum and
    no projection library to depend on.

    With the standard inverse stereographic at a south-polar aspect the latitude reduces to
    ``degrees(c) - 90`` for ``c = 2*atan(rho / 2R)``, and the longitude to ``atan2(x, y)``.
    """
    rho = math.hypot(x_m, y_m)
    if rho == 0.0:
        return (-90.0, 0.0)
    c = 2.0 * math.atan2(rho, 2.0 * radius_m)
    return (math.degrees(c) - 90.0, math.degrees(math.atan2(x_m, y_m)) % 360.0)


def _within(value: float, bounds: tuple[float, float], *, wrap: bool = False) -> bool:
    low, high = bounds
    if not wrap:
        return low <= value <= high
    # Longitude windows may be authored unwrapped to stay increasing (e.g. [350, 370] crosses the
    # prime meridian), so test the value at both its own turn and the next one.
    return low <= value <= high or low <= value + 360.0 <= high


class GriddedBelief:
    """A belief over a pinned prior, conditionable by Core readings and addressed by cell id.

    Immutable: :meth:`observe` returns a new belief, mirroring
    :meth:`~astro_mine.prospect.belief.field.BeliefField.update` and preserving its replay
    property — conditioning once over an ordered log and conditioning incrementally over the same
    log give the same posterior.
    """

    __slots__ = ("_field", "_grid", "_metadata")

    def __init__(self, field: BeliefField, metadata: FieldMetadata) -> None:
        if metadata.grid is None:
            raise ValueError(
                "a gridded belief needs the field's grid — this bundle's metadata declares none, "
                "so its cells cannot be enumerated or addressed"
            )
        self._field = field
        self._metadata = metadata
        self._grid = metadata.grid

    @property
    def species(self) -> str:
        """The resource species the belief models."""
        return self._metadata.species

    @property
    def unit(self) -> str:
        """The belief's SI unit token."""
        return self._metadata.unit

    def observe(
        self, readings: Iterable[tuple[SensorReading, Sequence[float], float]]
    ) -> GriddedBelief:
        """Condition on ``(reading, position, time_s)`` triples; return the updated belief.

        ``position`` is the observing agent's location in the field's own projected coordinates
        (SI metres). Each reading is adapted through
        :meth:`~astro_mine.prospect.belief.observation.FieldObservation.from_sensor_reading`, which
        resolves the instrument likelihood from the reading's ``sensor`` tag — so a neutron
        spectrometer conditions with its footprint and depth response rather than as a point
        sample.

        Readings that cannot be belief observations are **skipped, not guessed at**: a reading with
        no ``noise_sigma`` has no likelihood, and one flagged ``valid=False`` is the sensor saying
        it measured nothing. Inventing a likelihood for either would fabricate exactly the
        confidence this seam exists to avoid.
        """
        observations: list[FieldObservation] = []
        for reading, position, time_s in readings:
            if not reading.valid or reading.noise_sigma is None or not reading.values:
                continue
            if len(position) != 3:
                raise ValueError(
                    f"an observing position must be a 3-vector in the field's projected metres; "
                    f"got {len(position)} components"
                )
            at = (float(position[0]), float(position[1]), float(position[2]))
            observations.append(
                FieldObservation.from_sensor_reading(reading, position=at, time_s=time_s)
            )
        if not observations:
            return self
        return GriddedBelief(self._field.update(observations), self._metadata)

    def cells(self) -> Mapping[str, FieldDistribution]:
        """Every cell's current distribution, keyed by :func:`cell_id`.

        The full grid, so a consumer can compare a prior and a posterior cell-for-cell.
        """
        mean = self._field.mean_grid()
        variance = self._field.variance_grid()
        species, unit = self._metadata.species, self._metadata.unit
        return {
            cell_id(row, col): FieldDistribution(
                mean=float(mean[row, col]),
                variance=float(variance[row, col]),
                species=species,
                unit=unit,
            )
            for row in range(self._grid.n_rows)
            for col in range(self._grid.n_cols)
        }

    def cells_in_region(
        self, *, lat_deg: tuple[float, float], lon_deg: tuple[float, float]
    ) -> frozenset[str]:
        """The ids of cells whose centres fall inside a planetocentric lat/lon window.

        The projection lives here rather than in the caller because the CRS is the field's: a
        consumer pinning a region in degrees should not have to know the grid is south-polar
        stereographic, nor carry a projection dependency to find out.

        Bounds are inclusive ``(min, max)``. A longitude window may be authored unwrapped to keep
        ``min < max`` across the prime meridian (``[350, 370]``).
        """
        radius_m = self._metadata.crs.reference_radius_m
        found = set()
        for row in range(self._grid.n_rows):
            for col in range(self._grid.n_cols):
                x_m, y_m = _cell_centre_m(self._grid, row, col)
                lat, lon = _to_lat_lon(x_m, y_m, radius_m)
                if _within(lat, lat_deg) and _within(lon, lon_deg, wrap=True):
                    found.add(cell_id(row, col))
        return frozenset(found)


def belief_from_bundle(manifest: PluginManifest, layers: Mapping[str, bytes]) -> GriddedBelief:
    """Rebuild a conditionable :class:`GriddedBelief` from a pulled prior bundle.

    The ``prior_recipe`` entry-point factory, and the sibling of
    :func:`~astro_mine.prospect.publish.from_bundle`: that one yields a **queryable** field (the
    sealed realization a simulator's sensors sample), this one recovers the **prior** so a caller
    can replay an observation log against it — the capability ``prior_from_bundle`` was written for
    (RM-P1-PROSPECT-11).

    ``manifest`` is accepted for the Core factory contract (the consumer has already
    version-negotiated and verified it); the content lives entirely in the layers.
    """
    from astro_mine.prospect.publish._bundle import BUNDLE_MEDIA_TYPE, prior_from_bundle

    try:
        tar_bytes = layers[BUNDLE_MEDIA_TYPE]
    except KeyError:
        available = ", ".join(sorted(layers)) or "(none)"
        raise ValueError(
            f"no prior bundle found in the pulled layers (expected {BUNDLE_MEDIA_TYPE!r}); "
            f"available media types: {available}. A belief must be conditioned from the prior "
            "itself, which only that layer carries — a rebuilt queryable field is a posterior "
            "snapshot and cannot be re-conditioned."
        ) from None
    prior = prior_from_bundle(tar_bytes)
    return GriddedBelief(BeliefField.from_prior(prior), prior.metadata)
