"""RM-P0-PROSPECT-01 — the ResourceField contract Prospect adopts from the Core narrow waist.

Proves the two load-bearing properties at the Prospect boundary: a Prospect field built on
:class:`~astro_mine.prospect.field.BaseResourceField` satisfies the Core
:class:`~astro_mine.core.resource.ResourceField` contract, and there is no point-estimate-only
path (the uncertainty is structural). Also exercises the Prospect-side
:class:`~astro_mine.prospect.field.FieldMetadata` / :class:`~astro_mine.prospect.field.FieldGrid`
CRS/grid binding and its fail-loud validation.
"""

from __future__ import annotations

import pytest

from astro_mine.core.resource import FieldDistribution, ResourceField, check_resource_field
from astro_mine.core.units import MOON, MOON_BODY_FIXED, FrameClass, PlanetaryCRS, ReferenceFrame
from astro_mine.prospect.field import (
    BaseResourceField,
    FieldGrid,
    FieldMetadata,
    Position,
)

# A lunar south-pole CRS + grid (Shackleton vicinity), illustrative — consistent with Worlds.
_MOON_CRS = PlanetaryCRS(
    body=MOON,
    body_fixed_frame="MOON_ME",
    reference_radius_m=1_737_400.0,
    projection="+proj=stere +lat_0=-90 +R=1737400",
)
_GRID = FieldGrid(
    min_x_m=-5_000.0, min_y_m=-5_000.0, max_x_m=5_000.0, max_y_m=5_000.0, n_rows=100, n_cols=100
)


def _metadata(**overrides: object) -> FieldMetadata:
    fields: dict[str, object] = {
        "species": "water_equivalent_hydrogen",
        "unit": "mass_fraction",
        "frame": MOON_BODY_FIXED,
        "crs": _MOON_CRS,
        "grid": _GRID,
    }
    fields.update(overrides)
    return FieldMetadata(**fields)


class ConstantField(BaseResourceField):
    """A trivial Prospect field — flat mean, fixed variance — to drive the contract."""

    def __init__(self, *, mean: float = 0.3, variance: float = 0.01) -> None:
        super().__init__(_metadata())
        self._m = mean
        self._v = variance

    def mean(self, position: Position, *, epoch: object | None = None) -> float:
        return self._m

    def variance(self, position: Position, *, epoch: object | None = None) -> float:
        return self._v

    def quantile(self, position: Position, q: float, *, epoch: object | None = None) -> float:
        return self._m + (q - 0.5)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: object | None = None,
    ) -> tuple[float, ...]:
        return tuple(self._m for _ in range(n))


def test_constant_field_satisfies_core_contract() -> None:
    field = ConstantField()
    assert isinstance(field, ResourceField)
    assert check_resource_field(field) is None
    # metadata-backed contract properties resolve
    assert field.species == "water_equivalent_hydrogen"
    assert field.unit == "mass_fraction"
    assert field.frame is MOON_BODY_FIXED
    assert field.metadata.crs.body == MOON


def test_posterior_is_uncertainty_first() -> None:
    dist = ConstantField().posterior((0.0, 0.0, 0.0))
    assert isinstance(dist, FieldDistribution)
    # the mean is never returned without its variance and quantiles
    assert dist.mean == 0.3
    assert dist.variance == 0.01
    assert 0.5 in dist.quantiles
    assert dist.species == "water_equivalent_hydrogen"


def test_point_estimate_only_field_cannot_be_instantiated() -> None:
    # A field exposing only mean() omits variance/quantile/sample — the ABC forbids it,
    # so a point-estimate-only Prospect field is impossible to construct (prospect.md §2.1).
    class MeanOnly(BaseResourceField):
        def mean(self, position: Position, *, epoch: object | None = None) -> float:
            return 0.3

    with pytest.raises(TypeError):
        MeanOnly(_metadata())


def test_metadata_carries_crs_and_grid() -> None:
    md = _metadata()
    assert md.crs.body == MOON
    assert md.crs.reference_radius_m == 1_737_400.0
    assert md.grid is not None
    assert (md.grid.n_rows, md.grid.n_cols) == (100, 100)


def test_metadata_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError, match="not a recognized"):
        _metadata(unit="furlongs")


def test_metadata_rejects_frame_crs_body_mismatch() -> None:
    earth_frame = ReferenceFrame(name="ITRF93", frame_class=FrameClass.BODY_FIXED, center="EARTH")
    with pytest.raises(ValueError, match="must agree on the body"):
        _metadata(frame=earth_frame)


def test_field_grid_rejects_inverted_extent() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        FieldGrid(min_x_m=0.0, min_y_m=0.0, max_x_m=-1.0, max_y_m=1.0, n_rows=1, n_cols=1)
