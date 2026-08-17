# SPDX-License-Identifier: Apache-2.0
"""Prospect field metadata — the species/unit and CRS/grid binding (prospect.md §2/§5).

A Prospect resource field carries, alongside the Core :class:`~astro_mine.core.resource.\
ResourceField` query surface, the metadata that says *what* it models and *where*: the
resource :class:`FieldMetadata.species`, its SI ``unit`` token, the query ``frame`` and the
explicit planetary ``crs`` (the CRS/grid binding, consistent with Worlds — RM-P0-WORLDS-01),
and the optional :class:`FieldGrid` spatial domain. The CRS/frame primitives are Core's
(:mod:`astro_mine.core.units`); units are validated against the Core waist vocabulary, so
there is no implicit Earth/WGS84 and no typo'd unit (conventions.md §5).

Backlog: RM-P0-PROSPECT-01 — astro-mine-prospect#1
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.units import PlanetaryCRS, ReferenceFrame
from astro_mine.core.units.validate import is_si_unit

__all__ = ["FieldGrid", "FieldMetadata"]


class _Model(BaseModel):
    """Frozen base for the field-metadata models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldGrid(_Model):
    """The regular 2-D spatial grid a gridded resource field is defined over — the field's
    spatial domain, in the CRS's projected coordinates (SI metres), consistent with the
    Worlds reprojected grid (RM-P0-WORLDS-01).

    Bounds are half-open ``[min, max)`` on each axis; ``n_rows``/``n_cols`` are the grid
    shape. Depth/3-D and multi-species axes are deferred (P1+, out of scope for this item).
    """

    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float
    n_rows: int = Field(gt=0)
    n_cols: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_extent(self) -> FieldGrid:
        if self.max_x_m <= self.min_x_m or self.max_y_m <= self.min_y_m:
            raise ValueError(
                "grid max bound must be strictly greater than the min bound on each axis "
                f"(x: [{self.min_x_m}, {self.max_x_m}), y: [{self.min_y_m}, {self.max_y_m}))"
            )
        return self


class FieldMetadata(_Model):
    """The metadata a Prospect resource field carries (prospect.md §2/§5).

    ``species`` is the modeled resource (e.g. ``water_equivalent_hydrogen``); ``unit`` its SI
    unit token, validated against the Core waist vocabulary
    (:data:`astro_mine.core.units.enums.KNOWN_UNITS`). ``frame`` is the
    :class:`~astro_mine.core.units.ReferenceFrame` queried positions resolve in (it backs the
    Core contract's ``frame`` property), and ``crs`` the explicit
    :class:`~astro_mine.core.units.PlanetaryCRS` for reprojection; ``grid`` is the optional
    spatial domain. A sealed ground-truth realization and the belief posterior of one scenario
    share identical metadata, so their fields are directly comparable.
    """

    species: str = Field(min_length=1)
    unit: str
    frame: ReferenceFrame
    crs: PlanetaryCRS
    grid: FieldGrid | None = None

    @model_validator(mode="after")
    def _validate(self) -> FieldMetadata:
        if not is_si_unit(self.unit):
            raise ValueError(
                f"unit {self.unit!r} is not a recognized SI/dimensionless unit token "
                "(no implicit or typo'd units; see astro_mine.core.units.KNOWN_UNITS)"
            )
        if self.frame.center is not None and self.frame.center != self.crs.body:
            raise ValueError(
                f"field frame is centered on {self.frame.center!r} but the CRS body is "
                f"{self.crs.body!r} — frame and CRS must agree on the body"
            )
        return self
