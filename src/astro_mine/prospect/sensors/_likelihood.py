"""The sensor-likelihood contract — one model, used forward (Sim) and inverse (belief).

prospect.md §3 names **sensor/observation models** as a first-class extension point: "pluggable
likelihoods (neutron spectrometer, NIR reflectance, GPR, mass spec, drill assay) shared between
Prospect's updating and Sim's forward sensor model **so they stay consistent**". A
:class:`SensorLikelihood` is that shared object — *one* model, exposed in both directions:

- **forward** (Sim's sensor simulation, prospect.md §6) — :meth:`SensorLikelihood.sense` reads the
  sealed truth grid, applies the instrument's **footprint average** and **depth response**, adds
  seeded noise, and emits a Core :class:`~astro_mine.core.messages.model.SensorReading`: the very
  record Sim puts on the wire.
- **inverse** (belief conditioning, prospect.md §5) — :meth:`SensorLikelihood.conditioning_weights`
  and :meth:`SensorLikelihood.precision` say how that reading informs each grid cell, so
  :meth:`~astro_mine.prospect.belief.field.BeliefField.update` conditions with the *same* footprint
  and depth response Sim rendered with. There is no second copy of the instrument model to drift.

**The measurement model.** The field is the column-mean resource concentration over the reference
column (:data:`REFERENCE_COLUMN_DEPTH_M` — the ~1 m depth a neutron spectrometer integrates, which
is what a water-equivalent-hydrogen field *means*). An instrument does not see that column: it sees
its own depth window through its own vertical sensitivity, and it sees a spatial *footprint*, not a
point. So a reading is ``y = gain * (w . x) + noise`` with ``w`` the sum-normalized footprint
weights over grid cells, ``x`` the field, and ``gain`` the **depth-response gain**: the instrument's
sensitivity-weighted mean of the assumed :class:`VerticalProfile` over its depth window, divided by
that profile's mean over the reference column (:meth:`DepthResponse.gain`).

A neutron spectrometer integrating the whole column has ``gain`` near 1 (it is the definitional
instrument); a **surface-only** NIR reflectance sees the desiccated lag deposit and has ``gain`` far
below 1; a GPR or drill assay reaching buried ice has ``gain`` above 1. Inverting the model, the
reading informs the field's column value with precision ``gain**2 / sigma**2`` at the effective
value ``y / gain`` — which is exactly what the belief conditions on, and why a precise-but-shallow
NIR reading is honestly *weak* evidence about buried ice.

**Reduction to the Phase-0 model.** A zero-footprint, full-column, flat-sensitivity likelihood
(:data:`DEFAULT_LIKELIHOOD_NAME`) has ``gain == 1`` and a bilinear "footprint", so it reproduces the
previous single scalar-``noise_sigma`` Gaussian point measurement **exactly**: the default path is
unchanged and per-instrument behavior is opt-in.

Backlog: prospect.md §3, §6; LUNAR-FR-002; scenario §6 —
astro-mine-prospect#31
"""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.messages.model import SensorReading
from astro_mine.core.resource import Position
from astro_mine.core.sadf.enums import CapabilityTag, SensorKind
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata

__all__ = [
    "DEFAULT_LIKELIHOOD_NAME",
    "DEFAULT_PROFILE",
    "REFERENCE_COLUMN_DEPTH_M",
    "DepthResponse",
    "SensorLikelihood",
    "VerticalProfile",
]

#: The depth of the **reference column** the resource field is defined over (metres). A
#: water-equivalent-hydrogen field is a *column-mean* concentration, and the column it means is the
#: one an epithermal-neutron measurement integrates: roughly the top metre of regolith (the LEND /
#: LPNS sensing depth the anchor prior cites, ``priors/RECIPE.md``). Every instrument's depth
#: response is expressed as a **gain relative to this column**, so all the likelihoods speak about
#: the same quantity and a belief conditioned by a mixed instrument set stays coherent.
REFERENCE_COLUMN_DEPTH_M = 1.0

#: The registered name of the zero-footprint, full-column Gaussian point likelihood — the
#: backward-compatible default that an observation carrying no instrument tag is conditioned under
#: (it reproduces the Phase-0 scalar-sigma model exactly).
DEFAULT_LIKELIHOOD_NAME = "point_gaussian"

#: Quadrature resolution for the depth-response integrals. Deterministic and cheap: the integrals
#: are 1-D and evaluated once per likelihood, then cached on the frozen model.
_QUADRATURE_POINTS = 512


class _Model(BaseModel):
    """Frozen base for the sensor-likelihood models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VerticalProfile(_Model):
    """The assumed **relative** concentration of the resource with depth — a dimensionless shape.

    Lunar polar ice is not uniform with depth: sublimation leaves a **desiccated lag deposit** at
    the surface over comparatively ice-rich regolith below, so a surface-sensing instrument and a
    subsurface one measure genuinely different things at the same location. This is the
    reduced-order shape that difference is modeled with: relative concentration rises from
    :attr:`surface_fraction` at the surface toward 1, with an e-folding
    :attr:`desiccation_depth_m`. Only the *shape* matters — it is always used as a ratio
    (:meth:`DepthResponse.gain`), never as an absolute concentration, so the field's own units and
    magnitude are untouched.
    """

    surface_fraction: float = Field(gt=0.0, le=1.0)
    desiccation_depth_m: float = Field(gt=0.0)

    def relative(self, depth_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """The relative concentration at ``depth_m`` (dimensionless; surface fraction -> 1)."""
        z = np.asarray(depth_m, dtype=np.float64)
        deficit = 1.0 - self.surface_fraction
        return np.asarray(1.0 - deficit * np.exp(-z / self.desiccation_depth_m), dtype=np.float64)


#: The anchor scenario's regolith profile: a strongly desiccated top layer (10% of the buried
#: concentration at the very surface) recovering over ~0.2 m. Illustrative and reduced-order — a
#: recipe or scenario overrides it — but it is what makes an NIR surface reading and a drill assay
#: at the same cell honestly disagree.
DEFAULT_PROFILE = VerticalProfile(surface_fraction=0.1, desiccation_depth_m=0.2)


class DepthResponse(_Model):
    """An instrument's **vertical sensitivity**: an attenuated window ``[top_m, bottom_m)``.

    Sensitivity is ``exp(-(z - top_m) / attenuation_length_m)`` inside the window and zero outside;
    ``attenuation_length_m = None`` is a flat, uniformly-weighted window. The reduced-order stand-in
    for an instrument's true depth kernel — the neutron leakage profile, the radar two-way
    attenuation, the sampled interval of a drill core — sufficient to make the *ordering* and
    *magnitude* of the instruments' depth sensitivities honest.
    """

    top_m: float = Field(ge=0.0)
    bottom_m: float = Field(gt=0.0)
    attenuation_length_m: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_window(self) -> DepthResponse:
        if self.bottom_m <= self.top_m:
            raise ValueError(
                "depth-response bottom_m must be strictly below top_m "
                f"(got top {self.top_m}, bottom {self.bottom_m})"
            )
        return self

    def sensitivity(self, depth_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """The (unnormalized) sensitivity at ``depth_m`` — zero outside the window."""
        z = np.asarray(depth_m, dtype=np.float64)
        inside = (z >= self.top_m) & (z < self.bottom_m)
        if self.attenuation_length_m is None:
            return inside.astype(np.float64)
        decay = np.exp(-(z - self.top_m) / self.attenuation_length_m)
        return np.asarray(np.where(inside, decay, 0.0), dtype=np.float64)

    def gain(self, profile: VerticalProfile) -> float:
        """The **depth-response gain**: what this instrument reads, per unit of column-mean field.

        The sensitivity-weighted mean of ``profile`` over the instrument's window, divided by that
        profile's mean over the reference column (:data:`REFERENCE_COLUMN_DEPTH_M`). ``gain == 1``
        means the instrument reads the field's own quantity; below 1 that it under-reads it (a
        surface sensor over a desiccated layer); above 1 that it over-reads it (a subsurface sensor
        reaching buried ice). Strictly positive by construction, since the profile is.
        """
        z = np.linspace(self.top_m, self.bottom_m, _QUADRATURE_POINTS, dtype=np.float64)
        weights = self.sensitivity(np.clip(z, self.top_m, self.bottom_m - 1e-12))
        sensed = float(np.average(profile.relative(z), weights=weights))
        column = np.linspace(0.0, REFERENCE_COLUMN_DEPTH_M, _QUADRATURE_POINTS, dtype=np.float64)
        return sensed / float(np.mean(profile.relative(column)))


class SensorLikelihood(_Model):
    """One instrument's observation model — the plugin behind the ``sensors/`` extension point.

    Frozen, declarative, and content-addressable, so a likelihood round-trips through a Core
    ``observation_model`` :class:`~astro_mine.core.registry.PluginManifest`
    (:mod:`astro_mine.prospect.sensors._manifest`) and a scenario pins the exact instrument model it
    was run with. Instances are looked up by :attr:`name` through the registry
    (:func:`~astro_mine.prospect.sensors.get_likelihood`), so an observation names its likelihood
    and the belief resolves the matching model — no hard-coded instrument list in the conditioner.

    - :attr:`kind` is the **Core** :class:`~astro_mine.core.sadf.enums.SensorKind` (the
      narrow-waist vocabulary a Sim/SADF sensor declares) and :attr:`capability` the Core
      :class:`~astro_mine.core.sadf.enums.CapabilityTag` an asset carrying the instrument declares —
      so a Fleet asset, a Sim sensor, and this likelihood are provably the same instrument.
    - :attr:`footprint_sigma_m` is the Gaussian spatial footprint (``0`` == a point sensor).
    - :attr:`depth` is the vertical sensitivity and :attr:`profile` the regolith profile it is read
      against; together they fix :attr:`gain`.
    - :attr:`noise_sigma` is the instrument's nominal likelihood standard deviation, used when a
      caller does not override it per reading.
    """

    name: str = Field(min_length=1)
    kind: SensorKind
    capability: CapabilityTag
    footprint_sigma_m: float = Field(ge=0.0)
    depth: DepthResponse
    noise_sigma: float = Field(gt=0.0)
    profile: VerticalProfile = DEFAULT_PROFILE
    description: str = ""

    @cached_property
    def gain(self) -> float:
        """This instrument's depth-response gain against its regolith profile (the ratio above)."""
        return self.depth.gain(self.profile)

    # --- forward: Sim's sensor simulation (prospect.md §6) -------------------------------------

    def expected_reading(
        self, metadata: FieldMetadata, values: NDArray[np.float64], position: Position
    ) -> float:
        """The **noise-free** reading this instrument renders of ``values`` at ``position``.

        ``gain * (w . x)``: the footprint-weighted spatial average of the field, scaled by the
        depth-response gain. The forward operator that :meth:`conditioning_weights` and
        :meth:`precision` invert for belief updating.
        """
        weights = self.footprint_weights(_require_grid(metadata), position)
        flat = np.asarray(values, dtype=np.float64).ravel()
        return self.gain * float(np.dot(weights, flat))

    def sense(
        self,
        metadata: FieldMetadata,
        values: NDArray[np.float64],
        position: Position,
        *,
        rng: np.random.Generator | None = None,
        noise_sigma: float | None = None,
        sensor: str | None = None,
    ) -> SensorReading:
        """Render one noisy Core :class:`SensorReading` of ``values`` at ``position``.

        The forward sensor model **Sim consumes** (prospect.md §6): footprint average, then depth
        gain, then additive ``N(0, sigma**2)`` noise — emitted as the Core message Sim already puts
        on the wire, with the field's ``unit``/``resource_species`` and the realized ``noise_sigma``
        attached. The belief adapts it straight back into a
        :class:`~astro_mine.prospect.belief.observation.FieldObservation`
        (:meth:`FieldObservation.from_sensor_reading`) and conditions under *this* likelihood, so
        the two directions cannot drift apart. ``rng=None`` renders the noise-free expectation.

        ``sensor`` overrides the reading's provenance tag; it defaults to this likelihood's
        :attr:`name`, which is what lets the belief-side adapter resolve the model automatically.
        """
        sigma = self.noise_sigma if noise_sigma is None else float(noise_sigma)
        if sigma <= 0.0:
            raise ValueError(f"noise_sigma must be positive, got {sigma}")
        value = self.expected_reading(metadata, values, position)
        if rng is not None:
            value += float(rng.normal(0.0, sigma))
        return SensorReading(
            sensor=self.name if sensor is None else sensor,
            values=[value],
            unit=metadata.unit,
            resource_species=metadata.species,
            noise_sigma=sigma,
        )

    # --- inverse: belief conditioning (prospect.md §5) -----------------------------------------

    def to_field_value(self, reading: float) -> float:
        """Invert the depth response: the **column-mean field value** a ``reading`` implies."""
        return reading / self.gain

    def precision(self, noise_sigma: float) -> float:
        """The likelihood precision a reading of noise ``noise_sigma`` carries **about the field**.

        ``gain**2 / sigma**2``. The depth response rescales precision, not just the value: a
        surface-only instrument (gain far below 1) is *weakly* informative about the buried column
        even when its own readings are precise — exactly the honest behavior an over-confident
        scalar-sigma model gets wrong.
        """
        if noise_sigma <= 0.0:
            raise ValueError(f"noise_sigma must be positive, got {noise_sigma}")
        return (self.gain * self.gain) / (noise_sigma * noise_sigma)

    def conditioning_weights(
        self, grid: FieldGrid, position: Position, *, correlation_length_m: float
    ) -> NDArray[np.float64]:
        """The per-cell weights this reading's precision is spread over — **peak-normalized**.

        A Gaussian footprint convolved with a Gaussian spatial correlation is a Gaussian whose
        length scales add in quadrature, so the effective kernel is ``exp(-d**2 / (2 * (L**2 +
        f**2)))`` with ``L`` the belief's correlation length and ``f`` the instrument footprint. A
        point sensor (``f == 0``) reduces this **exactly** to the belief's own RBF weights — which
        is why the default conditioning path is bit-for-bit unchanged — while a broad-footprint
        instrument (a neutron spectrometer) correctly informs a wider neighbourhood per reading.
        """
        if correlation_length_m <= 0.0:
            raise ValueError(f"correlation_length_m must be positive, got {correlation_length_m}")
        effective = math.hypot(correlation_length_m, self.footprint_sigma_m)
        dist2 = _cell_distances_squared(grid, position)
        return np.asarray(np.exp(-dist2 / (2.0 * effective * effective)), dtype=np.float64)

    # --- the shared spatial footprint ----------------------------------------------------------

    def footprint_weights(self, grid: FieldGrid, position: Position) -> NDArray[np.float64]:
        """The instrument's spatial footprint over the grid cells — **sum-normalized** ``(M,)``.

        The averaging kernel of the *forward* model: a Gaussian disc of standard deviation
        :attr:`footprint_sigma_m`, or — for a point sensor (a drill core, the default likelihood) —
        the bilinear stencil of the four cells around ``position``, so that a point reading of a
        field is exactly that field's interpolated value with no spurious smoothing.
        """
        if self.footprint_sigma_m == 0.0:
            return _bilinear_stencil(grid, position)
        sigma = self.footprint_sigma_m
        weights = np.exp(-_cell_distances_squared(grid, position) / (2.0 * sigma * sigma))
        total = float(weights.sum())
        if total <= 0.0:  # a footprint far smaller than a cell, centred off-grid: fall back
            return _bilinear_stencil(grid, position)
        return np.asarray(weights / total, dtype=np.float64)


# --- grid geometry helpers (numpy + FieldGrid only — no backend dependency) ----------------------


def _require_grid(metadata: FieldMetadata) -> FieldGrid:
    grid = metadata.grid
    if grid is None:
        raise ValueError("a sensor likelihood requires metadata.grid (a FieldGrid spatial domain)")
    return grid


def _cell_centers(grid: FieldGrid) -> NDArray[np.float64]:
    """The ``(M, 2)`` cell-centre coordinates of ``grid``, in row-major (``ravel``) order."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _cell_distances_squared(grid: FieldGrid, position: Position) -> NDArray[np.float64]:
    """Squared planar distance from ``position`` to every cell centre — ``(M,)``."""
    diff = _cell_centers(grid) - np.array([position[0], position[1]], dtype=np.float64)
    return np.asarray(np.einsum("mi,mi->m", diff, diff), dtype=np.float64)


def _bilinear_stencil(grid: FieldGrid, position: Position) -> NDArray[np.float64]:
    """The four-cell, sum-1 bilinear stencil at ``position`` (edge-clamped), flattened to ``(M,)``.

    The zero-footprint limit of :meth:`SensorLikelihood.footprint_weights`: dotting this with a
    field's values reproduces the grid backend's bilinear query exactly, so a point-sensor reading
    of the sealed truth is the truth's own interpolated value.
    """
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    fc = min(max((position[0] - grid.min_x_m) / dx - 0.5, 0.0), float(grid.n_cols - 1))
    fr = min(max((position[1] - grid.min_y_m) / dy - 0.5, 0.0), float(grid.n_rows - 1))
    c0, r0 = int(np.floor(fc)), int(np.floor(fr))
    c1, r1 = min(c0 + 1, grid.n_cols - 1), min(r0 + 1, grid.n_rows - 1)
    tc, tr = fc - c0, fr - r0
    weights = np.zeros((grid.n_rows, grid.n_cols), dtype=np.float64)
    weights[r0, c0] += (1.0 - tr) * (1.0 - tc)
    weights[r0, c1] += (1.0 - tr) * tc
    weights[r1, c0] += tr * (1.0 - tc)
    weights[r1, c1] += tr * tc
    return weights.ravel()
