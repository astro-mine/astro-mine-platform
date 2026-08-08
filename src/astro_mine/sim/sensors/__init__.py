"""Sensor models — render observations *of* the resource field, never a point guess (RM-P0-SIM-06;
richer models + registry RM-P1-SIM-05; ISRU stored-mass gauge RM-P1-SIM-02).

A sensor model turns an agent's state (and the world's sealed resource field / its ISRU state) into
the :class:`~astro_mine.core.messages.model.SensorReading` an agent actually receives. Models are
**registry-discovered plugins** keyed by :class:`~astro_mine.core.sadf.enums.SensorKind`
(RM-P1-SIM-05), so adding or swapping a sensor model is *registration*, and routing a sensor is
*configuration* (which kind the SADF declares), not a change to :func:`render_sensor`. Two families:

- **Resource sensors** (a SADF :class:`~astro_mine.core.sadf.model.Sensor` carrying a ``resource``
  target — neutron/NIR spectrometer, GPR, drill assay, …): they **sample** the injected Core
  :class:`~astro_mine.core.resource.ResourceField` at the agent's position and add the declared
  measurement noise (the SADF ``observation_model.noise_sigma``). The field is the access-gated
  :class:`GroundTruthField` (prospect.md §9) — a *noisy draw of the sealed truth*, never a point
  estimate. The default model renders a single look; a **higher-fidelity** model (RM-P1-SIM-05, e.g.
  the NIR spectrometer) renders an **ensemble** of looks so the reading carries the measurement
  *distribution*, not one point.
- **Self-state sensors** (no resource target): IMU/odometry render the agent's own velocity;
  rangefinder/LIDAR/altimeter render its altitude over the body-fixed datum; contact renders a
  mode-derived touch flag; the thermal sensor reads the agent's temperature (set by RM-P0-SIM-07);
  the **ISRU stored-mass gauge** (RM-P1-SIM-02, kind ``resource_storage``) reports cumulative stored
  water (kg) + its extraction energy (J) from the agent's ISRU state.

Determinism is the caller's: each model draws from a seeded :class:`random.Random` stream the
stepping core derives per (agent, sensor), so same-seed runs reproduce byte-for-byte (CX-REPRO). A
declared-but-unmodelled sensor renders an explicit ``valid=False`` reading rather than a silent gap.

Backlog: RM-P0-SIM-06 -- astro-mine-sim#6
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.messages.model import SensorReading
from astro_mine.core.resource import FieldDistribution
from astro_mine.core.sadf.enums import SensorKind
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.core.world import IlluminationState

if TYPE_CHECKING:
    from collections.abc import Callable
    from random import Random

    from astro_mine.core.messages.model import StateSample
    from astro_mine.core.resource import Position, ResourceField
    from astro_mine.core.sadf.model import Sensor
    from astro_mine.core.units import Epoch, ReferenceFrame
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.isru import IsruState

__all__ = [
    "DEFAULT_ENSEMBLE_LOOKS",
    "DEFAULT_IMAGING_FEATURES",
    "DEFAULT_IMAGING_FOV_DEG",
    "DEFAULT_IMAGING_RANGE_M",
    "MIN_PASSIVE_IMAGING_FLUX_W_M2",
    "ReferenceResourceField",
    "SensorContext",
    "SensorModel",
    "imaging_footprint",
    "register_resource_model",
    "register_self_state_model",
    "render_sensor",
]

#: Independent looks a higher-fidelity ensemble resource sensor renders (RM-P1-SIM-05).
DEFAULT_ENSEMBLE_LOOKS = 8

#: The imaging sensor's field of view (deg) when the SADF ``observation_model`` declares none.
DEFAULT_IMAGING_FOV_DEG = 60.0
#: The imaging sensor's slant range (m) — how far down-boresight the frame's footprint is cast —
#: when the SADF ``observation_model`` declares no ``range_m``.
DEFAULT_IMAGING_RANGE_M = 10.0
#: Ground samples an imaging frame renders across its footprint: the **feature vector** that makes
#: the reading an observation *of* the field over an area, not a single point guess (sim.md §5).
DEFAULT_IMAGING_FEATURES = 5
#: Incident solar flux (W·m⁻²) below which a **passive** imager cannot form an exposure. A frame
#: cast into a permanently shadowed region with no auxiliary illumination is dark, so the reading is
#: rendered ``valid=False`` — degrade, don't lie (prospect.md §6). An *actively* illuminated look at
#: the same ground is the LIDAR/rangefinder model, which carries its own emitter.
MIN_PASSIVE_IMAGING_FLUX_W_M2 = 1.0


@dataclass(frozen=True, slots=True)
class SensorContext:
    """Everything a sensor model needs this tick: the SADF ``sensor``, the agent ``state``, the
    injected (sealed) ``field`` (``None`` when unwired), the agent's ``isru`` state (``None`` for a
    non-ISRU asset), and the seeded ``rng`` stream. Bundled so a model is a single-arg plugin.

    ``world`` and ``epoch`` are the Core :class:`~astro_mine.core.world.WorldProvider` and the
    tick's epoch — what a *framing* sensor (the RM-P0-SIM-06 imaging model) needs to resolve the
    terrain and illumination **inside its footprint**, since a frame observes ground the agent is
    not standing on.
    Both default to ``None``, so every existing single-point model is untouched."""

    sensor: Sensor
    state: StateSample
    field: ResourceField | None
    isru: IsruState | None
    rng: Random
    world: WorldProvider | None = None
    epoch: Epoch | None = None


if TYPE_CHECKING:
    #: A sensor forward-model: a pure function of the per-tick context to one reading.
    SensorModel = Callable[[SensorContext], SensorReading]

#: The plugin registries — resource-sensor models and self-state models, keyed by ``SensorKind``. A
#: resource kind with no registered model falls back to the default single-look render; a self-state
#: kind with none renders ``valid=False`` (RM-P1-SIM-05).
_RESOURCE_MODELS: dict[SensorKind, SensorModel] = {}
_SELF_STATE_MODELS: dict[SensorKind, SensorModel] = {}


def register_resource_model(kind: SensorKind) -> Callable[[SensorModel], SensorModel]:
    """Register a **resource** sensor model for ``kind`` (a plugin seam, RM-P1-SIM-05)."""

    def _register(model: SensorModel) -> SensorModel:
        _RESOURCE_MODELS[kind] = model
        return model

    return _register


def register_self_state_model(kind: SensorKind) -> Callable[[SensorModel], SensorModel]:
    """Register a **self-state** (proprioceptive/gauge) sensor model for ``kind`` (RM-P1-SIM-05)."""

    def _register(model: SensorModel) -> SensorModel:
        _SELF_STATE_MODELS[kind] = model
        return model

    return _register


def render_sensor(
    sensor: Sensor,
    state: StateSample,
    field: ResourceField | None,
    rng: Random,
    *,
    isru: IsruState | None = None,
    world: WorldProvider | None = None,
    epoch: Epoch | None = None,
) -> SensorReading:
    """The :class:`~astro_mine.core.messages.model.SensorReading` ``sensor`` produces this tick.

    Dispatches through the plugin registries. A ``resource`` target selects the *resource* reading
    **when the kind has one registered** — imaging and NIR both do, and for them a declared target
    genuinely means "sample the field through this instrument". A kind registered only as a
    *self-state* model keeps it: the target then names **what the instrument reports on**, not what
    it looks at.

    That distinction is the whole of astro-mine-sim#61's first defect. A ``resource_storage`` gauge
    declares ``ResourceTarget(species="water", si_unit="kg")`` to say *the tank holds water,
    measured in kilograms* — and the old dispatch read that as "sample the ice field here", so the
    plant's tank reported the **regolith's water-equivalent abundance under the plant** instead of
    its own stored mass, tagged ``unit="kg"``. Bench's ``water_mass`` filters on exactly
    ``(species, "kg")``, so the anchor's headline metric scored a mass *fraction* as kilograms and
    moved with the terrain rather than with anything the swarm did.

    Kinds with neither model registered still fall back to the single-look field render, which is
    what neutron / GPR / drill-assay want.

    ``isru`` is the agent's ISRU state, read by the ``resource_storage`` gauge (RM-P1-SIM-02);
    ``world`` / ``epoch`` are the world provider and tick epoch a *framing* sensor (imaging)
    resolves its footprint's terrain + illumination against.
    """
    ctx = SensorContext(
        sensor=sensor, state=state, field=field, isru=isru, rng=rng, world=world, epoch=epoch
    )
    if sensor.resource is not None:
        resource_model = _RESOURCE_MODELS.get(sensor.kind)
        if resource_model is not None:
            return resource_model(ctx)
        gauge = _SELF_STATE_MODELS.get(sensor.kind)
        return gauge(ctx) if gauge is not None else _render_resource_sample(ctx)
    model = _SELF_STATE_MODELS.get(sensor.kind)
    return model(ctx) if model is not None else SensorReading(sensor=sensor.name, valid=False)


# --- resource models -------------------------------------------------------------


def _render_resource_sample(ctx: SensorContext) -> SensorReading:
    """The default resource render — a single noisy look at the sealed field (RM-P0-SIM-06)."""
    resource = ctx.sensor.resource
    assert resource is not None  # dispatched only for a sensor with a resource target
    if ctx.field is None:
        # No ground truth wired in — degrade loudly via the reading's validity, never fabricate.
        return SensorReading(
            sensor=ctx.sensor.name,
            unit=resource.si_unit,
            resource_species=resource.species,
            valid=False,
        )
    sigma = _noise_sigma(ctx.sensor)
    position = _position(ctx.state)
    truth = ctx.field.sample(position, seed=ctx.rng.randrange(2**31))[0]
    reading = truth + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0)
    return SensorReading(
        sensor=ctx.sensor.name,
        values=[reading],
        unit=resource.si_unit,
        resource_species=resource.species,
        noise_sigma=sigma,
        valid=True,
    )


@register_resource_model(SensorKind.NIR_SPECTROMETER)
def _render_resource_ensemble(ctx: SensorContext) -> SensorReading:
    """A higher-fidelity resource render (RM-P1-SIM-05): an **ensemble** of independent noisy
    looks at the sealed field, so the reading carries the measurement *distribution* (its spread is
    the observation uncertainty) rather than a single point. Still an observation OF the field — a
    per-look noisy draw of the sealed truth — never a point ground-truth guess (prospect.md §6)."""
    resource = ctx.sensor.resource
    assert resource is not None
    if ctx.field is None:
        return SensorReading(
            sensor=ctx.sensor.name,
            unit=resource.si_unit,
            resource_species=resource.species,
            valid=False,
        )
    sigma = _noise_sigma(ctx.sensor)
    position = _position(ctx.state)
    looks = [
        ctx.field.sample(position, seed=ctx.rng.randrange(2**31))[0]
        + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0)
        for _ in range(DEFAULT_ENSEMBLE_LOOKS)
    ]
    return SensorReading(
        sensor=ctx.sensor.name,
        values=looks,
        unit=resource.si_unit,
        resource_species=resource.species,
        noise_sigma=sigma,
        valid=True,
    )


# --- imaging (a framing sensor: it observes ground the agent is not standing on) ---


def imaging_footprint(ctx: SensorContext) -> tuple[Position, ...]:
    """The ground points an imaging frame observes — its FOV footprint, in the agent's frame.

    Real (if reduced-order) frame geometry: the boresight is the sensor's mounted ``pose`` rotation
    applied to the camera's local ``-z`` (nadir when the SADF declares no pose); the footprint
    centre is cast ``range_m`` down that boresight, and its radius is ``range_m * tan(fov/2)``. The
    returned points are the centre plus a deterministic ring of ``DEFAULT_IMAGING_FEATURES - 1``
    samples across the disc — the frame's **feature grid**, so the reading is an observation *of* an
    area rather than one point (sim.md §5; prospect.md §6). Purely geometric: no RNG, so the
    geometry is reproducible.
    """
    model = ctx.sensor.observation_model
    fov_deg = DEFAULT_IMAGING_FOV_DEG if model is None or model.fov_deg is None else model.fov_deg
    range_m = DEFAULT_IMAGING_RANGE_M if model is None or model.range_m is None else model.range_m
    boresight = _boresight(ctx.sensor)
    origin = _position(ctx.state)
    centre = tuple(o + b * range_m for o, b in zip(origin, boresight, strict=True))
    radius = range_m * math.tan(math.radians(fov_deg) / 2.0)
    # Two axes spanning the frame plane (any pair orthogonal to the boresight).
    u, v = _frame_axes(boresight)
    points: list[Position] = [(centre[0], centre[1], centre[2])]
    ring = DEFAULT_IMAGING_FEATURES - 1
    for i in range(ring):
        angle = 2.0 * math.pi * i / ring
        du, dv = radius * math.cos(angle), radius * math.sin(angle)
        points.append(
            (
                centre[0] + u[0] * du + v[0] * dv,
                centre[1] + u[1] * du + v[1] * dv,
                centre[2] + u[2] * du + v[2] * dv,
            )
        )
    return tuple(points)


def _illumination_flux(ctx: SensorContext, point: Position) -> tuple[IlluminationState, float]:
    """The solar illumination state + incident flux at ``point``.

    With no world provider wired the frame is treated as **unlit** — the model never invents an
    exposure it cannot justify (degrade, don't lie); the caller sees ``valid=False``."""
    if ctx.world is None:
        return IlluminationState.SHADOW, 0.0
    surface = ctx.world.sample(point, epoch=ctx.epoch)
    return surface.illumination.state, surface.illumination.solar_flux_w_m2


@register_resource_model(SensorKind.IMAGING)
@register_self_state_model(SensorKind.IMAGING)
def _render_imaging(ctx: SensorContext) -> SensorReading:
    """Imaging — a **passive** framing sensor (RM-P0-SIM-06; sim.md §3, prospect.md §6).

    Three things make this a real model rather than a stub:

    - **Frame geometry.** The reading covers the sensor's FOV footprint at its declared slant range,
      resolved from the SADF ``observation_model`` (``fov_deg`` / ``range_m``) and the sensor's
      mounted ``pose`` (:func:`imaging_footprint`) — not the single point under the agent.
    - **Illumination-dependent validity.** The frame is exposed only if its footprint centre is lit.
      A frame cast into a permanently shadowed region (a PSR — the anchor scenario's whole point)
      has no solar flux, so a passive camera renders ``valid=False`` rather than a fabricated image;
      a ``PENUMBRA`` frame is *degraded*, not invalid — it stays valid but its reported
      ``noise_sigma`` is inflated by the flux shortfall, so a consumer weights it correctly. An
      actively illuminated look at the same ground is the LIDAR/rangefinder model, which carries its
      own emitter.
    - **An observation OF the world/Prospect state.** With a ``resource`` target the frame renders
    one
      noisy draw of the **sealed field** per footprint sample — the resource content of the imaged
      area, never a point ground-truth guess. With no resource target it renders the footprint's
      terrain relief (per-sample ground elevation), the geometric feature vector a visual-odometry
      or hazard consumer reads. Either way ``values`` is the frame's per-feature vector.
    """
    resource = ctx.sensor.resource
    unit = resource.si_unit if resource is not None else "m"
    species = resource.species if resource is not None else None
    footprint = imaging_footprint(ctx)
    state, flux = _illumination_flux(ctx, footprint[0])
    if state is IlluminationState.SHADOW or flux < MIN_PASSIVE_IMAGING_FLUX_W_M2:
        # A dark frame: no exposure to report. Loud invalidity, never a fabricated image.
        return SensorReading(
            sensor=ctx.sensor.name, unit=unit, resource_species=species, valid=False
        )
    if resource is not None and ctx.field is None:
        # Framed at a resource target with no ground truth wired in — same rule: degrade loudly.
        return SensorReading(
            sensor=ctx.sensor.name, unit=unit, resource_species=species, valid=False
        )
    sigma = _degraded_sigma(_noise_sigma(ctx.sensor), state, flux)
    values: list[float] = []
    for point in footprint:
        if resource is not None:
            assert ctx.field is not None  # guarded above
            truth = ctx.field.sample(point, seed=ctx.rng.randrange(2**31))[0]
        else:
            truth = _elevation(ctx, point)
        values.append(truth + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0))
    return SensorReading(
        sensor=ctx.sensor.name,
        values=values,
        unit=unit,
        resource_species=species,
        noise_sigma=sigma,
        valid=True,
    )


def _elevation(ctx: SensorContext, point: Position) -> float:
    """The ground elevation at ``point`` — the geometric feature a non-resource frame renders.

    Falls back to the point's own ``z`` when no world provider is wired (the frame is then only
    rendered at all if some provider reported it lit, so this is the flat-datum reduced-order case).
    """
    if ctx.world is None:
        return point[2]
    return ctx.world.sample(point, epoch=ctx.epoch).elevation_m


def _degraded_sigma(sigma: float | None, state: IlluminationState, flux: float) -> float | None:
    """Inflate the declared noise sigma by a partially-shadowed frame's flux shortfall.

    A ``PENUMBRA`` exposure is *degraded*, not invalid: the reading stays valid but carries a larger
    ``noise_sigma``, so a consumer (Prospect's belief update) down-weights it exactly as it should
    (prospect.md §3 — the observation model is shared). A ``LIT`` frame is undegraded."""
    if sigma is None or state is not IlluminationState.PENUMBRA:
        return sigma
    reference = _SOLAR_CONSTANT_W_M2
    ratio = max(flux, MIN_PASSIVE_IMAGING_FLUX_W_M2) / reference
    return sigma / math.sqrt(ratio) if 0.0 < ratio < 1.0 else sigma


#: Solar constant at 1 AU (W·m⁻²) — the reference flux a degraded (penumbral) exposure is scaled
#: against. Mirrors the same constant the power/thermal reference world provider uses.
_SOLAR_CONSTANT_W_M2 = 1361.0


def _boresight(sensor: Sensor) -> tuple[float, float, float]:
    """The sensor's viewing direction: its mounted ``pose`` rotation applied to the camera's local
    ``-z``, or straight nadir (``-z``) when the SADF declares no pose (the reduced-order
    default)."""
    if sensor.pose is None:
        return (0.0, 0.0, -1.0)
    q = sensor.pose.rotation_quat_xyzw
    return _rotate_by_quat((0.0, 0.0, -1.0), (q.x, q.y, q.z, q.w))


def _rotate_by_quat(
    v: tuple[float, float, float], q: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """Rotate ``v`` by the unit quaternion ``q`` (x, y, z, w) — ``v + 2 * qv x (qv x v + w *
    v)``."""
    x, y, z, w = q
    ux, uy, uz = x, y, z
    # t = 2 * (qv x v)
    tx = 2.0 * (uy * v[2] - uz * v[1])
    ty = 2.0 * (uz * v[0] - ux * v[2])
    tz = 2.0 * (ux * v[1] - uy * v[0])
    return (
        v[0] + w * tx + (uy * tz - uz * ty),
        v[1] + w * ty + (uz * tx - ux * tz),
        v[2] + w * tz + (ux * ty - uy * tx),
    )


def _frame_axes(
    boresight: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Two orthonormal axes spanning the image plane normal to ``boresight`` (Gram-Schmidt against
    whichever world axis is least parallel to it, so the construction never degenerates)."""
    seed = (1.0, 0.0, 0.0) if abs(boresight[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _normalize(_cross(seed, boresight))
    v = _normalize(_cross(boresight, u))
    return u, v


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(c * c for c in v))
    if magnitude == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


# --- self-state models -----------------------------------------------------------


@register_self_state_model(SensorKind.IMU)
@register_self_state_model(SensorKind.ODOMETRY)
def _render_velocity(ctx: SensorContext) -> SensorReading:
    """IMU / odometry — the agent's own linear velocity (reduced-order; angular rates are P1)."""
    velocity = ctx.state.linear_velocity_mps
    sigma = _noise_sigma(ctx.sensor)
    components = (0.0, 0.0, 0.0) if velocity is None else (velocity.x, velocity.y, velocity.z)
    values = [comp + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0) for comp in components]
    return SensorReading(sensor=ctx.sensor.name, values=values, unit="m/s", noise_sigma=sigma)


@register_self_state_model(SensorKind.RANGEFINDER)
@register_self_state_model(SensorKind.LIDAR)
@register_self_state_model(SensorKind.ALTIMETER)
def _render_altitude(ctx: SensorContext) -> SensorReading:
    """Rangefinder / LIDAR / altimeter — nadir altitude over the body-fixed datum (reduced-order
    until Worlds terrain/occlusion, RM-P0-WORLDS-06)."""
    sigma = _noise_sigma(ctx.sensor)
    altitude = ctx.state.pose.translation_m.z + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0)
    return SensorReading(sensor=ctx.sensor.name, values=[altitude], unit="m", noise_sigma=sigma)


@register_self_state_model(SensorKind.CONTACT)
def _render_contact(ctx: SensorContext) -> SensorReading:
    """Contact — a touch flag derived from the agent's operating mode (reduced-order)."""
    contacting = ctx.state.mode in _CONTACT_MODES
    return SensorReading(sensor=ctx.sensor.name, values=[1.0 if contacting else 0.0])


@register_self_state_model(SensorKind.THERMAL_SENSOR)
def _render_thermal(ctx: SensorContext) -> SensorReading:
    """Thermal sensor — the agent's temperature, set by power/thermal evolution (RM-P0-SIM-07);
    until an asset carries a thermal model the temperature is unset, so the reading is invalid."""
    if ctx.state.temperature_k is None:
        return SensorReading(sensor=ctx.sensor.name, unit="K", valid=False)
    sigma = _noise_sigma(ctx.sensor)
    value = ctx.state.temperature_k + (ctx.rng.gauss(0.0, sigma) if sigma else 0.0)
    return SensorReading(sensor=ctx.sensor.name, values=[value], unit="K", noise_sigma=sigma)


@register_self_state_model(SensorKind.RESOURCE_STORAGE)
def _render_storage(ctx: SensorContext) -> SensorReading:
    """ISRU stored-mass gauge (RM-P1-SIM-02): cumulative stored water (kg) + its extraction energy.

    ``values[0]`` is the cumulative stored water (kg, the primary quantity ``unit`` /
    ``resource_species`` describe); ``values[1]`` is the cumulative extraction energy (J), so a
    single MCAP channel carries both the ``water_mass`` and ``energy_per_kg`` inputs Bench scores
    (bench.md §3). **Channel 0 is the contract** — Bench reads stored mass from it, and reading the
    last channel instead scores joules as kilograms (astro-mine-sim#61).

    The species and unit are the gauge's **declared** :class:`~astro_mine.core.sadf.model.
    ResourceTarget`, not literals: the SADF says what the tank holds and in what unit, and a
    reading that renamed it would not match the scenario's own metric binding. The anchor's plant
    declares ``water`` / ``kg``, so this changes nothing there — it stops the model asserting it.
    An asset with no ISRU state (none declared) renders ``valid=False``."""
    resource = ctx.sensor.resource
    species = resource.species if resource is not None else "water"
    unit = resource.si_unit if resource is not None else "kg"
    if ctx.isru is None:
        return SensorReading(
            sensor=ctx.sensor.name, unit=unit, resource_species=species, valid=False
        )
    return SensorReading(
        sensor=ctx.sensor.name,
        values=[ctx.isru.stored_water_kg, ctx.isru.energy_used_j],
        unit=unit,
        resource_species=species,
        valid=True,
    )


#: Operating modes that imply physical contact (drives the reduced-order contact sensor).
_CONTACT_MODES = frozenset({"excavate", "drill", "dig"})


def _noise_sigma(sensor: Sensor) -> float | None:
    """The sensor's declared measurement-noise sigma (``observation_model.noise_sigma``), or
    ``None`` for a noiseless reduced-order reading."""
    model = sensor.observation_model
    if model is None or model.noise_sigma is None or model.noise_sigma <= 0.0:
        return None
    return model.noise_sigma


def _position(state: StateSample) -> Position:
    translation = state.pose.translation_m
    return (translation.x, translation.y, translation.z)


class ReferenceResourceField:
    """A deterministic, analytic Core :class:`~astro_mine.core.resource.ResourceField` for the
    always-works local tier — a single Gaussian ice "bump" so the prospecting loop runs offline
    before a real Prospect :class:`GroundTruthField` is wired in.

    It is uncertainty-first by contract (it implements the full distributional surface) but
    *deterministic* — zero variance, so :meth:`sample` returns the true value at a point and the
    sensor model adds the measurement noise on top, exactly as it would over a sealed ground-truth
    realization. Carries no ground-truth-access capability (it is a synthetic stand-in,
    not real sealed data)."""

    def __init__(
        self,
        *,
        species: str = "water_equivalent_hydrogen",
        unit: str = "mass_fraction",
        frame: ReferenceFrame = MOON_BODY_FIXED,
        peak: float = 0.1,
        center_m: Position = (0.0, 0.0, 0.0),
        length_scale_m: float = 10.0,
    ) -> None:
        if length_scale_m <= 0.0:
            raise ValueError(f"length_scale_m must be > 0, got {length_scale_m}")
        self._species = species
        self._unit = unit
        self._frame = frame
        self._peak = peak
        self._center = center_m
        self._length_scale = length_scale_m

    @property
    def species(self) -> str:
        return self._species

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def _value(self, position: Position) -> float:
        squared = sum((p - c) ** 2 for p, c in zip(position, self._center, strict=True))
        return self._peak * math.exp(-squared / (2.0 * self._length_scale**2))

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._value(position)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return 0.0  # a deterministic reference realization

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return self._value(position)  # zero-variance: every quantile is the mean

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return (self._value(position),) * n

    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        return FieldDistribution(
            mean=self._value(position), variance=0.0, species=self._species, unit=self._unit
        )
