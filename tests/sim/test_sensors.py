"""RM-P0-SIM-06 — sensor models that render observations *of* the resource field.

Covers the acceptance criteria: a scout's neutron/NIR spectrometer produces a noisy
observation sampled from the (sealed) ground-truth field — never a point estimate — and the
reading carries what Prospect's belief needs to update from it. Also covers the
proprioceptive models, the reference field's Core ResourceField contract, and determinism.
"""

from __future__ import annotations

import math
from random import Random

import pytest

from astro_mine.core.messages.model import (
    ActionBatch,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.resource import Position, ResourceField, check_resource_field
from astro_mine.core.sadf.enums import SensorKind
from astro_mine.core.sadf.model import ObservationModel, ResourceTarget, Sensor
from astro_mine.core.sadf.model import Quat as SadfQuat
from astro_mine.core.sadf.model import Transform as SadfTransform
from astro_mine.core.sadf.model import Vec3 as SadfVec3
from astro_mine.core.units import J2000_EPOCH, MOON_BODY_FIXED, Epoch, ReferenceFrame
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
)
from astro_mine.sim.isru import IsruState
from astro_mine.sim.runtime import AgentSpec, MobilityDynamics, Scenario, Simulator, run_episode
from astro_mine.sim.sensors import (
    DEFAULT_ENSEMBLE_LOOKS,
    DEFAULT_IMAGING_FEATURES,
    ReferenceResourceField,
    SensorContext,
    imaging_footprint,
    register_self_state_model,
    render_sensor,
)

_SPECIES = "water_equivalent_hydrogen"
_UNIT = "mass_fraction"


def _state(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] | None = None,
    mode: str | None = None,
    temperature_k: float | None = None,
) -> StateSample:
    return StateSample(
        agent_id="a",
        frame=MOON_BODY_FIXED,
        pose=Transform(
            translation_m=Vec3(x=position[0], y=position[1], z=position[2]),
            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        linear_velocity_mps=None
        if velocity is None
        else Vec3(x=velocity[0], y=velocity[1], z=velocity[2]),
        mode=mode,
        temperature_k=temperature_k,
    )


def _resource_sensor(*, noise_sigma: float | None) -> Sensor:
    model = None if noise_sigma is None else ObservationModel(noise_sigma=noise_sigma)
    return Sensor(
        name="neutron",
        kind=SensorKind.NEUTRON_SPECTROMETER,
        frame="body",
        observation_model=model,
        resource=ResourceTarget(species=_SPECIES, si_unit=_UNIT),
    )


# --- the reference field is a Core ResourceField ---------------------------------


def test_reference_field_satisfies_the_core_resource_field_contract() -> None:
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    assert isinstance(field, ResourceField)
    check_resource_field(field)  # the consumer-driven Core contract test
    assert field.species == _SPECIES and field.unit == _UNIT and field.frame == MOON_BODY_FIXED
    # deterministic + uncertainty-first: the bump peaks at its centre, variance is zero.
    assert field.mean((0.0, 0.0, 0.0)) == pytest.approx(0.1)
    assert field.variance((0.0, 0.0, 0.0)) == 0.0
    assert field.sample((1.0, 0.0, 0.0), n=3) == pytest.approx((field.mean((1.0, 0.0, 0.0)),) * 3)
    assert field.mean((50.0, 0.0, 0.0)) < field.mean((1.0, 0.0, 0.0))  # falls off with distance


def test_reference_field_rejects_a_non_positive_length_scale() -> None:
    with pytest.raises(ValueError, match="length_scale_m must be > 0"):
        ReferenceResourceField(length_scale_m=0.0)


# --- resource sensors render a noisy sample OF the sealed field -------------------


def test_resource_sensor_renders_a_noisy_sample_not_a_point_estimate() -> None:
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    sensor = _resource_sensor(noise_sigma=0.02)
    truth = field.mean((1.0, 0.0, 0.0))

    reading = render_sensor(sensor, _state(position=(1.0, 0.0, 0.0)), field, Random(1))
    assert reading.valid and reading.values is not None
    assert reading.resource_species == _SPECIES and reading.unit == _UNIT
    assert reading.noise_sigma == 0.02  # carried so Prospect's belief can update from it
    assert reading.values[0] != truth  # a noisy measurement, not the bare point estimate
    assert abs(reading.values[0] - truth) < 0.2  # but a measurement *of* the truth

    # independent streams -> independent noise: the sensor samples, it does not read a guess.
    other = render_sensor(sensor, _state(position=(1.0, 0.0, 0.0)), field, Random(2))
    assert other.values[0] != reading.values[0]


def test_resource_sensor_without_noise_returns_the_truth() -> None:
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    reading = render_sensor(_resource_sensor(noise_sigma=None), _state(), field, Random(0))
    assert reading.values == [pytest.approx(0.1)] and reading.noise_sigma is None


def test_resource_sensor_without_a_field_degrades_loudly() -> None:
    reading = render_sensor(_resource_sensor(noise_sigma=0.02), _state(), None, Random(0))
    assert reading.valid is False
    assert reading.resource_species == _SPECIES and reading.unit == _UNIT
    assert reading.values == []  # no fabricated value


# --- proprioceptive sensors ------------------------------------------------------


@pytest.mark.parametrize("kind", [SensorKind.IMU, SensorKind.ODOMETRY])
def test_imu_and_odometry_render_the_agents_velocity(kind: SensorKind) -> None:
    sensor = Sensor(name="motion", kind=kind, frame="body")
    reading = render_sensor(sensor, _state(velocity=(1.0, 2.0, 3.0)), None, Random(0))
    assert reading.unit == "m/s" and reading.values == [1.0, 2.0, 3.0]


def test_velocity_sensor_reads_zero_when_velocity_is_unset() -> None:
    sensor = Sensor(name="imu", kind=SensorKind.IMU, frame="body")
    assert render_sensor(sensor, _state(velocity=None), None, Random(0)).values == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("kind", [SensorKind.RANGEFINDER, SensorKind.LIDAR, SensorKind.ALTIMETER])
def test_range_sensors_render_nadir_altitude(kind: SensorKind) -> None:
    sensor = Sensor(name="range", kind=kind, frame="body")
    reading = render_sensor(sensor, _state(position=(0.0, 0.0, 5.0)), None, Random(0))
    assert reading.unit == "m" and reading.values == [5.0]


def test_contact_sensor_reflects_an_excavation_mode() -> None:
    sensor = Sensor(name="touch", kind=SensorKind.CONTACT, frame="body")
    assert render_sensor(sensor, _state(mode="excavate"), None, Random(0)).values == [1.0]
    assert render_sensor(sensor, _state(mode="idle"), None, Random(0)).values == [0.0]


def test_thermal_sensor_reads_temperature_when_set_else_degrades() -> None:
    sensor = Sensor(name="temp", kind=SensorKind.THERMAL_SENSOR, frame="body")
    hot = render_sensor(sensor, _state(temperature_k=250.0), None, Random(0))
    assert hot.valid and hot.unit == "K" and hot.values == [250.0]
    cold = render_sensor(sensor, _state(temperature_k=None), None, Random(0))
    assert cold.valid is False  # power/thermal (RM-P0-SIM-07) sets it


def test_an_unmodelled_sensor_kind_renders_invalid_not_nothing() -> None:
    # IMAGING is modelled now (RM-P0-SIM-06, below), so the unmodelled-kind regression is retargeted
    # at COMMS_LINK_STATE — a self-state kind Sim genuinely carries no model for (comms reachability
    # arrives as the RM-P0-SIM-08 observation mask, not as a sensor reading). The invariant under
    # test is unchanged: a declared-but-unmodelled sensor degrades *loudly*, never silently.
    sensor = Sensor(name="link", kind=SensorKind.COMMS_LINK_STATE, frame="body")
    reading = render_sensor(sensor, _state(), None, Random(0))
    assert reading.valid is False and reading.values == []


def test_noise_is_applied_to_proprioceptive_readings() -> None:
    sensor = Sensor(
        name="imu",
        kind=SensorKind.IMU,
        frame="body",
        observation_model=ObservationModel(noise_sigma=0.5),
    )
    reading = render_sensor(sensor, _state(velocity=(1.0, 0.0, 0.0)), None, Random(1))
    assert reading.values[0] != 1.0 and reading.noise_sigma == 0.5  # noisy about the true velocity


# --- integration through the Simulator -------------------------------------------


def _scout_scenario() -> Scenario:
    sensor = _resource_sensor(noise_sigma=0.02)
    return Scenario(
        name="scout",
        agents=(
            AgentSpec(
                agent_id="scout",
                initial_position_m=(2.0, 0.0, 0.0),
                velocity_mps=(0.5, 0.0, 0.0),
                battery_soc_j=200.0,
                dynamics=MobilityDynamics(mass_kg=120.0, max_speed_mps=0.5, max_traction_n=400.0),
                sensors=(sensor,),
            ),
        ),
        seed=5,
        horizon_steps=3,
    )


def test_simulator_attaches_sensor_readings_to_the_observation() -> None:
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    sim = Simulator(_scout_scenario(), resource_field=field)
    observation = sim.reset().observations["scout"]
    # the agent-facing observation carries only SensorReadings — the field never crosses it.
    assert observation.sensors and all(isinstance(r, SensorReading) for r in observation.sensors)
    assert observation.sensors[0].valid and observation.sensors[0].resource_species == _SPECIES


def test_resource_reading_tracks_the_moving_agent_over_the_field() -> None:
    # A noiseless sensor so the reading is exactly the truth at the agent's live position;
    # stepping the rover away from the bump centre must strictly drop the sampled ice fraction.
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    scenario = Scenario(
        name="track",
        agents=(
            AgentSpec(
                agent_id="scout",
                initial_position_m=(2.0, 0.0, 0.0),
                velocity_mps=(0.5, 0.0, 0.0),
                battery_soc_j=200.0,
                dynamics=MobilityDynamics(mass_kg=120.0, max_speed_mps=0.5, max_traction_n=400.0),
                sensors=(_resource_sensor(noise_sigma=None),),
            ),
        ),
        horizon_steps=3,
    )
    sim = Simulator(scenario, resource_field=field)
    values = [sim.reset().observations["scout"].sensors[0].values[0]]
    for _ in range(3):
        values.append(sim.step(ActionBatch()).observations["scout"].sensors[0].values[0])
    assert values[-1] < values[0]


# --- RM-P1-SIM-05: richer (ensemble) field render + the plugin registry -----------


def _nir_sensor(*, noise_sigma: float | None = 0.02) -> Sensor:
    model = None if noise_sigma is None else ObservationModel(noise_sigma=noise_sigma)
    return Sensor(
        name="nir",
        kind=SensorKind.NIR_SPECTROMETER,
        frame="body",
        observation_model=model,
        resource=ResourceTarget(species=_SPECIES, si_unit=_UNIT),
    )


def test_higher_fidelity_sensor_renders_the_field_as_an_ensemble_with_uncertainty() -> None:
    # A higher-fidelity resource sensor renders the measurement *distribution* (an ensemble of
    # noisy looks), never a single point — the uncertainty is explicit in the spread.
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    reading = render_sensor(_nir_sensor(), _state(position=(1.0, 0.0, 0.0)), field, Random(1))
    assert reading.valid and len(reading.values) == DEFAULT_ENSEMBLE_LOOKS
    assert len(set(reading.values)) > 1  # a genuine distribution, not a repeated point
    assert reading.noise_sigma == 0.02 and reading.resource_species == _SPECIES


def test_ensemble_sensor_is_deterministic_and_degrades_without_a_field() -> None:
    field = ReferenceResourceField()
    a = render_sensor(_nir_sensor(), _state(), field, Random(7))
    b = render_sensor(_nir_sensor(), _state(), field, Random(7))
    assert a.values == b.values  # same seed reproduces the ensemble
    degraded = render_sensor(_nir_sensor(), _state(), None, Random(7))
    assert degraded.valid is False and degraded.resource_species == _SPECIES


def test_sensor_models_are_registry_discovered_plugins() -> None:
    # Registering a model for a previously-unmodelled kind routes it with no change to
    # render_sensor — the "adding a sensor is a plugin, routing it is config" property. Uses
    # COMMS_LINK_STATE (still unmodelled) now that IMAGING carries a real model.
    link = Sensor(name="link", kind=SensorKind.COMMS_LINK_STATE, frame="body")
    assert render_sensor(link, _state(), None, Random(0)).valid is False  # unmodelled today

    @register_self_state_model(SensorKind.COMMS_LINK_STATE)
    def _fake_link_state(ctx: SensorContext) -> SensorReading:
        return SensorReading(sensor=ctx.sensor.name, values=[42.0], unit="dB")

    try:
        reading = render_sensor(link, _state(), None, Random(0))
        assert reading.valid and reading.values == [42.0] and reading.unit == "dB"
    finally:
        from astro_mine.sim.sensors import _SELF_STATE_MODELS

        del _SELF_STATE_MODELS[SensorKind.COMMS_LINK_STATE]


# --- RM-P1-SIM-02: the ISRU stored-mass gauge sensor ------------------------------


def _storage_sensor() -> Sensor:
    return Sensor(name="tank", kind=SensorKind.RESOURCE_STORAGE, frame="body")


def test_resource_storage_gauge_reports_stored_water_and_energy() -> None:
    reading = render_sensor(
        _storage_sensor(), _state(), None, Random(0), isru=IsruState(12.5, 25_000.0)
    )
    assert reading.valid and reading.unit == "kg" and reading.resource_species == "water"
    assert reading.values == [12.5, 25_000.0]  # values[0]=stored kg, values[1]=extraction J


def test_resource_storage_gauge_degrades_without_isru_state() -> None:
    reading = render_sensor(_storage_sensor(), _state(), None, Random(0), isru=None)
    assert reading.valid is False and reading.unit == "kg" and reading.resource_species == "water"


def test_sensor_rendering_is_deterministic_under_a_fixed_seed() -> None:
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    scenario = _scout_scenario()
    assert run_episode(scenario, resource_field=field).content_hash == (
        run_episode(scenario, resource_field=field).content_hash
    )


def test_a_sensorless_scenario_reports_no_readings() -> None:
    scenario = Scenario(name="bare", agents=(AgentSpec(agent_id="a", battery_soc_j=10.0),))
    assert Simulator(scenario).reset().observations["a"].sensors == []


# --- RM-P0-SIM-06: the imaging sensor model ---------------------------------------
# A *framing* sensor: it observes ground the agent is not standing on, so it is gated by frame
# geometry (FOV/pose/range) and by illumination — a passive camera cannot expose a PSR.


class _ConstantWorld:
    """A Core WorldProvider stand-in with a fixed illumination + a sloped datum, so a frame's
    footprint samples distinct elevations (the geometric feature vector) rather than one value."""

    def __init__(self, state: IlluminationState, flux: float) -> None:
        self._illumination = Illumination(state=state, solar_flux_w_m2=flux)

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def sample(self, position: Position, *, epoch: Epoch | None = None) -> SurfacePoint:
        return SurfacePoint(
            frame=MOON_BODY_FIXED,
            elevation_m=0.1 * position[0],  # a gentle slope, so footprint samples differ
            surface_normal=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -1.62),
            illumination=self._illumination,
            temperature_k=120.0,
            regolith=RegolithParams(bulk_density_kg_m3=1500.0),
        )


def _imaging_sensor(*, resource: bool = False, fov_deg: float = 60.0) -> Sensor:
    return Sensor(
        name="cam",
        kind=SensorKind.IMAGING,
        frame="body",
        observation_model=ObservationModel(noise_sigma=0.01, fov_deg=fov_deg, range_m=10.0),
        resource=ResourceTarget(species=_SPECIES, si_unit=_UNIT) if resource else None,
    )


def test_imaging_is_modelled_and_renders_a_feature_vector_of_the_terrain() -> None:
    lit = _ConstantWorld(IlluminationState.LIT, 1361.0)
    reading = render_sensor(
        _imaging_sensor(), _state(), None, Random(0), world=lit, epoch=J2000_EPOCH
    )
    # No longer the unmodelled `valid=False` case (the RM-P0-SIM-06 gap this closes).
    assert reading.valid is True
    # A frame, not a point: one value per footprint sample.
    assert len(reading.values) == DEFAULT_IMAGING_FEATURES
    assert len(set(reading.values)) > 1  # the footprint spans the slope — a real feature vector
    assert reading.unit == "m"


def test_imaging_renders_an_observation_of_the_sealed_resource_field_not_a_point_guess() -> None:
    lit = _ConstantWorld(IlluminationState.LIT, 1361.0)
    field = ReferenceResourceField(peak=0.1, length_scale_m=5.0)
    reading = render_sensor(
        _imaging_sensor(resource=True), _state(), field, Random(3), world=lit, epoch=J2000_EPOCH
    )
    assert reading.valid and reading.resource_species == _SPECIES and reading.unit == _UNIT
    assert len(reading.values) == DEFAULT_IMAGING_FEATURES
    # Each footprint sample is a *noisy draw of the sealed truth at that ground point* — so the
    # frame carries the field's spatial structure, never one ground-truth guess (prospect.md §6).
    truths = [field.mean(p) for p in imaging_footprint(_ctx_for(_imaging_sensor(resource=True)))]
    assert len(set(truths)) > 1  # the footprint really does span varying ground truth
    assert reading.values != truths  # and the reading is noisy about it, not the truth itself


def _ctx_for(sensor: Sensor) -> SensorContext:
    return SensorContext(sensor=sensor, state=_state(), field=None, isru=None, rng=Random(0))


def test_imaging_into_a_permanently_shadowed_region_is_invalid_not_fabricated() -> None:
    # The anchor scenario's whole point: a passive camera framed into a PSR has no exposure.
    psr = _ConstantWorld(IlluminationState.SHADOW, 0.0)
    reading = render_sensor(
        _imaging_sensor(resource=True),
        _state(),
        ReferenceResourceField(),
        Random(0),
        world=psr,
        epoch=J2000_EPOCH,
    )
    assert reading.valid is False and reading.values == []
    assert reading.resource_species == _SPECIES  # it still says *what* it failed to observe


def test_imaging_in_penumbra_is_degraded_not_invalid() -> None:
    penumbra = _ConstantWorld(IlluminationState.PENUMBRA, 100.0)
    lit = _ConstantWorld(IlluminationState.LIT, 1361.0)
    dim = render_sensor(
        _imaging_sensor(), _state(), None, Random(0), world=penumbra, epoch=J2000_EPOCH
    )
    bright = render_sensor(
        _imaging_sensor(), _state(), None, Random(0), world=lit, epoch=J2000_EPOCH
    )
    assert dim.valid and bright.valid
    assert dim.noise_sigma is not None and bright.noise_sigma is not None
    # Degraded, not invalid: the partially-lit frame reports a larger measurement sigma, so a
    # consumer (Prospect's belief update) down-weights it rather than trusting it equally.
    assert dim.noise_sigma > bright.noise_sigma


def test_imaging_without_a_world_provider_degrades_rather_than_inventing_an_exposure() -> None:
    reading = render_sensor(_imaging_sensor(), _state(), None, Random(0))
    assert reading.valid is False  # no illumination model wired ⇒ no fabricated frame


def test_imaging_footprint_follows_the_declared_fov_and_sensor_pose() -> None:
    # A wider FOV casts a wider footprint at the same slant range (real frame geometry).
    narrow = imaging_footprint(_ctx_for(_imaging_sensor(fov_deg=20.0)))
    wide = imaging_footprint(_ctx_for(_imaging_sensor(fov_deg=90.0)))
    spread = lambda pts: max(abs(p[0] - pts[0][0]) for p in pts)  # noqa: E731
    assert spread(wide) > spread(narrow)
    # Nadir boresight (no declared pose): the footprint centre is directly below the agent.
    assert narrow[0][0] == pytest.approx(0.0) and narrow[0][2] == pytest.approx(-10.0)
    # A pose that pitches the camera 90° about +y swings the boresight to +x (horizon-looking).
    pitched = Sensor(
        name="cam",
        kind=SensorKind.IMAGING,
        frame="body",
        observation_model=ObservationModel(fov_deg=20.0, range_m=10.0),
        pose=SadfTransform(
            translation_m=SadfVec3(x=0.0, y=0.0, z=0.0),
            # -90° about +y maps -z -> +x.
            rotation_quat_xyzw=SadfQuat(
                x=0.0, y=-math.sin(math.pi / 4), z=0.0, w=math.cos(math.pi / 4)
            ),
        ),
    )
    centre = imaging_footprint(_ctx_for(pitched))[0]
    assert centre[0] == pytest.approx(10.0, abs=1e-9) and centre[2] == pytest.approx(0.0, abs=1e-9)


def test_imaging_is_deterministic_under_a_fixed_seed() -> None:
    lit = _ConstantWorld(IlluminationState.LIT, 1361.0)
    field = ReferenceResourceField()
    a = render_sensor(
        _imaging_sensor(resource=True), _state(), field, Random(11), world=lit, epoch=J2000_EPOCH
    )
    b = render_sensor(
        _imaging_sensor(resource=True), _state(), field, Random(11), world=lit, epoch=J2000_EPOCH
    )
    assert a.values == b.values
