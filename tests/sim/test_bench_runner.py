"""RM-P0-SIM-11 / RM-P0-BENCH-04,05 — the Sim-backed Bench runners, on real physics.

The gap these close: Bench's determinism gate ran on "a pure, seeded function of the scenario hash
...
enough to exercise the reproducibility oracle *without a physics engine*", and its baseline scoring
path on "a deterministic stand-in ... *not a physics engine*". Both now have a real runner behind
them.

Everything here is exercised end to end against a **real content store**: Worlds/Fleet/Prospect
bundles are published to a local OCI-layout Hub registry (signed), the scenario pins them by content
hash, and the Sim runner resolves them through the RM-P1-SIM-01 ``ContentResolver`` — no
hand-authored
fixture anywhere in the path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from astro_mine.bench.baseline import (
    BaselinePolicy,
    ScoringRefused,
    assert_score_reproducible,
    run,
)
from astro_mine.bench.baseline import run as bench_run
from astro_mine.bench.baseline._runner import EpisodeRunner, reference_episode_runner
from astro_mine.bench.harness import Runner, assert_reproducible, lockfile_digest, reproduce
from astro_mine.bench.metrics import EpisodeTrace, ScoringContext
from astro_mine.bench.scenario import (
    ContentPins,
    ContentRef,
    EpisodeSpec,
    LatLonRegion,
    MetricRef,
    PlacementSpec,
    ResolvedScenario,
    ScenarioSpec,
    ScoringSpec,
    SeedSet,
    SitePlacement,
    resolve_scenario,
)
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import (
    ContactInterval,
    ContactNode,
    ContactPlan,
    SensorReading,
)
from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.resource import FieldDistribution
from astro_mine.core.sadf.enums import ContactElementKind, Regime, SensorKind
from astro_mine.core.sadf.model import (
    Actuator,
    Asset,
    Body,
    ContactElement,
    Identity,
    Inertia,
    Isru,
    Mobility,
    ObservationModel,
    PayloadSpec,
    PowerBudget,
    PowerStorage,
    Range,
    ResourceTarget,
    SadfDocument,
    Sensor,
    ThermalBudget,
    Vec3,
)
from astro_mine.core.units import J2000_EPOCH, MOON_BODY_FIXED, Epoch, ReferenceFrame, TimeScale
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
)
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.sim.bench import (
    SIM_RUNNER_ID,
    SimEpisodeRunner,
    SimHarnessRunner,
    dynamics_for_asset,
    sim_runner_provider,
    sim_scenario_from_spec,
)
from astro_mine.sim.bench._belief import prospecting_sensors
from astro_mine.sim.bench._scenario import _SURFACE_LAYOUT_RADIUS_M
from astro_mine.sim.comms import ReferenceConnectivitySampler
from astro_mine.sim.engines import kinematic_engine_factory
from astro_mine.sim.isru import DEFAULT_EXTRACTION_MODES
from astro_mine.sim.recording import read_recording
from astro_mine.sim.runtime import Simulator, run_episode
from astro_mine.sim.runtime._hub_adapter import HubBundleStore
from astro_mine.sim.runtime.content import (
    ContentPin,
    ContentResolver,
    ResolvedAsset,
    ScenarioContent,
)
from astro_mine.sim.runtime.provenance import engines_that_ran
from astro_mine.sim.runtime.scenario import MOON_RADIUS_M
from astro_mine.sim.sensors import ReferenceResourceField

_SADF_JSON = "application/vnd.astro-mine.sadf+json"
_WORLD_TAR = "application/vnd.astro-mine.world.bundle.v1.tar"
_FIELD_TAR = "application/vnd.astro-mine.resource-field.bundle.v1.tar"
_LINK_TAR = "application/vnd.astro-mine.contact-plan.bundle.v1.tar"
_WATER = "water"
_HYDROGEN = "water_equivalent_hydrogen"

#: A short diurnal period (s), so a test-length episode actually spans a night — the survival metric
#: needs a real dark window, and the world is what supplies it.
_PERIOD_S = 600.0


# --- the content the scenario pins (published to a real local Hub registry) -------


class _AnchorWorld:
    """A Core ``WorldProvider`` for the pinned world: regolith terramechanics, lunar gravity, and a
    diurnal illumination cycle (so a night really happens inside the episode)."""

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def sample(
        self, position: tuple[float, float, float], *, epoch: Epoch | None = None
    ) -> SurfacePoint:
        phase = 0.0 if epoch is None else (epoch.tdb_seconds % _PERIOD_S) / _PERIOD_S
        lit = phase < 0.5
        return SurfacePoint(
            frame=MOON_BODY_FIXED,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -1.62),
            illumination=Illumination(
                state=IlluminationState.LIT if lit else IlluminationState.SHADOW,
                solar_flux_w_m2=1361.0 * math.sin(math.pi * phase * 2.0) if lit else 0.0,
            ),
            temperature_k=350.0 if lit else 100.0,
            regolith=RegolithParams(
                bulk_density_kg_m3=1600.0,
                friction_angle_deg=35.0,
                bearing_capacity_pa=5.0e4,
            ),
        )


def _world_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> _AnchorWorld:
    return _AnchorWorld()


def _field_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> ReferenceResourceField:
    # The bump is centred on the SITE — the body-fixed south-polar surface — not on the origin.
    # `ReferenceResourceField` declares MOON_BODY_FIXED, so its `center_m` default of (0, 0, 0) is
    # the Moon's *centre*, ~1737 km underground. It only ever looked right because the agents were
    # mistakenly down there too (the placement bug this fixes): a Gaussian at the body centre and
    # rovers 25 m from the body centre happened to overlap. With the rovers correctly on the
    # surface, a field at the body centre reads zero everywhere — an honest consequence, not a
    # regression.
    return ReferenceResourceField(
        species=_HYDROGEN,
        peak=0.12,
        length_scale_m=30.0,
        center_m=(0.0, 0.0, -MOON_RADIUS_M),
    )


def _comms_factory(
    manifest: PluginManifest, layers: Mapping[str, bytes]
) -> ReferenceConnectivitySampler:
    """The ``comms_model`` provider factory, standing in for Link's (#53).

    Sim never imports Link — a producer self-registers its factory on the ``astro_mine.providers``
    entry-point group and Sim hands it the pinned bundle's layers verbatim (conventions.md §1.1).
    Decoding them is the *producer's* business: Link's real factory reads a ``contact_plan.pb`` out
    of a tar, which Sim cannot construct without importing Link. So this stub reads the plan
    straight off the layer. What is under test here is Sim's half — that the pinned link bundle is
    resolved and bound at all — not Link's tar format, which Link's own contract suite covers."""
    return ReferenceConnectivitySampler(ContactPlan.model_validate_json(layers[_LINK_TAR]))


def _excavator_document() -> SadfDocument:
    """An excavator: a digging **tool** is what makes it one, and the only asset in the fixture that
    routes to the granular ladder (reduced-order, or the DEM/surrogate tiers — see
    ``test_bench_speedup.py``)."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.excavator", name="Excavator", version="0.1.0", kind="excavator"
        ),
        root_frame="body",
        bodies=[_body(480.0)],
        actuators=[Actuator(name="arm", velocity=0.3, torque_nm=900.0)],
        mobility=Mobility(
            regimes=[Regime.SURFACE],
            contact=[
                ContactElement(
                    kind=ContactElementKind.WHEEL, dimensions_m=Vec3(x=0.12, y=0.12, z=0.30)
                ),
                ContactElement(
                    kind=ContactElementKind.TOOL, dimensions_m=Vec3(x=0.8, y=0.1, z=0.35)
                ),
            ],
        ),
        power=_power(6.0e6),
        thermal=_thermal(),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


def _excavator_asset(content: dict[str, Any]) -> ResolvedAsset:
    """The excavator as the resolver hands it back — asset + the digest it was pinned by."""
    return ResolvedAsset(
        asset=_excavator_document().asset, content_hash=content["excavator_digest"]
    )


def _power(capacity_j: float) -> PowerBudget:
    return PowerBudget(storage=[PowerStorage(name="bat", capacity_j=capacity_j)], floor_w=5.0)


def _thermal() -> ThermalBudget:
    return ThermalBudget(
        operating_range_k=Range(min=250.0, max=320.0),
        survival_range_k=Range(min=100.0, max=350.0),
    )


def _inertia() -> Inertia:
    return Inertia(ixx=1.0, iyy=1.0, izz=1.0)


def _body(mass_kg: float) -> Body:
    return Body(
        name="chassis",
        frame="body",
        mass_kg=mass_kg,
        center_of_mass_m=Vec3(x=0.0, y=0.0, z=0.0),
        inertia_kg_m2=_inertia(),
    )


def _rover_asset() -> SadfDocument:
    """A prospecting rover: wheels (so it drives), a neutron spectrometer (so it discovers)."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.prospecting-rover", name="Scout", version="0.1.0", kind="rover"
        ),
        root_frame="body",
        bodies=[_body(210.0)],
        actuators=[Actuator(name="drive", velocity=0.6, torque_nm=45.0)],
        mobility=Mobility(
            regimes=[Regime.SURFACE],
            contact=[
                ContactElement(
                    kind=ContactElementKind.WHEEL, dimensions_m=Vec3(x=0.1, y=0.1, z=0.28)
                )
            ],
        ),
        sensors=[
            Sensor(
                name="neutron",
                kind=SensorKind.NEUTRON_SPECTROMETER,
                frame="body",
                observation_model=ObservationModel(noise_sigma=0.005),
                resource=ResourceTarget(species=_HYDROGEN, si_unit="mass_fraction"),
            )
        ],
        power=_power(4.0e6),
        thermal=_thermal(),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


def _plant_asset() -> SadfDocument:
    """An ISRU plant: a payload with a declared throughput, and a storage gauge that reports it."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.isru-plant", name="Plant", version="0.1.0", kind="isru"
        ),
        root_frame="body",
        bodies=[_body(800.0)],
        payload=PayloadSpec(
            isru=Isru(throughput_kg_hr=3.6, plant_power_w=400.0), capacity_kg=250.0
        ),
        sensors=[
            Sensor(
                name="tank",
                kind=SensorKind.RESOURCE_STORAGE,
                frame="body",
                resource=ResourceTarget(species=_WATER, si_unit="kg"),
            )
        ],
        power=_power(9.0e6),
        thermal=_thermal(),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


def _relay_asset() -> SadfDocument:
    """A relay orbiter: it declares the PROXIMITY_ORBIT regime, so it propagates orbitally."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.relay-orbiter", name="Relay", version="0.1.0", kind="orbiter"
        ),
        root_frame="body",
        bodies=[_body(600.0)],
        mobility=Mobility(regimes=[Regime.PROXIMITY_ORBIT]),
        power=_power(8.0e6),
        thermal=_thermal(),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


def _publish(
    client: HubClient,
    key: bytes,
    *,
    name: str,
    artifact_kind: str,
    manifest_kind: PluginKind,
    core_interface: str,
    layers: list[Blob],
) -> str:
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        kind=manifest_kind,
        core_interfaces={core_interface: "0.1.0"},
        license="Apache-2.0",
    )
    artifact = client.publish(
        name=name,
        version="0.1.0",
        kind=artifact_kind,
        manifest=manifest,
        layers=layers,
        private_key_pem=key,
    )
    return str(artifact.digest)


#: The repo's own uv.lock — the lockfile that actually pins a *Sim-backed* run's toolchain.
_SIM_LOCKFILE = Path(__file__).resolve().parent.parent.parent / "uv.lock"


def publish_content(tmp_path: Path) -> dict[str, Any]:
    """Publish the world / fleet / prospect / link bundles to a real local Hub registry and return
    the store plus each pin's digest — exactly the content-addressed path a Bench scenario pins.

    A plain factory, not the fixture itself: a sibling test module needs the same published content
    (``test_bench_speedup.py``), and *importing* a fixture would shadow it at every use site."""
    private_key, public_key = generate_keypair()
    client = HubClient(Registry(tmp_path / "registry"), trusted_public_key_pem=public_key)
    store = HubBundleStore(client)

    digests: dict[str, str] = {}
    for document in (_relay_asset(), _rover_asset(), _plant_asset()):
        name = document.asset.identity.id
        digests[name] = _publish(
            client,
            private_key,
            name=name,
            artifact_kind="asset",
            manifest_kind=PluginKind.ASSET,
            core_interface="sadf",
            layers=[Blob(_SADF_JSON, document.model_dump_json().encode("utf-8"))],
        )
    digests["shackleton-de-gerlache-v1"] = _publish(
        client,
        private_key,
        name="shackleton-de-gerlache-v1",
        artifact_kind="world",
        manifest_kind=PluginKind.WORLD_PROVIDER,
        core_interface="world_provider",
        layers=[Blob(_WORLD_TAR, b"world-bundle-tar-bytes")],
    )
    digests["shackleton_water_ice_v1"] = _publish(
        client,
        private_key,
        name="shackleton_water_ice_v1",
        artifact_kind="plugin",
        manifest_kind=PluginKind.RESOURCE_FIELD_BACKEND,
        core_interface="resource_field",
        layers=[Blob(_FIELD_TAR, b"field-bundle-tar-bytes")],
    )
    # The pinned comms model (#53) — a real contact-plan artifact in the registry, resolved through
    # the same content path as the world and the field. Kept *out* of `digests`, which is the set of
    # pins `_spec()` carries: only `_spec_with_link()` pins the link, and the provenance tests below
    # assert that every pin in `digests` is byte-addressed in the trace.
    link_digest = _publish(
        client,
        private_key,
        name="astro-mine.link.lunar-polar-relay-dsn",
        artifact_kind="plugin",
        manifest_kind=PluginKind.COMMS_MODEL,
        core_interface="messages",
        layers=[Blob(_LINK_TAR, _contact_plan().model_dump_json().encode("utf-8"))],
    )
    # The excavator (#51) — likewise published but not pinned by `_spec()`: it is the only asset
    # that routes to the granular ladder, and only `test_bench_speedup.py` pins it.
    excavator = _excavator_document()
    excavator_digest = _publish(
        client,
        private_key,
        name=excavator.asset.identity.id,
        artifact_kind="asset",
        manifest_kind=PluginKind.ASSET,
        core_interface="sadf",
        layers=[Blob(_SADF_JSON, excavator.model_dump_json().encode("utf-8"))],
    )
    return {
        "store": store,
        "digests": digests,
        "link_digest": link_digest,
        "excavator_digest": excavator_digest,
        "factories": {
            PluginKind.WORLD_PROVIDER.value: _world_factory,
            PluginKind.RESOURCE_FIELD_BACKEND.value: _field_factory,
            PluginKind.COMMS_MODEL.value: _comms_factory,
        },
    }


@pytest.fixture
def content(tmp_path: Path) -> dict[str, Any]:
    return publish_content(tmp_path)


def _spec(content: dict[str, Any]) -> ScenarioSpec:
    """A Bench ScenarioSpec pinning that published content by hash — the anchor's shape, in
    miniature."""
    d = content["digests"]
    return ScenarioSpec(
        scenario_id="lunar-polar-ice-prospecting-test",
        name="Lunar Polar Water-Ice Prospecting (test)",
        core_interface={"env": "0.1.0", "messages": "0.1.0"},
        content=ContentPins(
            world=ContentRef(
                id="shackleton-de-gerlache-v1", content_hash=d["shackleton-de-gerlache-v1"]
            ),
            fleet=(
                ContentRef(
                    id="astro-mine.fleet.relay-orbiter",
                    content_hash=d["astro-mine.fleet.relay-orbiter"],
                ),
                ContentRef(
                    id="astro-mine.fleet.prospecting-rover",
                    content_hash=d["astro-mine.fleet.prospecting-rover"],
                ),
                ContentRef(
                    id="astro-mine.fleet.isru-plant", content_hash=d["astro-mine.fleet.isru-plant"]
                ),
            ),
            prospect=(
                ContentRef(id="shackleton_water_ice_v1", content_hash=d["shackleton_water_ice_v1"]),
            ),
        ),
        seeds=SeedSet(public=(1001, 1002)),
        episode=EpisodeSpec(horizon_steps=20, max_sim_seconds=1200.0),
        metrics=(
            MetricRef(name="water_mass"),
            MetricRef(name="energy_per_kg"),
            MetricRef(name="nights_survived"),
            MetricRef(name="discovery_latency"),
            MetricRef(name="comms_robustness"),
        ),
    )


def _runner(content: dict[str, Any], tmp_path: Path, **kwargs: Any) -> SimEpisodeRunner:
    return SimEpisodeRunner(
        store=content["store"],
        provider_factories=content["factories"],
        recording_dir=tmp_path / "mcap",
        **kwargs,
    )


# --- the runner satisfies Bench's seams ------------------------------------------


def test_the_sim_runner_satisfies_benchs_episode_runner_protocol(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Acceptance criterion: "a Sim-provided EpisodeRunner implementation is importable and satisfies
    # Bench's EpisodeRunner protocol". Structural, not nominal — Bench never imports Sim.
    runner: EpisodeRunner = _runner(content, tmp_path)
    assert callable(runner)
    harness: Runner = SimHarnessRunner(_runner(content, tmp_path))
    assert callable(harness)
    # And it identifies itself, so a Result records that real physics produced it.
    assert runner.__name__ == SIM_RUNNER_ID  # type: ignore[union-attr]


def test_the_runner_resolves_content_pins_rather_than_hand_authored_fixtures(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Acceptance criterion: "the adapter resolves scenario content pins via the ContentResolver".
    runner = _runner(content, tmp_path)
    resolved = resolve_scenario(_spec(content))
    run_ = runner.resolve(resolved, seed=1001)

    # Every pinned bundle resolved, and its digest rides in the map that goes into provenance.
    assert set(run_.content_hashes) == set(content["digests"])
    for content_id, digest in content["digests"].items():
        assert run_.content_hashes[content_id] == digest
    # The world and field are the *reconstructed* providers, not stand-ins the runner invented.
    assert isinstance(run_.world_provider, _AnchorWorld)
    assert isinstance(run_.resource_field, ReferenceResourceField)
    # One agent per pinned fleet asset, materialized from the SADF.
    assert tuple(a.agent_id for a in run_.scenario.agents) == tuple(
        ref.id for ref in resolved.spec.content.fleet
    )


def test_each_assets_engine_is_inferred_from_its_own_sadf(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Routing an engine is *configuration* — and the configuration is the pinned asset itself.
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec(content)), seed=1001)
    kinds = {a.agent_id: a.dynamics.kind for a in run_.scenario.agents}
    assert kinds["astro-mine.fleet.relay-orbiter"] == "orbital"  # declares PROXIMITY_ORBIT
    assert kinds["astro-mine.fleet.prospecting-rover"] == "mobility"  # declares wheels
    assert kinds["astro-mine.fleet.isru-plant"] == "kinematic"  # neither: it sits still


def test_the_dynamics_parameters_come_from_the_resolved_content(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # RM-P0-SIM-03's last gap: the physics parameters are the *content's*, not the scenario's.
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec(content)), seed=1001)
    rover = next(a for a in run_.scenario.agents if a.agent_id.endswith("prospecting-rover"))
    assert rover.dynamics.kind == "mobility"
    assert rover.dynamics.mass_kg == pytest.approx(210.0)  # the SADF body mass
    assert rover.dynamics.max_speed_mps == pytest.approx(0.6)  # the SADF actuator limit
    # Traction is the friction cone of that mass on the *pinned world's* regolith: mu * m * g.
    expected = math.tan(math.radians(35.0)) * 210.0 * 1.62
    assert rover.dynamics.max_traction_n == pytest.approx(expected)
    # The ISRU plant's extraction rate is its SADF throughput (3.6 kg/h -> 1e-3 kg/s).
    plant = next(a for a in run_.scenario.agents if a.agent_id.endswith("isru-plant"))
    assert plant.isru is not None
    assert plant.isru.extraction_rate_kg_s == pytest.approx(1.0e-3)
    assert plant.isru.capacity_kg == pytest.approx(250.0)


# --- the scoring path, on real physics -------------------------------------------


def test_benchs_baseline_run_scores_a_real_physics_episode(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Acceptance criterion: "Bench's baseline run(spec, policy) produces a real MCAP episode trace
    # and
    # metric scores using the Sim-backed runner, not the synthetic stand-in".
    spec = _spec(content)
    runner = _runner(content, tmp_path)

    card = run(spec, BaselinePolicy(), runner=runner)

    assert card.scenario_id == spec.scenario_id
    assert {m.metric for m in card.metrics} == {m.name for m in spec.metrics}
    assert card.content_hash.startswith("sha256:")
    # A REAL MCAP was written per seed — the artifact boundary Bench and Sim meet at.
    assert set(runner.recordings) == set(spec.seeds.public)
    for path in runner.recordings.values():
        assert path.exists() and path.stat().st_size > 0

    # The recording round-trips through Sim's own reader, carrying the run provenance + content
    # hashes.
    from astro_mine.sim.recording import read_recording

    recording = read_recording(runner.recordings[1001])
    assert recording.content_hash.startswith("sha") or recording.content_hash
    source_hashes = recording.provenance["run"]["source_content_hashes"]
    for content_id, digest in content["digests"].items():
        assert source_hashes[content_id] == digest  # byte-addressed to the content it ran on


def _tank_readings(trace: EpisodeTrace) -> list[SensorReading]:
    return [
        reading
        for observation in trace.observations
        for reading in observation.sensors
        if reading.sensor == "tank" and reading.valid
    ]


def test_the_storage_gauge_reports_the_tank_and_not_the_ice_field(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The gauge is a *self-state* sensor, even though its SADF declares a resource target (#61).

    This test replaces one that asserted `stored[-1] > stored[0]` and passed for entirely the wrong
    reason. `render_sensor` used to route any sensor with a `resource` target to the field-sample
    model, so the plant's tank rendered a noisy draw of the **prospect ice field** tagged
    `unit="kg"` — and the two samples it compared "grew" only because the kinematic engine's
    +/-0.01 m jitter moved the plant across the field's gradient. Bench's `water_mass` filters on
    exactly `(species, "kg")`, so the anchor's headline metric scored a mass *fraction* as
    kilograms and tracked the terrain rather than anything the swarm did.

    The shape is the tell: the real gauge emits **two** channels, the field sample emits one.
    """
    runner = _runner(content, tmp_path)
    trace: EpisodeTrace = runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)

    tanks = _tank_readings(trace)
    assert tanks, "the ISRU plant's storage gauge produced no readings"
    # [stored_water_kg, extraction_energy_j]; a one-value reading means the field model ran.
    assert all(len(r.values) == 2 for r in tanks), "the gauge was bypassed by the field sampler"
    assert trace.context.water_species == _WATER  # read off the SADF, not guessed
    assert all(r.unit == "kg" for r in tanks)  # the declared si_unit, not a literal


def test_an_idle_plant_stores_nothing_and_says_so(content: dict[str, Any], tmp_path: Path) -> None:
    """Zero is the honest reading for a swarm that never extracted (#61).

    `BaselinePolicy`'s "prospect" is not in `DEFAULT_EXTRACTION_MODES`, so nothing extracts — and
    the tank must report that as `0.0`, not as whatever the regolith under the plant happens to
    assay at.
    """
    runner = _runner(content, tmp_path)
    trace: EpisodeTrace = runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)

    stored = [r.values[0] for r in _tank_readings(trace)]
    assert stored and all(value == 0.0 for value in stored)


def test_an_extraction_mode_alone_no_longer_manufactures_water(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """A one-word mode change must not produce water (#64).

    This test used to assert the opposite — that driving the plant into `extract` accumulated
    water — and it was right to, because `IsruModel` was gated on the mode string alone. That was
    also the reason `CapabilityModePolicy` had to *refuse* to command an extraction mode: the
    policy was working around a physics gap.

    The gap is closed, so the workaround is gone and this asserts the physics instead. The plant
    is commanded into `extract` for the whole run, and stores nothing, because nothing dug or
    hauled any regolith to it.
    """
    runner = _runner(content, tmp_path)
    trace: EpisodeTrace = runner(
        resolve_scenario(_spec(content)), BaselinePolicy(mode="extract"), 1001
    )

    stored = [r.values[0] for r in _tank_readings(trace)]
    energy = [r.values[1] for r in _tank_readings(trace)]
    assert set(stored) == {0.0}, "an extraction mode alone produced water"
    assert set(energy) == {0.0}, "energy was spent extracting nothing"
    # The gauge still reports both channels — the path is live, it simply has nothing to report.
    assert all(len(r.values) == 2 for r in _tank_readings(trace))


def test_the_scoring_context_is_derived_from_the_pinned_world_and_fleet(
    content: dict[str, Any], tmp_path: Path
) -> None:
    runner = _runner(content, tmp_path)
    trace = runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)
    context = trace.context

    # Night windows are MEASURED against the pinned world's illumination, not assumed.
    assert context.night_intervals, "the pinned world's night was not measured"
    assert all(end > start for start, end in context.night_intervals)
    # The survival floor is the assets' SADF survival range (100 K), not the operating range (250
    # K).
    assert context.survivable_temperature_k == pytest.approx(100.0)
    # Species come off the SADF sensor declarations.
    assert context.water_species == _WATER
    assert context.discovery_species == _HYDROGEN


def test_sim_will_not_fabricate_a_belief_and_says_so(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The honest boundary: a belief is Prospect's, and Sim renders observations *of* a sealed field
    # rather than maintaining a posterior over it (sim.md §5). So the belief fields stay empty
    # unless
    # a caller injects them — Sim does not invent them to make a metric score.
    runner = _runner(content, tmp_path)
    trace = runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)
    assert trace.context.prior_belief == {}
    assert trace.context.belief_history == ()
    assert trace.context.psr_cells == frozenset()

    # ... and the injection seam works, for a caller that has a real belief to supply.
    belief = ScoringContext(
        prior_belief={"c0": FieldDistribution(mean=0.3, variance=1.0, species=_WATER, unit="kg")},
        psr_cells=frozenset({"c0"}),
        water_species=_WATER,
    )
    injected = _runner(content, tmp_path, scoring_context=belief)
    context = injected(resolve_scenario(_spec(content)), BaselinePolicy(), 1001).context
    # The injected belief arrives...
    assert context.prior_belief == belief.prior_belief
    assert context.psr_cells == frozenset({"c0"})
    # ...and it is *overlaid*, not substituted. The seam used to replace the derivation outright,
    # so a caller injecting only a belief silently discarded the night windows measured against the
    # pinned world and the survival floor read off the fleet's thermal budgets — and
    # `nights_survived` stopped scoring for reasons no one had asked for.
    assert context.night_intervals == trace.context.night_intervals
    assert context.night_intervals != ()
    assert context.survivable_temperature_k == trace.context.survivable_temperature_k


# --- the determinism gate, on real physics ---------------------------------------


def test_benchs_determinism_gate_passes_with_the_sim_runner_injected(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Acceptance criterion: "Bench's determinism gate passes with the Sim-backed runner injected, on
    # the anchor scenario, reproducibly across repeated runs (same seed => same result)".
    spec = _spec(content)
    gate = SimHarnessRunner(_runner(content, tmp_path))

    result = assert_reproducible(spec, gate, runner_id=SIM_RUNNER_ID)

    assert result.runner == SIM_RUNNER_ID  # the Result records that REAL PHYSICS produced it
    assert result.scenario_id == spec.scenario_id
    assert {s.seed for s in result.per_seed} == set(spec.seeds.public)
    report = reproduce(spec, gate, runner_id=SIM_RUNNER_ID, runs=3)
    assert report.reproducible and len(set(report.result_hashes)) == 1


def test_the_gate_pins_sims_own_lockfile(content: dict[str, Any], tmp_path: Path) -> None:
    """The Result pins *Sim's* environment — the dependency set that produced the physics.

    Bench is installed from site-packages here, never a source checkout. It used to derive the
    lockfile from its own ``__file__`` and raise FileNotFoundError from exactly this position, so
    Sim patched a Bench module global to get through the gate at all (bench#37). Bench now resolves
    the nearest ``uv.lock`` at or above the working directory, so an unpatched, installed Bench
    pins the lockfile that actually governs a Sim-backed run's reproducibility.
    """
    spec = _spec(content)
    gate = SimHarnessRunner(_runner(content, tmp_path))

    result = assert_reproducible(spec, gate, runner_id=SIM_RUNNER_ID)

    assert result.environment_lockfile == lockfile_digest(_SIM_LOCKFILE)


def test_the_gates_determinism_key_is_sims_own_trace_hash(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # One artifact, not two: Bench's reproducibility oracle compares the very hash Sim's own
    # RM-P0-SIM-10 gate compares, so the two gates cannot disagree.
    from astro_mine.sim.runtime import run_episode

    runner = _runner(content, tmp_path)
    gate = SimHarnessRunner(runner)
    resolved = resolve_scenario(_spec(content))

    outcome = gate(resolved, 1001)
    run_ = runner.resolve(resolved, seed=1001)
    trace = run_episode(
        run_.scenario,
        seed=1001,
        world_provider=run_.world_provider,
        resource_field=run_.resource_field,  # type: ignore[arg-type]
        content_hashes=run_.content_hashes,
    )
    assert outcome.determinism_key == trace.content_hash


def test_the_scoring_path_determinism_gate_passes_on_real_physics(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Bench's *scoring*-path gate (a content-addressed Scorecard, twice) — the other half of the
    # reproducibility story. Same inputs + same seeds => the identical scorecard hash.
    spec = _spec(content)
    card = assert_score_reproducible(
        spec, BaselinePolicy(), runner=_runner(content, tmp_path), runs=2
    )
    assert card.content_hash.startswith("sha256:")


def test_a_different_seed_really_produces_a_different_run(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # A guard that the gate has teeth: if every seed produced the same trace, "reproducible" would
    # be
    # vacuously true.
    gate = SimHarnessRunner(_runner(content, tmp_path))
    resolved = resolve_scenario(_spec(content))
    assert gate(resolved, 1001).determinism_key != gate(resolved, 1002).determinism_key


# --- the fixture is still the default (what changed, and what did not) ------------


def test_bench_still_defaults_to_the_reference_fixture_when_no_runner_is_injected(
    content: dict[str, Any],
) -> None:
    # Acceptance criterion: "documentation/tests make clear which Bench code paths still default to
    # the synthetic runner vs. the Sim-backed one." Injecting the runner is the CALLER's choice —
    # Bench's defaults are unchanged, which is what keeps its offline no-Sim tier working. So a
    # `run(spec, policy)` with no `runner=` still scores the fixture, by design.
    spec = _spec(content)
    card = bench_run(spec, BaselinePolicy())  # no runner= : the fixture
    assert card.scenario_id == spec.scenario_id
    # The fixture needs no content store at all — that is the property being preserved.
    assert reference_episode_runner is not None


def test_the_two_runners_disagree_which_is_the_point(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The fixture is "a deterministic stand-in ... not a physics engine". Real physics scores
    # differently — if it did not, wiring it in would have bought nothing.
    spec = _spec(content)
    fixture = run(spec, BaselinePolicy())
    physics = run(spec, BaselinePolicy(), runner=_runner(content, tmp_path))
    assert fixture.content_hash != physics.content_hash


def test_dynamics_for_asset_falls_back_to_the_kinematic_engine() -> None:
    # A bare asset (no mobility, no tool, no wheels) routes to the reference engine.
    asset = Asset(
        identity=Identity(id="bare", name="Bare", version="0.1.0", kind="probe"),
        root_frame="body",
    )
    resolved = ResolvedAsset(asset=asset, content_hash="sha256:" + "ab" * 32)
    assert dynamics_for_asset(resolved).kind == "kinematic"


def test_an_unknown_bench_export_raises_attribute_error() -> None:
    import astro_mine.sim.bench as bench_pkg

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = bench_pkg.nonexistent  # type: ignore[attr-defined]


def test_the_contact_tier_is_selectable_through_the_runner(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The fidelity dial: the same pinned rover can be routed to the articulated MuJoCo contact tier.
    pytest.importorskip("mujoco")
    runner = _runner(content, tmp_path, contact_tier=True)
    run_ = runner.resolve(resolve_scenario(_spec(content)), seed=1001)
    rover = next(a for a in run_.scenario.agents if a.agent_id.endswith("prospecting-rover"))
    assert rover.dynamics.kind == "mujoco_mobility"
    assert rover.dynamics.friction_angle_deg == pytest.approx(35.0)  # the pinned world's regolith


def test_the_horizon_can_be_capped_below_the_specs(content: dict[str, Any], tmp_path: Path) -> None:
    # The anchor's horizon is 43 200 ticks (a lunar month); a smoke run caps it.
    run_ = _runner(content, tmp_path, horizon_steps=4).resolve(
        resolve_scenario(_spec(content)), seed=1001
    )
    assert run_.scenario.horizon_steps == 4
    assert run_.scenario.dt_s == pytest.approx(
        60.0
    )  # max_sim_seconds / horizon_steps, from the spec


def test_resolved_scenario_is_the_bench_type_we_expect(content: dict[str, Any]) -> None:
    assert isinstance(resolve_scenario(_spec(content)), ResolvedScenario)


# --- the comms seam (issue #52) --------------------------------------------------

#: The agents the fixture's fleet pins resolve to — the ids a ContactPlan must name to mask them
#: (``Simulator._comms_agents`` intersects the plan's nodes with the scenario's agents).
_FLEET = (
    "astro-mine.fleet.relay-orbiter",
    "astro-mine.fleet.prospecting-rover",
    "astro-mine.fleet.isru-plant",
)
#: The spec runs 20 ticks x 60 s from the J2000 TDB origin; open a ground window over the first
#: half so the episode straddles the terminator — some ticks have Earth contact, some do not.
_CONTACT_END_TDB_S = 600.0


def _contact_plan() -> ContactPlan:
    """A ContactPlan over the fixture's own agent ids: Earth contact for the first half of the
    episode, blackout for the second."""
    ground = "dsn-goldstone"
    return ContactPlan(
        nodes=[
            ContactNode(id=ground, role=NodeRole.GROUND),
            *(ContactNode(id=agent, role=NodeRole.SPACE) for agent in _FLEET),
        ],
        intervals=[
            ContactInterval(
                node_a=agent,
                node_b=ground,
                start_tdb_s=0.0,
                end_tdb_s=_CONTACT_END_TDB_S,
                max_rate_bps=2.0e6,
                min_latency_s=1.3,
            )
            for agent in _FLEET
        ],
    )


def test_a_run_without_connectivity_cannot_score_comms_robustness(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The regression #52 fixes: with no ConnectivitySource the runner is comms-blind, so the one
    metric that measures degrade-not-collapse has nothing to measure."""
    card = run(_spec(content), BaselinePolicy(), runner=_runner(content, tmp_path))

    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value is None  # not applicable — no observation carried a mask


def test_the_injected_connectivity_masks_the_scored_run(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """With a ContactPlan injected, Sim masks the observations Bench scores — so
    ``comms_robustness`` becomes a real number rather than *not applicable*."""
    runner = _runner(content, tmp_path, connectivity=ReferenceConnectivitySampler(_contact_plan()))

    card = run(_spec(content), BaselinePolicy(), runner=runner)

    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value is not None, "the injected ContactPlan did not reach the recorded run"
    # The window covers half the episode, so the run is neither always-connected nor always-dark —
    # a constant would pass a `is not None` check while proving the mask is not actually varying.
    assert 0.0 < comms.value < 1.0, f"expected partial Earth contact, got {comms.value}"


def test_masking_is_deterministic_and_leaves_unmasked_runs_untouched(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """Injecting connectivity changes the trace (it must — the observations now carry masks), and
    the masked run still reproduces byte-for-byte."""
    spec = _spec(content)
    plain = _runner(content, tmp_path / "plain")
    masked = _runner(
        content,
        tmp_path / "masked",
        connectivity=ReferenceConnectivitySampler(_contact_plan()),
    )

    plain_trace, _ = plain.run(resolve_scenario(spec), BaselinePolicy(), seed=1001)
    masked_trace, _ = masked.run(resolve_scenario(spec), BaselinePolicy(), seed=1001)
    assert plain_trace.content_hash != masked_trace.content_hash  # the mask is really in the trace

    replay = _runner(
        content,
        tmp_path / "replay",
        connectivity=ReferenceConnectivitySampler(_contact_plan()),
    )
    replay_trace, _ = replay.run(resolve_scenario(spec), BaselinePolicy(), seed=1001)
    assert replay_trace.content_hash == masked_trace.content_hash


# --- the pinned link bundle resolves (issue #53) ----------------------------------


def _spec_with_link(content: dict[str, Any]) -> ScenarioSpec:
    """The same spec, now also pinning the link bundle — the anchor's shape.

    Kept separate from :func:`_spec` on purpose: pinning a comms model changes what a run *is*
    (its observations carry masks, so its content hash moves), and the un-pinned spec is what
    proves an unmasked run still behaves exactly as before."""
    spec = _spec(content)
    pins = spec.content
    return spec.model_copy(
        update={
            "content": pins.model_copy(
                update={
                    "link": ContentRef(
                        id="astro-mine.link.lunar-polar-relay-dsn",
                        content_hash=content["link_digest"],
                    )
                }
            )
        }
    )


def test_a_pinned_link_bundle_is_resolved_into_a_live_comms_model(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The #53 regression: the link pin resolved to a *digest* only, so the ContactPlan was pinned,
    hashed into provenance, and then never used — the anchor's comms model was dead weight."""
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec_with_link(content)), seed=1001)

    assert run_.connectivity is not None, "the pinned link bundle was not reconstructed"
    assert set(run_.connectivity.nodes) >= set(_FLEET)


def test_the_pinned_comms_model_masks_the_scored_run_with_nothing_injected(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The point of #53: a scenario that *pins* a link bundle scores comms_robustness on its own.

    Before this, only an explicitly injected ConnectivitySource (#52) could mask a run — the pinned
    plan named ``prospecting-rover`` while Sim's agents were ``astro-mine.fleet.prospecting-rover``,
    so the two id sets never intersected and the mask silently applied to nobody."""
    card = run(_spec_with_link(content), BaselinePolicy(), runner=_runner(content, tmp_path))

    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value is not None, "the pinned ContactPlan did not reach the recorded run"
    assert 0.0 < comms.value < 1.0, f"expected partial Earth contact, got {comms.value}"


def test_an_injected_comms_model_still_overrides_the_pinned_one(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """Resolution makes the pinned plan the *default*, not a mandate: an explicit source still wins,
    exactly as an injected world_provider / resource_field does."""
    blackout = ContactPlan(
        nodes=[ContactNode(id=agent, role=NodeRole.SPACE) for agent in _FLEET],
        intervals=[],  # no contact windows at all: every tick is dark
    )
    runner = _runner(content, tmp_path, connectivity=ReferenceConnectivitySampler(blackout))

    card = run(_spec_with_link(content), BaselinePolicy(), runner=runner)

    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value == 0.0, "the injected blackout plan did not override the pinned one"


def test_a_contact_plan_that_names_no_agent_fails_loudly(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The vocabulary check (#53). A plan whose nodes name none of the scenario's agents masks
    nothing — the run would look healthy and score *not applicable* forever. That is always an id
    error, never a legitimate scenario, so it raises instead of silently scoring nothing.

    These are exactly the node ids the *published* anchor plan carried before Link was fixed."""
    wrong_vocabulary = ContactPlan(
        nodes=[
            ContactNode(id="prospecting-rover", role=NodeRole.SPACE),  # not the SADF identity.id
            ContactNode(id="isru-plant", role=NodeRole.SPACE),
        ],
        intervals=[],
    )
    runner = _runner(content, tmp_path, connectivity=ReferenceConnectivitySampler(wrong_vocabulary))

    with pytest.raises(ValueError, match="name none of this scenario's agents"):
        runner.run(resolve_scenario(_spec(content)), BaselinePolicy(), seed=1001)


def test_a_contact_plan_whose_window_misses_the_episode_fails_loudly(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The temporal twin of the vocabulary check (astro-mine-bench#48).

    A ContactPlan is a plan **over a window of time**, so whether it applies depends on *when* the
    episode runs. Nothing checked that, and the real failure was ugly: Bench's `EpisodeSpec` carried
    no start epoch, so this seam left `Scenario` on its `J2000_EPOCH` default (TDB 0.0) while the
    published anchor plan covers 24 h at TDB 946_728_000 (2030-01-01). Thirty years apart. Every
    interval was inactive at every tick, `earth_contact` was false forever, and `comms_robustness`
    scored a confident **0.0**.

    Note the asymmetry with the node-id guard, and why both are needed: a plan whose *nodes* name no
    agent masks nothing, so the metric reads `not applicable` — which at least announces itself. A
    plan whose *window* misses the episode masks *everything*, so the metric reads a confident zero,
    which does not.

    The guard is on the **pinned** plan, so the spec pins the link bundle (the anchor's shape). The
    fixture plan's contacts run 0..600 s TDB; the episode is declared to start a year later.
    """
    spec = _spec_with_link(content)
    a_year_later = Epoch(tdb_seconds=365 * 86_400.0, scale=TimeScale.TDB)
    spec = spec.model_copy(
        update={"episode": spec.episode.model_copy(update={"start_epoch": a_year_later})}
    )
    runner = _runner(content, tmp_path)

    with pytest.raises(ValueError, match="never in contact at any point in this episode"):
        runner.run(resolve_scenario(spec), BaselinePolicy(), seed=1001)


def test_a_pinned_plan_whose_window_covers_the_declared_epoch_is_accepted(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The other side of the guard: an epoch the plan actually covers runs, and masks for real."""
    spec = _spec_with_link(content)
    inside = Epoch(tdb_seconds=60.0, scale=TimeScale.TDB)  # within the fixture plan's 0..600 s
    spec = spec.model_copy(
        update={"episode": spec.episode.model_copy(update={"start_epoch": inside})}
    )

    card = run(spec, BaselinePolicy(), runner=_runner(content, tmp_path))

    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value is not None, "the pinned plan did not reach the recorded run"
    # Strictly inside (0, 1) is the whole point. `not applicable` was the symptom before the plan's
    # nodes bound to the agents; a confident **0.0** was the symptom after they bound but the
    # episode ran outside the plan's window. Only a partial-contact number proves both are fixed.
    assert 0.0 < comms.value < 1.0, f"expected partial Earth contact, got {comms.value}"


def test_the_episode_runs_at_the_epoch_the_scenario_declares(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The task decides *when* it runs — the runner does not pick for it."""
    declared = Epoch(tdb_seconds=120.0, scale=TimeScale.TDB)  # inside the fixture plan's window
    spec = _spec(content)
    spec = spec.model_copy(
        update={"episode": spec.episode.model_copy(update={"start_epoch": declared})}
    )

    resolved_run = sim_scenario_from_spec(
        resolve_scenario(spec),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )
    assert resolved_run.scenario.start_epoch == declared


def test_a_scenario_that_declares_no_epoch_keeps_the_runner_default(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """Unset stays unset: the pre-existing behaviour, and right when nothing time-dependent is
    pinned."""
    resolved_run = sim_scenario_from_spec(
        resolve_scenario(_spec(content)),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )
    assert resolved_run.scenario.start_epoch == J2000_EPOCH


def test_surface_assets_are_placed_on_the_body_fixed_SURFACE_not_around_its_centre(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The placement bug (astro-mine-bench#31), pinned so it cannot come back.

    Core's `WorldProvider` resolves queried positions in the **body-fixed** frame, `AgentSpec` says
    the same of `initial_position_m`, and `Scenario.frame` defaults to MOON_BODY_FIXED. Orbiters
    already honoured that. Surface assets did not: they were laid out on a 25 m ring about the
    *origin*, which in a body-fixed frame is 25 m from the Moon's **centre**, on the equator.

    Worlds read that literally and — because `sample()` is deliberately total, returning a default
    point rather than raising for an out-of-grid query — handed back gravity evaluated 25 m from the
    body centre (3.3e24 m/s^2), `None` regolith, and zero elevation. Nothing raised. The
    reduced-order engines integrate no particle bed so it stayed invisible; the DEM granular tier
    exploded its settle to NaN on the first step.

    So: a surface asset must sit ~one body radius from the origin, not ~25 m from it.
    """
    run_ = sim_scenario_from_spec(
        resolve_scenario(_spec(content)),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )
    surface = [a for a in run_.scenario.agents if a.dynamics.kind != "orbital"]
    assert surface, "the fixture pins no surface asset"

    for agent in surface:
        x, y, z = agent.initial_position_m
        radius = math.sqrt(x * x + y * y + z * z)
        assert radius == pytest.approx(MOON_RADIUS_M, rel=1e-3), (
            f"{agent.agent_id} sits {radius:.1f} m from the body centre; a surface asset belongs "
            f"~{MOON_RADIUS_M:.0f} m out, on the ground"
        )
        # ...and the deterministic ring survives: the tangential offset is what it always was.
        assert math.hypot(x, y) == pytest.approx(_SURFACE_LAYOUT_RADIUS_M, rel=1e-6)
        assert z < 0.0, "the anchor site is the SOUTH pole"


def test_the_pinned_worlds_soil_and_gravity_actually_reach_the_physics(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The consequence that made the bug matter: with the asset off-world, every `world.sample()`
    fell through to the out-of-grid default, so the **pinned world's regolith never reached the
    dynamics** — Sim silently used its own reduced-order constants instead."""
    run_ = sim_scenario_from_spec(
        resolve_scenario(_spec(content)),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )
    world = run_.world_provider
    for agent in run_.scenario.agents:
        if agent.dynamics.kind == "orbital":
            continue
        point = world.sample(agent.initial_position_m)
        gravity = math.sqrt(sum(c * c for c in point.gravity))
        assert 0.5 < gravity < 5.0, (
            f"gravity at {agent.agent_id}'s site is {gravity:.3g} m/s^2 — a surface asset must not "
            "be sampling the gravity field from inside the body"
        )


# --- the runner provider: Bench's `astro_mine.bench.runners` entry point (RM-P0-SIM-11) ----------
#
# `astro-mine-bench score --runner sim` resolves this provider by name through the entry-point group
# bench#58 defined, and injects the runner it returns. Bench never imports Sim (conventions.md §1.1;
# bench.md §2.2) — the direction is one-way.

_REGISTRY_ENV = "ASTRO_MINE_HUB_REGISTRY"


def test_provider_reports_the_sim_runner_identity() -> None:
    assert sim_runner_provider.runner_id == SIM_RUNNER_ID


def test_provider_builds_the_sim_runners_from_an_explicit_store() -> None:
    # A store passed explicitly is honoured as-is (no env, no open); the runners are the real ones.
    store = object()  # opaque: SimEpisodeRunner holds it until a run resolves content
    assert isinstance(sim_runner_provider.episode_runner(store), SimEpisodeRunner)
    assert isinstance(sim_runner_provider.harness_runner(store), SimHarnessRunner)


def test_provider_resolves_the_store_from_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When Bench passes no store (its `score` CLI does not), the provider opens the one named by
    # $ASTRO_MINE_HUB_REGISTRY — the same convention `astro-mine-sim run` uses.
    monkeypatch.setenv(_REGISTRY_ENV, str(tmp_path))
    assert isinstance(sim_runner_provider.episode_runner(), SimEpisodeRunner)
    assert isinstance(sim_runner_provider.harness_runner(), SimHarnessRunner)


def test_provider_without_a_store_or_env_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_REGISTRY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=_REGISTRY_ENV):
        sim_runner_provider.episode_runner()


def test_bench_resolves_the_sim_runner_through_its_entry_point_group() -> None:
    # The end-to-end registration: the *installed* Bench discovers Sim's provider by name and
    # accepts it as a BenchRunnerProvider (load_runner_provider raises if it does not conform).
    from astro_mine.bench.baseline import load_runner_provider

    provider = load_runner_provider("sim")
    assert provider.runner_id == SIM_RUNNER_ID
    assert provider is sim_runner_provider


# --- the anchor baseline's golden scorecard (#61, CX-REPRO) --------------------------------------


def _baseline_policy(content: dict[str, Any], spec: ScenarioSpec) -> Any:
    """The shipped baseline, built the way Bench's `score --runner sim` builds it."""
    return sim_runner_provider.default_policy(spec, content["store"])


def test_the_anchor_baseline_scores_a_reproducible_scorecard(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The artifact G1.3 says does not exist: a Sim-scored baseline with numbers in it (#61).

    **This asserts reproducibility, not a pinned hash** — the pinned hash it replaced was wrong on
    both counts (#65). The fixture routes its assets to `orbital` and `mobility`, both `TOLERANCE`
    engines, and `validation/determinism.py` is explicit that a bit-exact golden is not portable
    across builds for those; pinning one asserted a guarantee the engines do not make.

    It was also not testing what it claimed. A `Scorecard`'s content hash covers the metric
    *values*, which here are coarse enough (`0.0`, several `None`s, an integer night count) that the
    hash survived the entire engine substitution this issue is about — three engines swapped under
    it and the golden never noticed. The scorecard's meaning is asserted by the metric-level test
    below, and the physics by `test_the_declared_engines_are_the_engines_that_ran`.

    Runs at a capped horizon on purpose: the anchor's declared 43 200 steps are a benchmark run.
    """
    spec = _spec(content)

    def score() -> Any:
        return run(
            spec,
            _baseline_policy(content, spec),
            runner=_runner(content, tmp_path),
            runner_id=SIM_RUNNER_ID,
            seeds=(1001,),
        )

    card = score()
    assert card.runner == SIM_RUNNER_ID
    # Same seed, same inputs ⇒ same scorecard (CX-REPRO). This is the guarantee the engines *do*
    # make, and it still fails loudly on an unseeded draw or a wall-clock leak.
    assert score().content_hash == card.content_hash


def test_the_baseline_scorecard_is_not_empty(content: dict[str, Any], tmp_path: Path) -> None:
    """Non-empty is the point of G1.3 — and each not-applicable has a recorded reason.

    `water_mass` is `0.0` **by policy**, not by accident: `IsruModel` is uncoupled from excavation
    (#64), so commanding the plant into `extract` would manufacture water that no digging earned.
    `energy_per_kg` follows from it (Bench returns None when water <= 0), and the two belief
    metrics are not-applicable by design — Sim will not fabricate a posterior (#66).
    """
    spec = _spec(content)
    card = run(
        spec,
        _baseline_policy(content, spec),
        runner=_runner(content, tmp_path),
        runner_id=SIM_RUNNER_ID,
        seeds=(1001,),
    )
    scored = {m.metric: m.value for m in card.metrics}

    assert scored["water_mass"] == 0.0  # honest zero: nothing was extracted, and nothing claims it
    assert scored["energy_per_kg"] is None  # derived from water <= 0
    assert scored["nights_survived"] is not None  # measured against the pinned world
    assert scored["discovery_latency"] is not None  # the rover's prospecting sensors really read


def test_the_baseline_commands_only_modes_the_pinned_assets_declare(
    content: dict[str, Any],
) -> None:
    """The defect `BaselinePolicy` has: it commands "prospect" at assets that never declare it, so
    the power model prices a tick the asset cannot be in."""
    spec = _spec(content)
    policy = _baseline_policy(content, spec)
    # Fleet-only, exactly as `default_policy` resolves: pulling the world/prospect/link pins too
    # would materialize providers this assertion has no use for (and, for the anchor, ~460 MB of
    # terrain).
    fleet = ScenarioContent(
        fleet=tuple(ContentPin(id=r.id, reference=r.content_hash) for r in spec.content.fleet)
    )
    resolved = ContentResolver(content["store"]).resolve(fleet)

    for agent_id, mode in policy.modes.items():
        asset = resolved.assets[agent_id].asset
        declared = {load.mode for load in asset.power.loads_by_mode} if asset.power else set()
        assert mode in declared or not declared, f"{agent_id} cannot be in {mode!r}: {declared}"


def test_the_baseline_never_puts_an_isru_plant_in_an_extraction_mode(
    content: dict[str, Any],
) -> None:
    spec = _spec(content)
    policy = _baseline_policy(content, spec)
    plant = next(a for a in policy.modes if a.endswith("isru-plant"))
    assert policy.modes[plant] not in DEFAULT_EXTRACTION_MODES


def test_the_declared_engines_are_the_engines_that_ran(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The #65 regression: a Bench-scored run must execute the dynamics its SADF routes to.

    Before this, `SimEpisodeRunner` left `engine_factory` unset, so `kinematic_engine_factory` built
    a single `KinematicEngine` for *every* agent — the orbiter did not propagate orbitally, the
    excavator could not dig — while the trace's fidelity record and engine versions were computed
    from `AgentSpec.dynamics.kind` and therefore claimed engines that never ran.

    The fixture's three assets route to three different kinds, so this is a real routing assertion
    rather than a homogeneous no-op.
    """
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec(content)), 1001)
    declared = {a.dynamics.kind for a in run_.scenario.agents}
    assert declared == {"orbital", "mobility", "kinematic"}  # the fixture is heterogeneous

    simulator = Simulator(run_.scenario, world_provider=run_.world_provider)
    simulator.reset(seed=1001)

    ran = {d.name for d in engines_that_ran(simulator.engine)}
    assert ran == {
        "astro-mine.sim.orbital",
        "astro-mine.sim.mobility",
        "astro-mine.sim.kinematic",
    }


def test_the_recorded_engine_versions_come_from_the_engines_that_ran(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """Provenance is a measurement now, not a claim (#65).

    `engine_versions` read `AgentSpec.dynamics.kind`, so a recording could name an orbital engine
    for a run in which only the kinematic engine ever stepped — and `provenance.py`'s own docstring
    asserted the opposite guarantee. It now reads the constructed engine.
    """
    runner = _runner(content, tmp_path)
    runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)
    recording = read_recording(runner.recordings[1001])

    assert set(recording.provenance["run"]["engine_versions"]) == {
        "astro-mine.sim.orbital",
        "astro-mine.sim.mobility",
        "astro-mine.sim.kinematic",
    }


def test_a_homogeneous_scenario_is_unperturbed_by_the_coupler(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """Adopting the coupler as the default must not move an existing all-kinematic trace (CX-REPRO).

    This is what made the default flip safe to take globally rather than only in the Bench adapter:
    the coupler over one kind yields that kind's single sub-engine and reproduces the reference
    stepping core byte-for-byte.
    """
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec(content)), 1001)
    kinematic_only = run_.scenario.model_copy(
        update={"agents": tuple(a for a in run_.scenario.agents if a.dynamics.kind == "kinematic")}
    )

    coupled = run_episode(kinematic_only, seed=1001, world_provider=run_.world_provider)
    reference = run_episode(
        kinematic_only,
        seed=1001,
        world_provider=run_.world_provider,
        engine_factory=kinematic_engine_factory,
    )

    assert coupled.content_hash == reference.content_hash


def test_scoring_refuses_a_run_whose_pinned_providers_did_not_rebuild(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """A scorecard is a claim, so the scoring path refuses to produce one from a blind run (#67).

    `astro-mine-bench fetch` obtains the content; rebuilding a world bundle into a `WorldProvider`
    is astro-mine-worlds' job. A user with every digest and no producers installed would otherwise
    score `nights_survived` for a mission with no measured night and `comms_robustness` for a swarm
    nothing masked — while the provenance says the content was all there.
    """
    blind = SimEpisodeRunner(store=content["store"], provider_factories={})

    # Bench's own type, not a bare RuntimeError: it is what lets `astro-mine-bench score` present a
    # deliberate refusal as an actionable error while an engine bug keeps its traceback
    # (astro-mine-bench#79). A caller must never have to match on message text to tell them apart.
    with pytest.raises(ScoringRefused, match="refusing to score") as excinfo:
        blind.resolve(resolve_scenario(_spec(content)), 1001)

    message = str(excinfo.value)
    assert "astro-mine-worlds" in message  # the package to install
    assert "nights_survived" in message  # what it costs
    assert "allow_unresolved_content=True" in message  # the deliberate opt-out


def test_the_refusal_is_benchs_type_so_the_cli_can_present_it(content: dict[str, Any]) -> None:
    """The seam this crosses: Sim raises, Bench catches, neither matches on a string.

    `ScoringRefused` subclasses RuntimeError, so a caller written against the old behaviour keeps
    working — but a caller that wants to tell a refusal from a bug now can.
    """
    blind = SimEpisodeRunner(store=content["store"], provider_factories={})

    with pytest.raises(ScoringRefused) as excinfo:
        blind.resolve(resolve_scenario(_spec(content)), 1001)

    assert isinstance(excinfo.value, RuntimeError)  # nothing that caught it before is broken


def test_scoring_a_blind_run_is_possible_when_asked_for_deliberately(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The refusal is a default, not a wall — an explicit opt-out still scores."""
    runner = SimEpisodeRunner(
        store=content["store"], provider_factories={}, allow_unresolved_content=True
    )

    run_ = runner.resolve(resolve_scenario(_spec(content)), 1001)

    assert run_.world_provider is None
    assert {u.kind for u in run_.unresolved} >= {"world_provider"}


def test_a_fully_resolved_scoring_run_is_unaffected(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The refusal must not fire on the path every existing test uses (injected factories)."""
    run_ = _runner(content, tmp_path).resolve(resolve_scenario(_spec(content)), 1001)
    assert run_.unresolved == ()
    assert run_.world_provider is not None


def test_a_blind_recording_says_so_in_its_own_provenance(
    content: dict[str, Any], tmp_path: Path
) -> None:
    """The #65 lesson applied to #67: a trace carries what happened, not what was declared.

    The key is written only when something failed to rebuild — provenance rides inside the
    determinism hash, so an always-present key would re-hash every existing trace to record that
    nothing was missing.
    """
    runner = SimEpisodeRunner(
        store=content["store"],
        provider_factories={},
        allow_unresolved_content=True,
        recording_dir=tmp_path / "mcap",
    )
    runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)

    recorded = read_recording(runner.recordings[1001]).provenance["run"]
    kinds = {entry["kind"] for entry in recorded["unresolved_providers"]}
    assert "world_provider" in kinds

    # ...and a fully-resolved run carries no such key at all.
    ok = _runner(content, tmp_path)
    ok(resolve_scenario(_spec(content)), BaselinePolicy(), 1001)
    assert "unresolved_providers" not in read_recording(ok.recordings[1001]).provenance["run"]


# --- a scenario pins where the swarm stands (astro-mine-bench#63) ----------------


def _placed(spec: ScenarioSpec, *sites: SitePlacement) -> ScenarioSpec:
    return spec.model_copy(update={"placement": PlacementSpec(sites=sites)})


def _resolve_run(spec: ScenarioSpec, content: dict[str, Any]) -> Any:
    return sim_scenario_from_spec(
        resolve_scenario(spec),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )


def _agent(run_: Any, agent_id: str) -> Any:
    return next(a for a in run_.scenario.agents if a.agent_id == agent_id)


def test_a_pinned_site_places_the_asset_there_not_on_the_ring(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The point of the change: siting is the scenario's, not a function of the content digest.
    rover = "astro-mine.fleet.prospecting-rover"
    spec = _placed(
        _spec(content),
        SitePlacement(asset=rover, lat_deg=-89.9, lon_deg=0.0, elevation_m=-3800.0),
    )
    x, y, z = _agent(_resolve_run(spec, content), rover).initial_position_m
    radius = math.sqrt(x * x + y * y + z * z)
    assert radius == pytest.approx(MOON_RADIUS_M - 3800.0, rel=1e-9)
    assert math.degrees(math.asin(z / radius)) == pytest.approx(-89.9, abs=1e-6)
    # On the pinned meridian, not on a 25 m ring: at lon 0 the y-component vanishes.
    assert abs(y) < 1e-6


def test_an_unplaced_asset_still_falls_back_to_the_deterministic_ring(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # A scenario places only what it cares about; everything else keeps the prior behaviour.
    rover = "astro-mine.fleet.prospecting-rover"
    plant = "astro-mine.fleet.isru-plant"
    spec = _placed(
        _spec(content),
        SitePlacement(asset=rover, lat_deg=-89.9, lon_deg=0.0, elevation_m=-3800.0),
    )
    run_ = _resolve_run(spec, content)
    px, py, _pz = _agent(run_, plant).initial_position_m
    # The unplaced plant is still on the 25 m ring about the pole.
    assert math.hypot(px, py) == pytest.approx(25.0, rel=1e-6)


def test_an_omitted_elevation_is_taken_from_the_pinned_terrain(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The DEM is pinned by hash, so sampling it is as reproducible as restating the number — and
    # cannot drift from the world the run actually uses.
    rover = "astro-mine.fleet.prospecting-rover"
    lat, lon = -89.9, 0.0
    spec = _placed(_spec(content), SitePlacement(asset=rover, lat_deg=lat, lon_deg=lon))
    run_ = _resolve_run(spec, content)
    x, y, z = _agent(run_, rover).initial_position_m
    radius = math.sqrt(x * x + y * y + z * z)

    # The contract is *"take it from the pinned world"*, so assert against what that world actually
    # says at this site rather than against a hard-coded height — the fixture's terrain is flat
    # here, and demanding a non-zero elevation would test the fixture, not the code.
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    reference = (
        MOON_RADIUS_M * math.cos(lat_r) * math.cos(lon_r),
        MOON_RADIUS_M * math.cos(lat_r) * math.sin(lon_r),
        MOON_RADIUS_M * math.sin(lat_r),
    )
    assert run_.world_provider is not None
    expected = float(run_.world_provider.sample(reference).elevation_m)
    assert radius == pytest.approx(MOON_RADIUS_M + expected, rel=1e-12)
    assert math.degrees(math.asin(z / radius)) == pytest.approx(lat, abs=1e-6)


def test_a_surface_site_pinned_for_an_orbiter_is_refused(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # Silently ignoring it would place the orbiter somewhere the scenario did not ask for while
    # claiming it was placed — the same silent-contradiction class this whole change removes.
    spec = _placed(
        _spec(content),
        SitePlacement(asset="astro-mine.fleet.relay-orbiter", lat_deg=-89.9, lon_deg=0.0),
    )
    with pytest.raises(ValueError, match="cannot place an orbiter"):
        _resolve_run(spec, content)


def test_pinned_placement_survives_a_re_pin(content: dict[str, Any], tmp_path: Path) -> None:
    """The regression the issue is actually about.

    `scenario_hash` derives from `spec_hash`, so under the ring layout *any* re-pin moved every
    surface asset — and `water_mass`, `discovery_latency` and the illumination-dependent metrics
    moved with it, for reasons having nothing to do with the policy. A pinned site must not care.
    """
    rover = "astro-mine.fleet.prospecting-rover"
    site = SitePlacement(asset=rover, lat_deg=-89.9, lon_deg=0.0, elevation_m=-3800.0)
    before = _placed(_spec(content), site)
    # Simulate a re-pin: a different task identity, same siting.
    after = before.model_copy(update={"description": "a re-pinned edition of the same task"})
    assert after.spec_hash != before.spec_hash, "the fixture did not actually re-pin"
    assert (
        _agent(_resolve_run(after, content), rover).initial_position_m
        == _agent(_resolve_run(before, content), rover).initial_position_m
    )


# --- a scenario pins how its belief is scored (astro-mine-bench#63) --------------


def test_spec_pinned_scoring_parameters_reach_the_scorecard_context(
    content: dict[str, Any], tmp_path: Path
) -> None:
    spec = _spec(content).model_copy(
        update={
            "scoring": ScoringSpec(
                cell_area_m2=62_500.0,
                characterized_variance_threshold=2.1e-4,
                discovery_threshold=0.01,
            )
        }
    )
    context = _runner(content, tmp_path)(resolve_scenario(spec), BaselinePolicy(), 1001).context
    assert context.cell_area_m2 == 62_500.0
    assert context.characterized_variance_threshold == 2.1e-4
    assert context.discovery_threshold == 0.01


def test_what_the_scenario_pins_outranks_the_runners_own_default(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # A runner constructed by someone who did not know what the task wanted must not override the
    # task. The constructor argument is the fallback, not the authority.
    spec = _spec(content).model_copy(update={"scoring": ScoringSpec(discovery_threshold=0.02)})
    runner = _runner(content, tmp_path, discovery_threshold=0.5)
    context = runner(resolve_scenario(spec), BaselinePolicy(), 1001).context
    assert context.discovery_threshold == 0.02


def test_a_scenario_pinning_no_scoring_block_keeps_the_runners_argument(
    content: dict[str, Any], tmp_path: Path
) -> None:
    runner = _runner(content, tmp_path, discovery_threshold=0.5)
    context = runner(resolve_scenario(_spec(content)), BaselinePolicy(), 1001).context
    assert context.discovery_threshold == 0.5


# --- a real Prospect belief, driven by the run's own observations (#66) ----------


class _StubBelief:
    """A conditionable belief with the shape Prospect's ``GriddedBelief`` has.

    Deliberately not Prospect: Sim depends on the *shape*, never on the package, so the test
    proves the seam works for anything satisfying it. Variance halves per observation, which is
    enough to make `information_gain` positive without pretending to be Bayes.
    """

    def __init__(self, variance: float = 1.0, seen: list[Any] | None = None) -> None:
        self._variance = variance
        #: Every reading handed across the seam, shared with the belief this one returns, so a
        #: test can inspect what was actually conditioned on.
        self.seen: list[Any] = [] if seen is None else seen

    def observe(self, readings: Any) -> _StubBelief:
        taken = list(readings)
        if not taken:
            return self
        self.seen.extend(taken)
        return _StubBelief(self._variance / (2 ** len(taken)), self.seen)

    def cells(self) -> dict[str, FieldDistribution]:
        return {
            f"r{r:04d}c{c:04d}": FieldDistribution(
                mean=0.03, variance=self._variance, species=_HYDROGEN, unit="mass_fraction"
            )
            for r in range(2)
            for c in range(2)
        }

    def cells_in_region(self, *, lat_deg: Any, lon_deg: Any) -> frozenset[str]:
        return frozenset({"r0000c0000", "r0000c0001"})


def _belief_content(content: dict[str, Any], belief: object) -> dict[str, Any]:
    factories = dict(content["factories"])
    factories[PluginKind.PRIOR_RECIPE.value] = lambda manifest, layers: belief
    return {**content, "factories": factories}


def _scored(content: dict[str, Any], tmp_path: Path, spec: ScenarioSpec) -> Any:
    runner = SimEpisodeRunner(
        store=content["store"],
        provider_factories=content["factories"],
        recording_dir=tmp_path / "mcap",
    )
    return runner(resolve_scenario(spec), BaselinePolicy(), 1001).context


def test_a_run_conditions_the_pinned_prior_on_what_it_observed(
    content: dict[str, Any], tmp_path: Path
) -> None:
    stub = _StubBelief()
    ctx = _scored(_belief_content(content, stub), tmp_path, _spec(content))
    # The belief fields Sim refused to fabricate are now filled — from a real prior and real
    # readings, not invented.
    assert ctx.prior_belief, "no prior belief reached the scoring context"
    assert len(ctx.belief_history) == 1, "one fused update, not a per-tick chain"
    posterior = ctx.belief_history[0].cells
    assert posterior, "the posterior is empty"
    # Uncertainty fell where the run observed: the property `information_gain` scores.
    cell = next(iter(ctx.prior_belief))
    assert posterior[cell].variance < ctx.prior_belief[cell].variance


def test_without_a_producer_the_belief_metrics_stay_not_applicable(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The honest degraded case: no `prior_recipe` factory installed, so no belief. Sim must leave
    # the fields empty rather than synthesize one — not-applicable, never a fabricated zero.
    ctx = _scored(content, tmp_path, _spec(content))
    assert ctx.prior_belief == {}
    assert ctx.belief_history == ()
    assert ctx.psr_cells == frozenset()


def test_the_derived_context_survives_a_belief(content: dict[str, Any], tmp_path: Path) -> None:
    # Filling the belief must not cost the fields Sim measured honestly — the same overlay
    # property the injection seam has.
    with_belief = _scored(_belief_content(content, _StubBelief()), tmp_path, _spec(content))
    without = _scored(content, tmp_path, _spec(content))
    assert with_belief.night_intervals == without.night_intervals
    assert with_belief.survivable_temperature_k == without.survivable_temperature_k
    assert with_belief.water_species == without.water_species


def test_a_pinned_psr_region_scopes_the_belief_to_the_target(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # The anchor asks for uncertainty reduced "over the target PSR", not over the whole grid — a
    # field-wide sum is dominated by cells nobody flew over. Both metrics also end up on one cell
    # set, so the area scored is the ground the gain was measured on.
    spec = _spec(content).model_copy(
        update={
            "scoring": ScoringSpec(
                psr_region=LatLonRegion(lat_deg=(-90.0, -89.0), lon_deg=(0.0, 360.0)),
                cell_area_m2=62_500.0,
                characterized_variance_threshold=0.5,
            )
        }
    )
    ctx = _scored(_belief_content(content, _StubBelief()), tmp_path, spec)
    assert ctx.psr_cells == frozenset({"r0000c0000", "r0000c0001"})
    assert set(ctx.prior_belief) == ctx.psr_cells, "the belief was not scoped to the region"
    assert set(ctx.belief_history[0].cells) == ctx.psr_cells


def test_only_prospecting_readings_condition_the_belief(
    content: dict[str, Any], tmp_path: Path
) -> None:
    # A storage gauge reads a tank, not the ground. Conditioning a resource field on the swarm's
    # own inventory would infer ice from how much water the swarm had already stored.
    stub = _StubBelief()
    _scored(_belief_content(content, stub), tmp_path, _spec(content))
    assert stub.seen, "nothing was conditioned on at all"

    run_ = sim_scenario_from_spec(
        resolve_scenario(_spec(content)),
        store=content["store"],
        provider_factories=content["factories"],
        seed=1001,
    )
    allowed = prospecting_sensors(run_.scenario)
    assert allowed, "the fixture pins no prospecting sensor"
    # The plant's tank gauge is not a prospecting sensor, so it must never appear in the log.
    assert "astro-mine.fleet.isru-plant" not in allowed
    every_allowed_name = {name for names in allowed.values() for name in names}
    handed = {reading.sensor for reading, _position, _t in stub.seen}
    assert handed <= every_allowed_name, f"a non-prospecting reading was conditioned on: {handed}"
    # The subset assertion above only means something if the fixture *has* a sensor that should be
    # excluded — otherwise it passes trivially on a roster where everything is a prospecting sensor.
    declared = {s.name for agent in run_.scenario.agents for s in agent.sensors}
    assert declared - every_allowed_name, (
        "the fixture declares no non-prospecting sensor to exclude"
    )


def test_one_pin_resolves_two_different_providers(content: dict[str, Any], tmp_path: Path) -> None:
    # A prospect bundle rebuilds both the sealed field (what sensors sample) and the belief (what a
    # posterior is inferred over). The provider cache was keyed by digest alone, so the second
    # lookup returned the first provider — the same object, silently, for a different contract.
    stub = _StubBelief()
    prepared = _belief_content(content, stub)
    run_ = sim_scenario_from_spec(
        resolve_scenario(_spec(content)),
        store=prepared["store"],
        provider_factories=prepared["factories"],
        seed=1001,
    )
    assert run_.belief is stub, "the belief factory's provider did not survive resolution"
    assert run_.resource_field is not None, "the sealed field did not resolve"
    # The point: two contracts, two objects, from one pin. Under a digest-only cache key these
    # were the same object and the belief silently *was* the sealed field.
    assert run_.resource_field is not run_.belief
