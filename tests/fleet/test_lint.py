"""Physical-plausibility lint (RM-P0-FLEET-03).

Engine unit tests over resolved ``model.Asset`` objects, plus CLI integration proving
the headline property: a document that passes Core's schema gate (``validate``) but is
physically impossible still fails ``lint``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.core.sadf import enums, model
from astro_mine.fleet import cli
from astro_mine.fleet.lint import PlausibilityFinding, lint_asset

# --- builders --------------------------------------------------------------------


def make_asset(
    *,
    bodies: list[model.Body] | None = None,
    power: model.PowerBudget | None = None,
    sensors: list[model.Sensor] | None = None,
) -> model.Asset:
    return model.Asset(
        identity=model.Identity(id="test.asset", name="Test", version="0.1.0", kind="rover"),
        core_interface_versions={"sadf": "0.1.0"},
        # Declare the 'base' frame so bodies/sensors referencing it are referentially closed
        # (Core loader closure check, RM-P1-CORE-05) when the CLI writes + reloads the doc.
        frames=[model.Frame(name="base")],
        root_frame="base",
        bodies=bodies or [],
        power=power,
        sensors=sensors or [],
    )


def body(
    *,
    name: str = "hull",
    mass: float = 10.0,
    ixx: float = 1.0,
    iyy: float = 1.0,
    izz: float = 1.0,
    ixy: float = 0.0,
    ixz: float = 0.0,
    iyz: float = 0.0,
) -> model.Body:
    return model.Body(
        name=name,
        frame="base",
        mass_kg=mass,
        center_of_mass_m=model.Vec3(x=0.0, y=0.0, z=0.0),
        inertia_kg_m2=model.Inertia(ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz),
    )


def sensor(om: model.ObservationModel | None) -> model.Sensor:
    return model.Sensor(
        name="cam", kind=enums.SensorKind.IMAGING, frame="base", observation_model=om
    )


def rules(asset: model.Asset) -> set[str]:
    return {f.rule for f in lint_asset(asset)}


# --- clean cases -----------------------------------------------------------------


def test_empty_asset_is_clean() -> None:
    assert lint_asset(make_asset()) == []


def test_well_formed_asset_is_clean() -> None:
    asset = make_asset(
        bodies=[body()],
        power=model.PowerBudget(
            sources=[
                model.PowerSource(
                    name="solar", kind=enums.PowerSourceKind.SOLAR, nominal_power_w=200.0
                )
            ],
            storage=[model.PowerStorage(name="bat", capacity_j=1.0e6, max_discharge_w=150.0)],
            floor_w=20.0,
            loads_by_mode=[model.ModeLoad(mode="drive", power_w=120.0)],
        ),
        sensors=[
            sensor(
                model.ObservationModel(
                    fov_deg=60.0,
                    range_m=100.0,
                    noise_sigma=0.0,
                    footprint_m2=1.0,
                    depth_response_m=0.5,
                )
            )
        ],
    )
    assert lint_asset(asset) == []


# --- inertia / mass --------------------------------------------------------------


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_non_positive_mass_flagged(mass: float) -> None:
    assert "mass.positive" in rules(make_asset(bodies=[body(mass=mass)]))


def test_inertia_minor1_not_positive_definite() -> None:
    assert "inertia.positive_definite" in rules(make_asset(bodies=[body(ixx=-1.0)]))


def test_inertia_minor2_not_positive_definite() -> None:
    # ixx*iyy - ixy^2 = 1 - 4 < 0
    assert "inertia.positive_definite" in rules(make_asset(bodies=[body(ixy=2.0)]))


def test_inertia_minor3_not_positive_definite() -> None:
    # leading minors 1 and 2 positive, full determinant negative (iyz dominates)
    asset = make_asset(bodies=[body(iyz=2.0)])
    found = lint_asset(asset)
    assert [f.rule for f in found] == ["inertia.positive_definite"]
    assert found[0].path == "asset.bodies[0].inertia_kg_m2"


def test_zero_inertia_tensor_not_positive_definite() -> None:
    # strict positive-definiteness: a degenerate (zero) tensor fails minor 1
    assert "inertia.positive_definite" in rules(make_asset(bodies=[body(ixx=0.0)]))


# --- power balance ---------------------------------------------------------------


def test_negative_source_power_flagged() -> None:
    power = model.PowerBudget(
        sources=[
            model.PowerSource(name="rtg", kind=enums.PowerSourceKind.RTG, nominal_power_w=-5.0)
        ]
    )
    assert "power.negative" in rules(make_asset(power=power))


def test_negative_storage_fields_flagged() -> None:
    power = model.PowerBudget(
        storage=[
            model.PowerStorage(name="bat", capacity_j=-1.0, max_charge_w=-2.0, max_discharge_w=-3.0)
        ]
    )
    found = [f for f in lint_asset(make_asset(power=power)) if f.rule == "power.negative"]
    assert len(found) == 3  # one per negative field


def test_negative_load_and_floor_flagged() -> None:
    power = model.PowerBudget(
        floor_w=-1.0, loads_by_mode=[model.ModeLoad(mode="idle", power_w=-2.0)]
    )
    assert rules(make_asset(power=power)) == {"power.negative"}


def test_peak_load_exceeds_supply_is_a_deficit() -> None:
    power = model.PowerBudget(
        sources=[
            model.PowerSource(name="solar", kind=enums.PowerSourceKind.SOLAR, nominal_power_w=100.0)
        ],
        loads_by_mode=[model.ModeLoad(mode="dig", power_w=500.0)],
    )
    assert "power.deficit" in rules(make_asset(power=power))


def test_storage_discharge_credits_peak_supply() -> None:
    # 50 W sources + 100 W discharge = 150 W supply covers a 120 W load: no deficit
    power = model.PowerBudget(
        sources=[
            model.PowerSource(name="solar", kind=enums.PowerSourceKind.SOLAR, nominal_power_w=50.0)
        ],
        storage=[model.PowerStorage(name="bat", capacity_j=1.0e6, max_discharge_w=100.0)],
        loads_by_mode=[model.ModeLoad(mode="drive", power_w=120.0)],
    )
    assert lint_asset(make_asset(power=power)) == []


def test_undeclared_discharge_gives_no_peak_credit() -> None:
    # storage present but no declared discharge limit -> 0 creditable peak supply
    power = model.PowerBudget(
        storage=[model.PowerStorage(name="bat", capacity_j=1.0e6)],
        loads_by_mode=[model.ModeLoad(mode="drive", power_w=10.0)],
    )
    assert "power.deficit" in rules(make_asset(power=power))


def test_floor_exceeding_supply_flagged() -> None:
    power = model.PowerBudget(
        sources=[
            model.PowerSource(name="solar", kind=enums.PowerSourceKind.SOLAR, nominal_power_w=100.0)
        ],
        floor_w=200.0,
    )
    assert "power.floor" in rules(make_asset(power=power))


def test_externally_powered_asset_is_not_a_deficit() -> None:
    # loads but no on-board supply -> treated as externally powered, no deficit/floor flag
    power = model.PowerBudget(
        floor_w=50.0,
        loads_by_mode=[model.ModeLoad(mode="run", power_w=500.0)],
    )
    assert lint_asset(make_asset(power=power)) == []


# --- sensor sanity ---------------------------------------------------------------


def test_sensor_without_observation_model_is_clean() -> None:
    assert lint_asset(make_asset(sensors=[sensor(None)])) == []


def test_sensor_with_all_fields_unset_is_clean() -> None:
    assert lint_asset(make_asset(sensors=[sensor(model.ObservationModel())])) == []


@pytest.mark.parametrize("fov", [0.0, 361.0])
def test_sensor_fov_out_of_range_flagged(fov: float) -> None:
    assert "sensor.fov" in rules(make_asset(sensors=[sensor(model.ObservationModel(fov_deg=fov))]))


def test_sensor_non_positive_range_flagged() -> None:
    assert "sensor.range" in rules(
        make_asset(sensors=[sensor(model.ObservationModel(range_m=0.0))])
    )


def test_sensor_negative_noise_flagged() -> None:
    assert "sensor.noise" in rules(
        make_asset(sensors=[sensor(model.ObservationModel(noise_sigma=-0.1))])
    )


def test_sensor_non_positive_footprint_flagged() -> None:
    assert "sensor.footprint" in rules(
        make_asset(sensors=[sensor(model.ObservationModel(footprint_m2=0.0))])
    )


def test_sensor_non_positive_depth_flagged() -> None:
    assert "sensor.depth" in rules(
        make_asset(sensors=[sensor(model.ObservationModel(depth_response_m=-1.0))])
    )


def test_finding_is_an_immutable_record() -> None:
    f = PlausibilityFinding("mass.positive", "asset.bodies[0]", "boom")
    with pytest.raises(AttributeError):
        f.rule = "other"  # type: ignore[misc]


# --- CLI integration -------------------------------------------------------------


def run(*argv: str) -> int:
    """Invoke the CLI, returning the process exit code (0 when it does not exit)."""
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def write_doc(path: Path, asset: model.Asset) -> Path:
    doc = model.SadfDocument(sadf_version="0.1", asset=asset)
    data = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    path.write_text(json.dumps(data), encoding="utf-8")  # JSON is valid YAML
    return path


def test_cli_lint_passes_a_plausible_asset(tmp_path: Path) -> None:
    path = write_doc(tmp_path / "ok.sadf.json", make_asset(bodies=[body()]))
    assert run("lint", str(path)) == 0


def test_cli_lint_rejects_schema_valid_but_implausible_asset(tmp_path: Path) -> None:
    # negative inertia: structurally fine, physically impossible
    path = write_doc(tmp_path / "bad.sadf.json", make_asset(bodies=[body(ixx=-1.0)]))
    assert run("validate", str(path)) == 0  # passes Core's schema gate
    assert run("lint", str(path)) == 1  # but fails physical-plausibility


def test_cli_lint_json_carries_the_rule_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_doc(tmp_path / "bad.sadf.json", make_asset(bodies=[body(ixx=-1.0)]))
    assert run("lint", str(path), "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "[inertia.positive_definite]" in payload["diagnostics"][0]["message"]


def test_a_gauge_declaring_an_unknown_unit_is_a_finding() -> None:
    """`mass_kg` looks plausible and is not a unit the platform knows (astro-mine-fleet#40).

    Sim renders a gauge's *declared* unit verbatim and Bench matches `water_mass` on `kg`, so a
    gauge declaring `mass_kg` emits readings that are silently filtered out — the tank reads empty
    however full it is. The schema accepts the string, so the check has to live here.
    """
    gauge = model.Sensor(
        name="water_gauge",
        kind=enums.SensorKind.RESOURCE_STORAGE,
        frame="base",
        resource=model.ResourceTarget(species="water", si_unit="mass_kg"),
    )
    findings = lint_asset(make_asset(sensors=[gauge]))
    assert [f.rule for f in findings] == ["sensor.resource_unit"]
    assert "mass_kg" in findings[0].message


def test_a_gauge_declaring_a_known_unit_is_clean() -> None:
    gauge = model.Sensor(
        name="water_gauge",
        kind=enums.SensorKind.RESOURCE_STORAGE,
        frame="base",
        resource=model.ResourceTarget(species="water", si_unit="kg"),
    )
    assert lint_asset(make_asset(sensors=[gauge])) == []
