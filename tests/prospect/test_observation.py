"""RM-P0-PROSPECT-04 — the typed sensor return and the CSV observation feed.

Proves the belief-log entry (:class:`FieldObservation`): a located, noisy reading with a mandatory
likelihood, the Core ``SensorReading`` adapter (the Sim/Ops seam), and :func:`load_observations`
reading an ordered log from a CSV of ``(location, sensor reading)`` rows (the acceptance feed).
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import SensorReading
from astro_mine.prospect.belief import FieldObservation, load_observations

_HEADER = "x_m,y_m,z_m,value,noise_sigma,time_s,sensor"


# --- FieldObservation ------------------------------------------------------------------------


def test_position_property() -> None:
    obs = FieldObservation(x_m=1.0, y_m=2.0, z_m=3.0, value=0.5, noise_sigma=0.01)
    assert obs.position == (1.0, 2.0, 3.0)


def test_noise_sigma_must_be_positive() -> None:
    # A noiseless reading is ground truth, not an observation.
    with pytest.raises(ValueError, match="noise_sigma"):
        FieldObservation(x_m=0.0, y_m=0.0, value=0.5, noise_sigma=0.0)


def test_observation_is_frozen_and_strict() -> None:
    obs = FieldObservation(x_m=0.0, y_m=0.0, value=0.5, noise_sigma=0.01)
    with pytest.raises(ValueError, match=r"frozen|immutable"):
        obs.value = 0.9  # type: ignore[misc]
    with pytest.raises(ValueError, match=r"extra|forbidden|not permitted"):
        FieldObservation(x_m=0.0, y_m=0.0, value=0.5, noise_sigma=0.01, bogus=1)  # type: ignore[call-arg]


# --- the Core SensorReading adapter ----------------------------------------------------------


def test_from_sensor_reading_maps_a_scalar_value() -> None:
    reading = SensorReading(sensor="neutron", values=[0.4, 0.56], noise_sigma=0.02)
    obs = FieldObservation.from_sensor_reading(
        reading, position=(1.0, 2.0, 0.0), time_s=10.0, index=1
    )
    assert obs.value == 0.56
    assert obs.noise_sigma == 0.02
    assert obs.sensor == "neutron"
    assert obs.position == (1.0, 2.0, 0.0)
    assert obs.time_s == 10.0


def test_from_sensor_reading_requires_a_likelihood() -> None:
    reading = SensorReading(sensor="ns", values=[0.4], noise_sigma=None)
    with pytest.raises(ValueError, match="no noise_sigma"):
        FieldObservation.from_sensor_reading(reading, position=(0.0, 0.0, 0.0))


def test_from_sensor_reading_rejects_a_missing_value_index() -> None:
    reading = SensorReading(sensor="ns", values=[0.4], noise_sigma=0.02)
    with pytest.raises(ValueError, match="no value at index"):
        FieldObservation.from_sensor_reading(reading, position=(0.0, 0.0, 0.0), index=5)


# --- load_observations (CSV) -----------------------------------------------------------------


def test_load_from_an_iterable_of_lines_preserves_order() -> None:
    rows = [
        _HEADER,
        "0,0,0,0.5,0.01,1,ns",
        "100,200,0,0.3,0.02,2,gpr",
    ]
    obs = load_observations(rows)
    assert len(obs) == 2
    assert obs[0].value == 0.5 and obs[0].sensor == "ns"
    assert obs[1].position == (100.0, 200.0, 0.0)
    assert obs[1].time_s == 2.0


def test_load_from_a_file_path(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "obs.csv"  # type: ignore[attr-defined]
    path.write_text(f"{_HEADER}\n0,0,0,0.5,0.01,1,ns\n")
    obs = load_observations(str(path))
    assert len(obs) == 1 and obs[0].value == 0.5


def test_optional_columns_default() -> None:
    # Only the required columns are present; z_m/time_s default to 0 and sensor to None.
    obs = load_observations(["x_m,y_m,value,noise_sigma", "1,2,0.4,0.03"])
    assert obs[0].position == (1.0, 2.0, 0.0)
    assert obs[0].time_s == 0.0
    assert obs[0].sensor is None


def test_blank_sensor_becomes_none() -> None:
    obs = load_observations([_HEADER, "0,0,0,0.5,0.01,1,"])
    assert obs[0].sensor is None


def test_missing_required_column_fails_loudly() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        load_observations(["x_m,y_m,value", "0,0,0.5"])  # no noise_sigma column


def test_empty_csv_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no header row"):
        load_observations([])


def test_empty_required_field_fails_loudly() -> None:
    with pytest.raises(ValueError, match="empty required field 'value'"):
        load_observations(["x_m,y_m,value,noise_sigma", "0,0,,0.01"])


def test_non_numeric_field_fails_loudly() -> None:
    with pytest.raises(ValueError, match="is not a number"):
        load_observations(["x_m,y_m,value,noise_sigma", "0,0,high,0.01"])


def test_constraint_violation_in_csv_reports_the_line() -> None:
    with pytest.raises(ValueError, match="line 2"):
        load_observations(["x_m,y_m,value,noise_sigma", "0,0,0.5,0"])  # noise_sigma <= 0
