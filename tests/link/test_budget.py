"""Parametric link budget — CCSDS-aligned (RM-P0-LINK-03).

Unit-level: FSPL/C-N0/rate arithmetic, mod/cod selection, and the fail-loud boundaries are
checked against closed-form values and synthetic SADF radios. No geometry layer is needed — the
budget takes a slant range as a scalar.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.sadf.enums import CommsBand
from astro_mine.core.sadf.model import Antenna, Comms
from astro_mine.link.budget import (
    BOLTZMANN_DBW_PER_K_HZ,
    CCSDS_MODCODS,
    SPEED_OF_LIGHT_M_S,
    LinkBudgetError,
    ModCod,
    ModCodError,
    ModCodTable,
    band_frequency_hz,
    compute_link_budget,
)

_QPSK = ["qpsk_r1_2", "qpsk_r3_4"]


def _tx(**kw: object) -> Comms:
    return Comms(name="tx", band=CommsBand.S_BAND, **kw)  # type: ignore[arg-type]


def _rx(**kw: object) -> Comms:
    return Comms(name="rx", band=CommsBand.S_BAND, **kw)  # type: ignore[arg-type]


# --- FSPL / C-N0 / EIRP arithmetic ------------------------------------------------------


def test_fspl_matches_closed_form() -> None:
    budget = compute_link_budget(
        _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"]),
        _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"]),
        range_m=1000.0,
        frequency_hz=SPEED_OF_LIGHT_M_S,  # wavelength = 1 m, so FSPL = 20·log10(4π·range)
    )
    assert budget.fspl_db == pytest.approx(20.0 * math.log10(4.0 * math.pi * 1000.0))
    assert budget.frequency_hz == SPEED_OF_LIGHT_M_S


def test_cn0_combines_eirp_loss_gt_and_boltzmann() -> None:
    budget = compute_link_budget(
        _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"]),
        _rx(gt_db_per_k=5.0, modcod_supported=["qpsk_r1_2"]),
        range_m=1000.0,
        frequency_hz=SPEED_OF_LIGHT_M_S,
        extra_loss_db=2.0,
    )
    expected = 10.0 - budget.fspl_db + 5.0 - BOLTZMANN_DBW_PER_K_HZ - 2.0
    assert budget.cn0_dbhz == pytest.approx(expected)


def test_eirp_derived_from_power_and_antenna_gain() -> None:
    budget = compute_link_budget(
        _tx(tx_power_w=10.0, antenna=Antenna(gain_dbi=3.0), modcod_supported=["qpsk_r1_2"]),
        _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"]),
        range_m=1000.0,
        frequency_hz=SPEED_OF_LIGHT_M_S,
    )
    assert budget.eirp_dbw == pytest.approx(13.0)  # 10·log10(10) + 3


# --- mod/cod selection + rate -----------------------------------------------------------


def test_rate_selects_best_modcod_and_reports_consistent_ebn0() -> None:
    tx = _tx(eirp_dbw=50.0, modcod_supported=_QPSK)
    rx = _rx(gt_db_per_k=10.0, modcod_supported=_QPSK)
    budget = compute_link_budget(tx, rx, range_m=1.0e6, frequency_hz=2.2e9, margin_db=3.0)
    assert budget.feasible
    assert budget.modcod == "qpsk_r1_2"  # lowest required Eb/N0 → highest energy-limited rate
    assert budget.rate_bps == pytest.approx(10.0 ** ((budget.cn0_dbhz - 1.0 - 3.0) / 10.0))
    assert budget.margin_db == pytest.approx(3.0)
    assert budget.ebn0_db == pytest.approx(budget.cn0_dbhz - 10.0 * math.log10(budget.rate_bps))


def test_rate_selection_prefers_higher_rate_regardless_of_listed_order() -> None:
    tx = _tx(eirp_dbw=50.0, modcod_supported=["qpsk_r3_4", "qpsk_r1_2"])
    rx = _rx(gt_db_per_k=10.0, modcod_supported=["qpsk_r3_4", "qpsk_r1_2"])
    budget = compute_link_budget(tx, rx, range_m=1.0e6, frequency_hz=2.2e9)
    assert budget.modcod == "qpsk_r1_2"  # higher rate wins even though it is listed second


def test_rate_clamps_to_the_lower_radio_max_rate() -> None:
    tx = _tx(eirp_dbw=80.0, modcod_supported=["qpsk_r1_2"], max_rate_bps=1.0e6)
    rx = _rx(gt_db_per_k=30.0, modcod_supported=["qpsk_r1_2"], max_rate_bps=2.0e6)
    budget = compute_link_budget(tx, rx, range_m=1.0e5, frequency_hz=2.2e9)
    assert budget.rate_bps == pytest.approx(1.0e6)  # min(tx, rx) cap
    assert budget.margin_db is not None and budget.margin_db > 3.0  # capped link keeps surplus


def test_weak_geometry_is_infeasible_not_an_error() -> None:
    tx = _tx(eirp_dbw=0.0, modcod_supported=["qpsk_r1_2"], min_rate_bps=1.0e3)
    rx = _rx(gt_db_per_k=-20.0, modcod_supported=["qpsk_r1_2"], min_rate_bps=1.0e3)
    budget = compute_link_budget(tx, rx, range_m=3.8e8, frequency_hz=2.2e9)
    assert not budget.feasible
    assert budget.modcod is None
    assert budget.rate_bps == 0.0
    assert budget.ebn0_db is None
    assert budget.margin_db is None
    assert budget.cn0_dbhz < 60.0  # geometry/C-N0 are still reported for the planner


# --- latency ----------------------------------------------------------------------------


def test_latency_is_light_time_plus_turnaround() -> None:
    budget = compute_link_budget(
        _tx(eirp_dbw=50.0, modcod_supported=["qpsk_r1_2"]),
        _rx(gt_db_per_k=10.0, modcod_supported=["qpsk_r1_2"]),
        range_m=3.0e8,
        frequency_hz=2.2e9,
        turnaround_s=0.5,
    )
    assert budget.light_time_s == pytest.approx(3.0e8 / SPEED_OF_LIGHT_M_S)
    assert budget.latency_s == pytest.approx(budget.light_time_s + 0.5)


# --- frequency resolution ---------------------------------------------------------------


def test_explicit_frequency_bypasses_the_band_check() -> None:
    tx = _tx(eirp_dbw=50.0, modcod_supported=["qpsk_r1_2"])
    rx = Comms(name="rx", band=CommsBand.X_BAND, gt_db_per_k=10.0, modcod_supported=["qpsk_r1_2"])
    budget = compute_link_budget(tx, rx, range_m=1.0e6, frequency_hz=2.2e9)
    assert budget.feasible


def test_frequency_defaults_to_the_shared_band_center() -> None:
    tx = _tx(eirp_dbw=50.0, modcod_supported=["qpsk_r1_2"])
    rx = _rx(gt_db_per_k=10.0, modcod_supported=["qpsk_r1_2"])
    budget = compute_link_budget(tx, rx, range_m=1.0e6)  # both S-band, no explicit frequency
    assert budget.frequency_hz == pytest.approx(2.2e9)
    assert budget.feasible


def test_band_mismatch_without_frequency_raises() -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"])
    rx = Comms(name="rx", band=CommsBand.X_BAND, gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="band"):
        compute_link_budget(tx, rx, range_m=1000.0)


def test_band_frequency_known_and_optical_unsupported() -> None:
    assert band_frequency_hz(CommsBand.X_BAND) == pytest.approx(8.4e9)
    with pytest.raises(LinkBudgetError, match="optical"):
        band_frequency_hz(CommsBand.OPTICAL)


# --- fail-loud boundaries ---------------------------------------------------------------


def test_missing_eirp_raises() -> None:
    tx = _tx(modcod_supported=["qpsk_r1_2"])  # no eirp_dbw, no tx_power_w
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="no EIRP"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=2.2e9)


def test_power_without_antenna_gain_cannot_derive_eirp() -> None:
    tx = _tx(tx_power_w=10.0, modcod_supported=["qpsk_r1_2"])  # power but no antenna gain
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="no EIRP"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=2.2e9)


def test_missing_gt_raises() -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"])
    rx = _rx(modcod_supported=["qpsk_r1_2"])  # no gt_db_per_k
    with pytest.raises(LinkBudgetError, match="G/T"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=2.2e9)


def test_unknown_modcod_raises_modcod_error() -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["exotic_r9_9"])
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["exotic_r9_9"])
    with pytest.raises(ModCodError, match="unknown mod/cod"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=2.2e9)


def test_no_shared_modcod_raises() -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"])
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["gmsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="share no mod/cod"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=2.2e9)


@pytest.mark.parametrize("bad_range", [0.0, -1.0])
def test_non_positive_range_raises(bad_range: float) -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"])
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="range_m must be positive"):
        compute_link_budget(tx, rx, range_m=bad_range, frequency_hz=2.2e9)


@pytest.mark.parametrize("bad_freq", [0.0, -1.0])
def test_non_positive_frequency_raises(bad_freq: float) -> None:
    tx = _tx(eirp_dbw=10.0, modcod_supported=["qpsk_r1_2"])
    rx = _rx(gt_db_per_k=0.0, modcod_supported=["qpsk_r1_2"])
    with pytest.raises(LinkBudgetError, match="frequency_hz must be positive"):
        compute_link_budget(tx, rx, range_m=1000.0, frequency_hz=bad_freq)


# --- the CCSDS table + ModCodTable ------------------------------------------------------


def test_ccsds_table_is_internally_monotonic() -> None:
    table = CCSDS_MODCODS
    assert {
        "qpsk_r1_2",
        "qpsk_r3_4",
        "gmsk_r1_2",
        "gmsk_r3_4",
        "bpsk_r1_2",
        "8psk_r3_4",
    } <= table.names
    assert table.get("qpsk_r3_4").required_ebn0_db > table.get("qpsk_r1_2").required_ebn0_db
    assert table.get("gmsk_r3_4").required_ebn0_db > table.get("gmsk_r1_2").required_ebn0_db


def test_modcod_table_rejects_duplicate_names() -> None:
    with pytest.raises(LinkBudgetError, match="duplicate"):
        ModCodTable([ModCod("x", 1.0, 1.0), ModCod("x", 2.0, 1.0)])


def test_modcod_table_get_unknown_raises_and_contains_works() -> None:
    table = ModCodTable([ModCod("x", 1.0, 1.0)])
    assert "x" in table
    assert "y" not in table
    assert table.names == frozenset({"x"})
    with pytest.raises(ModCodError, match="unknown"):
        table.get("y")


def test_a_custom_modcod_table_is_honoured() -> None:
    table = ModCodTable(
        [ModCod("custom_r1_2", required_ebn0_db=0.5, spectral_efficiency_bps_per_hz=1.0)]
    )
    tx = _tx(eirp_dbw=50.0, modcod_supported=["custom_r1_2"])
    rx = _rx(gt_db_per_k=10.0, modcod_supported=["custom_r1_2"])
    budget = compute_link_budget(tx, rx, range_m=1.0e6, frequency_hz=2.2e9, modcods=table)
    assert budget.modcod == "custom_r1_2"
