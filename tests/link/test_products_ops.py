"""Forward-looking Earth-link schedule + export-control gate (RM-P1-LINK-13).

Ops (P2) consumes upcoming Earth ground-station windows with uplink/downlink latency; the
open path uses public ephemerides, and the live-mission prediction path is gated behind the
Core capability vocabulary value.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.products import (
    LIVE_MISSION_LINK_PREDICTION,
    EarthLinkSchedule,
    LinkProductsError,
    earth_link_schedule,
)


def _now(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


def _plan() -> ContactPlan:
    return ContactPlan(
        nodes=[
            ContactNode(id="S", role=NodeRole.SPACE),
            ContactNode(id="R", role=NodeRole.SPACE),
            ContactNode(id="R2", role=NodeRole.SPACE),
            ContactNode(id="G", role=NodeRole.GROUND),
        ],
        intervals=[
            ContactInterval(node_a="S", node_b="R", start_tdb_s=0.0, end_tdb_s=50.0),  # Moon-side
            ContactInterval(node_a="R", node_b="G", start_tdb_s=50.0, end_tdb_s=90.0),  # past
            ContactInterval(
                node_a="R2",
                node_b="G",
                start_tdb_s=90.0,
                end_tdb_s=150.0,
                max_rate_bps=800.0,
                mean_latency_s=2.0,
            ),  # active at now=100
            ContactInterval(
                node_a="R",
                node_b="G",
                start_tdb_s=120.0,
                end_tdb_s=180.0,
                max_rate_bps=800.0,
                mean_latency_s=2.0,
            ),  # upcoming
        ],
    )


def test_schedule_is_forward_looking_and_earth_only() -> None:
    schedule = earth_link_schedule(_plan(), _now(100.0))
    assert isinstance(schedule, EarthLinkSchedule)
    # The Moon-side S-R link and the already-closed R-G [50,90) window are excluded.
    spans = [(w.station, w.target, w.window.start.tdb_seconds) for w in schedule.windows]
    assert spans == [("G", "R2", 90.0), ("G", "R", 120.0)]
    assert schedule.stations == ("G",)


def test_schedule_carries_typed_epochs() -> None:
    # ``now`` and each window's span are typed Core Epoch/EpochWindow, not bare TDB floats
    # (RM-P1-LINK-14, RFC-0007).
    schedule = earth_link_schedule(_plan(), _now(100.0))
    assert isinstance(schedule.now, Epoch) and schedule.now.tdb_seconds == 100.0
    win = schedule.windows[0]
    assert isinstance(win.window, EpochWindow)
    assert win.window.start.scale is TimeScale.TDB
    assert (win.window.start.tdb_seconds, win.window.end.tdb_seconds) == (90.0, 150.0)


def test_uplink_downlink_latency_and_rate_are_carried() -> None:
    window = earth_link_schedule(_plan(), _now(100.0)).windows[0]
    assert window.uplink_latency_s == 2.0 and window.downlink_latency_s == 2.0
    assert window.max_rate_bps == 800.0


def test_active_and_upcoming_partition_windows() -> None:
    schedule = earth_link_schedule(_plan(), _now(100.0))
    assert [w.target for w in schedule.active(_now(100.0))] == ["R2"]  # 90 ≤ 100 < 150
    assert [w.target for w in schedule.upcoming(_now(100.0))] == ["R"]  # starts at 120


def test_horizon_bounds_the_schedule() -> None:
    schedule = earth_link_schedule(_plan(), _now(100.0), horizon_s=10.0)
    # Only the already-open R2-G window survives a 10 s horizon; the 120 s window is dropped.
    assert [w.target for w in schedule.windows] == ["R2"]


def test_provenance_is_carried() -> None:
    schedule = earth_link_schedule(_plan(), _now(100.0), provenance={"kernels": "meta@abc"})
    assert schedule.provenance == {"kernels": "meta@abc"}


# --- export control --------------------------------------------------------------


def test_live_mission_capability_value_matches_core_vocabulary() -> None:
    assert LIVE_MISSION_LINK_PREDICTION == "comms.live_mission_link_prediction"


def test_live_mission_prediction_is_gated_without_capability() -> None:
    with pytest.raises(LinkProductsError, match="export-controlled"):
        earth_link_schedule(_plan(), _now(100.0), live_mission=True)


def test_live_mission_prediction_allowed_when_authorized() -> None:
    schedule = earth_link_schedule(
        _plan(),
        _now(100.0),
        live_mission=True,
        authorized_capabilities=[LIVE_MISSION_LINK_PREDICTION],
    )
    assert schedule.stations == ("G",)


def test_open_path_needs_no_capability() -> None:
    # The default (public-ephemeris) path is always allowed with no authorized capabilities.
    assert earth_link_schedule(_plan(), _now(100.0)).windows
