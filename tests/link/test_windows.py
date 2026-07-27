"""Contact windows — relay-orbiter + DSN ground-station (RM-P0-LINK-02).

Unit-level: the rise/set reduction, bisection refinement, and the two callers (relay LOS,
ground-station elevation mask) are exercised with fakes/monkeypatch, so no SPICE kernels or DEM
are needed. Visibility correctness lives in the geometry/spice layers; here we verify the search
reduces a series into intervals correctly and degrades loudly when a predicate raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from astro_mine import spice
from astro_mine.core.units import (
    EARTH,
    MOON_BODY_FIXED,
    Epoch,
    EpochWindow,
    FrameClass,
    ReferenceFrame,
    TimeScale,
    require_frame,
)
from astro_mine.core.world import Vector
from astro_mine.link.geometry import EphemerisNode, LinkGeometryError, SurfaceNode
from astro_mine.link.windows import (
    EARTH_BODY_FIXED,
    ContactWindow,
    GroundStation,
    LinkWindowError,
    SpiceTopocentric,
    dsn_contact_windows,
    relay_contact_windows,
    search_windows,
)


def _epoch(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


_WINDOW = EpochWindow(start=_epoch(0.0), end=_epoch(100.0))
_PAIR = ("a", "b")


# --- search_windows: rise/set reduction -------------------------------------------------


def test_interior_window_uses_coarse_sample_boundaries() -> None:
    # visible on [25, 55): rise sampled at 30, set sampled at 60 with a 10 s step.
    out = search_windows(_PAIR, lambda e: 25.0 <= e.tdb_seconds < 55.0, _WINDOW, 10.0)
    assert len(out) == 1
    w = out[0]
    assert (w.observer, w.target) == _PAIR
    assert (w.start.tdb_seconds, w.end.tdb_seconds) == (30.0, 60.0)
    assert w.duration_s == pytest.approx(30.0)


def test_contact_open_at_window_start_clamps_to_start() -> None:
    out = search_windows(_PAIR, lambda e: e.tdb_seconds < 35.0, _WINDOW, 10.0)
    assert len(out) == 1
    assert out[0].start.tdb_seconds == 0.0  # window.start — cannot be refined earlier
    assert out[0].end.tdb_seconds == 40.0


def test_contact_open_at_window_end_clamps_to_end() -> None:
    out = search_windows(_PAIR, lambda e: e.tdb_seconds >= 65.0, _WINDOW, 10.0)
    assert len(out) == 1
    assert out[0].start.tdb_seconds == 70.0
    assert out[0].end.tdb_seconds == 100.0  # window.end — still open at the last sample


def test_two_disjoint_windows() -> None:
    out = search_windows(
        _PAIR, lambda e: e.tdb_seconds < 15.0 or e.tdb_seconds >= 75.0, _WINDOW, 10.0
    )
    assert [(w.start.tdb_seconds, w.end.tdb_seconds) for w in out] == [(0.0, 20.0), (80.0, 100.0)]


def test_always_visible_is_one_full_window() -> None:
    out = search_windows(_PAIR, lambda e: True, _WINDOW, 10.0)
    assert len(out) == 1
    assert (out[0].start.tdb_seconds, out[0].end.tdb_seconds) == (0.0, 100.0)


def test_never_visible_is_empty() -> None:
    assert search_windows(_PAIR, lambda e: False, _WINDOW, 10.0) == []


def test_single_sample_window_visible() -> None:
    # window shorter than the step → one grid sample at start.
    win = EpochWindow(start=_epoch(0.0), end=_epoch(5.0))
    assert search_windows(_PAIR, lambda e: True, win, 10.0)[0].end.tdb_seconds == 5.0
    assert search_windows(_PAIR, lambda e: False, win, 10.0) == []


# --- search_windows: bisection refinement -----------------------------------------------


def test_refinement_sharpens_both_boundaries() -> None:
    out = search_windows(_PAIR, lambda e: 25.0 <= e.tdb_seconds < 55.0, _WINDOW, 10.0, refine_s=0.1)
    assert len(out) == 1
    assert out[0].start.tdb_seconds == pytest.approx(25.0, abs=0.1)
    assert out[0].end.tdb_seconds == pytest.approx(55.0, abs=0.1)


def test_refinement_does_not_cross_the_window_start() -> None:
    # An open-at-start contact has no interior rise to refine — the start stays clamped.
    out = search_windows(_PAIR, lambda e: e.tdb_seconds < 33.0, _WINDOW, 10.0, refine_s=0.1)
    assert out[0].start.tdb_seconds == 0.0
    assert out[0].end.tdb_seconds == pytest.approx(33.0, abs=0.1)


# --- search_windows: validation + degrade-loudly ----------------------------------------


@pytest.mark.parametrize("step", [0.0, -10.0])
def test_non_positive_step_raises(step: float) -> None:
    with pytest.raises(LinkWindowError, match="step_s must be positive"):
        search_windows(_PAIR, lambda e: True, _WINDOW, step)


@pytest.mark.parametrize("refine", [0.0, -1.0, 20.0])
def test_refine_outside_range_raises(refine: float) -> None:
    with pytest.raises(LinkWindowError, match="refine_s must be in"):
        search_windows(_PAIR, lambda e: True, _WINDOW, 10.0, refine_s=refine)


def test_predicate_error_propagates_not_swallowed() -> None:
    def boom(_: Epoch) -> bool:
        raise RuntimeError("no kernel furnished")

    with pytest.raises(RuntimeError, match="no kernel furnished"):
        search_windows(_PAIR, boom, _WINDOW, 10.0)


# --- relay_contact_windows --------------------------------------------------------------


class _EpochWorld:
    """A WorldProvider whose horizon LOS is gated by an epoch predicate (ignores geometry)."""

    def __init__(self, visible_when: object, frame: ReferenceFrame = MOON_BODY_FIXED) -> None:
        self._visible_when = visible_when
        self._frame = frame

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        assert epoch is not None
        return bool(self._visible_when(epoch))  # type: ignore[operator]


class _FakeEphemeris:
    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        return (0.0, 0.0, 2.0e6)


_ROVER = SurfaceNode(name="rover-a", position_m=(0.0, 0.0, 1.7e6))
_RELAY = EphemerisNode(name="relay", target="-999")


def test_relay_windows_track_los_over_the_pass() -> None:
    world = _EpochWorld(lambda e: 25.0 <= e.tdb_seconds < 55.0)
    out = relay_contact_windows(
        _ROVER, _RELAY, _WINDOW, 10.0, world=world, ephemeris=_FakeEphemeris()
    )
    assert len(out) == 1
    assert (out[0].observer, out[0].target) == ("rover-a", "relay")
    assert (out[0].start.tdb_seconds, out[0].end.tdb_seconds) == (30.0, 60.0)


def test_relay_windows_degrade_loudly_without_ephemeris() -> None:
    # An ephemeris orbiter with no provider must raise, never silently report "no contact".
    world = _EpochWorld(lambda e: True)
    with pytest.raises(LinkGeometryError, match="no ephemeris provider"):
        relay_contact_windows(_ROVER, _RELAY, _WINDOW, 10.0, world=world)


# --- EARTH_BODY_FIXED regression (RM-P1-LINK-14, RFC-0007 rule 6) ------------------------


def test_earth_body_fixed_constructs_and_passes_require_frame() -> None:
    """``EARTH_BODY_FIXED`` is a legitimate Earth body-fixed frame and MUST keep working.

    It is exactly the case Core's guard rule 6 protects (``body == EARTH`` ⇒ an Earth frame is
    valid, not an implicit-WGS84 defaulting bug — conventions.md §5). ``require_frame`` returns it
    unchanged rather than raising."""
    assert EARTH_BODY_FIXED.name == "IAU_EARTH"
    assert EARTH_BODY_FIXED.frame_class is FrameClass.BODY_FIXED
    assert EARTH_BODY_FIXED.center == EARTH
    assert require_frame(EARTH_BODY_FIXED) is EARTH_BODY_FIXED


# --- dsn_contact_windows + GroundStation + SpiceTopocentric -----------------------------


class _ElevTopo:
    """A TopocentricProvider returning a preset elevation per epoch, recording its calls."""

    def __init__(self, elevation_of: object) -> None:
        self._elevation_of = elevation_of
        self.calls: list[tuple[str, float]] = []

    def elevation_deg(self, target: str, site: spice.Site, epoch: Epoch) -> float:
        self.calls.append((target, epoch.tdb_seconds))
        return float(self._elevation_of(epoch.tdb_seconds))  # type: ignore[operator]


def test_ground_station_from_latlon_places_the_site() -> None:
    equator = GroundStation.from_latlon("DSS-eq", 0.0, 0.0)
    assert equator.site.body == EARTH
    assert equator.site.frame == EARTH_BODY_FIXED
    assert equator.site.position_m[0] == pytest.approx(6_371_000.0)
    pole = GroundStation.from_latlon("DSS-pole", 90.0, 0.0, min_elevation_deg=5.0)
    assert pole.site.position_m[2] == pytest.approx(6_371_000.0)
    assert pole.min_elevation_deg == 5.0


def test_dsn_windows_open_above_the_elevation_mask() -> None:
    station = GroundStation.from_latlon("DSS-14", 35.0, 243.0, min_elevation_deg=10.0)
    topo = _ElevTopo(lambda t: 20.0 if 25.0 <= t < 55.0 else -5.0)
    out = dsn_contact_windows(station, "-999", _WINDOW, 10.0, topocentric=topo)
    assert len(out) == 1
    assert (out[0].observer, out[0].target) == ("DSS-14", "-999")
    assert (out[0].start.tdb_seconds, out[0].end.tdb_seconds) == (30.0, 60.0)
    assert topo.calls[0] == ("-999", 0.0)


def test_dsn_windows_degrade_loudly_on_provider_error() -> None:
    class _Boom:
        def elevation_deg(self, target: str, site: spice.Site, epoch: Epoch) -> float:
            raise RuntimeError("no kernel furnished")

    station = GroundStation.from_latlon("DSS-14", 35.0, 243.0)
    with pytest.raises(RuntimeError, match="no kernel furnished"):
        dsn_contact_windows(station, "-999", _WINDOW, 10.0, topocentric=_Boom())


def test_spice_topocentric_resolves_via_body_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_body_geometry(target, site, epoch, *, abcorr):  # type: ignore[no-untyped-def]
        captured.update(target=target, site=site, abcorr=abcorr)
        return SimpleNamespace(elevation_deg=42.0)

    monkeypatch.setattr("astro_mine.spice.body_geometry", fake_body_geometry)
    site = spice.Site(body=EARTH, position_m=(6.4e6, 0.0, 0.0), frame=EARTH_BODY_FIXED)
    assert SpiceTopocentric().elevation_deg("EARTH", site, _epoch(0.0)) == pytest.approx(42.0)
    assert captured["target"] == "EARTH"
    assert captured["site"] is site
    assert captured["abcorr"] == spice.DEFAULT_ABCORR


def test_spice_topocentric_honours_explicit_abcorr(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_body_geometry(target, site, epoch, *, abcorr):  # type: ignore[no-untyped-def]
        captured.update(abcorr=abcorr)
        return SimpleNamespace(elevation_deg=0.0)

    monkeypatch.setattr("astro_mine.spice.body_geometry", fake_body_geometry)
    site = spice.Site(body=EARTH, position_m=(6.4e6, 0.0, 0.0), frame=EARTH_BODY_FIXED)
    SpiceTopocentric(abcorr="NONE").elevation_deg("EARTH", site, _epoch(0.0))
    assert captured["abcorr"] == "NONE"


def test_contact_window_duration() -> None:
    w = ContactWindow("a", "b", _epoch(10.0), _epoch(30.0))
    assert w.duration_s == pytest.approx(20.0)
