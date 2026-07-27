"""Geometric LOS + terrain occlusion (RM-P0-LINK-01).

Unit-level: the LOS composition (SPICE ephemeris position + Core ``WorldProvider`` terrain
occlusion) and the degrade-loudly contract are exercised with fakes, so no SPICE kernels or
DEM are needed. The ``SpiceEphemeris`` adapter is covered by monkeypatching
``astro_mine.spice.body_position``. Terrain-occlusion correctness itself lives in (and is
tested by) astro-mine-worlds; here we verify Link wires it correctly and never defaults to
"connected".
"""

from __future__ import annotations

import pytest

from astro_mine.core.units import (
    INERTIAL_J2000,
    MOON,
    MOON_BODY_FIXED,
    Epoch,
    ReferenceFrame,
    TimeScale,
)
from astro_mine.core.world import Vector
from astro_mine.link.geometry import (
    EphemerisNode,
    LinkGeometryError,
    LosResult,
    SpiceEphemeris,
    SurfaceNode,
    compute_los,
)

_EPOCH = Epoch(tdb_seconds=0.0, scale=TimeScale.TDB)
_RADIUS = 1_737_400.0
_SURFACE = SurfaceNode(name="rover-a", position_m=(0.0, 0.0, _RADIUS))
_RIM = SurfaceNode(name="rim-tower", position_m=(10.0, 0.0, _RADIUS))
_EARTH = EphemerisNode(name="earth", target="EARTH")


class FakeWorld:
    """A minimal WorldProvider: ``line_of_sight`` returns a preset verdict, recording its args."""

    def __init__(self, *, visible: bool, frame: ReferenceFrame = MOON_BODY_FIXED) -> None:
        self._visible = visible
        self._frame = frame
        self.calls: list[tuple[Vector, Vector]] = []

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def line_of_sight(self, observer: Vector, target: Vector, *, epoch: object = None) -> bool:
        self.calls.append((observer, target))
        return self._visible


class FakeEphemeris:
    """An EphemerisProvider returning a preset position, recording its args."""

    def __init__(self, position: Vector) -> None:
        self._position = position
        self.calls: list[tuple[str, ReferenceFrame]] = []

    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        self.calls.append((target, frame))
        return self._position


def test_surface_to_surface_visible_when_world_clears() -> None:
    world = FakeWorld(visible=True)
    result = compute_los(_SURFACE, _RIM, _EPOCH, world=world)
    assert isinstance(result, LosResult)
    assert result.visible is True
    assert (result.observer, result.target) == ("rover-a", "rim-tower")
    assert result.range_m == pytest.approx(10.0)
    assert result.frame == MOON_BODY_FIXED
    assert world.calls == [(_SURFACE.position_m, _RIM.position_m)]


def test_psr_interior_loses_los_to_earth() -> None:
    # A surface agent in a PSR: the world reports no horizon-clear line of sight to Earth.
    world = FakeWorld(visible=False)
    earth = FakeEphemeris((3.8e8, 0.0, 0.0))
    result = compute_los(_SURFACE, _EARTH, _EPOCH, world=world, ephemeris=earth)
    assert result.visible is False
    assert earth.calls == [("EARTH", MOON_BODY_FIXED)]


def test_surface_to_earth_range_uses_ephemeris_position() -> None:
    world = FakeWorld(visible=True)
    earth = FakeEphemeris((0.0, 0.0, _RADIUS + 3.8e8))
    result = compute_los(_SURFACE, _EARTH, _EPOCH, world=world, ephemeris=earth)
    assert result.visible is True
    assert result.range_m == pytest.approx(3.8e8)


def test_missing_world_raises() -> None:
    with pytest.raises(LinkGeometryError, match="no world provider"):
        compute_los(_SURFACE, _RIM, _EPOCH, world=None)


def test_ephemeris_node_without_provider_raises() -> None:
    world = FakeWorld(visible=True)
    with pytest.raises(LinkGeometryError, match="no ephemeris provider was given"):
        compute_los(_SURFACE, _EARTH, _EPOCH, world=world)


def test_ephemeris_error_propagates_not_swallowed() -> None:
    # A missing kernel (etc.) must surface, never be silently treated as "connected".
    class Boom:
        def position_body_fixed(
            self, target: str, epoch: Epoch, *, frame: ReferenceFrame
        ) -> Vector:
            raise RuntimeError("no kernel furnished")

    world = FakeWorld(visible=True)
    with pytest.raises(RuntimeError, match="no kernel furnished"):
        compute_los(_SURFACE, _EARTH, _EPOCH, world=world, ephemeris=Boom())


# --- SpiceEphemeris adapter (monkeypatched astro_mine.spice; no kernels) ----------------


def test_spice_ephemeris_resolves_via_body_position(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_body_position(target, observer, epoch, *, frame, abcorr):
        captured.update(target=target, observer=observer, frame=frame, abcorr=abcorr)
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr("astro_mine.spice.body_position", fake_body_position)
    pos = SpiceEphemeris().position_body_fixed("EARTH", _EPOCH, frame=MOON_BODY_FIXED)
    assert pos == (1.0, 2.0, 3.0)
    assert captured["target"] == "EARTH"
    assert captured["observer"] == MOON  # defaulted from frame.center
    assert captured["frame"] == MOON_BODY_FIXED


def test_spice_ephemeris_honours_explicit_body_and_abcorr(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_body_position(target, observer, epoch, *, frame, abcorr):
        captured.update(observer=observer, abcorr=abcorr)
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr("astro_mine.spice.body_position", fake_body_position)
    SpiceEphemeris(body="MOON", abcorr="NONE").position_body_fixed(
        "301", _EPOCH, frame=MOON_BODY_FIXED
    )
    assert captured["observer"] == "MOON"
    assert captured["abcorr"] == "NONE"


def test_spice_ephemeris_frame_without_centre_raises() -> None:
    # INERTIAL_J2000 has no centre body — refuse rather than guess an observer.
    with pytest.raises(LinkGeometryError, match="no centre body"):
        SpiceEphemeris().position_body_fixed("EARTH", _EPOCH, frame=INERTIAL_J2000)
