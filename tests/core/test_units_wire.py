"""units frame/CRS/time types on the wire: Protobuf (units.proto) + Cap'n Proto parity.

RFC-0007 / RM-P1-CORE-07. The four SPICE-shaped value types round-trip byte-exactly
through both of Core's wire formats, the ``TimeScale.ET`` alias survives (ET == TDB), and
the Protobuf encoding is byte-stable. The Cap'n Proto structs (``PlanetaryCRS`` /
``EpochWindow`` new for parity; ``ReferenceFrame`` / ``Epoch`` pre-existing) live in the
messages-owned ``observation.capnp`` hot-path schema.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import capnp
import pytest

from astro_mine.core.units import wire
from astro_mine.core.units.enums import FrameClass, TimeScale
from astro_mine.core.units.model import Epoch, EpochWindow, PlanetaryCRS, ReferenceFrame

# One representative value per type, exercising present/absent optionals and the ET alias.
_FRAMES = [
    ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON"),
    ReferenceFrame(name="J2000", frame_class=FrameClass.INERTIAL),  # center absent
]
_CRSS = [
    PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0),
    PlanetaryCRS(
        body="MOON",
        body_fixed_frame="MOON_ME",
        reference_radius_m=1737400.0,
        projection="+proj=stere +lat_0=-90 +R=1737400",
        datum="D_MOON",
    ),
]
_EPOCHS = [
    Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
    Epoch(tdb_seconds=1.23456789e8, scale=TimeScale.ET),  # ET alias must survive
]
_WINDOWS = [
    EpochWindow(
        start=Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=86400.0, scale=TimeScale.ET),
    ),
]


# --- Protobuf round-trips ----------------------------------------------------------


@pytest.mark.parametrize("frame", _FRAMES, ids=lambda f: f.name)
def test_reference_frame_proto_round_trip(frame: ReferenceFrame) -> None:
    assert wire.reference_frame_from_proto(wire.reference_frame_to_proto(frame)) == frame
    assert wire.reference_frame_from_wire(wire.reference_frame_to_wire(frame)) == frame


@pytest.mark.parametrize("crs", _CRSS, ids=["geographic", "projected"])
def test_planetary_crs_proto_round_trip(crs: PlanetaryCRS) -> None:
    assert wire.planetary_crs_from_proto(wire.planetary_crs_to_proto(crs)) == crs
    assert wire.planetary_crs_from_wire(wire.planetary_crs_to_wire(crs)) == crs


@pytest.mark.parametrize("epoch", _EPOCHS, ids=lambda e: e.scale.value)
def test_epoch_proto_round_trip(epoch: Epoch) -> None:
    got = wire.epoch_from_wire(wire.epoch_to_wire(epoch))
    assert got == epoch
    assert got.scale is epoch.scale  # ET stays ET, TDB stays TDB


@pytest.mark.parametrize("window", _WINDOWS, ids=["tdb-et"])
def test_epoch_window_proto_round_trip(window: EpochWindow) -> None:
    assert wire.epoch_window_from_wire(wire.epoch_window_to_wire(window)) == window


def test_proto_wire_is_byte_stable() -> None:
    e = Epoch(tdb_seconds=42.0, scale=TimeScale.ET)
    assert wire.epoch_to_wire(e) == wire.epoch_to_wire(Epoch(tdb_seconds=42.0, scale=TimeScale.ET))
    rf = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")
    assert wire.reference_frame_to_wire(rf) == wire.reference_frame_to_wire(
        ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")
    )


# --- Cap'n Proto parity round-trips ------------------------------------------------
# The units structs live in the messages-owned observation.capnp; load it the same way
# astro_mine.core.messages.hotpath does. units/wire.py stays Protobuf-only (RFC-0007
# "Migration path"); these encode/decode helpers exercise the hot-path struct parity.


@lru_cache(maxsize=1)
def _capnp_schema() -> Any:
    with resources.as_file(
        resources.files("astro_mine.core.messages").joinpath("schema/observation.capnp")
    ) as path:
        return capnp.load(str(path))


def _frame_capnp_round_trip(frame: ReferenceFrame) -> ReferenceFrame:
    schema = _capnp_schema()
    d: dict[str, Any] = {"name": frame.name, "frameClass": frame.frame_class.value}
    if frame.center is not None:
        d["center"] = frame.center
    with schema.ReferenceFrame.from_bytes(schema.ReferenceFrame.new_message(**d).to_bytes()) as r:
        return ReferenceFrame(
            name=r.name,
            frame_class=FrameClass(r.frameClass),
            center=r.center if r._has("center") else None,
        )


def _crs_capnp_round_trip(crs: PlanetaryCRS) -> PlanetaryCRS:
    schema = _capnp_schema()
    d: dict[str, Any] = {
        "body": crs.body,
        "bodyFixedFrame": crs.body_fixed_frame,
        "referenceRadiusM": crs.reference_radius_m,
    }
    if crs.projection is not None:
        d["projection"] = crs.projection
    if crs.datum is not None:
        d["datum"] = crs.datum
    with schema.PlanetaryCRS.from_bytes(schema.PlanetaryCRS.new_message(**d).to_bytes()) as r:
        return PlanetaryCRS(
            body=r.body,
            body_fixed_frame=r.bodyFixedFrame,
            reference_radius_m=r.referenceRadiusM,
            projection=r.projection if r._has("projection") else None,
            datum=r.datum if r._has("datum") else None,
        )


def _epoch_dict(epoch: Epoch) -> dict[str, Any]:
    return {"tdbSeconds": epoch.tdb_seconds, "scale": epoch.scale.value}


def _r_epoch(r: Any) -> Epoch:
    return Epoch(tdb_seconds=r.tdbSeconds, scale=TimeScale(r.scale))


def _epoch_capnp_round_trip(epoch: Epoch) -> Epoch:
    schema = _capnp_schema()
    with schema.Epoch.from_bytes(schema.Epoch.new_message(**_epoch_dict(epoch)).to_bytes()) as r:
        return _r_epoch(r)


def _window_capnp_round_trip(window: EpochWindow) -> EpochWindow:
    schema = _capnp_schema()
    msg = schema.EpochWindow.new_message(
        start=_epoch_dict(window.start), end=_epoch_dict(window.end)
    )
    with schema.EpochWindow.from_bytes(msg.to_bytes()) as r:
        return EpochWindow(start=_r_epoch(r.start), end=_r_epoch(r.end))


@pytest.mark.parametrize("frame", _FRAMES, ids=lambda f: f.name)
def test_reference_frame_capnp_round_trip(frame: ReferenceFrame) -> None:
    assert _frame_capnp_round_trip(frame) == frame


@pytest.mark.parametrize("crs", _CRSS, ids=["geographic", "projected"])
def test_planetary_crs_capnp_round_trip(crs: PlanetaryCRS) -> None:
    assert _crs_capnp_round_trip(crs) == crs


@pytest.mark.parametrize("epoch", _EPOCHS, ids=lambda e: e.scale.value)
def test_epoch_capnp_round_trip(epoch: Epoch) -> None:
    got = _epoch_capnp_round_trip(epoch)
    assert got == epoch
    assert got.scale is epoch.scale


@pytest.mark.parametrize("window", _WINDOWS, ids=["tdb-et"])
def test_epoch_window_capnp_round_trip(window: EpochWindow) -> None:
    assert _window_capnp_round_trip(window) == window
