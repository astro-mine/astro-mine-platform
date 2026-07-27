"""Tests for ``astro_mine.core.world`` — the WorldProvider contract, a trivial conforming
provider, and the conformance utility (D8; worlds.md §5/§6)."""

from __future__ import annotations

import pytest

from astro_mine.core import compat
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, ReferenceFrame
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
    WorldProvider,
    WorldProviderContractError,
    check_world_provider,
)


class FlatWorld:
    """A trivial, deterministic WorldProvider: a flat lit plane at z=0 with uniform gravity
    and regolith. Enough to exercise the whole query surface."""

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return SurfacePoint(
            frame=MOON_BODY_FIXED,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -1.62),
            illumination=Illumination(state=IlluminationState.LIT, solar_flux_w_m2=1361.0),
            temperature_k=210.0,
            regolith=RegolithParams(bulk_density_kg_m3=1500.0, thermal_inertia_tiu=55.0),
        )

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        # A ray pointing down hits the z=0 plane; anything else misses.
        if direction[2] < 0.0:
            return (origin[0], origin[1], 0.0)
        return None

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return True


def test_trivial_provider_satisfies_protocol() -> None:
    assert isinstance(FlatWorld(), WorldProvider)


def test_check_world_provider_passes() -> None:
    assert check_world_provider(FlatWorld()) is None


def test_check_world_provider_rejects_non_provider() -> None:
    with pytest.raises(WorldProviderContractError):
        check_world_provider(object())  # type: ignore[arg-type]


def test_surface_point_carries_the_expected_handles() -> None:
    point = FlatWorld().sample((1.0, 2.0, 0.0))
    assert isinstance(point, SurfacePoint)
    assert point.frame is MOON_BODY_FIXED
    assert point.illumination.state is IlluminationState.LIT
    assert point.regolith.bulk_density_kg_m3 == 1500.0


def test_ray_intersect_miss_returns_none() -> None:
    assert FlatWorld().ray_intersect((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) is None


class _BadFrame(FlatWorld):
    @property
    def frame(self):  # type: ignore[override]
        return "MOON_ME"


class _BadSampleType(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None):  # type: ignore[override]
        return {"elevation_m": 0.0}


class _BadNormal(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        base = super().sample(position)
        return SurfacePoint(
            frame=base.frame,
            elevation_m=base.elevation_m,
            surface_normal=(0.0, 1.0),  # type: ignore[arg-type]
            gravity=base.gravity,
            illumination=base.illumination,
            temperature_k=base.temperature_k,
            regolith=base.regolith,
        )


class _BadLineOfSight(FlatWorld):
    def line_of_sight(self, observer: Vector, target: Vector, *, epoch: Epoch | None = None):  # type: ignore[override]
        return "yes"


def _surface_point(**overrides: object) -> SurfacePoint:
    base: dict[str, object] = {
        "frame": MOON_BODY_FIXED,
        "elevation_m": 0.0,
        "surface_normal": (0.0, 0.0, 1.0),
        "gravity": (0.0, 0.0, -1.62),
        "illumination": Illumination(state=IlluminationState.LIT, solar_flux_w_m2=1361.0),
        "temperature_k": 210.0,
        "regolith": RegolithParams(),
    }
    base.update(overrides)
    return SurfacePoint(**base)  # type: ignore[arg-type]


class _BadSurfaceFrame(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return _surface_point(frame="MOON_ME")


class _BadElevation(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return _surface_point(elevation_m=0)


class _BadIllumination(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return _surface_point(illumination="lit")


class _BadRegolith(FlatWorld):
    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return _surface_point(regolith={"bulk_density_kg_m3": 1500.0})


@pytest.mark.parametrize(
    "bad",
    [
        _BadFrame,
        _BadSampleType,
        _BadNormal,
        _BadLineOfSight,
        _BadSurfaceFrame,
        _BadElevation,
        _BadIllumination,
        _BadRegolith,
    ],
)
def test_check_world_provider_rejects_violations(bad: type[FlatWorld]) -> None:
    with pytest.raises(WorldProviderContractError):
        check_world_provider(bad())


def test_world_provider_is_a_registered_core_interface() -> None:
    assert compat.CORE_INTERFACE_VERSIONS["world_provider"] == "0.1.0"
