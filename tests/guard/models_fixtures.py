"""Shared builders for the ``astro_mine.guard.models`` adapter tests.

A minimal Core ``SadfDocument`` and a scriptable in-memory ``WorldProvider`` — enough to exercise
the Fleet/Worlds constraint-source adapters without a live Sim/Fleet/Worlds. Core-typed only."""

from __future__ import annotations

from astro_mine.core.sadf.enums import ContactElementKind
from astro_mine.core.sadf.model import (
    Actuator,
    Asset,
    ContactElement,
    Identity,
    Mobility,
    PowerBudget,
    PowerStorage,
    Range,
    SadfDocument,
    ThermalBudget,
)
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.core.world.model import Illumination, IlluminationState, SurfacePoint

Vector = tuple[float, float, float]

MOON_ME = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")


def sadf_document(
    *,
    floor_w: float | None = 15.0,
    capacity_j: float | None = 1_200_000.0,
    operating_range_k: tuple[float, float] | None = (120.0, 320.0),
    survival_range_k: tuple[float, float] | None = (100.0, 350.0),
    torque_nm: float | None = 42.0,
    max_slope_deg: float | None = 22.0,
) -> SadfDocument:
    """A minimal lunar-rover SADF document; pass ``None`` to omit a subsystem/budget."""
    power = None
    if floor_w is not None or capacity_j is not None:
        storage = (
            [PowerStorage(name="bat", capacity_j=capacity_j)] if capacity_j is not None else []
        )
        power = PowerBudget(floor_w=floor_w, storage=storage)

    thermal = None
    if operating_range_k is not None:
        thermal = ThermalBudget(
            operating_range_k=Range(min=operating_range_k[0], max=operating_range_k[1]),
            survival_range_k=(
                Range(min=survival_range_k[0], max=survival_range_k[1])
                if survival_range_k is not None
                else None
            ),
        )

    actuators = [Actuator(name="drill", torque_nm=torque_nm)] if torque_nm is not None else []
    mobility = None
    if max_slope_deg is not None:
        mobility = Mobility(
            contact=[
                ContactElement(kind=ContactElementKind.WHEEL, max_slope_deg=max_slope_deg + 5.0),
                ContactElement(kind=ContactElementKind.WHEEL, max_slope_deg=max_slope_deg),
            ]
        )

    return SadfDocument(
        sadf_version="0.1",
        asset=Asset(
            identity=Identity(id="rover", name="Prospector", version="0.1", kind="rover"),
            root_frame="body",
            actuators=actuators,
            power=power,
            thermal=thermal,
            mobility=mobility,
        ),
    )


class FakeWorldProvider:
    """An in-memory ``WorldProvider`` returning a fixed (or per-position) ``SurfacePoint``."""

    def __init__(
        self,
        *,
        frame: ReferenceFrame = MOON_ME,
        illumination: IlluminationState = IlluminationState.LIT,
        surface_normal: Vector = (0.0, 0.0, 1.0),
        gravity: Vector = (0.0, 0.0, -1.62),
        temperature_k: float = 110.0,
        raise_on_sample: bool = False,
    ) -> None:
        self._frame = frame
        self._illumination = illumination
        self._surface_normal = surface_normal
        self._gravity = gravity
        self._temperature_k = temperature_k
        self._raise = raise_on_sample

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def sample(self, position: Vector, *, epoch: object | None = None) -> SurfacePoint:
        if self._raise:
            raise RuntimeError("world sampling failed (out of raster bounds)")
        return SurfacePoint(
            frame=self._frame,
            elevation_m=0.0,
            surface_normal=self._surface_normal,
            gravity=self._gravity,
            illumination=Illumination(state=self._illumination, solar_flux_w_m2=0.0),
            temperature_k=self._temperature_k,
        )

    # The WorldProvider protocol also declares ray_intersect / line_of_sight; the adapters use only
    # sample(), so minimal stubs keep the object structurally a provider for the frame guard.
    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:  # pragma: no cover
        return None

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: object | None = None
    ) -> bool:  # pragma: no cover
        return True
