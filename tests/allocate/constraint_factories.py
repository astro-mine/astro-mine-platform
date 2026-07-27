"""Fakes + fixtures for the RM-P1-ALLOC-03 constraint-builder tests.

Minimal in-memory implementations of the Core contracts the builders consume — a
:class:`~astro_mine.core.world.protocol.WorldProvider`, a
:class:`~astro_mine.core.resource.protocol.ResourceField`, a
:class:`~astro_mine.core.messages.model.ContactPlan`, and SADF
:class:`~astro_mine.core.sadf.model.Asset` handles — so the builders run end to end without
importing any sibling package (that *is* the narrow-waist contract under test).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from astro_mine.allocate import AssetRef, ConstraintContext
from astro_mine.allocate.constraints import CostEntry, CostTable
from astro_mine.core.messages.enums import ContactConfidence, NodeRole
from astro_mine.core.messages.model import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.resource.model import FieldDistribution, Position
from astro_mine.core.sadf.enums import ContactElementKind
from astro_mine.core.sadf.model import (
    Asset,
    ContactElement,
    Identity,
    Mobility,
    PowerBudget,
    PowerStorage,
)
from astro_mine.core.units import J2000_EPOCH, Epoch, FrameClass, ReferenceFrame
from astro_mine.core.world.model import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
)

MOON_FRAME = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")


class FakeWorld:
    """A uniform :class:`~astro_mine.core.world.protocol.WorldProvider` — one slope/light/bearing
    everywhere, tuned to make a location traversable or a keep-out."""

    def __init__(
        self,
        *,
        slope_deg: float = 5.0,
        illumination: IlluminationState = IlluminationState.LIT,
        bearing_capacity_pa: float | None = 5.0e4,
        solar_flux_w_m2: float = 1361.0,
    ) -> None:
        rad = math.radians(slope_deg)
        # Gravity points down (-z); a normal tilted `slope_deg` off +z yields exactly that slope.
        self._normal: Vector = (math.sin(rad), 0.0, math.cos(rad))
        self._gravity: Vector = (0.0, 0.0, -1.62)
        self._illumination = illumination
        self._bearing = bearing_capacity_pa
        self._flux = solar_flux_w_m2

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_FRAME

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        return SurfacePoint(
            frame=MOON_FRAME,
            elevation_m=0.0,
            surface_normal=self._normal,
            gravity=self._gravity,
            illumination=Illumination(state=self._illumination, solar_flux_w_m2=self._flux),
            temperature_k=100.0,
            regolith=RegolithParams(bearing_capacity_pa=self._bearing),
        )

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        return None

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return True


class FakeField:
    """A uniform :class:`~astro_mine.core.resource.protocol.ResourceField` — one posterior."""

    def __init__(self, *, mean: float = 0.5, variance: float = 0.01) -> None:
        self._mean = mean
        self._variance = variance

    @property
    def species(self) -> str:
        return "water_equivalent_hydrogen"

    @property
    def unit(self) -> str:
        return "mass_fraction"

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_FRAME

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._mean

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._variance

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return self._mean

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return (self._mean,) * n

    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        return FieldDistribution(mean=self._mean, variance=self._variance)


def sadf_asset(
    asset_id: str,
    *,
    max_slope_deg: float | None = None,
    ground_pressure_pa: float | None = None,
    floor_w: float | None = None,
    storage_j: float | None = None,
) -> Asset:
    """A minimal SADF :class:`~astro_mine.core.sadf.model.Asset` carrying just the mobility/power
    the terrain and power builders read."""
    mobility = None
    if max_slope_deg is not None or ground_pressure_pa is not None:
        mobility = Mobility(
            contact=[
                ContactElement(
                    kind=ContactElementKind.WHEEL,
                    max_slope_deg=max_slope_deg,
                    max_ground_pressure_pa=ground_pressure_pa,
                )
            ]
        )
    power = None
    if floor_w is not None or storage_j is not None:
        power = PowerBudget(
            floor_w=floor_w,
            storage=[PowerStorage(name="battery", capacity_j=storage_j)] if storage_j else [],
        )
    return Asset(
        identity=Identity(id=asset_id, name=asset_id, version="0.1.0", kind="rover"),
        root_frame="body",
        mobility=mobility,
        power=power,
    )


def contact_plan(
    windows: dict[str, tuple[float, float]],
    *,
    relay_id: str = "relay-1",
    epoch0: Epoch = J2000_EPOCH,
) -> ContactPlan:
    """A :class:`~astro_mine.core.messages.model.ContactPlan` with a ground relay node and one
    contact interval per ``asset_id → window`` (episode seconds, shifted into TDB by ``epoch0``
    — the typed Core :class:`~astro_mine.core.units.Epoch` that mirrors ``CommsPolicy.epoch0``)."""
    nodes = [ContactNode(id=relay_id, role=NodeRole.GROUND, kind="ground_station")]
    intervals = []
    for asset_id, (start_s, end_s) in sorted(windows.items()):
        nodes.append(ContactNode(id=asset_id, role=NodeRole.SPACE, kind="surface_agent"))
        intervals.append(
            ContactInterval(
                node_a=asset_id,
                node_b=relay_id,
                start_tdb_s=start_s + epoch0.tdb_seconds,
                end_tdb_s=end_s + epoch0.tdb_seconds,
                max_rate_bps=1.0e6,
                confidence=ContactConfidence.HIGH,
            )
        )
    return ContactPlan(nodes=nodes, intervals=intervals)


def context(
    *,
    world: FakeWorld | None = None,
    contacts: ContactPlan | None = None,
    resource: FakeField | None = None,
    assets: dict[str, Asset] | None = None,
    info_values: dict[str, float] | None = None,
) -> ConstraintContext:
    """Assemble a :class:`~astro_mine.allocate.ConstraintContext` from the fakes."""
    return ConstraintContext(
        world=world,
        contacts=contacts,
        resource=resource,
        assets=assets or {},
        info_values=info_values,
    )


def cost_table(costs: Mapping[tuple[str, str], tuple[float | None, float | None]]) -> CostTable:
    """A :class:`~astro_mine.allocate.constraints.CostTable` from ``{(task, asset): (dur, e)}``."""
    return CostTable.of(
        {pair: CostEntry(duration_s=d, energy_j=e) for pair, (d, e) in costs.items()}
    )


def anchor_asset_refs_with_budget(energy_j: float = 1.0e7) -> list[AssetRef]:
    """The anchor's three assets with a uniform energy budget (a convenience for power tests)."""
    from astro_mine.core.sadf import CapabilityTag

    return [
        AssetRef(
            asset_id="prospector-rover-1",
            capability_tags=[CapabilityTag.PROSPECTING_NEUTRON, CapabilityTag.MOBILITY_WHEELED],
            budgets={"energy_j": energy_j, "time_s": 18000.0},
        ),
        AssetRef(
            asset_id="excavator-1",
            capability_tags=[CapabilityTag.EXCAVATION_BUCKET, CapabilityTag.MOBILITY_TRACKED],
            budgets={"energy_j": energy_j},
        ),
        AssetRef(
            asset_id="hauler-1",
            capability_tags=[CapabilityTag.MOBILITY_WHEELED, CapabilityTag.RETURN_BULK_HAULER],
            budgets={"energy_j": energy_j},
        ),
    ]
