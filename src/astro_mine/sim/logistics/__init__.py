# SPDX-License-Identifier: Apache-2.0
"""The value chain between a digger and a plant: excavate -> haul -> feedstock (#64).

`IsruModel` used to produce water from a mode string alone — no dig target, no delivered mass, no
proximity check — so a plant flipped into ``extract`` manufactured a confident ``water_mass`` that
no excavation and no haulage had earned. `GranularEngine`'s excavated mass, meanwhile, accrued on
engine-private accessors and reached nothing. The two halves of the anchor's value chain sat on
opposite sides of a gap.

This module is that gap closed, at the same **reduced order** as the models either side of it: no
bucket kinematics, no queueing, no route planning. Regolith accumulates where it is dug, moves to a
hauler that is physically next to the digger, moves again to a plant the hauler has physically
driven to, and is consumed there. Every step is gated on mass that exists and distance that was
actually covered.

**Grade travels with the material, and that is the substantive change.** Extraction previously
scaled by the resource field sampled *at the plant's own position* — modelling a fixed surface
plant as if it mined the ground under its own footprint, which the anchor's own SADF contradicts
("extracts water from **hauled** regolith"). Excavation yields *bulk regolith*, not water, so the
water content is a property of **where it was dug**. The grade is sampled at the excavator and
carried, blending by mass on every transfer, until the plant extracts against it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_TRANSFER_RADIUS_M",
    "DEFAULT_TRANSFER_RATE_KG_S",
    "Material",
    "transfer",
]

#: How close two assets must be for material to move between them (m). Reduced-order stand-in for
#: docking, bucket reach and manoeuvring: within this distance the transfer is possible, beyond it
#: it is not. Generous enough that a hauler need not hit a site exactly, tight enough that it must
#: genuinely have driven there — the anchor's dig site and plant are ~12 km apart.
DEFAULT_TRANSFER_RADIUS_M = 50.0

#: How fast material moves once in range (kg/s). Bounds a transfer to something that takes time
#: rather than teleporting a full cargo bin in one tick.
DEFAULT_TRANSFER_RATE_KG_S = 5.0


@dataclass(frozen=True, slots=True)
class Material:
    """A quantity of regolith and the water grade it carries.

    ``water_fraction`` is the mass fraction of water-equivalent in this material — sampled from the
    resource field where it was excavated, not where it is stored. Empty material carries no grade;
    a blend carries the mass-weighted mean, so a plant fed from two dig sites extracts against what
    it was actually given.
    """

    mass_kg: float = 0.0
    water_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_kg < 0.0:
            raise ValueError(f"material mass must be >= 0, got {self.mass_kg}")
        if not 0.0 <= self.water_fraction <= 1.0:
            raise ValueError(f"water fraction must be in [0, 1], got {self.water_fraction}")

    @property
    def water_kg(self) -> float:
        """The water-equivalent mass this material contains."""
        return self.mass_kg * self.water_fraction

    def blended_with(self, other: Material) -> Material:
        """This material plus ``other``, carrying the mass-weighted mean grade."""
        total = self.mass_kg + other.mass_kg
        if total <= 0.0:
            return Material()
        fraction = (self.water_kg + other.water_kg) / total
        return Material(mass_kg=total, water_fraction=fraction)

    def take(self, mass_kg: float) -> tuple[Material, Material]:
        """Split off up to ``mass_kg``; return ``(taken, remaining)``, both at this grade."""
        taken = max(0.0, min(mass_kg, self.mass_kg))
        return (
            Material(mass_kg=taken, water_fraction=self.water_fraction),
            Material(mass_kg=self.mass_kg - taken, water_fraction=self.water_fraction),
        )


def transfer(
    source: Material,
    sink: Material,
    *,
    dt_s: float,
    rate_kg_s: float = DEFAULT_TRANSFER_RATE_KG_S,
    sink_capacity_kg: float | None = None,
) -> tuple[Material, Material]:
    """Move material from ``source`` to ``sink`` for one tick; return the updated pair.

    Bounded by the transfer rate, by what the source actually holds, and by the sink's remaining
    capacity — so a full hauler stops loading and a transfer never creates mass. The caller decides
    *whether* a transfer may happen (proximity); this decides how much.
    """
    if dt_s <= 0.0 or rate_kg_s <= 0.0 or source.mass_kg <= 0.0:
        return source, sink
    room = float("inf") if sink_capacity_kg is None else max(0.0, sink_capacity_kg - sink.mass_kg)
    moved, remaining = source.take(min(rate_kg_s * dt_s, room))
    if moved.mass_kg <= 0.0:
        return source, sink
    return remaining, sink.blended_with(moved)
