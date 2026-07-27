"""Per-asset ISRU extraction/storage — the productivity the anchor scenario is *for* (RM-P1-SIM-02).

A reduced-order in-situ-resource-utilization process model, mirroring the power/thermal state
pattern (RM-P0-SIM-07): it turns an asset's operating mode and the **feedstock the swarm delivered
to it** into an evolving **stored-water mass** and its cumulative **extraction energy** — so a run
reports the ``water_mass`` and ``energy_per_kg`` [Bench](bench.md) scores (bench.md §3).

The process is deliberately reduced-order (high-fidelity ISRU chemistry/thermodynamics and
multi-species extraction are out of scope): while the asset is in an **extraction mode** it
converts delivered regolith to water at ``extraction_rate_kg_s``, bounded by the water that
regolith actually carries and by an optional tank ``capacity_kg``; each kilogram extracted costs
``specific_energy_j_per_kg``. **Stored mass is monotonic** (extraction only adds; offload is out of
scope), so the reported series is non-degenerate and reproducible. The model holds **no RNG** — it
is deterministic given the same modes and deliveries.

**The mode gate is no longer sufficient, which is the point (#64).** This model once produced water
from a mode string alone, scaled by the resource field sampled at the plant's own position — so a
fixed surface plant mined the ground under its own footprint, and any policy that flipped it to
``extract`` manufactured a confident ``water_mass`` that no excavation and no haulage had earned.
Feedstock now has to arrive, and it carries the grade of wherever it was dug
(:mod:`astro_mine.sim.logistics`).

The ISRU energy is tracked on a **separate accounting** (a dedicated process bus), not drawn from
the survival battery the power/thermal model evolves — so productivity accounting never perturbs
the night-survival termination (RM-P0-SIM-07). Sim consumes the field only through the Core
``ResourceField`` contract; it never imports Prospect (conventions.md §1.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.sim.logistics import Material

__all__ = ["DEFAULT_EXTRACTION_MODES", "IsruModel", "IsruState"]

#: Operating modes that imply active extraction (matches the reduced-order contact/excavation
#: vocabulary the P0 sensor models already use).
DEFAULT_EXTRACTION_MODES = frozenset({"excavate", "drill", "dig", "extract", "isru"})


@dataclass(frozen=True, slots=True)
class IsruState:
    """An asset's evolving ISRU state: cumulative stored water (kg) and the cumulative
    electrical energy (J) spent extracting it. Both are monotonic non-decreasing."""

    stored_water_kg: float = 0.0
    energy_used_j: float = 0.0


class IsruModel:
    """The per-asset reduced-order ISRU extraction/storage evolution (RM-P1-SIM-02)."""

    def __init__(
        self,
        *,
        extraction_rate_kg_s: float,
        specific_energy_j_per_kg: float,
        capacity_kg: float | None = None,
        extraction_modes: frozenset[str] = DEFAULT_EXTRACTION_MODES,
    ) -> None:
        if extraction_rate_kg_s < 0.0:
            raise ValueError(f"extraction_rate_kg_s must be >= 0, got {extraction_rate_kg_s}")
        if specific_energy_j_per_kg < 0.0:
            raise ValueError(
                f"specific_energy_j_per_kg must be >= 0, got {specific_energy_j_per_kg}"
            )
        if capacity_kg is not None and capacity_kg < 0.0:
            raise ValueError(f"capacity_kg must be >= 0, got {capacity_kg}")
        self._rate = extraction_rate_kg_s
        self._specific_energy = specific_energy_j_per_kg
        self._capacity = capacity_kg
        self._modes = extraction_modes

    def initial_state(self) -> IsruState:
        """The empty starting state — no stored water, no energy spent."""
        return IsruState()

    def step(
        self, state: IsruState, dt_s: float, mode: str | None, feedstock: Material
    ) -> tuple[IsruState, Material]:
        """Advance the ISRU state by ``dt_s``, consuming delivered ``feedstock``.

        Returns the new state **and the feedstock that remains** — extraction is now a conversion
        of material the swarm delivered, not a function of a mode string. Water produced this step
        is ``extraction_rate_kg_s * dt_s``, bounded by the water actually present in the feedstock
        and by the tank's remaining ``capacity_kg``; the regolith consumed to yield it is
        ``produced / water_fraction``. Energy costs ``specific_energy_j_per_kg`` per kg produced.

        **The grade comes from the material, not from under the plant.** Previously this took an
        ``abundance`` sampled at the plant's own position, so a fixed surface plant produced water
        from the ground beneath its footprint — a number invariant to everything the swarm did.
        The feedstock carries the grade of wherever it was dug (:mod:`astro_mine.sim.logistics`).

        Stored mass and energy remain monotonic. A non-extraction mode, an empty or barren
        feedstock, or a full tank all return the state unchanged — the mode gate is still real, it
        is simply no longer *sufficient*.
        """
        if mode not in self._modes or dt_s <= 0.0 or self._rate == 0.0:
            return state, feedstock
        available_water = feedstock.water_kg
        if available_water <= 0.0:
            return state, feedstock
        produced = min(self._rate * dt_s, available_water)
        if self._capacity is not None:
            produced = min(produced, self._capacity - state.stored_water_kg)
        if produced <= 0.0:
            return state, feedstock
        _consumed, remaining = feedstock.take(produced / feedstock.water_fraction)
        return (
            IsruState(
                stored_water_kg=state.stored_water_kg + produced,
                energy_used_j=state.energy_used_j + self._specific_energy * produced,
            ),
            remaining,
        )
