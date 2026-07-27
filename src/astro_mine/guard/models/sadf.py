"""Fleet SADF budget adapter — the *untrusted* constraint-source binding (RM-P1-GUARD-04).

Extracts the power / energy / thermal / torque / slope budgets a ``SafetySpec`` threshold binds
to from a Core :class:`~astro_mine.core.sadf.model.SadfDocument` (guard.md §3, §6). This lives
**outside** the trusted core (guard.md §3 — ``models/`` are constraint/dynamics adapters, not TCB):
it only *reads* Core-typed SADF, producing plain SI scalars a profile author binds into the spec's
thresholds at authoring time. The safety guarantee stays in the Rust core; a wrong budget here can
only make a *threshold* wrong, which is caught by review of the reviewed SafetySpec artifact.

Every intermediate SADF field is optional, so extraction is null-safe: a missing subsystem yields a
``None`` field rather than raising, and the profile author decides whether an absent budget is an
authoring error. Values are SI and frame-agnostic (scalars); no sibling imports — only
``astro_mine.core`` (the narrow waist)."""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.sadf.model import SadfDocument

__all__ = ["SadfBudgets"]


@dataclass(frozen=True, slots=True)
class SadfBudgets:
    """The SI budget scalars a lunar-polar SafetySpec threshold binds to, extracted from a Core
    ``SadfDocument`` (Fleet). Every field is ``None`` when the source SADF does not declare it.

    - ``power_floor_w`` — ``asset.power.floor_w`` (the hard survival power floor).
    - ``battery_capacity_j`` — Σ ``asset.power.storage[*].capacity_j`` (total stored-energy
      capacity; a night-survival *energy floor* is authored as a fraction of this).
    - ``thermal_operating_{max,min}_k`` — ``asset.thermal.operating_range_k``.
    - ``thermal_survival_{max,min}_k`` — ``asset.thermal.survival_range_k``.
    - ``max_actuator_torque_nm`` — the largest ``asset.actuators[*].torque_nm`` (the anchoring
      drill's actuator torque ceiling).
    - ``max_slope_deg`` — the *most conservative* (smallest)
      ``asset.mobility.contact[*].max_slope_deg`` (the traverse slope limit on polar terrain)."""

    power_floor_w: float | None = None
    battery_capacity_j: float | None = None
    thermal_operating_max_k: float | None = None
    thermal_operating_min_k: float | None = None
    thermal_survival_max_k: float | None = None
    thermal_survival_min_k: float | None = None
    max_actuator_torque_nm: float | None = None
    max_slope_deg: float | None = None

    @classmethod
    def from_document(cls, document: SadfDocument) -> SadfBudgets:
        """Extract the budgets from a Core ``SadfDocument`` (null-safe on each subsystem)."""
        asset = document.asset

        power_floor_w: float | None = None
        battery_capacity_j: float | None = None
        if asset.power is not None:
            power_floor_w = asset.power.floor_w
            caps = [s.capacity_j for s in asset.power.storage]
            battery_capacity_j = sum(caps) if caps else None

        thermal_operating_max_k = thermal_operating_min_k = None
        thermal_survival_max_k = thermal_survival_min_k = None
        if asset.thermal is not None:
            thermal_operating_max_k = asset.thermal.operating_range_k.max
            thermal_operating_min_k = asset.thermal.operating_range_k.min
            if asset.thermal.survival_range_k is not None:
                thermal_survival_max_k = asset.thermal.survival_range_k.max
                thermal_survival_min_k = asset.thermal.survival_range_k.min

        torques = [a.torque_nm for a in asset.actuators if a.torque_nm is not None]
        max_actuator_torque_nm = max(torques) if torques else None

        max_slope_deg: float | None = None
        if asset.mobility is not None:
            slopes = [
                c.max_slope_deg for c in asset.mobility.contact if c.max_slope_deg is not None
            ]
            max_slope_deg = min(slopes) if slopes else None

        return cls(
            power_floor_w=power_floor_w,
            battery_capacity_j=battery_capacity_j,
            thermal_operating_max_k=thermal_operating_max_k,
            thermal_operating_min_k=thermal_operating_min_k,
            thermal_survival_max_k=thermal_survival_max_k,
            thermal_survival_min_k=thermal_survival_min_k,
            max_actuator_torque_nm=max_actuator_torque_nm,
            max_slope_deg=max_slope_deg,
        )

    def energy_floor_j(self, fraction: float) -> float | None:
        """A night-survival stored-energy floor: ``fraction`` of the battery capacity (or ``None``
        when the SADF declares no storage). ``fraction`` is the reserve the survival floor holds."""
        if self.battery_capacity_j is None:
            return None
        return self.battery_capacity_j * fraction
