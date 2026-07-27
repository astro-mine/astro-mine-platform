"""``WorldsFleetSignalResolver`` — the GUARD-04 signal resolver (untrusted marshalling).

The concrete :class:`~astro_mine.guard.wrap.shield.SignalResolver` that turns a compiled model's
abstract signal keys into the per-tick float vector the trusted core reads, resolving each from its
declared source (guard.md §5, §6):

1. a **Worlds**-derived signal (charging window / slope / surface temperature) — computed from the
   wrapped :class:`~astro_mine.guard.models.worlds.WorldsTerrain` at the agent's current position;
2. a static **Fleet SADF** budget value bound to the key (rare — most SADF budgets bind *thresholds*
   at authoring time, not runtime signals);
3. otherwise an **observation** channel (a ``SensorReading`` or a known ``StateSample`` scalar), via
   the same :class:`~astro_mine.guard.wrap.shield.DefaultSignalResolver` semantics.

**Fail-safe by construction** (guard.md §2 principle 4, §9.1): any signal that cannot be resolved —
no binding, no observation, a missing position for a Worlds signal, or a Worlds sampling error —
resolves to ``NaN``. The trusted core treats a ``NaN`` signal as bad input and falls back to a
verified safe action; a resolver *never* substitutes a silent unsafe default. No sibling imports;
only ``astro_mine.core`` and Guard's own ``wrap`` seam."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from astro_mine.core.messages.model import Observation
from astro_mine.guard.models.sadf import SadfBudgets
from astro_mine.guard.models.worlds import Vector, WorldsTerrain
from astro_mine.guard.wrap.shield import DefaultSignalResolver

__all__ = ["WorldsFleetSignalResolver", "WorldsSignalKind"]


class WorldsSignalKind(StrEnum):
    """Which Worlds-derived quantity a signal key binds to."""

    CHARGING_WINDOW = "charging_window"
    SLOPE_DEG = "slope_deg"
    SURFACE_TEMPERATURE_K = "surface_temperature_k"


def _position_of(observation: Observation | None) -> Vector | None:
    """The agent's position from ``self_state.pose.translation_m`` (``None`` when unavailable — a
    Worlds signal then resolves to ``NaN``)."""
    if observation is None:
        return None
    t = observation.self_state.pose.translation_m
    return (t.x, t.y, t.z)


class WorldsFleetSignalResolver:
    """Resolve a compiled model's signals from Worlds terrain + Fleet SADF + observation; ``NaN`` on
    anything unresolved (fail-safe). Implements the ``SignalResolver`` protocol the ``PolicyShield``
    injects."""

    def __init__(
        self,
        *,
        terrain: WorldsTerrain | None = None,
        budgets: SadfBudgets | None = None,
        worlds_bindings: Mapping[str, WorldsSignalKind] | None = None,
        sadf_bindings: Mapping[str, float] | None = None,
    ) -> None:
        self._terrain = terrain
        #: Retained for provenance/introspection (the SADF budgets thresholds were bound from).
        self.budgets = budgets
        self._worlds: dict[str, WorldsSignalKind] = {
            key: WorldsSignalKind(kind) for key, kind in (worlds_bindings or {}).items()
        }
        self._sadf: dict[str, float] = {
            key: float(value) for key, value in (sadf_bindings or {}).items()
        }
        self._observation = DefaultSignalResolver()

    def resolve(self, signal_keys: Sequence[str], observation: Observation | None) -> list[float]:
        """One float per key in ``signal_keys`` order (``NaN`` where unresolvable)."""
        position = _position_of(observation)
        return [self._resolve_one(key, observation, position) for key in signal_keys]

    def _resolve_one(
        self, key: str, observation: Observation | None, position: Vector | None
    ) -> float:
        kind = self._worlds.get(key)
        if kind is not None:
            if self._terrain is None or position is None:
                return math.nan
            try:
                return self._terrain_value(kind, position)
            except Exception:
                # A Worlds sampling failure (out-of-bounds query, provider fault) is unresolved →
                # NaN → the core fails the tick closed. Never a silent unsafe default.
                return math.nan
        if key in self._sadf:
            return self._sadf[key]
        return self._observation.resolve([key], observation)[0]

    def _terrain_value(self, kind: WorldsSignalKind, position: Vector) -> float:
        assert self._terrain is not None
        if kind == WorldsSignalKind.CHARGING_WINDOW:
            return self._terrain.charging_window_active(position)
        if kind == WorldsSignalKind.SLOPE_DEG:
            return self._terrain.slope_deg(position)
        return self._terrain.surface_temperature_k(position)
