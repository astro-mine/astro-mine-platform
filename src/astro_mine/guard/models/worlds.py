# SPDX-License-Identifier: Apache-2.0
"""Worlds terrain / illumination adapter — the *untrusted* constraint-source binding (GUARD-04).

Wraps a Core :class:`~astro_mine.core.world.protocol.WorldProvider` and derives the per-tick signals
a lunar-polar ``SafetySpec`` reads from terrain and illumination (guard.md §3, §6): the
charging-window key (from the permanently-shadowed-region illumination state), the local surface
slope (from the terrain normal vs. gravity), and the surface temperature. It can also build Core
:class:`~astro_mine.core.messages.model.Volume` keep-out geometry in the body-fixed frame.

Like all of ``models/`` this is **outside** the trusted core (guard.md §3): it only reads Core-typed
Worlds outputs — no sibling imports, only ``astro_mine.core`` (the narrow waist). Every value
resolves in an explicitly named **planetary body-fixed** frame; the constructor rejects a provider
whose frame is not ``BODY_FIXED`` (and, when given, whose ``center`` is not the expected body), so a
non-lunar / Earth-referenced provider can never silently supply a constraint input (LUNAR-TR-001 —
no implicit Earth/WGS84 assumption; conventions.md §5)."""

from __future__ import annotations

import math

from astro_mine.core.messages.model import Vec3, Volume
from astro_mine.core.units.enums import FrameClass
from astro_mine.core.units.model import Epoch
from astro_mine.core.world.model import IlluminationState
from astro_mine.core.world.protocol import WorldProvider

__all__ = ["WorldsTerrain", "slope_deg_from_normal"]

#: The 3-vector position type the Core WorldProvider samples at.
Vector = tuple[float, float, float]


def slope_deg_from_normal(surface_normal: Vector, gravity: Vector) -> float:
    """The terrain slope in degrees: the angle between the surface normal and local "up" (the
    direction opposing ``gravity``). Flat ground (normal anti-parallel to gravity) is ``0°``.

    Returns ``nan`` for a degenerate (zero-length) normal or gravity — the fail-safe value, which
    the core treats as bad input and falls back on rather than trusting a bogus slope."""
    n = math.sqrt(sum(c * c for c in surface_normal))
    g = math.sqrt(sum(c * c for c in gravity))
    if n == 0.0 or g == 0.0:
        return math.nan
    # up = -gravity/|gravity|; cos(theta) = n_hat . up_hat, clamped for acos domain safety.
    cos_theta = -sum(sc * gc for sc, gc in zip(surface_normal, gravity, strict=True)) / (n * g)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


class WorldsTerrain:
    """Derives terrain/illumination signals from a Core ``WorldProvider`` (guarded to a planetary
    body-fixed frame). Sampling errors surface as exceptions to the caller, which the resolver turns
    into ``NaN`` (fail-safe)."""

    def __init__(
        self,
        provider: WorldProvider,
        *,
        expected_center: str | None = None,
        epoch: Epoch | None = None,
    ) -> None:
        frame = provider.frame
        if frame.frame_class != FrameClass.BODY_FIXED:
            raise ValueError(
                f"WorldsTerrain requires a body-fixed provider frame (LUNAR-TR-001), got "
                f"{frame.frame_class!r} for frame {frame.name!r} — no implicit Earth/inertial frame"
            )
        if expected_center is not None and frame.center != expected_center:
            raise ValueError(
                f"WorldsTerrain expected a provider centred on {expected_center!r}, got "
                f"{frame.center!r} (frame {frame.name!r})"
            )
        self._provider = provider
        self._epoch = epoch

    @property
    def frame_name(self) -> str:
        """The name of the provider's body-fixed frame (the CRS every derived value lives in)."""
        return self._provider.frame.name

    def charging_window_active(self, position: Vector) -> float:
        """``1.0`` while the surface point at ``position`` is receiving sunlight (illumination state
        is not ``SHADOW`` — i.e. a solar charging window is available), else ``0.0``. The
        night-survival ``until`` clause keys off this (PSR ⇒ ``0.0``)."""
        sample = self._provider.sample(position, epoch=self._epoch)
        return 0.0 if sample.illumination.state == IlluminationState.SHADOW else 1.0

    def slope_deg(self, position: Vector) -> float:
        """The local terrain slope (degrees) at ``position`` — surface normal vs. gravity."""
        sample = self._provider.sample(position, epoch=self._epoch)
        return slope_deg_from_normal(sample.surface_normal, sample.gravity)

    def surface_temperature_k(self, position: Vector) -> float:
        """The surface temperature (K) at ``position`` — the thermal signal when the chassis is
        surface-coupled (SADF ``ThermalBudget.surface_coupling``)."""
        return self._provider.sample(position, epoch=self._epoch).temperature_k

    def keep_out_box(self, center_m: Vector, dimensions_m: Vector) -> Volume:
        """Build a Core axis-aligned :class:`~astro_mine.core.messages.model.Volume` keep-out box in
        the provider's body-fixed frame — the frame-explicit way to author a Worlds-derived
        exclusion region a ``KeepOutConstraint`` consumes (no cross-frame ambiguity)."""
        return Volume(
            frame=self.frame_name,
            center_m=Vec3(x=center_m[0], y=center_m[1], z=center_m[2]),
            dimensions_m=Vec3(x=dimensions_m[0], y=dimensions_m[1], z=dimensions_m[2]),
        )
