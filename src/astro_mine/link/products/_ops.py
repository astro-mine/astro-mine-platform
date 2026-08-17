# SPDX-License-Identifier: Apache-2.0
"""Forward-looking Earth-link windows for Ops (RM-P1-LINK-13).

[Ops](ops.md) lands in Phase 2, but its **delay-tolerant supervisory scheduling** needs a
forward-looking product now: the upcoming Earth ground-station contact windows, each with its
uplink/downlink latency and rate, over a horizon from "now" (link.md §6; charter §8). This
module derives that schedule from a Core :class:`~astro_mine.core.messages.ContactPlan` (the
ground-node intervals), keeping Link the **constraint model**, not the transport — Ops/Bridge
own the actual data plane (link.md §1).

**Export control (link.md §9, conventions.md §12).** The default open path predicts contacts
from *public* ephemerides + parametric antenna models — no gated capability. A
**live-mission** prediction (tying high-fidelity prediction to a real mission's assets) is
operational availability intelligence and is gated behind the Core capability
:data:`LIVE_MISSION_LINK_PREDICTION` (:class:`~astro_mine.core.sadf.enums.CapabilityTag`); this
module refuses to produce a live-mission schedule unless that capability is explicitly
authorized, so an open-commons caller gets only the public-ephemeris product.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from astro_mine.core.messages import ContactInterval, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.products._errors import LinkProductsError

__all__ = [
    "LIVE_MISSION_LINK_PREDICTION",
    "EarthLinkSchedule",
    "EarthLinkWindow",
    "earth_link_schedule",
]

#: The Core capability-vocabulary value that gates live-mission link prediction out of the open
#: commons (RM-P1-LINK-13; RFC-0003). This is exactly the ``CapabilityTag`` member
#: ``COMMS_LIVE_MISSION_LINK_PREDICTION`` ("comms.live_mission_link_prediction"): Link references
#: the canonical Core vocabulary *value* while its ``astro-mine-core`` pin is the frozen v0.1.0
#: tag; once the Core release carrying RFC-0003 is cut and pinned, swap this for the enum member.
LIVE_MISSION_LINK_PREDICTION = "comms.live_mission_link_prediction"


@dataclass(frozen=True, slots=True)
class EarthLinkWindow:
    """One upcoming Earth ground-station contact, from Ops' point of view.

    ``station`` is the Earth antenna (a ``GROUND`` node); ``target`` is the Moon-side node it
    tracks (a relay or a surface agent). ``window`` is the contact's typed
    :class:`~astro_mine.core.units.EpochWindow` span (RFC-0007; conventions.md §5).
    ``uplink_latency_s`` / ``downlink_latency_s`` are the one-way delays (symmetric light-time in
    the reduced-order lunar model); ``max_rate_bps`` is the achievable rate."""

    station: str
    target: str
    window: EpochWindow
    uplink_latency_s: float | None = None
    downlink_latency_s: float | None = None
    max_rate_bps: float | None = None


@dataclass(frozen=True)
class EarthLinkSchedule:
    """A forward-looking Earth-link schedule Ops consumes for supervisory planning.

    ``windows`` are ordered by start time and include every contact still open or in the future
    at ``now`` (optionally bounded by ``horizon_s``). ``now`` is a typed
    :class:`~astro_mine.core.units.Epoch` (RFC-0007; conventions.md §5). ``provenance`` echoes the
    kernel/DEM/config hashes of the plan it was derived from."""

    now: Epoch
    windows: tuple[EarthLinkWindow, ...]
    horizon_s: float | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def stations(self) -> tuple[str, ...]:
        """The distinct ground stations that appear in the schedule, in first-seen order."""
        seen: dict[str, None] = {}
        for window in self.windows:
            seen.setdefault(window.station, None)
        return tuple(seen)

    def active(self, at: Epoch) -> tuple[EarthLinkWindow, ...]:
        """The windows open at ``at`` (``start ≤ t < end``)."""
        t = at.tdb_seconds
        return tuple(
            w for w in self.windows if w.window.start.tdb_seconds <= t < w.window.end.tdb_seconds
        )

    def upcoming(self, at: Epoch) -> tuple[EarthLinkWindow, ...]:
        """The windows that have not yet started at ``at`` (``start > t``)."""
        t = at.tdb_seconds
        return tuple(w for w in self.windows if w.window.start.tdb_seconds > t)


def _ground_ids(plan: ContactPlan) -> frozenset[str]:
    return frozenset(node.id for node in plan.nodes if node.role is NodeRole.GROUND)


def earth_link_schedule(
    plan: ContactPlan,
    now: Epoch,
    *,
    horizon_s: float | None = None,
    live_mission: bool = False,
    authorized_capabilities: Collection[str] = (),
    provenance: Mapping[str, str] | None = None,
) -> EarthLinkSchedule:
    """The forward-looking Earth-link schedule from ``plan``, from ``now`` over ``horizon_s``.

    Extracts every contact interval touching a ``GROUND`` node whose window has not yet closed
    at ``now`` (and, if ``horizon_s`` is given, starts within the horizon), as
    :class:`EarthLinkWindow`\\ s with uplink/downlink latency and rate — the product Ops (P2)
    schedules against. Deterministic: ordered by ``(start, station, target)``.

    **Gated.** ``live_mission=True`` requires :data:`LIVE_MISSION_LINK_PREDICTION` in
    ``authorized_capabilities`` (export-control, link.md §9); otherwise it raises
    :class:`~astro_mine.link.products.LinkProductsError`. The default (``live_mission=False``)
    open path — public ephemerides + parametric antennas — is always allowed.
    """
    if live_mission and LIVE_MISSION_LINK_PREDICTION not in set(authorized_capabilities):
        raise LinkProductsError(
            "live-mission Earth-link prediction is export-controlled: it requires the "
            f"{LIVE_MISSION_LINK_PREDICTION!r} capability, which is not authorized; "
            "the open path (live_mission=False) uses public ephemerides only"
        )
    ground = _ground_ids(plan)
    now_s = now.tdb_seconds
    windows: list[EarthLinkWindow] = []
    for interval in plan.intervals:
        if interval.node_a in ground:
            station, target = interval.node_a, interval.node_b
        elif interval.node_b in ground:
            station, target = interval.node_b, interval.node_a
        else:
            continue  # a Moon-side inter-agent/relay link, not an Earth link
        if interval.end_tdb_s <= now_s:
            continue  # already closed — not forward-looking
        if horizon_s is not None and interval.start_tdb_s > now_s + horizon_s:
            continue  # beyond the planning horizon
        latency = (
            interval.mean_latency_s
            if interval.mean_latency_s is not None
            else interval.min_latency_s
        )
        windows.append(
            EarthLinkWindow(
                station=station,
                target=target,
                window=_interval_window(interval, now.scale),
                uplink_latency_s=latency,
                downlink_latency_s=latency,
                max_rate_bps=interval.max_rate_bps,
            )
        )
    windows.sort(key=lambda w: (w.window.start.tdb_seconds, w.station, w.target))
    return EarthLinkSchedule(
        now=now,
        windows=tuple(windows),
        horizon_s=horizon_s,
        provenance=dict(provenance or {}),
    )


def _interval_window(interval: ContactInterval, scale: TimeScale) -> EpochWindow:
    """The interval's span as a typed :class:`EpochWindow` (RFC-0007).

    Prefers the interval's typed ``window`` when the producer populated it (RFC-0007: consumers
    MUST prefer the typed field), and reconstructs one from the ``*_tdb_s`` primitives in the
    schedule's own ``scale`` only as a fallback."""
    if interval.window is not None:
        return interval.window
    return EpochWindow(
        start=Epoch(tdb_seconds=interval.start_tdb_s, scale=scale),
        end=Epoch(tdb_seconds=interval.end_tdb_s, scale=scale),
    )
