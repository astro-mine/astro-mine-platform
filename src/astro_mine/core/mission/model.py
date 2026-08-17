# SPDX-License-Identifier: Apache-2.0
"""Mission-architecture schema v0.1 — typed Pydantic models (RM-P1-CORE-04 / RFC-0001).

The **reserved** RFC-0001 mission hooks: a :class:`MissionSpec` is an ordered set of
:class:`Phase`\\ s, each in a :class:`~astro_mine.core.mission.enums.Regime`, with
descriptive design-time :class:`TrajectoryRef` / :class:`ManeuverBudget` artifacts on the
inter-phase :class:`Leg`\\ s and a typed :class:`PhaseTransition` handoff event. **Schema
only — no mechanism** (mission-model.md §3): Core learns the *schema* of a Mission, never
how to fly one. A single-``surface``-phase Mission is exactly today's campaign, so the
whole schema is additive.

Dual-use (RFC-0001 R3): :class:`TrajectoryRef` and :class:`Maneuver` are descriptive
(epoch, Δv magnitude + direction in a named SPICE frame, reference state samples,
feasibility margins). They **omit — by schema — actuator/thruster command channels,
control gains, closed-loop guidance laws, and any onboard-flight-clock binding**;
operational maneuver targeting stays behind the ``operational_targeting`` gate.

The canonical schema is ``schema/mission.schema.json`` (shipped in-package); these models
mirror it and a consistency test (``tests/test_mission_consistency.py``) asserts the two
agree.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.mission.enums import ManeuverType, Regime
from astro_mine.core.units import Epoch, EpochWindow, ReferenceFrame

__all__ = [
    "Leg",
    "Maneuver",
    "ManeuverBudget",
    "MissionConstraints",
    "MissionDocument",
    "MissionSpec",
    "Phase",
    "PhaseBoundary",
    "PhaseTransition",
    "Provenance",
    "ReferenceState",
    "TrajectoryRef",
    "TrajectorySegment",
    "Vec3",
]

MISSION_VERSION = "0.1"


class _Model(BaseModel):
    """Base for every mission model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class Vec3(_Model):
    x: float
    y: float
    z: float


class Provenance(_Model):
    """Reproducibility provenance of a mission / trajectory artifact (conventions.md §5)."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


# --- descriptive design-time trajectory artifacts (NOT executable guidance) ------


class ManeuverBudget(_Model):
    """A Δv / time-of-flight budget for a leg (design-time; produced by Trajectory,
    consumed by Sizing/Allocate/Studio). Descriptive — not a command."""

    total_delta_v_mps: float
    time_of_flight_s: float | None = None
    margin_mps: float | None = None


class Maneuver(_Model):
    """A descriptive reference maneuver: an epoch, a Δv **magnitude**, and a unit direction
    in a named SPICE frame, classified by :class:`ManeuverType`. Carries **no** actuator/
    thruster channel, control gain, or closed-loop guidance law (RFC-0001 R3)."""

    epoch_tdb_s: float
    delta_v_mps: float
    direction: Vec3
    maneuver_type: ManeuverType
    # Information-preserving typed sibling of epoch_tdb_s — no onboard-clock binding,
    # no guidance capability (RFC-0007 R3; RFC-0001 R3).
    epoch: Epoch | None = None


class ReferenceState(_Model):
    """A reference position/velocity sample along a trajectory at a bounded cadence
    (design-time reference arc, not an onboard-clock-bound command)."""

    epoch_tdb_s: float
    position_m: Vec3
    velocity_mps: Vec3
    epoch: Epoch | None = None  # typed sibling of epoch_tdb_s (RFC-0007)


class TrajectorySegment(_Model):
    """One ordered segment of a reference trajectory, bounded by TDB epochs."""

    id: str
    start_epoch_tdb_s: float
    end_epoch_tdb_s: float
    kind: str | None = None
    window: EpochWindow | None = None  # typed sibling of start/end_epoch_tdb_s (RFC-0007)


class TrajectoryRef(_Model):
    """A descriptive design-time reference trajectory: ordered segments, reference
    maneuvers, reference state samples, and a feasibility margin, all in a named SPICE
    ``frame``. A *descriptive artifact, not executable guidance* — actuator/thruster/
    closed-loop-guidance channels are omitted by schema (mission-model.md §4; RFC-0001
    R3), and operational use is gated by the ``operational_targeting`` capability tag."""

    id: str
    frame: str
    segments: list[TrajectorySegment] = Field(default_factory=list)
    maneuvers: list[Maneuver] = Field(default_factory=list)
    reference_states: list[ReferenceState] = Field(default_factory=list)
    feasibility_margin: float | None = None
    provenance: Provenance | None = None
    frame_ref: ReferenceFrame | None = None  # typed sibling of `frame` (RFC-0007)


# --- mission / phase structure ---------------------------------------------------


class PhaseBoundary(_Model):
    """A phase entry/exit boundary: a descriptive ``condition`` and an optional reference
    to the boundary state (the handoff is the typed :class:`PhaseTransition` event)."""

    condition: str | None = None
    state_ref: str | None = None


class Leg(_Model):
    """A trajectory/maneuver-budget connection between phases (mission-model.md §1)."""

    id: str
    trajectory_ref: TrajectoryRef | None = None
    maneuver_budget: ManeuverBudget | None = None


class PhaseTransition(_Model):
    """The typed Environment-API handoff event carrying one phase's terminal state as the
    next phase's initial state (mission-model.md §2.2). **Reserved** — no P1 runtime
    consumes it; it is the schema hook the multi-regime track builds on in Phase 3."""

    from_phase: str
    to_phase: str
    terminal_state_ref: str | None = None
    initial_state_ref: str | None = None


class Phase(_Model):
    """One phase of a mission, in a single :class:`Regime`. ``environment_ref`` names its
    world (Worlds body) or free-space (Transit) environment; ``assets_active`` the fleet
    subset present; ``legs`` the trajectory/maneuver budgets connecting it onward."""

    id: str
    regime: Regime
    environment_ref: str | None = None
    assets_active: list[str] = Field(default_factory=list)
    entry: PhaseBoundary | None = None
    exit: PhaseBoundary | None = None
    campaign_ref: str | None = None
    legs: list[Leg] = Field(default_factory=list)


class MissionConstraints(_Model):
    """Global mission limits — budget, schedule, launch capacity — and the ``export_gated``
    flag marking a mission whose artifacts cross the design-time→operational line."""

    budget: float | None = None
    schedule_s: float | None = None
    launch_capacity_kg: float | None = None
    export_gated: bool = False


class MissionSpec(_Model):
    """A declarative mission: an ordered set of phases over a fleet, toward an objective,
    under global constraints. It **does not encode an optimization formulation** (RFC-0001
    R4) — it carries an ``objective_ref``, the ``fleet``, the ``phases`` (each with its
    per-leg budgets), and ``constraints``. A single-``surface``-phase mission is exactly a
    Phase-0 campaign (mission-model.md §5)."""

    id: str
    name: str
    description: str | None = None
    phases: list[Phase] = Field(default_factory=list)
    fleet: list[str] = Field(default_factory=list)
    objective_ref: str | None = None
    constraints: MissionConstraints | None = None
    provenance: Provenance | None = None


class MissionDocument(_Model):
    """Top-level mission document. ``mission_version`` pins the schema minor."""

    mission_version: Literal["0.1"]
    mission: MissionSpec
