"""Message-catalog v0.1 — closed vocabularies (Core-owned).

The closed enums of the runtime message vocabulary. Like the SADF vocabularies
(``astro_mine.core.sadf.enums``) these grow only by RFC: members are append-only and
never removed or repurposed (conventions.md §3).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ActionKind",
    "ChargeSource",
    "ContactConfidence",
    "ControlMode",
    "ExcavationPattern",
    "ExcavationTool",
    "NodeRole",
    "SampleMethod",
    "TaskKind",
]


class ActionKind(StrEnum):
    """The top-level kind of an :class:`~astro_mine.core.messages.model.Action`
    (policy.md §"composable sub-interfaces": controller / TAMP / mission)."""

    ACTUATOR = "actuator"  # low-level actuator setpoint
    MODE = "mode"  # named operating-mode command
    TASK = "task"  # high-level task directive (allocator/planner output)


class ControlMode(StrEnum):
    """Control mode of an actuator setpoint (the interpretation of ``setpoint``)."""

    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"  # torque/force
    IMPEDANCE = "impedance"  # stiffness/damping about a setpoint
    TRAJECTORY = "trajectory"  # follow a referenced trajectory


class TaskKind(StrEnum):
    """Typed task-directive kinds. Grows by RFC as autonomy components (Mind/Allocate)
    define the directives they emit (mind.md §1; allocate.md §3)."""

    GOTO = "goto"
    SAMPLE = "sample"
    EXCAVATE = "excavate"
    HAUL = "haul"
    DOCK = "dock"
    HOP = "hop"
    CHARGE = "charge"
    PROSPECT = "prospect"
    DEPLOY = "deploy"
    STANDBY = "standby"
    CUSTOM = "custom"  # carries a free directive + params


class SampleMethod(StrEnum):
    """Sample-collection method (SADF ``sample_collection.*`` capabilities)."""

    DRILL = "drill"
    SCOOP = "scoop"
    AUGER = "auger"
    PNEUMATIC = "pneumatic"
    CAPTURE_BAG = "capture_bag"


class ExcavationTool(StrEnum):
    """Excavation tool (SADF ``excavation.*`` capabilities)."""

    BUCKET = "bucket"
    AUGER = "auger"
    SCOOP = "scoop"
    DRILL = "drill"


class ExcavationPattern(StrEnum):
    """Excavation pattern for an excavate task (scenario §7)."""

    TRENCH = "trench"
    BENCH = "bench"
    SPIRAL = "spiral"


class ChargeSource(StrEnum):
    """Energy source for a charge task (scenario §10 power forks)."""

    SOLAR = "solar"
    RELAY = "relay"  # beamed/relayed power
    PLANT = "plant"
    FISSION = "fission"


class NodeRole(StrEnum):
    """Comms node role in a contact plan (link.md §6). Mirrors the SADF comms node
    role; carried here as the contact-graph node's role."""

    SPACE = "space"
    GROUND = "ground"


class ContactConfidence(StrEnum):
    """Qualitative confidence band for a predicted contact interval (link.md §3 —
    contacts carry margin + confidence; degrade loudly, never assume connected)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
