# SPDX-License-Identifier: Apache-2.0
"""Message-catalog v0.1 — typed Pydantic models (the runtime vocabulary).

The typed cross-component message vocabulary every plane exchanges: the per-tick
**observation** family (state, sensor, comms-mask) on the hot path, and the
**action** and **contact-plan** families on the control plane (system.md §3.4;
core.md §3). Co-designed with the Environment API (RM-P0-CORE-02): an
:class:`Observation` is exactly what ``step()`` returns and an :class:`Action` is what
it consumes.

Encoding follows latency class (conventions.md §3):

- **hot path / per-tick** (:class:`Observation`, :class:`StateSample`,
  :class:`SensorReading`, :class:`CommsObservationMask`) — canonical **Cap'n Proto**
  (``schema/observation.capnp``), zero-copy decode via
  :mod:`astro_mine.core.messages.hotpath`;
- **control plane** (:class:`ActionBatch`, :class:`ContactPlan`) — canonical **JSON
  Schema** (``schema/messages.schema.json``) + **Protobuf** wire form
  (:mod:`astro_mine.core.messages.wire`), the SADF pattern.

These models are **purely structural**; semantic/cross-reference checks live in
:mod:`astro_mine.core.messages.loader`. All quantities are SI. On the hot path,
frame identity and absolute time are the **typed** RM-P0-CORE-06 primitives:
:class:`StateSample.frame <StateSample>` carries a
:class:`~astro_mine.core.units.ReferenceFrame` and :class:`Observation.epoch <Observation>`
a :class:`~astro_mine.core.units.Epoch` (``tdb_seconds`` + ``scale``). Elapsed durations
(``sim_time_s``) stay bare SI seconds — they are intervals, not epochs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.messages.enums import (
    ActionKind,
    ChargeSource,
    ContactConfidence,
    ControlMode,
    ExcavationPattern,
    ExcavationTool,
    NodeRole,
    SampleMethod,
    TaskKind,
)
from astro_mine.core.units import Epoch, EpochWindow, ReferenceFrame

__all__ = [
    "Action",
    "ActionBatch",
    "ActuatorCommand",
    "ChargeTask",
    "CommsObservationMask",
    "ContactInterval",
    "ContactNode",
    "ContactPlan",
    "DockTask",
    "ExcavateTask",
    "GotoTask",
    "HaulTask",
    "HopTask",
    "LinkBudget",
    "ModeCommand",
    "Observation",
    "PeerLink",
    "ProspectTask",
    "Quat",
    "Route",
    "SampleTask",
    "SensorReading",
    "StateSample",
    "TaskDirective",
    "Transform",
    "Vec3",
    "Volume",
]


class _Model(BaseModel):
    """Base for every message model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


# --- value types (canonical frame/time types are RM-P0-CORE-06, in astro_mine.core.units)


class Vec3(_Model):
    x: float
    y: float
    z: float


class Quat(_Model):
    """Unit quaternion, scalar-last (x, y, z, w)."""

    x: float
    y: float
    z: float
    w: float


class Transform(_Model):
    """A rigid transform relative to a named frame (SI metres)."""

    translation_m: Vec3
    rotation_quat_xyzw: Quat


class Volume(_Model):
    """An axis-aligned box region in a named frame (SI metres) — a target region for
    excavate/prospect tasks."""

    frame: str
    center_m: Vec3
    dimensions_m: Vec3
    frame_ref: ReferenceFrame | None = None  # typed sibling of `frame` (RFC-0007)


# --- hot-path / per-tick observation family (Cap'n Proto) -------------------------


class StateSample(_Model):
    """A per-agent dynamical + system-state sample (one tick). The per-agent unit of
    both ``self_state`` and observed ``neighbors`` in an :class:`Observation`. ``frame``
    is the typed :class:`~astro_mine.core.units.ReferenceFrame` the ``pose`` resolves in."""

    agent_id: str
    frame: ReferenceFrame
    pose: Transform
    linear_velocity_mps: Vec3 | None = None
    angular_velocity_rps: Vec3 | None = None
    battery_soc_j: float | None = None
    temperature_k: float | None = None
    mode: str | None = None


class SensorReading(_Model):
    """One sensor's reading this tick. Renders an observation *of* a Prospect field —
    a value plus its realized likelihood — never a point ground-truth guess
    (prospect.md §6; sim.md §3). ``sensor`` resolves against the SADF sensor suite."""

    sensor: str
    values: list[float] = Field(default_factory=list)
    unit: str | None = None
    resource_species: str | None = None
    noise_sigma: float | None = None
    valid: bool = True


class PeerLink(_Model):
    """Connectivity to one peer this tick (an entry of a :class:`CommsObservationMask`)."""

    peer: str
    reachable: bool
    rate_bps: float | None = None
    latency_s: float | None = None
    margin_db: float | None = None


class CommsObservationMask(_Model):
    """Per-agent connectivity/observation mask for one tick — the constraint that makes
    coordination hard, applied through the Environment API (link.md §3; LUNAR-TR-003).
    ``earth_contact`` flags reachability to an Earth/DSN gateway this tick."""

    agent_id: str
    links: list[PeerLink] = Field(default_factory=list)
    earth_contact: bool = False


class Observation(_Model):
    """The per-tick, per-agent observation — what the Environment API's ``step()``
    returns (RM-P0-CORE-02). Partial by construction: ``observable`` flags whether the
    agent is observable this tick, and ``comms`` masks what it can exchange.

    Hot-path payload: encoded zero-copy via Cap'n Proto
    (:mod:`astro_mine.core.messages.hotpath`)."""

    tick: int
    sim_time_s: float
    agent_id: str
    observable: bool = True
    self_state: StateSample
    sensors: list[SensorReading] = Field(default_factory=list)
    comms: CommsObservationMask | None = None
    neighbors: list[StateSample] = Field(default_factory=list)
    epoch: Epoch | None = None


# --- control-plane: action family (Protobuf) -------------------------------------


class ActuatorCommand(_Model):
    """A low-level actuator setpoint. ``setpoint`` is interpreted per ``control_mode``;
    ``stiffness``/``damping`` apply to IMPEDANCE, ``trajectory_ref`` to TRAJECTORY."""

    target: str  # joint/actuator name (resolves against SADF)
    control_mode: ControlMode
    setpoint: list[float] = Field(default_factory=list)
    unit: str | None = None
    stiffness: float | None = None
    damping: float | None = None
    feedforward: list[float] = Field(default_factory=list)
    trajectory_ref: str | None = None


class ModeCommand(_Model):
    """A named operating-mode command (matches a SADF ``loads_by_mode`` mode)."""

    mode: str
    params: dict[str, str] = Field(default_factory=dict)


class GotoTask(_Model):
    target_frame: str
    target_pose: Transform
    position_tolerance_m: float | None = None
    heading_tolerance_rad: float | None = None
    max_speed_mps: float | None = None
    target_frame_ref: ReferenceFrame | None = None  # typed sibling of `target_frame` (RFC-0007)


class SampleTask(_Model):
    site_frame: str
    target_point_m: Vec3
    method: SampleMethod
    depth_m: float | None = None
    sample_mass_kg: float | None = None
    site_frame_ref: ReferenceFrame | None = None  # typed sibling of `site_frame` (RFC-0007)


class ExcavateTask(_Model):
    region: Volume
    tool: ExcavationTool
    pattern: ExcavationPattern
    target_volume_m3: float | None = None


class HaulTask(_Model):
    from_frame: str
    to_frame: str
    payload_kg: float | None = None
    resource_species: str | None = None
    from_frame_ref: ReferenceFrame | None = None  # typed sibling of `from_frame` (RFC-0007)
    to_frame_ref: ReferenceFrame | None = None  # typed sibling of `to_frame` (RFC-0007)


class DockTask(_Model):
    target_asset_id: str
    port_name: str | None = None
    approach_frame: str | None = None
    approach_frame_ref: ReferenceFrame | None = None  # typed sibling of `approach_frame` (RFC-0007)


class HopTask(_Model):
    launch_frame: str
    target_point_m: Vec3
    range_m: float | None = None
    apoapsis_m: float | None = None
    launch_frame_ref: ReferenceFrame | None = None  # typed sibling of `launch_frame` (RFC-0007)


class ChargeTask(_Model):
    source: ChargeSource
    target_soc_j: float | None = None
    max_power_w: float | None = None


class ProspectTask(_Model):
    region: Volume
    sensor_kinds: list[str] = Field(default_factory=list)
    info_gain_target: float | None = None


class TaskDirective(_Model):
    """A high-level task directive — a tagged union over :class:`~enums.TaskKind`.

    Exactly one typed task field SHOULD be set, matching ``task_kind`` (enforced in the
    loader). This is the allocator/planner output (allocate.md §3; mind.md §1); typed
    kinds grow by RFC as those components define what they emit."""

    task_kind: TaskKind
    priority: int = 0
    deadline_s: float | None = None
    goto: GotoTask | None = None
    sample: SampleTask | None = None
    excavate: ExcavateTask | None = None
    haul: HaulTask | None = None
    dock: DockTask | None = None
    hop: HopTask | None = None
    charge: ChargeTask | None = None
    prospect: ProspectTask | None = None
    directive: str | None = None  # for TaskKind.CUSTOM
    params: dict[str, str] = Field(default_factory=dict)  # for TaskKind.CUSTOM


class Action(_Model):
    """A single decision for one agent — a tagged union over :class:`~enums.ActionKind`.

    Exactly one of ``actuator``/``mode``/``task`` SHOULD be set, matching ``kind``
    (enforced in the loader). Composable across the policy sub-interfaces
    (controller / mission-and-task planner; policy.md §"composable sub-interfaces")."""

    agent_id: str
    kind: ActionKind
    sim_time_s: float | None = None
    actuator: ActuatorCommand | None = None
    mode: ModeCommand | None = None
    task: TaskDirective | None = None


class ActionBatch(_Model):
    """A batch of per-agent actions for one decision step (the multi-agent ``step()``
    input; RM-P0-CORE-02)."""

    actions: list[Action] = Field(default_factory=list)


# --- control-plane: contact-plan family (Protobuf) -------------------------------


class LinkBudget(_Model):
    """A parametric RF link-budget breakdown for a contact (link.md §3, CCSDS-aligned)."""

    eirp_dbw: float | None = None
    path_loss_db: float | None = None
    gt_db_per_k: float | None = None
    required_ebn0_db: float | None = None
    margin_db: float | None = None
    modcod: str | None = None


class ContactNode(_Model):
    """A node in the contact graph (a relay orbiter, ground station, or surface agent)."""

    id: str
    role: NodeRole
    kind: str | None = None  # free label: relay_orbiter | ground_station | surface_agent | ...


class ContactInterval(_Model):
    """A predicted contact window between two nodes (link.md §3). Carries rate, latency,
    margin, and confidence — never a bare boolean. Times are SI seconds past a SPICE
    TDB epoch (``epoch_tdb_s``; frame/epoch types in RM-P0-CORE-06)."""

    node_a: str
    node_b: str
    start_tdb_s: float
    end_tdb_s: float
    max_rate_bps: float | None = None
    min_latency_s: float | None = None
    mean_latency_s: float | None = None
    margin_db: float | None = None
    confidence: ContactConfidence = ContactConfidence.HIGH
    band: str | None = None
    modcod: str | None = None
    link_budget: LinkBudget | None = None
    window: EpochWindow | None = None  # typed sibling of start/end_tdb_s (RFC-0007)


class Route(_Model):
    """A multi-hop store-and-forward route over the contact graph (link.md §"DeliveryModel",
    CGR-style). ``hops`` is the ordered list of node ids from ``source`` to ``dest``."""

    source: str
    dest: str
    hops: list[str] = Field(default_factory=list)
    store_and_forward: bool = False
    earliest_delivery_tdb_s: float | None = None
    total_latency_s: float | None = None
    earliest_delivery: Epoch | None = None  # typed sibling of earliest_delivery_tdb_s (RFC-0007)


class ContactPlan(_Model):
    """A precomputed comms-availability product from Link (link.md §3): the contact
    graph (nodes), the predicted contact windows (intervals), and optional multi-hop
    delivery routes. Content-addressed and reproducible from pinned inputs (link.md §5)."""

    nodes: list[ContactNode] = Field(default_factory=list)
    intervals: list[ContactInterval] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    epoch_start_tdb_s: float | None = None
    epoch_end_tdb_s: float | None = None
    window: EpochWindow | None = None  # typed sibling of epoch_start/end_tdb_s (RFC-0007)
