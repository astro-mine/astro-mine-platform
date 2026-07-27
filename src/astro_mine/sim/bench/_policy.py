"""The anchor baseline — a capability-aware mode policy (astro-mine-sim#61, G1.3).

The floor a Bench leaderboard needs something to beat. It emits **one Core ``MODE`` action per
observed agent per tick**, choosing each agent's mode from that asset's own SADF: its capability
tags say what the asset is *for*, and its ``power.loads_by_mode`` says which mode names it actually
publishes a power draw for. Nothing here is hardcoded to the anchor's roster.

**Why a mode policy and not something cleverer.** Sim gates its reduced-order ISRU extraction, its
contact sensor, and its power/thermal draw on the agent's mode string, so a mode is the smallest
command that makes the shipped `RM-P1-SIM-02` machinery do anything at all. Bench's own
``BaselinePolicy`` emits ``MODE("prospect")`` for *every* agent — a mode five of the anchor's six
assets never declare, so the power model prices a tick the asset cannot be in. This picks a
declared mode per asset instead.

**Why its ``water_mass`` is what it is.** This policy once had to *refuse* to command an extraction
mode, because `IsruModel` was gated on the mode string alone — no dig target, no delivered
feedstock, no proximity check — so a one-word change produced a confident number that no excavation
and no haulage had earned. That refusal was a policy working around a physics gap.

Since #64 the gap is closed: extraction consumes regolith that was dug, carried and delivered, at
the grade of the ground it came from. The refusal is therefore gone, and a plant is commanded into
whatever mode its capabilities imply. What it *produces* is now decided by the world — and this
policy commands no motion, so on a scenario whose dig site and plant are kilometres apart nothing
is delivered and ``water_mass`` is still ``0.0``. That zero is now a physical fact about a swarm
that hauled nothing, not a guard rail. Making it non-zero is a job for a policy that drives.

This is a *replaceable example*, not a good policy (conventions.md §1.3): it does not plan,
allocate, navigate, or react to anything it observes. It is the conformance floor for the
Sim-backed path, the counterpart to Bench's ``BaselinePolicy`` on the fixture path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from astro_mine.core.messages.enums import (
    ActionKind,
    ExcavationPattern,
    ExcavationTool,
    TaskKind,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ExcavateTask,
    GotoTask,
    ModeCommand,
    Observation,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
    Volume,
)
from astro_mine.core.policy import DecisionContext
from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.core.sadf.model import Asset

__all__ = ["CapabilityModePolicy", "mode_for_asset", "mode_table"]

#: The fallback when an asset declares no mode this policy recognizes. Every anchor asset declares
#: it; an asset that does not falls back again to its first declared mode.
_IDLE = "idle"
#: The frame goto/excavate directives are expressed in — the body-fixed frame agents live in.
_BODY_FRAME = "MOON_ME"

#: Capability class -> the mode names that class implies, in preference order. Ordered by role, and
#: **prospecting outranks excavation deliberately**: the anchor's prospecting rover declares both
#: (its drill is an assay instrument), and a surveyor that spends the campaign in ``drill`` is not
#: what the scenario means by prospecting. Within a class, the first *declared* name wins.
_MODES_BY_CAPABILITY: tuple[tuple[frozenset[CapabilityTag], tuple[str, ...]], ...] = (
    (
        frozenset(
            {
                CapabilityTag.PROSPECTING_NEUTRON,
                CapabilityTag.PROSPECTING_NIR,
                CapabilityTag.PROSPECTING_GPR,
                CapabilityTag.PROSPECTING_MASS_SPEC,
                CapabilityTag.PROSPECTING_DRILL_ASSAY,
            }
        ),
        ("prospect", "survey", "perceive"),
    ),
    (
        frozenset(
            {
                CapabilityTag.EXCAVATION_BUCKET,
                CapabilityTag.EXCAVATION_AUGER,
                CapabilityTag.EXCAVATION_SCOOP,
                CapabilityTag.EXCAVATION_DRILL,
            }
        ),
        ("excavate", "dig", "drill"),
    ),
    (
        # An ISRU plant. Absent until #64: while `IsruModel` was gated on the mode string alone,
        # commanding a plant into `extract` manufactured water, so this policy deliberately refused
        # to — and the plant fell through to `idle`. Extraction now consumes delivered feedstock,
        # so commanding the mode is safe and *necessary*: an idle plant processes nothing however
        # much regolith is hauled to it.
        frozenset(
            {
                CapabilityTag.ISRU_THERMAL_EXTRACTION,
                CapabilityTag.ISRU_ELECTROLYSIS,
                CapabilityTag.ISRU_PURIFICATION,
                CapabilityTag.ISRU_STORAGE,
            }
        ),
        ("extract", "isru", "purify"),
    ),
    (
        frozenset(
            {
                CapabilityTag.COMMS_RELAY,
                CapabilityTag.COMMS_DIRECT_TO_EARTH,
                CapabilityTag.COMMS_DSN,
                CapabilityTag.COMMS_DTN,
            }
        ),
        ("downlink", "nominal"),
    ),
    (
        frozenset(
            {
                CapabilityTag.MOBILITY_WHEELED,
                CapabilityTag.MOBILITY_TRACKED,
                CapabilityTag.MOBILITY_LEGGED,
                CapabilityTag.MOBILITY_HOP,
            }
        ),
        ("drive_empty", "drive"),
    ),
)


def _declared_modes(asset: Asset) -> tuple[str, ...]:
    """The mode names ``asset`` publishes a power draw for, in declaration order (deduplicated)."""
    if asset.power is None or not asset.power.loads_by_mode:
        return ()
    seen: dict[str, None] = {}
    for load in asset.power.loads_by_mode:
        seen.setdefault(load.mode, None)
    return tuple(seen)


def mode_for_asset(asset: Asset) -> str:
    """The mode this baseline commands ``asset`` into — always one the asset declares.

    Picks the first declared mode implied by the asset's highest-ranked capability class, falling
    back to ``idle`` and then to whatever it declares first.

    **An ISRU plant is no longer excluded from extraction modes.** It used to be: `IsruModel` was
    gated on the mode string alone, so commanding `extract` manufactured water that no excavation
    had earned, and this function refused to do it. That was a policy working around a physics gap.
    Since #64 the gap is closed — extraction consumes delivered feedstock — so the exclusion is
    gone and a plant may be commanded into the mode it declares. It still produces nothing until
    something digs and hauls to it, which is now a fact about the world rather than about the
    policy.
    """
    declared = _declared_modes(asset)
    if not declared:
        return _IDLE
    capabilities = set(asset.capabilities)
    for tags, candidates in _MODES_BY_CAPABILITY:
        if not capabilities & tags:
            continue
        for candidate in candidates:
            if candidate in declared:
                return candidate
    return _IDLE if _IDLE in declared else declared[0]


def mode_table(assets: Mapping[str, Asset]) -> dict[str, str]:
    """``agent_id -> mode`` for a resolved roster.

    The agent id is the asset's SADF ``identity.id`` — the same string Sim uses as ``agent_id``."""
    return {agent_id: mode_for_asset(asset) for agent_id, asset in assets.items()}


@dataclass(frozen=True, slots=True)
class CapabilityModePolicy:
    """A Core :class:`~astro_mine.core.policy.Policy` that holds each agent in its capability mode.

    Deterministic and stateless: the same observations always produce the same batch, and actions
    are emitted in sorted ``agent_id`` order so a batch does not depend on observation iteration
    order. An agent absent from ``modes`` — one the scenario spawned but the roster did not pin — is
    commanded to ``idle`` rather than skipped, so every observed agent gets a decision.
    """

    modes: Mapping[str, str]

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        return ActionBatch(
            actions=[
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.MODE,
                    mode=ModeCommand(mode=self.modes.get(agent_id, _IDLE)),
                )
                for agent_id in sorted(observations)
            ]
        )

    def modes_for(self, agent_ids: Iterable[str]) -> dict[str, str]:
        """The modes this policy would command — for provenance and tests, not the decide path."""
        return {agent_id: self.modes.get(agent_id, _IDLE) for agent_id in agent_ids}


#: How close a carrier must get to its destination before it turns around. Larger than the
#: logistics transfer radius so a carrier that has arrived stays put long enough to load or unload
#: rather than oscillating on the boundary.
_ARRIVAL_RADIUS_M = 40.0


@dataclass
class ValueChainPolicy:
    """A baseline that actually runs the value chain: dig, haul, extract (#64).

    :class:`CapabilityModePolicy` commands one mode per asset and nothing else, so no surface asset
    ever moves and no regolith ever reaches the plant — on the anchor the dig site and the plant are
    ~12 km apart. A mode-only policy therefore scores ``water_mass = 0.0`` no matter how the physics
    is coupled, which makes it a poor floor for a benchmark whose headline metric is water.

    This policy adds the two commands the chain needs and nothing more:

    - **Diggers** get a continuous ``excavate`` task, so the granular engine removes regolith.
      Without a task the engine has no dig target and idles however the mode is set.
    - **Carriers** get a ``goto`` shuttling between a digger and a plant, flipping destination on
      arrival. That is the whole logistics policy — no routing, no scheduling, no queueing.

    Everything else keeps its capability mode. It is a **floor, not a good policy**: the shuttle
    ignores how full the cargo is, which plant is nearest, and whether the trip is worth taking. A
    submission that plans its haulage properly should beat it comfortably, which is the point of a
    baseline.

    Deterministic: destinations are a pure function of the observed poses, so the same observations
    always produce the same batch, and actions are emitted in sorted ``agent_id`` order.
    """

    modes: Mapping[str, str]
    diggers: frozenset[str] = frozenset()
    carriers: frozenset[str] = frozenset()
    plants: frozenset[str] = frozenset()
    #: Per-carrier destination, carried across ticks. Derived from observed geometry, never random.
    _destination: dict[str, str] = field(default_factory=dict)

    def _pose(self, observations: Mapping[str, Observation], agent_id: str) -> Vec3 | None:
        observation = observations.get(agent_id)
        return None if observation is None else observation.self_state.pose.translation_m

    def _retarget(self, observations: Mapping[str, Observation], carrier: str) -> str | None:
        """Where this carrier should head next — a plant when loaded up, a digger when empty.

        Cargo is not observable (it is Sim-internal), so the shuttle flips on *arrival* instead:
        reaching a digger means the next stop is a plant, and vice versa. Coarse, deterministic,
        and enough to move material across a 12 km gap.
        """
        here = self._pose(observations, carrier)
        if here is None:
            return None
        current = self._destination.get(carrier)
        if current is not None:
            target = self._pose(observations, current)
            if target is not None and _distance(here, target) > _ARRIVAL_RADIUS_M:
                return current  # still en route
            # Arrived: swap ends of the chain.
            nxt = sorted(self.plants) if current in self.diggers else sorted(self.diggers)
            return nxt[0] if nxt else current
        first = sorted(self.diggers) or sorted(self.plants)
        return first[0] if first else None

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        actions: list[Action] = []
        for agent_id in sorted(observations):
            if agent_id in self.diggers:
                actions.append(Action(agent_id=agent_id, kind=ActionKind.TASK, task=_dig_task()))
                continue
            if agent_id in self.carriers:
                destination = self._retarget(observations, agent_id)
                target = None if destination is None else self._pose(observations, destination)
                if destination is not None and target is not None:
                    self._destination[agent_id] = destination
                    actions.append(
                        Action(agent_id=agent_id, kind=ActionKind.TASK, task=_goto_task(target))
                    )
                    continue
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.MODE,
                    mode=ModeCommand(mode=self.modes.get(agent_id, _IDLE)),
                )
            )
        return ActionBatch(actions=actions)


def _dig_task() -> TaskDirective:
    """An open-ended excavate directive — dig until told otherwise (no target volume)."""
    return TaskDirective(
        task_kind=TaskKind.EXCAVATE,
        excavate=ExcavateTask(
            region=Volume(
                frame=_BODY_FRAME,
                center_m=Vec3(x=0.0, y=0.0, z=0.0),
                dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
            ),
            tool=ExcavationTool.BUCKET,
            pattern=ExcavationPattern.TRENCH,
            target_volume_m3=None,
        ),
    )


def _goto_task(target: Vec3) -> TaskDirective:
    return TaskDirective(
        task_kind=TaskKind.GOTO,
        goto=GotoTask(
            target_frame=_BODY_FRAME,
            target_pose=Transform(
                translation_m=target, rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
        ),
    )


def _distance(a: Vec3, b: Vec3) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
