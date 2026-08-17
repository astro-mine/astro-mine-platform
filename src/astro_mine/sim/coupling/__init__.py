# SPDX-License-Identifier: Apache-2.0
"""Multi-domain coupler — explicit co-simulation across engines (RM-P0-SIM-04).

The heterogeneous co-step the reduced-order engine set (RM-P0-SIM-03) was built for but could not
yet run: the stepping core drives **one** :class:`~astro_mine.sim.engines.RegimeEngine`, so a
scenario mixing an orbital relay, a surface rover, and a granular excavator had no way to advance
all three together. The :class:`CoupledEngine` is that composite — *itself* a ``RegimeEngine`` (so
it routes behind the same Core Environment waist, no new public surface), owning a set of labelled
sub-engines and co-stepping them with:

- **multi-rate sub-stepping** — each sub-engine advances over the macro step in its own
  number of sub-steps (orbital minutes, mobility seconds, granular sub-seconds; sim.md §4), declared
  per label in a :class:`CouplingSchedule`;
- **named coupling boundaries** (:class:`CouplingBoundary`) — a one-directional pose handoff
  of shared agents from a producer sub-engine to a consumer sub-engine after the macro step (the
  Jacobi co-simulation scheme: advance all, then exchange);
- **explicit frame / SI bridging** — a boundary exchange across two *different* reference frames
  (the relay orbiter propagates in inertial ``J2000``; the surface swarm lives in rotating
  body-fixed ``MOON_ME``) is resolved by a :class:`~astro_mine.sim.coupling.FrameBridge`. The
  default :class:`~astro_mine.sim.coupling.SpiceFrameBridge` resolves the rotating-frame rotation
  through **``astro-mine-spice``**
  ([RFC-0002](https://github.com/astro-mine/docs/blob/main/rfc/0002-shared-spice-foundation.md)) —
  the one SPICE realization the platform shares — so an orbital↔surface co-simulation is now
  expressible. SI is invariant across the platform, so only the rotation crosses (conventions.md
  §5). Degrade, don't lie: when the SPICE kernel pool cannot resolve the rotation (no orientation
  kernel furnished for the body), the boundary raises :class:`FrameBridgeError` naming both frames
  rather than silently mixing them;
- **tracked coupling residuals** (:class:`CouplingResidual`) — the pose discontinuity each
  boundary corrects, surfaced via :attr:`CoupledEngine.residuals` so a host can assert it stays
  bounded (sim.md §3, §11).

Multi-regime propagation (free-space ↔ proximity ↔ surface coupling across regimes) is the Phase-3
extension
([RFC-0001](https://github.com/astro-mine/docs/blob/main/rfc/0001-multi-regime-missions.md)); here
every sub-engine shares one body-fixed/inertial regime and the coupler is the same-body
co-simulation mechanism.

Backlog: RM-P0-SIM-04 -- astro-mine-sim#4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier
from astro_mine.core.units import J2000_EPOCH, Epoch
from astro_mine.sim.coupling._frames import FrameBridge, FrameBridgeError, SpiceFrameBridge
from astro_mine.sim.engines import (
    CouplingState,
    EngineDescriptor,
    FidelityDescriptor,
    RegimeEngine,
    brax_contact_engine_factory,
    dem_granular_engine_factory,
    granular_engine_factory,
    kinematic_engine_factory,
    manipulation_engine_factory,
    mjx_contact_engine_factory,
    mobility_engine_factory,
    mujoco_mobility_engine_factory,
    orbital_engine_factory,
    orekit_orbital_engine_factory,
)
from astro_mine.sim.engines._vecmath import norm, sub

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from astro_mine.core.messages.model import ActionBatch, StateSample
    from astro_mine.sim.engines.registry import EngineFactory
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import AgentSpec, Scenario

__all__ = [
    "COUPLED_ENGINE_NAME",
    "CoupledEngine",
    "CouplingBoundary",
    "CouplingResidual",
    "CouplingSchedule",
    "FrameBridge",
    "FrameBridgeError",
    "SpiceFrameBridge",
    "coupled_engine_factory",
]

#: The composite engine's name (rendered into its synthesized descriptor / manifest).
COUPLED_ENGINE_NAME = "astro-mine.sim.coupled"

#: Coarsest-to-finest fidelity order — the composite reports the *coarsest* tier among its
#: sub-engines (a chain is only as trustworthy as its weakest rung).
_TIER_ORDER: tuple[FidelityTier, ...] = (
    FidelityTier.MASSMODEL,
    FidelityTier.KINEMATIC,
    FidelityTier.ARTICULATED,
    FidelityTier.SURROGATE,
)


@dataclass(frozen=True, slots=True)
class CouplingBoundary:
    """A named, one-directional coupling boundary between two sub-engines.

    After the macro step, the ``producer`` sub-engine's **pose** for each agent in ``agents`` is
    handed to the ``consumer`` sub-engine (the dig site follows the moving excavator; the lander
    pose anchors the deployed rover). Only the pose crosses — battery, thermal, and mode stay each
    engine's own — and only if both engines resolve the agent in the same frame (else
    :class:`FrameBridgeError`). ``producer`` / ``consumer`` are
    sub-engine **labels** (the keys of :class:`CoupledEngine`'s engine map)."""

    name: str
    producer: str
    consumer: str
    agents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CouplingResidual:
    """The pose discontinuity a boundary exchange corrected this step.

    ``position_residual_m`` is the distance between the consumer's pre-exchange position and the
    producer's authoritative position — the co-simulation residual a host asserts stays
    inside a tolerance (sim.md §11). Zero means the engines already agreed."""

    boundary: str
    agent_id: str
    position_residual_m: float


@dataclass(frozen=True, slots=True)
class CouplingSchedule:
    """Per-label multi-rate sub-step counts.

    ``substeps[label]`` is how many equal sub-steps that sub-engine takes per macro ``dt`` (default
    1). A faster regime (granular, contact) declares more sub-steps than a slower one (orbital) so
    each integrates at a stable rate without forcing the whole co-sim to the
    finest rate."""

    substeps: Mapping[str, int]

    def for_label(self, label: str) -> int:
        """The sub-step count for ``label`` (default 1)."""
        return self.substeps.get(label, 1)


class CoupledEngine:
    """A composite :class:`~astro_mine.sim.engines.RegimeEngine` co-stepping sub-engines.

    Owns a label→sub-engine map and drives each through the same coupling triad it exposes upward,
    so the stepping core sees one engine. ``advance`` sub-steps each sub-engine at its scheduled
    rate, then performs the boundary pose handoffs and records their residuals. Determinism is the
    *weakest* of the sub-engines' classes; the reported fidelity is the
    coarsest tier."""

    def __init__(
        self,
        engines: Mapping[str, RegimeEngine],
        *,
        boundaries: Sequence[CouplingBoundary] = (),
        schedule: CouplingSchedule | None = None,
        frame_bridge: FrameBridge | None = None,
        start_epoch: Epoch = J2000_EPOCH,
    ) -> None:
        if not engines:
            raise ValueError("a coupled engine needs at least one sub-engine")
        self._engines: dict[str, RegimeEngine] = dict(engines)
        self._boundaries = tuple(boundaries)
        self._schedule = schedule or CouplingSchedule(substeps={})
        # The rotating-frame transform for a cross-frame boundary (RFC-0002). Defaults to the shared
        # SPICE realization; a host may inject any FrameBridge. ``start_epoch`` anchors the absolute
        # epoch the rotation is resolved *at* — a body-fixed frame's orientation is time-dependent,
        # so a boundary exchange is only meaningful against a real epoch, not elapsed seconds.
        self._frame_bridge: FrameBridge = frame_bridge or SpiceFrameBridge()
        self._start_epoch = start_epoch
        for boundary in self._boundaries:
            for label in (boundary.producer, boundary.consumer):
                if label not in self._engines:
                    raise ValueError(
                        f"coupling boundary {boundary.name!r} references unknown sub-engine "
                        f"{label!r} (known: {sorted(self._engines)})"
                    )
        for label, count in self._schedule.substeps.items():
            if label not in self._engines:
                raise ValueError(f"schedule references unknown sub-engine {label!r}")
            if count < 1:
                raise ValueError(f"sub-step count for {label!r} must be >= 1, got {count}")
        self._elapsed_s = 0.0
        self._residuals: tuple[CouplingResidual, ...] = ()
        self._descriptor = self._synthesize_descriptor()

    # --- RegimeEngine contract ---------------------------------------------------

    @property
    def descriptor(self) -> EngineDescriptor:
        return self._descriptor

    @property
    def engines(self) -> Mapping[str, RegimeEngine]:
        """The sub-engines actually built, by ``dynamics.kind`` label.

        Public because provenance reads it: a run's recorded engine versions must come from the
        engines that *ran*, not from the kinds the scenario declared (#65). The composite
        :attr:`descriptor` deliberately reports one synthesized identity, which is the wrong
        granularity for that record."""
        return dict(self._engines)

    @property
    def residuals(self) -> tuple[CouplingResidual, ...]:
        """The coupling residuals recorded at the last :meth:`advance` (empty before any)."""
        return self._residuals

    def apply_actions(self, actions: ActionBatch) -> None:
        """Fan the batch to every sub-engine; each actuates only the agents it owns."""
        for engine in self._engines.values():
            engine.apply_actions(actions)

    def advance(self, dt_s: float) -> None:
        """Advance every sub-engine over ``dt_s`` (each at its scheduled sub-rate), then run
        the boundary handoffs and record their residuals — the Jacobi co-simulation step."""
        for label, engine in self._engines.items():
            steps = self._schedule.for_label(label)
            h = dt_s / steps
            for _ in range(steps):
                engine.advance(h)
        self._residuals = self._exchange_boundaries(dt_s)
        self._elapsed_s += dt_s

    def export_coupling_state(self) -> CouplingState:
        """The union of the sub-engines' samples, deduplicated by agent id (first sub-engine
        in insertion order wins), at the coupler's macro elapsed time."""
        samples: list[StateSample] = []
        seen: set[str] = set()
        # Excavation rides the same union (#64): it is per-agent, no two sub-engines own the same
        # agent, so a digging engine contributes and everything else contributes nothing.
        excavated: dict[str, float] = {}
        for engine in self._engines.values():
            state = engine.export_coupling_state()
            excavated.update(state.excavated_kg)
            for sample in state.samples:
                if sample.agent_id in seen:
                    continue
                seen.add(sample.agent_id)
                samples.append(sample)
        return CouplingState(
            sim_time_s=self._elapsed_s, samples=tuple(samples), excavated_kg=excavated
        )

    def import_coupling_state(self, state: CouplingState) -> None:
        """Fan a boundary snapshot to every sub-engine (each ignores agents it does not own)
        and adopt its elapsed time."""
        for engine in self._engines.values():
            engine.import_coupling_state(state)
        self._elapsed_s = state.sim_time_s

    def retire(self, agent_ids: Iterable[str]) -> None:
        """Retire the agents from every sub-engine that owns them."""
        ids = tuple(agent_ids)
        for engine in self._engines.values():
            engine.retire(ids)

    # --- coupling internals ------------------------------------------------------

    @property
    def epoch(self) -> Epoch:
        """The absolute epoch the coupler has advanced to (``start_epoch + elapsed``).

        A body-fixed frame's orientation is time-dependent, so a cross-frame boundary exchange is
        resolved *at this epoch* — not at elapsed seconds."""
        return Epoch(
            tdb_seconds=self._start_epoch.tdb_seconds + self._elapsed_s,
            scale=self._start_epoch.scale,
        )

    def _exchange_boundaries(self, dt_s: float) -> tuple[CouplingResidual, ...]:
        residuals: list[CouplingResidual] = []
        # The sub-engines have already advanced over the macro step, so the state being exchanged is
        # valid at ``start + elapsed + dt`` — resolve the rotation there. (``_elapsed_s`` is
        # incremented *after* the exchange, and the handed CouplingState keeps stamping the
        # pre-increment ``sim_time_s`` exactly as before, so same-frame traces stay byte-identical.)
        epoch = Epoch(
            tdb_seconds=self._start_epoch.tdb_seconds + self._elapsed_s + dt_s,
            scale=self._start_epoch.scale,
        )
        for boundary in self._boundaries:
            producer = self._engines[boundary.producer].export_coupling_state().by_agent
            consumer_engine = self._engines[boundary.consumer]
            consumer = consumer_engine.export_coupling_state().by_agent
            handed: list[StateSample] = []
            for agent_id in boundary.agents:
                src = producer.get(agent_id)
                dst = consumer.get(agent_id)
                if src is None or dst is None:
                    continue  # the boundary's agent is not in both engines this step
                bridged = self._frame_bridge.bridge(src, dst.frame, epoch)
                residuals.append(
                    CouplingResidual(
                        boundary=boundary.name,
                        agent_id=agent_id,
                        position_residual_m=_position_residual(dst, bridged),
                    )
                )
                # Hand over the pose only — keep the consumer's own battery/thermal/mode.
                handed.append(dst.model_copy(update={"pose": bridged.pose}))
            if handed:
                consumer_engine.import_coupling_state(
                    CouplingState(sim_time_s=self._elapsed_s, samples=tuple(handed))
                )
        return tuple(residuals)

    def _synthesize_descriptor(self) -> EngineDescriptor:
        descriptors = [engine.descriptor for engine in self._engines.values()]
        regimes = sorted(
            {regime for d in descriptors for regime in d.regimes}, key=lambda r: r.value
        )
        frames = sorted(
            {frame.name: frame for d in descriptors for frame in d.frames}.values(),
            key=lambda f: f.name,
        )
        determinism = (
            DeterminismClass.TOLERANCE
            if any(d.determinism_class is DeterminismClass.TOLERANCE for d in descriptors)
            else DeterminismClass.BIT_EXACT
        )
        coarsest = min(
            (d.fidelity.tier for d in descriptors),
            key=_TIER_ORDER.index,
        )
        return EngineDescriptor(
            name=COUPLED_ENGINE_NAME,
            version="0.1.0",
            regimes=tuple(regimes),
            frames=tuple(frames),
            determinism_class=determinism,
            fidelity=FidelityDescriptor(tier=coarsest),
        )


def _position_residual(consumer: StateSample, producer: StateSample) -> float:
    a = consumer.pose.translation_m
    b = producer.pose.translation_m
    return norm(sub((a.x, a.y, a.z), (b.x, b.y, b.z)))


#: ``dynamics.kind`` -> the engine factory that owns that regime's agents (RM-P0-SIM-03).
_KIND_FACTORIES: dict[str, EngineFactory] = {
    "kinematic": kinematic_engine_factory,
    "orbital": orbital_engine_factory,
    "orekit_orbital": orekit_orbital_engine_factory,
    "mobility": mobility_engine_factory,
    "mujoco_mobility": mujoco_mobility_engine_factory,
    "manipulation": manipulation_engine_factory,
    "granular": granular_engine_factory,
    "dem_granular": dem_granular_engine_factory,
    "brax_contact": brax_contact_engine_factory,
    "mjx_contact": mjx_contact_engine_factory,
}


def coupled_engine_factory(
    *,
    boundaries: Sequence[CouplingBoundary] = (),
    schedule: CouplingSchedule | None = None,
    frame_bridge: FrameBridge | None = None,
) -> EngineFactory:
    """An :class:`~astro_mine.sim.engines.registry.EngineFactory` that co-steps a scenario's
    heterogeneous agents.

    It partitions the scenario's agents by ``dynamics.kind`` (in first-appearance order) and builds
    one sub-engine per kind from the matching RM-P0-SIM-03 factory, labelled by kind — so
    ``boundaries`` / ``schedule`` reference ``"orbital"`` / ``"mobility"`` / ``"granular"`` /
    ``"manipulation"`` / ``"kinematic"``. A homogeneous all-``kinematic`` scenario yields a single
    kinematic sub-engine and reproduces the reference stepping core byte-for-byte (CX-REPRO), so
    adopting the coupler never perturbs an existing trace.

    ``frame_bridge`` resolves a **cross-frame** boundary — the orbital↔surface handoff, where the
    producer propagates in inertial ``J2000`` and the consumer lives in body-fixed ``MOON_ME``. It
    defaults to the shared SPICE realization (:class:`SpiceFrameBridge`, RFC-0002); the coupler
    anchors it at the scenario's ``start_epoch``, since a body-fixed frame's orientation is
    time-dependent."""

    def build(scenario: Scenario, rng: RngStreams) -> CoupledEngine:
        order: list[str] = []
        groups: dict[str, list[AgentSpec]] = {}
        for spec in scenario.agents:
            kind = spec.dynamics.kind
            if kind not in groups:
                groups[kind] = []
                order.append(kind)
            groups[kind].append(spec)
        engines: dict[str, RegimeEngine] = {}
        for label in order:
            sub_scenario = scenario.model_copy(update={"agents": tuple(groups[label])})
            engines[label] = _KIND_FACTORIES[label](sub_scenario, rng)
        return CoupledEngine(
            engines,
            boundaries=boundaries,
            schedule=schedule,
            frame_bridge=frame_bridge,
            start_epoch=scenario.start_epoch,
        )

    return build
