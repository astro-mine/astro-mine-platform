"""The deterministic episode loop — Sim's Core Environment implementation (RM-P0-SIM-01).

:class:`Simulator` is Sim's reference stepping core: it loads a
:class:`~astro_mine.sim.runtime.scenario.Scenario`, seeds its
:class:`~astro_mine.sim.runtime.rng.RngStreams`, advances a
:class:`~astro_mine.sim.runtime.clock.SimClock`, and renders per-agent Core
:class:`~astro_mine.core.messages.model.Observation`\\ s through ``reset`` / ``step`` —
honoring the Core Environment API contract (RM-P0-CORE-02, asserted by
:func:`astro_mine.core.env.check_environment`), terminations and a shrinking active set
included.

Dynamics route entirely behind the public
:class:`~astro_mine.sim.engines.RegimeEngine` adapter (RM-P0-SIM-02): the Simulator is
engine-agnostic, building one per episode from an injectable ``engine_factory``
(the multi-domain :func:`~astro_mine.sim.coupling.coupled_engine_factory` by default),
advancing it through the coupling triad and *projecting* its coupling state onto Core
observations — so no engine *type* leaks past the Environment surface. Termination is the
Simulator's policy (each agent's battery floor, from the scenario), not the engine's. The
action batch is actuated into the engine each step via ``apply_actions`` (RM-P0-SIM-03)
before the advance; the environment is reward-free by default (scoring is trace-based via
ObjectiveSpec/Bench).

:func:`run_episode` rolls a whole scenario into a canonical, serializable :class:`Trace`
whose :attr:`Trace.content_hash` is the single byte-for-byte reproducibility key the
CI determinism gate (RM-P0-SIM-10) checks and the MCAP recording (RM-P0-SIM-09) carries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

from astro_mine.core.env import ResetResult, StepResult
from astro_mine.core.messages.model import ActionBatch, Observation, SensorReading, StateSample
from astro_mine.core.policy import DecisionContext
from astro_mine.sim.comms import apply_comms_mask
from astro_mine.sim.coupling import coupled_engine_factory
from astro_mine.sim.engines import EngineFactory, RegimeEngine
from astro_mine.sim.isru import IsruModel, IsruState
from astro_mine.sim.logistics import DEFAULT_TRANSFER_RADIUS_M, Material, transfer
from astro_mine.sim.power_thermal import (
    PowerThermalModel,
    PowerThermalState,
    ReferenceWorldProvider,
    default_initial_temperature,
)
from astro_mine.sim.runtime.clock import SimClock
from astro_mine.sim.runtime.content import UnresolvedProvider
from astro_mine.sim.runtime.provenance import engine_versions, scenario_digest
from astro_mine.sim.runtime.rng import RngStreams
from astro_mine.sim.runtime.scenario import Scenario
from astro_mine.sim.runtime.timing import TimingRecorder, timed_engine_factory
from astro_mine.sim.scheduler import Scheduler
from astro_mine.sim.sensors import render_sensor

if TYPE_CHECKING:  # static proof that Simulator satisfies the Core Environment contract
    from collections.abc import Callable, Iterator, Sequence

    from astro_mine.core.env import Environment
    from astro_mine.core.policy import Policy
    from astro_mine.core.resource import ResourceField
    from astro_mine.core.sadf.model import Sensor
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.comms import ConnectivitySource
    from astro_mine.sim.engines import CouplingState

__all__ = ["CORE_INTERFACES", "Simulator", "Trace", "run_episode"]


def _distance(a: Any, b: Any) -> float:
    """Straight-line distance between two poses' translations, or infinity if either is absent."""
    if a is None or b is None:
        return math.inf
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


#: The Core interfaces the stepping core is built against — proven by
#: ``assert_core_compatible`` (RM-P0-CORE-07) and stamped into a :class:`Trace`'s
#: provenance. Units/frames/time are shared *primitives*, not version-negotiated, so absent.
CORE_INTERFACES: dict[str, str] = {"env": "0.1.0", "messages": "0.1.0"}


class Simulator:
    """A deterministic scenario runtime implementing the Core Environment API.

    The stepping core is engine-agnostic: it drives a
    :class:`~astro_mine.sim.engines.RegimeEngine` built per-episode by ``engine_factory``
    (the multi-domain :func:`~astro_mine.sim.coupling.coupled_engine_factory` by default),
    advances the clock, integrates dynamics behind the engine adapter, projects the
    engine's coupling state onto Core observations, and applies the battery-floor
    termination policy. No engine *type* leaks past the Environment surface."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        engine_factory: EngineFactory | None = None,
        resource_field: ResourceField | None = None,
        world_provider: WorldProvider | None = None,
        connectivity: ConnectivitySource | None = None,
    ) -> None:
        self._scenario = scenario
        # The multi-domain coupler is the default, not the kinematic reference engine (#65). It
        # partitions agents by their declared `dynamics.kind` and builds the matching RM-P0-SIM-03
        # engine for each, so an orbiter really propagates and an excavator really digs. For a
        # homogeneous all-kinematic scenario it yields a single kinematic sub-engine and reproduces
        # the reference stepping core byte-for-byte, so adopting it never perturbs an existing trace
        # (asserted in tests/test_coupling.py). Before this, the per-asset dynamics a scenario
        # declared were inferred, stamped into provenance, and then never executed.
        self._engine_factory = engine_factory or coupled_engine_factory()
        self._resource_field = resource_field
        self._world_provider = world_provider or ReferenceWorldProvider()
        self._possible_agents = tuple(a.agent_id for a in scenario.agents)
        # Comms masking (RM-P0-SIM-08): only agents that are contact-graph nodes carry a per-tick
        # CommsObservationMask; an agent with no modeled comms (or no ContactPlan at all) keeps
        # ``comms`` unset, so scenarios without connectivity produce byte-identical traces.
        #
        # The binding is an exact id match, and that is the whole convention (issue #53): a
        # contact-graph node that *is* a fleet asset is named by its SADF ``identity.id`` — the
        # same string Bench pins as content and Sim uses as ``agent_id`` — so a node and an agent
        # name the same robot. Nodes that are not agents (DSN ground stations, a relay outside the
        # fleet) simply do not bind, and agents that are not nodes (a lander with no modeled radio)
        # keep ``comms`` unset. No aliasing, no suffix matching: two vocabularies that must agree,
        # checked here.
        self._connectivity = connectivity
        self._comms_agents: frozenset[str] = (
            frozenset(connectivity.nodes) & frozenset(self._possible_agents)
            if connectivity is not None
            else frozenset()
        )
        if connectivity is not None and not self._comms_agents:
            # Fail loud. A ContactPlan whose nodes name no agent masks nothing, so the run would
            # score *not applicable* for comms_robustness while looking perfectly healthy — the
            # silent-comms-blind bug of #53. An empty intersection is always a vocabulary error
            # (a plan authored against a different id namespace), never a legitimate scenario:
            # a caller who wants no masking passes no ConnectivitySource at all. Link takes the
            # same stance from the producer side — "a resolver must never fall back to an empty
            # (fully connected, or fully denied) comms model".
            raise ValueError(
                "the ContactPlan's nodes name none of this scenario's agents, so no observation "
                "would be masked. A contact node that is a fleet asset must be named by its SADF "
                "identity.id — the same id Sim uses as agent_id.\n"
                f"  plan nodes: {sorted(connectivity.nodes)}\n"
                f"  scenario agents: {sorted(self._possible_agents)}\n"
                "Pass connectivity=None for an unmasked run."
            )
        self._battery_floor = {a.agent_id: a.battery_floor_j for a in scenario.agents}
        self._sensors: dict[str, tuple[Sensor, ...]] = {
            a.agent_id: a.sensors for a in scenario.agents
        }
        # Power/thermal evolution (RM-P0-SIM-07) is engaged per-agent only when a PowerBudget
        # is declared; an un-budgeted agent keeps its engine's placeholder battery draw, so
        # existing scenarios evolve exactly as before.
        self._pt_models: dict[str, PowerThermalModel] = {}
        self._pt_initial: dict[str, tuple[float, float | None]] = {}
        for agent in scenario.agents:
            if agent.power is None:
                continue
            self._pt_models[agent.agent_id] = PowerThermalModel(agent.power, agent.thermal)
            temperature = (
                agent.initial_temperature_k
                if agent.initial_temperature_k is not None
                else default_initial_temperature(agent.thermal)
            )
            self._pt_initial[agent.agent_id] = (agent.battery_soc_j, temperature)
        self._pt_state: dict[str, PowerThermalState] = {}
        # ISRU extraction/storage (RM-P1-SIM-02): engaged per-agent only when an ``isru`` block is
        # declared; the stored-water mass + extraction energy are read by a ``resource_storage``
        # sensor. An un-declared agent has no ISRU state (byte-identical to before).
        self._isru_models: dict[str, IsruModel] = {}
        self._isru_nominal: dict[str, float] = {}
        for agent in scenario.agents:
            if agent.isru is None:
                continue
            self._isru_models[agent.agent_id] = IsruModel(
                extraction_rate_kg_s=agent.isru.extraction_rate_kg_s,
                specific_energy_j_per_kg=agent.isru.specific_energy_j_per_kg,
                capacity_kg=agent.isru.capacity_kg,
                extraction_modes=frozenset(agent.isru.extraction_modes),
            )
            self._isru_nominal[agent.agent_id] = agent.isru.nominal_abundance
        self._isru_state: dict[str, IsruState] = {}
        # The value chain (#64): where material sits, and how much each engine has dug so far. The
        # stores are per-agent and role-free — an excavator's stockpile, a hauler's cargo and a
        # plant's feedstock are the same kind of thing in different places, which is why one map
        # serves all three. `_cargo_capacity` marks who may carry, from the SADF payload slots that
        # accept regolith; `None` means "not a carrier" and never "unbounded".
        self._material: defaultdict[str, Material] = defaultdict(Material)
        self._excavated_seen: dict[str, float] = {}
        self._cargo_capacity: dict[str, float | None] = {
            a.agent_id: a.cargo_capacity_kg for a in scenario.agents
        }
        self._clock = SimClock(start_epoch=scenario.start_epoch, dt_s=scenario.dt_s)
        self._engine: RegimeEngine | None = None
        self._rng: RngStreams | None = None
        self._active: tuple[str, ...] = ()

    @property
    def engine(self) -> RegimeEngine:
        """The engine built for the current episode — the source of a run's engine provenance (#65).

        Available only after :meth:`reset`, because the engine is built from the seeded RNG. A run's
        recorded engine versions read from *this*, not from the kinds the scenario declared.
        """
        if self._engine is None:
            raise RuntimeError("no engine yet: call reset() before reading the episode's engine")
        return self._engine

    @property
    def possible_agents(self) -> tuple[str, ...]:
        """Every agent id that may appear in this environment."""
        return self._possible_agents

    @property
    def agents(self) -> tuple[str, ...]:
        """Agent ids currently active — shrinks as agents terminate at the battery floor."""
        return self._active

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        """Reset the episode (re-seeding, rebuilding the engine, and rewinding the clock)
        and return the initial per-agent observations. ``seed`` overrides the scenario
        default when given."""
        effective_seed = self._scenario.seed if seed is None else seed
        self._rng = RngStreams(effective_seed)
        self._engine = self._engine_factory(self._scenario, self._rng)
        self._clock = SimClock(start_epoch=self._scenario.start_epoch, dt_s=self._scenario.dt_s)
        self._active = self._possible_agents
        self._pt_state = {
            agent_id: PowerThermalState(soc_j=soc, temperature_k=temperature)
            for agent_id, (soc, temperature) in self._pt_initial.items()
        }
        self._isru_state = {aid: model.initial_state() for aid, model in self._isru_models.items()}
        self._material = defaultdict(Material)
        self._excavated_seen = {}
        coupling = self._engine.export_coupling_state()
        samples = self._apply_power_thermal(coupling.samples, 0.0)
        self._apply_logistics(coupling, samples, 0.0)
        self._apply_isru(samples, 0.0)
        observations = {s.agent_id: self._observe(s) for s in samples}
        return ResetResult(observations=observations)

    def step(self, actions: ActionBatch) -> StepResult:
        """Advance one tick. The actions are actuated into the engine (``apply_actions``,
        RM-P0-SIM-03), the clock advances, the engine integrates every live agent behind the
        adapter under those commands, and any agent at its battery floor terminates and
        leaves the active set (the Gym/PettingZoo terminal pattern)."""
        engine = self._engine
        if engine is None:
            raise RuntimeError("reset() must be called before step()")
        self._clock = self._clock.advanced()
        dt_s = self._clock.dt_s
        engine.apply_actions(actions)
        engine.advance(dt_s)
        coupling = engine.export_coupling_state()
        samples = self._apply_power_thermal(coupling.samples, dt_s)  # post-advance, pre-retire
        self._apply_logistics(coupling, samples, dt_s)
        self._apply_isru(samples, dt_s)
        observations = {s.agent_id: self._observe(s) for s in samples}
        truncated = self._clock.tick >= self._scenario.horizon_steps
        terminations = {
            s.agent_id: s.battery_soc_j is not None
            and s.battery_soc_j <= self._battery_floor[s.agent_id]
            for s in samples
        }
        truncations = {s.agent_id: truncated for s in samples}
        retired = [aid for aid, done in terminations.items() if done]
        engine.retire(retired)
        self._active = tuple(aid for aid in self._active if aid not in retired)
        return StepResult(
            observations=observations,
            sim_time_s=self._clock.sim_time_s,
            terminations=terminations,
            truncations=truncations,
            dt_s=dt_s,
        )

    def _observe(self, sample: StateSample) -> Observation:
        # Project the engine's (richer) coupling sample onto the observation surface: the
        # agent perceives its own system state — pose, battery, temperature (set by
        # power/thermal, RM-P0-SIM-07), and mode — but not the coupling-only velocity the
        # coupler exchanges across boundaries (that is proprioceptive, via an odometry sensor).
        self_state = StateSample(
            agent_id=sample.agent_id,
            frame=sample.frame,
            pose=sample.pose,
            battery_soc_j=sample.battery_soc_j,
            temperature_k=sample.temperature_k,
            mode=sample.mode,
        )
        epoch = self._clock.now_epoch()
        observation = Observation(
            tick=self._clock.tick,
            sim_time_s=self._clock.sim_time_s,
            agent_id=sample.agent_id,
            self_state=self_state,
            sensors=self._render_sensors(sample),
            epoch=epoch,
        )
        # Apply Link connectivity as a per-tick comms mask (RM-P0-SIM-08): a contact-graph agent
        # sees its reachable/denied peers this tick, so a policy cannot observe or message an
        # unreachable peer. Agents outside the plan pass through with ``comms`` unset.
        if sample.agent_id in self._comms_agents:
            assert self._connectivity is not None  # _comms_agents is empty when no source
            mask = self._connectivity.comms_mask(sample.agent_id, epoch)
            observation = apply_comms_mask(observation, mask)
        return observation

    def _render_sensors(self, sample: StateSample) -> list[SensorReading]:
        """Render the agent's SADF sensor suite into per-tick readings (RM-P0-SIM-06).

        Each sensor draws from a seeded per-(agent, sensor) stream so readings reproduce;
        resource sensors sample the injected (sealed) field, proprioceptive sensors render
        from the agent's own state, and *framing* sensors (imaging) resolve their footprint's
        terrain + illumination against the world provider. An agent with no declared sensors
        reports none."""
        sensors = self._sensors.get(sample.agent_id, ())
        if not sensors:
            return []
        assert self._rng is not None  # _observe runs only after reset() seeds the streams
        isru = self._isru_state.get(sample.agent_id)
        epoch = self._clock.now_epoch()
        readings: list[SensorReading] = []
        for sensor in sensors:
            stream = self._rng.stream(f"sensor:{sample.agent_id}:{sensor.name}")
            readings.append(
                render_sensor(
                    sensor,
                    sample,
                    self._resource_field,
                    stream,
                    isru=isru,
                    world=self._world_provider,
                    epoch=epoch,
                )
            )
        return readings

    def _grade_at(self, sample: StateSample) -> float:
        """The water-equivalent mass fraction of regolith dug at this agent's position.

        Sampled from the injected (sealed) resource field through the Core
        :class:`ResourceField` contract, never a Prospect import. This is the *only* place the
        field's abundance enters the value chain now: it is a property of the ground being dug, so
        it is read at the excavator and then travels with the material (#64).
        """
        if self._resource_field is None:
            return self._isru_nominal.get(sample.agent_id, 0.0)
        t = sample.pose.translation_m
        grade = self._resource_field.mean((t.x, t.y, t.z), epoch=self._clock.now_epoch())
        return max(0.0, min(1.0, grade))

    def _apply_logistics(
        self, coupling: CouplingState, samples: Sequence[StateSample], dt_s: float
    ) -> None:
        """Move regolith along the value chain: dig site -> hauler -> plant (#64).

        Three reduced-order steps, each gated on something physical:

        1. **Accrual.** The increment in an engine's cumulative excavated mass becomes material at
           the excavator's own stockpile, carrying the grade of the ground it came out of.
        2. **Delivery.** An ISRU plant draws from any holder within
           :data:`DEFAULT_TRANSFER_RADIUS_M` of it — usually a hauler that drove there, but a
           digger parked beside the plant feeds it directly, because requiring a carrier between
           two co-located assets would be a fiction of its own.
        3. **Pickup.** A carrier does the same, bounded by the cargo capacity its SADF declares.

        Distance is the whole gate. On the anchor the dig site and the plant are ~12 km apart, so
        nothing reaches the plant unless a hauler actually drove the distance; co-location is the
        degenerate case, not the design centre. There is no route planning and no queueing here —
        that is autonomy's, and the chain only has to be *physical*, not clever.
        """
        pose = {s.agent_id: s.pose.translation_m for s in samples}

        for sample in samples:
            total = coupling.excavated_kg.get(sample.agent_id)
            if total is None:
                continue
            dug = total - self._excavated_seen.get(sample.agent_id, 0.0)
            self._excavated_seen[sample.agent_id] = total
            if dug > 0.0:
                self._material[sample.agent_id] = self._material[sample.agent_id].blended_with(
                    Material(mass_kg=dug, water_fraction=self._grade_at(sample))
                )

        def _pull(sink: str, capacity: float | None, *, from_plants: bool = True) -> None:
            """Draw material into ``sink`` from every holder within transfer range of it.

            ``from_plants=False`` marks a plant as a **sink only**: feedstock that has been
            delivered has arrived, and a carrier parked beside a plant must not pick it back up.
            Without that, material ping-pongs between the two on consecutive ticks and never gets
            extracted — the plant pulls the cargo in, the carrier pulls it straight back out.
            """
            for source in sorted(self._material):
                if source == sink or self._material[source].mass_kg <= 0.0:
                    continue
                if not from_plants and source in self._isru_models:
                    continue
                if _distance(pose.get(sink), pose.get(source)) > DEFAULT_TRANSFER_RADIUS_M:
                    continue
                self._material[source], self._material[sink] = transfer(
                    self._material[source],
                    self._material[sink],
                    dt_s=dt_s,
                    sink_capacity_kg=capacity,
                )

        # Plants pull first, so material a carrier has just delivered is available to extract on
        # the same tick it arrives rather than a tick later.
        for plant in sorted(self._isru_models):
            _pull(plant, None)
        for carrier, capacity in sorted(self._cargo_capacity.items()):
            if capacity is not None:
                _pull(carrier, capacity, from_plants=False)

    def _apply_isru(self, samples: Sequence[StateSample], dt_s: float) -> None:
        """Evolve each ISRU agent's stored-water mass + extraction energy (RM-P1-SIM-02).

        Extraction converts the **feedstock delivered to the plant** — regolith some excavator dug
        and some hauler drove over — at the grade that material carries. It no longer samples the
        resource field under the plant's own footprint, which modelled a fixed plant as if it mined
        the ground it stands on and made ``water_mass`` a siting constant invariant to everything
        the swarm did (#64). The state is read by the agent's ``resource_storage`` sensor; agents
        with no ``isru`` block are untouched.
        """
        if not self._isru_models:
            return
        for sample in samples:
            model = self._isru_models.get(sample.agent_id)
            if model is None:
                continue
            self._isru_state[sample.agent_id], self._material[sample.agent_id] = model.step(
                self._isru_state[sample.agent_id],
                dt_s,
                sample.mode,
                self._material[sample.agent_id],
            )

    def _apply_power_thermal(
        self, samples: Sequence[StateSample], dt_s: float
    ) -> tuple[StateSample, ...]:
        """Evolve each budgeted agent's battery SoC and temperature (RM-P0-SIM-07) and write
        them onto its sample, so the observation and the battery-floor termination see the
        power/thermal state. Agents with no PowerBudget pass through untouched (their engine's
        placeholder draw stands), so existing scenarios are unaffected."""
        if not self._pt_models:
            return tuple(samples)
        epoch = self._clock.now_epoch()
        evolved: list[StateSample] = []
        for sample in samples:
            model = self._pt_models.get(sample.agent_id)
            if model is None:
                evolved.append(sample)
                continue
            translation = sample.pose.translation_m
            surface = self._world_provider.sample(
                (translation.x, translation.y, translation.z), epoch=epoch
            )
            state = model.step(self._pt_state[sample.agent_id], dt_s, surface, sample.mode)
            self._pt_state[sample.agent_id] = state
            # Power/thermal owns the budgeted agent's battery, superseding the engine's
            # placeholder draw: a storage asset gets its evolved SoC, a loads-only asset its
            # externally-supplied (unchanging) SoC — never the engine's stand-in drain.
            evolved.append(
                sample.model_copy(
                    update={"battery_soc_j": state.soc_j, "temperature_k": state.temperature_k}
                )
            )
        return tuple(evolved)


@dataclass(frozen=True, slots=True)
class Trace:
    """A canonical, serializable record of a full episode — the reproducibility artifact.

    ``frames`` is the deterministic per-tick payload (reset frame plus one per step);
    ``provenance`` is the run envelope (seed, scenario, Core interface versions) that
    RM-P0-SIM-09 extends with input hashes, engine versions/tiers, and error-budget
    outcomes. :attr:`content_hash` over the whole record is the single key the determinism
    gate (RM-P0-SIM-10) compares and the MCAP recording (RM-P0-SIM-09) carries — there is
    one canonical artifact, not two.

    ``timing`` is the run's measured wall-clock (:mod:`astro_mine.sim.runtime.timing`) — the
    real-time factor sim.md §10 calls for, and the numerator of the surrogate speedup claim
    (surrogate.md §8). It is deliberately **outside** the reproducibility key: it is not a
    field of :meth:`to_canonical_json`, and it is declared ``compare=False`` so two traces of
    the same seeded run stay equal however long each took. Wall-clock is non-deterministic by
    nature; a hash that moved with it would fail the determinism gate (RM-P0-SIM-10) on every
    run. Recorded for audit, excluded from the content hash — exactly as the environment stamp
    is (:func:`astro_mine.sim.recording.run_provenance`)."""

    scenario_name: str
    seed: int
    frames: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any]
    timing: Mapping[str, Any] | None = dataclass_field(default=None, compare=False)

    def to_canonical_json(self) -> str:
        """A stable, sorted, compact JSON serialization for byte-equality comparison.

        The fields are enumerated, not reflected off the dataclass — so a non-deterministic
        field (``timing``) cannot leak into the determinism key by being added to the class."""
        return json.dumps(
            {
                "provenance": dict(self.provenance),
                "scenario": self.scenario_name,
                "seed": self.seed,
                "frames": list(self.frames),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def content_hash(self) -> str:
        """The SHA-256 of the canonical serialization — the determinism-gate key."""
        return hashlib.sha256(self.to_canonical_json().encode()).hexdigest()


def run_episode(
    scenario: Scenario,
    *,
    seed: int | None = None,
    resource_field: ResourceField | None = None,
    world_provider: WorldProvider | None = None,
    connectivity: ConnectivitySource | None = None,
    policy: Policy | None = None,
    engine_factory: EngineFactory | None = None,
    content_hashes: Mapping[str, str] | None = None,
    unresolved: Sequence[UnresolvedProvider] = (),
    on_frame: Callable[[dict[str, Any]], None] | None = None,
    timing: TimingRecorder | None = None,
) -> Trace:
    """Run ``scenario`` to its horizon and capture the canonical :class:`Trace`.

    ``seed`` overrides the scenario default; ``resource_field`` is the (sealed) field the
    agents' resource sensors observe (RM-P0-SIM-06); ``connectivity`` is the (optional) Link
    connectivity source whose per-tick :class:`CommsObservationMask` masks each agent's
    observation (RM-P0-SIM-08). Two calls with the same effective seed produce byte-identical
    traces (equal :attr:`Trace.content_hash`).

    ``world_provider`` is the (optionally content-resolved) Core
    :class:`~astro_mine.core.world.WorldProvider` the run's power/thermal evolution (RM-P0-SIM-07)
    and framing sensors (RM-P0-SIM-06) sample — the *pinned world*, when a scenario resolves one
    (RM-P1-SIM-01). Omitted, the Simulator falls back to its analytic reference world, exactly as
    before.

    ``policy`` (a Core :class:`~astro_mine.core.policy.Policy`) closes the decision loop — each
    step is ``policy.decide(observations, context)`` (RM-P0-SIM-11); omitted, the environment is
    stepped open-loop. ``engine_factory`` selects the regime engine(s) (default: the reference
    kinematic engine; the anchor uses the multi-domain coupler). ``content_hashes`` (the
    ``id -> sha256:`` map a :class:`~astro_mine.sim.runtime.content.ContentResolver` returns) is
    folded into the run provenance so a content-resolved run is byte-addressed to its pinned
    bundles (RM-P1-SIM-01).

    ``on_frame``, if given, is invoked with each canonical frame *as it is produced* — the
    one seam that makes headless and interactive a single runtime (sim.md §2.6): a headless
    run passes none; an interactive/recording run (RM-P0-SIM-09) passes a sink (a live view,
    an MCAP writer) without forking the stepping loop.

    ``timing`` is the wall-clock sink (:class:`~astro_mine.sim.runtime.timing.TimingRecorder`).
    Every run is timed — the real-time factor is an always-on observability signal (sim.md §10),
    and the two ``perf_counter_ns`` reads per step are free against the physics — so the
    returned :attr:`Trace.timing` is always populated. Pass a recorder to *keep* the handle
    (a caller comparing tiers needs both runs' numbers; the speedup runner does exactly this);
    omit it and one is made per run. Timing never enters :attr:`Trace.content_hash`."""
    effective_seed = scenario.seed if seed is None else seed
    recorder = TimingRecorder() if timing is None else timing
    sim = Simulator(
        scenario,
        engine_factory=timed_engine_factory(engine_factory or coupled_engine_factory(), recorder),
        resource_field=resource_field,
        world_provider=world_provider,
        connectivity=connectivity,
    )
    frames: list[dict[str, Any]] = []
    for frame in _iter_frames(sim, scenario, effective_seed, policy):
        if on_frame is not None:
            on_frame(frame)
        frames.append(frame)
    return Trace(
        scenario_name=scenario.name,
        seed=effective_seed,
        frames=tuple(frames),
        provenance=_provenance(
            scenario,
            effective_seed,
            sim.engine,
            resource_field,
            connectivity,
            content_hashes,
            unresolved,
        ),
        timing=recorder.as_provenance(),
    )


def _iter_frames(
    sim: Simulator, scenario: Scenario, effective_seed: int, policy: Policy | None = None
) -> Iterator[dict[str, Any]]:
    """Yield the canonical frames of one episode — the reset frame then one per step.

    The single source of the per-tick stream both the batch :class:`Trace` and the streaming
    MCAP recorder consume, so headless and interactive never diverge (sim.md §2.6). When a Core
    :class:`~astro_mine.core.policy.Policy` is injected it closes the loop —
    ``sim.step(policy.decide(observations, context))`` — feeding each step's observations to the
    next decision (RM-P0-SIM-11); with no policy the environment is stepped open-loop (an empty
    batch), the byte-identical default for existing scenarios. The policy is a Core contract, so
    Sim drives any planner without importing it (conventions.md §1.1)."""
    reset = sim.reset(seed=effective_seed)
    yield _reset_frame(reset)
    observations = reset.observations
    sim_time_s = 0.0
    for _ in range(scenario.horizon_steps):
        if policy is None:
            actions = ActionBatch()
        else:
            context = DecisionContext(sim_time_s=sim_time_s, seed=effective_seed)
            actions = policy.decide(observations, context)
        result = sim.step(actions)
        observations = result.observations
        sim_time_s = result.sim_time_s
        yield _step_frame(result)


def _provenance(
    scenario: Scenario,
    seed: int,
    engine: RegimeEngine,
    resource_field: ResourceField | None = None,
    connectivity: ConnectivitySource | None = None,
    content_hashes: Mapping[str, str] | None = None,
    unresolved: Sequence[UnresolvedProvider] = (),
) -> dict[str, Any]:
    # The reproducibility envelope: SIM-01's seed/scenario/interface fields, extended by
    # RM-P0-SIM-09 with the input content hashes and engine versions that make a run
    # reproducible (the MCAP recording carries the same Trace.content_hash). Every value is
    # deterministic across same-seed runs, so it stays inside the determinism hash. The
    # multi-fidelity scheduler (RM-P0-SIM-05) records the per-agent tier selection +
    # implied-error outcome — the "error-budget outcomes" RM-P0-SIM-09 surfaces.
    fidelity = Scheduler(scenario.fidelity).resolve(scenario)
    source_content_hashes = {"scenario": scenario_digest(scenario)}
    resource_hash = getattr(resource_field, "content_hash", None)
    if isinstance(resource_hash, str):
        source_content_hashes["resource_field"] = resource_hash
    # The ContactPlan is a reproducibility-relevant input (RM-P0-SIM-08), so its content hash rides
    # in provenance when the source exposes one (Sim's ReferenceConnectivitySampler does); a source
    # without one is simply absent, exactly like an un-hashed resource field.
    contact_plan_hash = getattr(connectivity, "content_hash", None)
    if isinstance(contact_plan_hash, str):
        source_content_hashes["contact_plan"] = contact_plan_hash
    # Content-pinned construction (RM-P1-SIM-01): the resolved world/fleet/prospect content hashes
    # ride in provenance so a content-resolved run is byte-addressed to the exact bundles it used.
    if content_hashes:
        source_content_hashes.update(content_hashes)
    provenance: dict[str, Any] = {
        "seed": seed,
        "scenario": scenario.name,
        "dt_s": scenario.dt_s,
        "horizon_steps": scenario.horizon_steps,
        "core_interfaces": dict(CORE_INTERFACES),
        "fidelity": {aid: selection.as_provenance() for aid, selection in fidelity.items()},
        "source_content_hashes": source_content_hashes,
        "engine_versions": engine_versions(engine),
    }
    if unresolved:
        # Only when non-empty, deliberately: provenance rides inside the determinism hash, so an
        # always-present key would re-hash every existing trace to record a fact that is almost
        # always "nothing was missing". A run that *was* blind says so in its own recording, which
        # is the #65 lesson applied to #67 — a trace should carry what happened, not what was
        # declared.
        provenance["unresolved_providers"] = [
            {"content_id": u.content_id, "kind": u.kind, "producer": u.producer} for u in unresolved
        ]
    return provenance


def _dump_observations(observations: Mapping[str, Observation]) -> dict[str, Any]:
    return {aid: obs.model_dump(mode="json") for aid, obs in observations.items()}


def _reset_frame(result: ResetResult) -> dict[str, Any]:
    return {"kind": "reset", "observations": _dump_observations(result.observations)}


def _step_frame(result: StepResult) -> dict[str, Any]:
    return {
        "kind": "step",
        "sim_time_s": result.sim_time_s,
        "dt_s": result.dt_s,
        "observations": _dump_observations(result.observations),
        "terminations": dict(result.terminations),
        "truncations": dict(result.truncations),
    }


if TYPE_CHECKING:

    def _assert_environment(sim: Simulator) -> Environment:
        # mypy fails here if Simulator drifts from the Core Environment Protocol.
        return sim
