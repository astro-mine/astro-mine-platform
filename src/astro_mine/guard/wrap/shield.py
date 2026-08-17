# SPDX-License-Identifier: Apache-2.0
"""``PolicyShield`` — the runtime-assurance wrapper over a Core Policy/Planner (RM-P1-GUARD-03).

The headline abstraction (guard.md §3, §6). ``PolicyShield`` *implements* the
:class:`~astro_mine.core.policy.protocol.Policy` contract and *wraps* another ``Policy`` (a Mind
planner, an Allocate allocation, a Learn :class:`~astro_mine.core.policy.OnnxPolicy`), so from
everywhere else — Sim, Studio, Ops — a shielded policy *is* just a policy. Each tick it reads only
the wrapped policy's **proposed** :class:`~astro_mine.core.messages.ActionBatch` (never its
internals — the wrapped policy is adversarial input, guard.md §9.1), routes each agent's proposed
action through the trusted Rust safety core, and re-emits the core's **certified** action.

**Minimal TCB.** This wrapper does *zero* certification — it is pure marshalling: pull ``position``
/ ``velocity`` from the :class:`~astro_mine.core.messages.Observation`, resolve the compiled
model's ``signals`` vector, classify the proposed :class:`~astro_mine.core.messages.Action` for the
core's action gate, call :meth:`SafetyCore.step`, and write the certified command back. Every
guarantee lives in the Rust core (RM-P1-GUARD-02); adding this wrapper adds no trust surface
(guard.md §2.1, §9.1).

**Fail-safe, never fail-open.** *Every* action in the batch crosses the core — there is no
pass-through branch (LUNAR-FR-006, "every path to actuation crosses Guard"). When the core cannot
certify (infeasible filter, bad input, missing observation, watchdog miss, an action kind it cannot
model) it has already substituted the verified backup command *inside the TCB* (guard.md §9.1); the
wrapper just forwards it. A missing observation or a malformed signal vector is marshalled into the
core as ``NaN``, so the core fail-closes to the backup — the fail-safe decision stays inside the
TCB.

Action ↔ command convention --------------------------- The core certifies a **commanded
quantity**, and Core's :class:`~astro_mine.core.messages.Action` is a tagged union (actuator / mode
/ task). :class:`ActionCodec` classifies every action into exactly one of the core's four
dispositions — the mapping is the *whole* surface, and it is closed:

+-----------------------------------------------+-------------------------------------------------+
| Proposed action                               | Disposition                                     |
+===============================================+=================================================+
| ``ACTUATOR`` / ``EFFORT``, ``spatial_dim``    | **shielded** — commanded acceleration, projected|
| setpoint                                      | by the HOCBF double-integrator filter           |
+-----------------------------------------------+-------------------------------------------------+
| ``ACTUATOR`` / ``VELOCITY``, ``spatial_dim``  | **shielded** — commanded velocity, projected    |
| setpoint                                      | onto the reviewed speed ceiling ∩ the keep-out  |
|                                               | safe set (the realised next pose is certified)  |
+-----------------------------------------------+-------------------------------------------------+
| ``ACTUATOR`` / ``POSITION``, ``spatial_dim``  | **shielded** — commanded target, projected onto |
| setpoint                                      | the reviewed step cap ∩ the safe set            |
+-----------------------------------------------+-------------------------------------------------+
| ``ACTUATOR`` / ``IMPEDANCE`` or               | **rejected** — the TCB has no plant model for   |
| ``TRAJECTORY``; or any actuator setpoint the  | these, so no certificate is possible. The core  |
| spatial plant cannot evaluate (wrong arity, a | substitutes a verified safe command. *Not* a    |
| non-spatial model, a malformed union)         | pass-through — see below.                       |
+-----------------------------------------------+-------------------------------------------------+
| ``MODE`` / ``TASK``                           | **gated** — certified only if the directive is  |
|                                               | on the *effective* allowlist (the reviewed      |
|                                               | ``SafetySpec.admissible_directives`` ∩ the      |
|                                               | configured :class:`ActionPolicy`); else         |
|                                               | rejected as above. A continuous projection is   |
|                                               | meaningless for a discrete directive;           |
|                                               | enumeration of a pre-certified set is the       |
|                                               | corresponding discipline.                       |
+-----------------------------------------------+-------------------------------------------------+

**Why ``IMPEDANCE``/``TRAJECTORY`` are rejected rather than passed through.** Guard's guarantee is
that a buggy or adversarial policy *cannot* cause a hard-constraint violation (guard.md §9.1). A
control mode the core cannot model is a control mode whose effect on the safe set it cannot bound —
so "let it through" would be precisely the fail-open the component exists to prevent, and the
narrower the shield's coverage, the more attractive that mode becomes to an adversarial policy. The
defined treatment is therefore to **reject**: the core answers with a verified safe command in the
:attr:`ActionPolicy.fallback_control_mode` channel. Extending the TCB with an impedance/trajectory
plant model (so those modes can be *certified* rather than refused) is the follow-up; until it
lands the refusal is the honest posture, not a gap.

**Answer in the plant's own channel.** A rejected *actuator* command is answered in **its own**
control mode (a ``VELOCITY`` proposal yields a certified velocity command, not an ``EFFORT`` brake
a velocity-tracking actuator would silently ignore) — a "safe" command in a channel nobody reads is
a fail-open in disguise.

**Best-effort audit (RM-P1-GUARD-06).** After the certified action is fixed, the per-tick
:class:`~astro_mine.guard.audit.model.SafetyVerdict` is handed to an optional sink; any emission
fault is swallowed and never gates the certified action (guard.md §8, §9.1).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.core.compat import assert_core_compatible
from astro_mine.core.hashing import content_hash_json
from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand, Observation
from astro_mine.guard import __version__ as _GUARD_VERSION
from astro_mine.guard._core import SafetyCore
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.wire import compiled_to_wire

if TYPE_CHECKING:
    from astro_mine.core.policy import AgentId, DecisionContext, Policy
    from astro_mine.guard._core import Verdict
    from astro_mine.guard.audit.sink import VerdictSink

__all__ = [
    "ACCEL_CONTROL_MODE",
    "MODELLED_CONTROL_MODES",
    "SHIELD_CORE_INTERFACES",
    "ActionCodec",
    "ActionPolicy",
    "CoreConfig",
    "DefaultSignalResolver",
    "MappingSignalResolver",
    "PolicyShield",
    "SignalResolver",
]

_LOG = logging.getLogger(__name__)

#: The Core :class:`~astro_mine.core.messages.enums.ControlMode` whose ``ActuatorCommand.setpoint``
#: carries the flat commanded-acceleration vector (see the module docstring). Retained as the
#: canonical "effort *is* acceleration" binding; :data:`MODELLED_CONTROL_MODES` is the full set the
#: core can now certify.
ACCEL_CONTROL_MODE = ControlMode.EFFORT

#: The Core control modes the trusted core has a plant model for, mapped to the core's
#: `action_kind` token. **This mapping is the shield's coverage**: an actuator command in any
#: *other* control mode (``IMPEDANCE`` / ``TRAJECTORY``) carries no certificate the core can
#: compute and is rejected (never passed through) — the module docstring states the rationale.
MODELLED_CONTROL_MODES: dict[ControlMode, str] = {
    ControlMode.EFFORT: "effort",
    ControlMode.VELOCITY: "velocity",
    ControlMode.POSITION: "position",
}

#: The core's token for an action it cannot model — the fail-closed classification.
_OPAQUE = "opaque"

#: The SI unit each certified command is emitted with (RFC-0007, units on the wire).
_UNIT_OF: dict[ControlMode, str] = {
    ControlMode.EFFORT: "m/s^2",
    ControlMode.VELOCITY: "m/s",
    ControlMode.POSITION: "m",
}

#: The Core interfaces ``PolicyShield`` is built against — it *is* a Policy (``policy``) mapping
#: the message vocabulary (``messages``). Negotiated against this Core at construction (fail loud
#: on an incompatible Core, never mid-episode), and the consumer-driven contract-test assertion.
SHIELD_CORE_INTERFACES: dict[str, str] = {"policy": "0.1.0", "messages": "0.1.0"}

#: Known scalar fields a :class:`~astro_mine.core.messages.StateSample` carries directly,
#: resolvable by an exact signal-key match (the rest are SADF/Worlds signals GUARD-04 resolves).
_STATE_SCALAR_FIELDS: frozenset[str] = frozenset({"battery_soc_j", "temperature_k"})


@runtime_checkable
class SignalResolver(Protocol):
    """Resolves a compiled model's ordered ``signals`` keys to their float values for one tick.

    The seam where the abstract signal keys a ``SafetySpec`` references (SADF budgets, Worlds
    rasters, observation fields) become the concrete vector the core reads. The Phase-1 MVP ships a
    minimal observation-only default (:class:`DefaultSignalResolver`); the real Worlds/Fleet
    resolution is RM-P1-GUARD-04. Returning ``NaN`` for an unresolvable signal is the correct
    fail-safe: the core treats it as bad input and falls back (never a silent unsafe default)."""

    def resolve(self, signal_keys: Sequence[str], observation: Observation | None) -> list[float]:
        """Return one float per key in ``signal_keys`` order (``NaN`` where unresolvable)."""
        ...


class DefaultSignalResolver:
    """The minimal, observation-only signal resolver (RM-P1-GUARD-04 fills the real resolution).

    Resolves each signal key, in order, from the agent's
    :class:`~astro_mine.core.messages.Observation`:

    1. a :class:`~astro_mine.core.messages.SensorReading` whose ``sensor`` equals the key → its
       first value;
    2. else a known :class:`~astro_mine.core.messages.StateSample` scalar field of the same name
       (:data:`_STATE_SCALAR_FIELDS`) → that value;
    3. else ``NaN`` — an unresolvable safety signal fails the tick *closed* inside the core.

    SADF/Worlds-sourced signals (power floors, torque budgets, terrain) resolve to ``NaN`` here by
    design until GUARD-04 supplies the Fleet/Worlds resolution — so a spec that references them is
    conservatively fail-safe rather than silently satisfied."""

    def resolve(self, signal_keys: Sequence[str], observation: Observation | None) -> list[float]:
        if observation is None:
            return [math.nan] * len(signal_keys)
        sensors = {r.sensor: r for r in observation.sensors}
        out: list[float] = []
        for key in signal_keys:
            reading = sensors.get(key)
            if reading is not None and reading.values:
                out.append(float(reading.values[0]))
            elif key in _STATE_SCALAR_FIELDS:
                value = getattr(observation.self_state, key, None)
                out.append(float(value) if value is not None else math.nan)
            else:
                out.append(math.nan)
        return out


class MappingSignalResolver:
    """A signal resolver over an explicit static ``{key: value}`` map (unknown keys → ``NaN``).

    Deterministic and observation-independent — the injection point for tests/fixtures and for
    GUARD-04 to hand pre-resolved SADF/Worlds values in. Unknown keys resolve to ``NaN``
    (fail-safe)."""

    def __init__(self, values: Mapping[str, float]) -> None:
        self._values = dict(values)

    def resolve(self, signal_keys: Sequence[str], observation: Observation | None) -> list[float]:
        return [float(self._values.get(key, math.nan)) for key in signal_keys]


@dataclass(frozen=True, slots=True)
class ActionPolicy:
    """The **configured** allowlist for discrete directives, plus the channel a rejected action is
    answered in (RM-P1-GUARD-03; RFC-0004 Amendment 2; guard.md §3, §5).

    A ``MODE`` / ``TASK`` action carries no continuous quantity to project, so the shield cannot
    *correct* it — it can only certify it by **enumeration**, and an admitted directive is
    re-emitted **untouched**. That makes the *grant* itself the safety decision, so the grant lives
    in the reviewed, content-addressed, signed
    :attr:`~astro_mine.guard.spec.model.SafetySpec.admissible_directives` — **not here**.

    **Configuration may only NARROW the reviewed grant.** The allowlist the trusted core actually
    gates against is the *effective* one, resolved once at core construction::

        effective = spec.admissible_directives  ∩  config.action_policy

    So this dataclass can *tighten* the contract — the legitimate "run this deployment stricter than
    the contract allows" case — but it can never **create** a permission. Where the spec is
    **silent**, the effective allowlist is **empty**: nothing is certifiable, however permissive the
    configuration is. (Contrast the ``kinematic_limit`` ceiling, where an *unauthored* limit leaves
    the configured one standing: both merges are the greatest-lower-bound, but the identity of a
    ceiling's meet is ``+∞`` and of a permission set's is ``∅``. Silence must never grant
    authority.)

    Both allowlists are still empty by default, so an unconfigured Guard certifies **no** directive
    — and a wrapped policy can never widen either side, so the independence property
    (guard.md §9.1) holds structurally.

    ``fallback_control_mode`` / ``fallback_target`` are genuine, and deliberate, *configuration*:
    they name the actuation **channel** a rejected *directive* (or an unmodelled actuator command
    with no channel of its own) is answered in — a property of the plant and its wiring, not a
    permission. No setting of them can widen what is certifiable, and ``__post_init__`` already
    constrains the mode to one the TCB has a plant model for. A rejected *actuator* command is
    always answered in its own control mode, so they only apply to ``MODE``/``TASK`` and to
    malformed actions."""

    #: Pre-certified :attr:`~astro_mine.core.messages.model.ModeCommand.mode` names.
    certified_modes: frozenset[str] = frozenset()
    #: Pre-certified :class:`~astro_mine.core.messages.enums.TaskKind` values.
    certified_tasks: frozenset[str] = frozenset()
    #: The actuation channel a rejected directive is answered in.
    fallback_control_mode: ControlMode = ControlMode.EFFORT
    #: The ``ActuatorCommand.target`` a substituted safe command addresses.
    fallback_target: str = "body"

    def __post_init__(self) -> None:
        if self.fallback_control_mode not in MODELLED_CONTROL_MODES:
            raise ValueError(
                f"fallback_control_mode must be one of {
                    sorted(m.value for m in MODELLED_CONTROL_MODES)
                }"
                f", got {self.fallback_control_mode.value!r} — the core has no plant model for it"
            )


class ActionCodec:
    """Marshals between a Core :class:`~astro_mine.core.messages.Action` and the core's flat
    commanded vectors, per the module's Action ↔ command convention.

    Stateless; a single instance is shared across agents. Override to adopt a different
    action-space convention without touching the shield's control flow — but note that the
    *classification* is the shield's coverage: anything a subclass classifies as ``opaque`` is
    rejected by the core, never passed through."""

    def classify(self, action: Action, spatial_dim: int) -> tuple[str, list[float], str | None]:
        """Classify ``action`` for the core's action gate.

        Returns ``(action_kind, setpoint, directive)`` where ``action_kind`` is one of ``effort`` |
        ``velocity`` | ``position`` (a modelled command, with its ``setpoint``), ``mode`` |
        ``task`` (a discrete directive, named by ``directive``), or ``opaque`` — an action the core
        has no plant model for and therefore **rejects**. A malformed action (a ``kind`` whose
        union arm is unset) is ``opaque``: the safety core, not the marshal layer, decides what
        to do about it, and it decides *closed*."""
        if action.kind == ActionKind.ACTUATOR and action.actuator is not None:
            kind = MODELLED_CONTROL_MODES.get(action.actuator.control_mode)
            setpoint = [float(x) for x in action.actuator.setpoint]
            # A setpoint the spatial plant cannot evaluate (a non-spatial model, or the wrong
            # arity) carries no certificate the core can compute — reject rather than guess at
            # padding.
            if kind is None or spatial_dim <= 0 or len(setpoint) != spatial_dim:
                return _OPAQUE, [], None
            return kind, setpoint, None
        if action.kind == ActionKind.MODE and action.mode is not None:
            return "mode", [], action.mode.mode
        if action.kind == ActionKind.TASK and action.task is not None:
            return "task", [], str(action.task.task_kind.value)
        return _OPAQUE, [], None

    def read_position(self, observation: Observation | None, spatial_dim: int) -> list[float]:
        """The agent's position from ``self_state.pose.translation_m`` (zeros when unavailable)."""
        if observation is None:
            return [0.0] * spatial_dim
        t = observation.self_state.pose.translation_m
        return self._fit([t.x, t.y, t.z], spatial_dim)

    def read_velocity(self, observation: Observation | None, spatial_dim: int) -> list[float]:
        """The linear velocity from ``self_state.linear_velocity_mps`` (zeros when absent)."""
        if observation is None or observation.self_state.linear_velocity_mps is None:
            return [0.0] * spatial_dim
        v = observation.self_state.linear_velocity_mps
        return self._fit([v.x, v.y, v.z], spatial_dim)

    def write_certified_action(self, action: Action, certified: Sequence[float]) -> Action:
        """Return a copy of ``action`` with the certified command as its setpoint (only).

        Used when the shield *corrected* a modelled actuator command: the control mode, target, unit
        and every other field are preserved — only the numbers the core certified change."""
        assert action.actuator is not None  # guaranteed by classify() returning a modelled kind
        actuator = action.actuator.model_copy(update={"setpoint": [float(x) for x in certified]})
        return action.model_copy(update={"actuator": actuator})

    def write_safe_action(
        self,
        action: Action,
        certified: Sequence[float],
        control_mode: ControlMode,
        target: str,
    ) -> Action:
        """Return the **verified safe command** the core substituted for a rejected action.

        The core has already decided *what* the safe command is and *which channel* it is expressed
        in (``control_mode``); this only builds the Core message that carries it. A rejected
        actuator command keeps its own ``target`` (the safe command must reach the actuator the
        policy was addressing); a rejected ``MODE``/``TASK`` directive has no actuator of its own
        and uses the
        :attr:`ActionPolicy.fallback_target`."""
        original = action.actuator
        return Action(
            agent_id=action.agent_id,
            kind=ActionKind.ACTUATOR,
            actuator=ActuatorCommand(
                target=original.target if original is not None else target,
                control_mode=control_mode,
                setpoint=[float(x) for x in certified],
                unit=_UNIT_OF[control_mode],
            ),
        )

    @staticmethod
    def _fit(values: list[float], n: int) -> list[float]:
        """Pad with zeros / truncate ``values`` to exactly ``n`` elements (spatial_dim ≤ 3)."""
        return [float(x) for x in (values + [0.0] * n)[:n]]


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """The trusted-core construction parameters (mirrors :meth:`SafetyCore.from_wire` defaults).

    Shared across every agent's core so all agents enforce the same spec under the same tuning.
    ``max_history_cap`` bounds a temporal monitor's ring buffer — raise it (or compile the spec at
    a coarser ``sample_period_s``) for long-horizon survival clauses.

    **Configuration may only narrow the reviewed contract — never widen it.** Two knobs here are
    merged with the reviewed ``SafetySpec`` inside the TCB, and both merges are the
    greatest-lower-bound of *(configured, authored)*:

    - ``u_max`` / ``v_max`` are the *configured* control-authority ceilings. A ``kinematic_limit``
      in the spec lowers to an :class:`~astro_mine.guard.spec.ir.ActionLimits` that **tightens**
      them (``min(config, authored)``). An *absent* authored limit leaves the configured ceiling
      standing — the identity of a ceiling's meet is ``+∞``.
    - ``action_policy`` is the *configured* directive allowlist. The spec's
      :attr:`~astro_mine.guard.spec.model.SafetySpec.admissible_directives` **intersects** it
      (``spec ∩ config``; RFC-0004 Amendment 2). An *absent* authored grant admits **nothing** — the
      identity of a permission set's meet is ``∅``, so a spec-silent contract certifies **no**
      directive however permissive this configuration is.

    That asymmetry on silence is deliberate: a ceiling with no opinion constrains nothing, but a
    permission with no opinion grants nothing."""

    u_max: float = 20.0
    v_max: float = 2.0
    k0: float = 9.0
    k1: float = 6.0
    k_brake: float = 4.0
    predictive_horizon_samples: int = 5
    deadline_us: int | None = None
    max_history_cap: int = 1_048_576
    action_policy: ActionPolicy = field(default_factory=ActionPolicy)

    def build(self, wire: bytes) -> SafetyCore:
        """Decode ``wire`` (a ``CompiledSafetyModel`` protobuf) into a fresh :class:`SafetyCore`."""
        return SafetyCore.from_wire(
            wire,
            u_max=self.u_max,
            v_max=self.v_max,
            k0=self.k0,
            k1=self.k1,
            k_brake=self.k_brake,
            predictive_horizon_samples=self.predictive_horizon_samples,
            deadline_us=self.deadline_us,
            max_history_cap=self.max_history_cap,
            certified_modes=sorted(self.action_policy.certified_modes),
            certified_tasks=sorted(self.action_policy.certified_tasks),
            fallback_control_mode=self.action_policy.fallback_control_mode.value,
        )


class PolicyShield:
    """A :class:`~astro_mine.core.policy.protocol.Policy` that shields another policy's actions.

    Construct with the wrapped policy and the
    :class:`~astro_mine.guard.spec.ir.CompiledSafetyModel` the trusted core enforces; call
    :meth:`decide` exactly as any Core policy. One :class:`SafetyCore` is created per agent
    (temporal monitors are stateful across ticks) and decoded once from the shared compiled wire.
    """

    #: The Core interfaces this shield is built against (the contract-test assertion).
    CORE_INTERFACES = SHIELD_CORE_INTERFACES

    def __init__(
        self,
        wrapped: Policy,
        compiled: CompiledSafetyModel,
        *,
        signal_resolver: SignalResolver | None = None,
        codec: ActionCodec | None = None,
        sink: VerdictSink | None = None,
        watchdog: bool = False,
        core_config: CoreConfig | None = None,
    ) -> None:
        assert_core_compatible(self.CORE_INTERFACES)
        self._wrapped = wrapped
        self._compiled = compiled
        self._wire = compiled_to_wire(compiled)
        self._signal_order: tuple[str, ...] = tuple(compiled.predicate_table.signals)
        self._resolver: SignalResolver = signal_resolver or DefaultSignalResolver()
        self._codec = codec or ActionCodec()
        self._sink = sink
        self._watchdog = watchdog
        self._core_config = core_config or CoreConfig()
        # A prototype validates the wire (fail loud at construction) and carries the provenance
        # metadata; it is never stepped — every agent gets its own fresh, stateful core.
        prototype = self._core_config.build(self._wire)
        self._spatial_dim: int = prototype.spatial_dim
        self._spec_id: str = prototype.spec_id
        self._spec_content_hash: str = prototype.spec_content_hash
        self._v_max: float | None = prototype.v_max
        self._compiled_content_hash: str = compiled.content_hash()
        self._cores: dict[AgentId, SafetyCore] = {}
        self._decision_index = 0

    # --- introspection --------------------------------------------------------------
    @property
    def wrapped(self) -> Policy:
        """The wrapped policy (read-only; the shield never reaches into its internals)."""
        return self._wrapped

    @property
    def spatial_dim(self) -> int:
        """The dimensionality of the commanded vectors the core certifies."""
        return self._spatial_dim

    @property
    def v_max(self) -> float | None:
        """The enforced commanded-speed ceiling — the reviewed spec's ``max_velocity_mps`` when it
        authors one, else the configured :attr:`CoreConfig.v_max`. ``None`` for a non-spatial model
        (no shield, so no kinematic envelope)."""
        return self._v_max

    @property
    def action_policy(self) -> ActionPolicy:
        """The certified-directive allowlist in force (empty by default — fail-closed)."""
        return self._core_config.action_policy

    @property
    def spec_id(self) -> str:
        """The id of the SafetySpec in force."""
        return self._spec_id

    @property
    def spec_content_hash(self) -> str:
        """The content address of the source SafetySpec (as reported by the trusted core)."""
        return self._spec_content_hash

    @property
    def compiled_content_hash(self) -> str:
        """The content address of the CompiledSafetyModel the core enforces."""
        return self._compiled_content_hash

    # --- the Policy contract --------------------------------------------------------
    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Shield the wrapped policy's proposed batch and return the certified batch.

        Reads *only* the wrapped policy's proposed :class:`ActionBatch` (adversarial input), then
        certifies each agent's action through its core, preserving the batch's action order."""
        proposed = self._wrapped.decide(observations, context)
        certified = [
            self._certify(action, observations.get(action.agent_id), context)
            for action in proposed.actions
        ]
        self._decision_index += 1
        return ActionBatch(actions=certified)

    # --- internals ------------------------------------------------------------------
    def _core_for(self, agent_id: AgentId) -> SafetyCore:
        core = self._cores.get(agent_id)
        if core is None:
            core = self._core_config.build(self._wire)
            self._cores[agent_id] = core
        return core

    def _certify(
        self, action: Action, observation: Observation | None, context: DecisionContext
    ) -> Action:
        """Route one proposed action through the trusted core and emit its verdict.

        There is **no** branch that returns the proposal without asking the core — that is what
        makes "every path to actuation crosses Guard" (LUNAR-FR-006) a property of the code rather
        than a
        claim about it."""
        action_kind, proposed_vec, directive = self._codec.classify(action, self._spatial_dim)
        position = self._codec.read_position(observation, self._spatial_dim)
        velocity = self._codec.read_velocity(observation, self._spatial_dim)
        signals = _fit_signals(
            self._resolver.resolve(self._signal_order, observation), len(self._signal_order)
        )
        core = self._core_for(action.agent_id)
        start = time.perf_counter_ns()
        verdict = core.step(
            signals,
            position,
            velocity,
            proposed_vec,
            action_kind=action_kind,
            directive=directive,
            watchdog=self._watchdog,
        )
        latency_us = (time.perf_counter_ns() - start) / 1000.0
        certified = [float(x) for x in verdict["certified_action"]]
        self._emit(
            action,
            observation,
            context,
            proposed_vec,
            signals,
            position,
            velocity,
            verdict,
            latency_us,
        )
        return self._emit_action(action, certified, verdict)

    def _emit_action(self, action: Action, certified: list[float], verdict: Verdict) -> Action:
        """Build the Core action carrying the core's verdict — the one place a decision becomes an
        emitted command. The routing is entirely the core's (``intervention``), never this
        layer's."""
        intervention = verdict["intervention"]
        if intervention == "fallback":
            # The core rejected the proposal and substituted a verified safe command, in the
            # channel it named. This is the only branch that *replaces* an action.
            return self._codec.write_safe_action(
                action,
                certified,
                ControlMode(verdict["certified_control_mode"]),
                self.action_policy.fallback_target,
            )
        if not certified:
            # A certified *directive* (an allowlisted MODE/TASK) carries no numeric command —
            # re-emit the proposal exactly as authored.
            return action
        return self._codec.write_certified_action(action, certified)

    def _emit(
        self,
        action: Action,
        observation: Observation | None,
        context: DecisionContext,
        proposed_vec: list[float],
        signals: list[float],
        position: list[float],
        velocity: list[float],
        verdict: Verdict,
        latency_us: float,
    ) -> None:
        """Hand the verdict to the audit sink — best-effort; never gates the certified action."""
        if self._sink is None:
            return
        try:
            from astro_mine.guard.audit.model import SafetyVerdict

            certified = [float(x) for x in verdict["certified_action"]]
            record = SafetyVerdict(
                verdict_version="0.1",
                agent_id=action.agent_id,
                tick=observation.tick if observation is not None else self._decision_index,
                sim_time_s=(
                    observation.sim_time_s if observation is not None else context.sim_time_s
                ),
                spec_id=self._spec_id,
                spec_content_hash=self._spec_content_hash,
                compiled_content_hash=self._compiled_content_hash,
                guard_code_version=_GUARD_VERSION,
                layer=verdict["layer"],
                intervention=verdict["intervention"],
                reason=verdict["reason"],
                backup_kind=verdict["backup_kind"],
                constraint_ids=list(verdict["fired"]),
                certified_action=certified,
                min_barrier_margin=verdict["min_barrier_margin"],
                action_divergence=_l2_distance(certified, proposed_vec),
                inputs_content_hash=content_hash_json(
                    {
                        "signals": signals,
                        "position": position,
                        "velocity": velocity,
                        "proposed_action": proposed_vec,
                    }
                ),
                shield_latency_us=latency_us,
            )
            self._sink.write_verdict(record)
        except Exception:
            _LOG.warning(
                "SafetyVerdict emission failed (best-effort telemetry); "
                "the certified action is unaffected",
                exc_info=True,
            )


def _fit_signals(values: Sequence[float], n: int) -> list[float]:
    """Coerce a resolver's output to exactly ``n`` floats (``NaN`` pad / truncate).

    A misbehaving resolver returning the wrong length is marshalled into ``NaN`` slots rather than
    crashing the control loop — the core then fails the tick *closed* (bad input ⇒ backup)."""
    fitted = list(values[:n]) + [math.nan] * max(0, n - len(values))
    return [float(x) for x in fitted]


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance ‖a - b‖ (the derived shielding-cost divergence). 0.0 for equal vectors.

    Over the common prefix: a rejected directive has an empty proposed vector (there was no
    commanded quantity to diverge *from*), so its divergence is 0.0 by construction — the shielding
    *cost* of a refusal is not a distance in command space, and the ``reason`` field is what
    records
    it."""
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=False)))
