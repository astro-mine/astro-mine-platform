"""PolicyShield — the runtime-assurance wrapper over a Core Policy/Planner (RM-P1-GUARD-03).

Proves the headline contract (guard.md §3, §6, §9.1): a shielded policy *is* a Core Policy, it is
fail-safe (never fail-open), it reads only the proposed action, one core per agent, and a
Learn-exported ONNX policy wrapped by the shield is a drop-in, Sim-consumable ``ActionBatch`` — all
without importing ``astro_mine.sim``. All certification stays in the trusted Rust core.
"""

from __future__ import annotations

import math
import sys

import pytest

from astro_mine.core.compat import assert_core_compatible
from astro_mine.core.messages.enums import ActionKind, ControlMode, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ModeCommand,
    TaskDirective,
)
from astro_mine.core.policy import DecisionContext, OnnxPolicy, check_composition, check_policy
from astro_mine.core.policy.model import ModelRef, PolicyPackage
from astro_mine.core.policy.protocol import Policy
from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.spec import SafetyDocument, compile_spec
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.model import AdmissibleDirectives
from astro_mine.guard.wrap import (
    ACCEL_CONTROL_MODE,
    MODELLED_CONTROL_MODES,
    SHIELD_CORE_INTERFACES,
    ActionCodec,
    ActionPolicy,
    CoreConfig,
    DefaultSignalResolver,
    MappingSignalResolver,
    PolicyShield,
    SignalResolver,
)
from tests.guard.conftest import (
    COARSE_SAMPLE_PERIOD_S as COARSE_PERIOD_S,
)
from tests.guard.conftest import (
    SAFE_SIGNALS,
    StubPolicy,
    make_effort_action,
    make_observation,
)

pytest.importorskip("astro_mine.guard._core", reason="Rust safety core not built (run `uv sync`)")


def _shield(compiled: CompiledSafetyModel, batch: ActionBatch, **kwargs: object) -> PolicyShield:
    return PolicyShield(StubPolicy(batch), compiled, **kwargs)  # type: ignore[arg-type]


# --- the Policy contract & drop-in composition -----------------------------------------------


def test_shield_is_a_core_policy(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    shield = _shield(anchor_compiled_coarse, ActionBatch())
    assert isinstance(shield, Policy)
    assert shield.CORE_INTERFACES == SHIELD_CORE_INTERFACES


def test_declares_compatible_core_interfaces() -> None:
    # The consumer-driven contract-test assertion (RM-P0-CORE-07): the versions the shield claims
    # are satisfied by this Core.
    assert_core_compatible(SHIELD_CORE_INTERFACES)


def test_check_policy_contract(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    batch = ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])])
    shield = _shield(
        anchor_compiled_coarse, batch, signal_resolver=MappingSignalResolver(SAFE_SIGNALS)
    )
    result = check_policy(shield, {"rover": make_observation("rover")}, DecisionContext())
    assert isinstance(result, ActionBatch)
    assert result.actions[0].actuator is not None


def test_check_composition_places_shield_in_a_stack(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # An allocator (TASK) composed before a shield wrapping a controller (EFFORT): the composed
    # stack's final, certified batch honors the Policy contract end-to-end.
    controller = StubPolicy(ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])]))
    allocator = StubPolicy(
        ActionBatch(
            actions=[
                Action(
                    agent_id="rover",
                    kind=ActionKind.TASK,
                    task=TaskDirective(task_kind=TaskKind.STANDBY),
                )
            ]
        )
    )
    shield = PolicyShield(
        controller, anchor_compiled_coarse, signal_resolver=MappingSignalResolver(SAFE_SIGNALS)
    )
    result = check_composition(
        allocator,
        shield,
        observations={"rover": make_observation("rover")},
        context=DecisionContext(),
    )
    assert result.actions[0].kind == ActionKind.ACTUATOR


def test_onnx_policy_wrapped_is_drop_in(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    # A Learn ONNX policy (OnnxPolicy + a stub infer) wrapped by the shield satisfies the Policy
    # contract and yields a Sim-consumable ActionBatch — with NO astro_mine.sim import.
    package = PolicyPackage(
        name="learned-rover",
        version="0.1",
        onnx_model=ModelRef(digest="sha256:" + "0" * 64),
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
    )

    def infer(observations: object, context: object) -> ActionBatch:
        return ActionBatch(actions=[make_effort_action(a, [0.0, 0.0, 0.0]) for a in observations])  # type: ignore[union-attr]

    onnx = OnnxPolicy(package, infer)
    shield = PolicyShield(
        onnx, anchor_compiled_coarse, signal_resolver=MappingSignalResolver(SAFE_SIGNALS)
    )
    result = check_policy(shield, {"rover": make_observation("rover")}, DecisionContext())
    assert result.actions[0].actuator is not None
    assert "astro_mine.sim" not in sys.modules


# --- certify / correct / fail-safe -----------------------------------------------------------


def test_safe_action_certified_passthrough(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    proposed = [0.0, 0.0, 0.0]
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", proposed)]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
        sink=sink,
    )
    out = shield.decide(
        {"rover": make_observation("rover", position=(100.0, 100.0, 100.0))}, DecisionContext()
    )
    assert out.actions[0].actuator is not None
    assert out.actions[0].actuator.setpoint == proposed  # certified unchanged
    v = sink.verdicts[0]
    assert (v.layer, v.intervention, v.reason) == ("primary", "none", "certified")
    assert v.action_divergence == 0.0


def test_shield_corrects_action_toward_keepout(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    proposed = [-5.0, 0.0, 0.0]  # driving into the lander safety sphere at the origin
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", proposed)]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
        sink=sink,
    )
    out = shield.decide(
        {"rover": make_observation("rover", position=(31.0, 0.0, 0.0))}, DecisionContext()
    )
    certified = out.actions[0].actuator.setpoint  # type: ignore[union-attr]
    assert certified != proposed  # the shield modified the action
    assert sink.verdicts[0].intervention == "modified"
    assert sink.verdicts[0].layer == "shield"
    assert sink.verdicts[0].action_divergence > 0.0


def test_unresolved_signals_fail_safe_not_open(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    # DefaultSignalResolver cannot resolve the SADF signals (GUARD-04) → NaN → the core fails the
    # tick closed and substitutes the backup — never the raw proposed action.
    proposed = [9.0, 0.0, 0.0]
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", proposed)]),
        signal_resolver=DefaultSignalResolver(),
        sink=sink,
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert out.actions[0].actuator.setpoint != proposed  # type: ignore[union-attr]  # NOT fail-open
    v = sink.verdicts[0]
    assert v.intervention == "fallback"
    assert v.layer == "backup"
    assert v.reason == "bad_input"
    assert v.backup_kind == "brake_to_stop"


def test_scalar_violation_falls_back_with_constraint_id(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    sink = CollectingSink()
    signals = {**SAFE_SIGNALS, "anchor_torque_nm": 100.0}  # > 40 N·m torque ceiling
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])]),
        signal_resolver=MappingSignalResolver(signals),
        sink=sink,
    )
    shield.decide({"rover": make_observation("rover")}, DecisionContext())
    v = sink.verdicts[0]
    assert v.reason == "scalar_violated"
    assert v.constraint_ids == ["c_anchor_torque"]


def test_missing_observation_fails_safe(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    # A shieldable action whose agent has no observation must NOT pass through — it fails safe.
    proposed = [7.0, 0.0, 0.0]
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("ghost", proposed)]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
        sink=sink,
    )
    out = shield.decide({}, DecisionContext())  # no observation for "ghost"
    assert out.actions[0].actuator.setpoint != proposed  # type: ignore[union-attr]
    assert sink.verdicts[0].intervention == "fallback"


# --- the action gate: MODE / TASK / unmodelled actuator commands (RM-P1-GUARD-03) ------------
#
# The Phase-1 MVP passed every non-EFFORT action through unshielded. It no longer does: a MODE/TASK
# directive is certified only by *enumeration* (the reviewed allowlist), and an actuator command
# the TCB has no plant model for is *rejected*, never passed through — that would be the fail-open
# the component exists to prevent (LUNAR-FR-006; guard.md §9.1).


def _safe_kwargs(**extra: object) -> dict[str, object]:
    return {"signal_resolver": MappingSignalResolver(SAFE_SIGNALS), **extra}


def test_unlisted_mode_action_is_rejected(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    mode = Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="drill_hard"))
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse, ActionBatch(actions=[mode]), sink=sink, **_safe_kwargs()
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    # Not passed through: replaced by the verified safe command, in the configured channel.
    emitted = out.actions[0]
    assert emitted.kind == ActionKind.ACTUATOR
    assert emitted.actuator is not None
    assert emitted.actuator.control_mode == ControlMode.EFFORT  # the default fallback channel
    assert emitted.actuator.target == ActionPolicy().fallback_target
    v = sink.verdicts[0]
    assert (v.layer, v.intervention, v.reason) == ("backup", "fallback", "not_certifiable")


def test_allowlisted_mode_action_is_certified_untouched(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    mode = Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="safe_hold"))
    policy = ActionPolicy(certified_modes=frozenset({"safe_hold"}))
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[mode]),
        sink=sink,
        **_safe_kwargs(core_config=CoreConfig(action_policy=policy)),
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert out.actions[0] is mode  # pre-certified: re-emitted exactly as authored
    assert sink.verdicts[0].reason == "certified"


def test_task_action_is_gated_against_the_task_allowlist(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    task = Action(
        agent_id="rover", kind=ActionKind.TASK, task=TaskDirective(task_kind=TaskKind.STANDBY)
    )
    # Unlisted → rejected.
    rejecting = _shield(anchor_compiled_coarse, ActionBatch(actions=[task]), **_safe_kwargs())
    rejected = rejecting.decide({"rover": make_observation("rover")}, DecisionContext())
    assert rejected.actions[0].kind == ActionKind.ACTUATOR  # substituted safe command

    # Listed (by TaskKind value) → certified untouched.
    policy = ActionPolicy(certified_tasks=frozenset({TaskKind.STANDBY.value}))
    accepting = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[task]),
        **_safe_kwargs(core_config=CoreConfig(action_policy=policy)),
    )
    accepted = accepting.decide({"rover": make_observation("rover")}, DecisionContext())
    assert accepted.actions[0] is task


def test_impedance_actuator_action_is_rejected(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    # IMPEDANCE has no plant model in the TCB, so no certificate is computable → reject, never pass
    # through. (The documented, deliberately narrow treatment — see wrap/shield.py.)
    imped = Action(
        agent_id="rover",
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="arm", control_mode=ControlMode.IMPEDANCE, setpoint=[1.0, 2.0, 3.0]
        ),
    )
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse, ActionBatch(actions=[imped]), sink=sink, **_safe_kwargs()
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    emitted = out.actions[0]
    assert emitted.actuator is not None
    assert emitted.actuator.setpoint != [1.0, 2.0, 3.0]  # NOT fail-open
    assert emitted.actuator.target == "arm"  # answered at the actuator the policy addressed
    assert sink.verdicts[0].reason == "not_certifiable"


def test_trajectory_actuator_action_is_rejected(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    traj = Action(
        agent_id="rover",
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.TRAJECTORY, setpoint=[1.0, 2.0, 3.0]
        ),
    )
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse, ActionBatch(actions=[traj]), sink=sink, **_safe_kwargs()
    )
    shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert sink.verdicts[0].reason == "not_certifiable"


def test_wrong_dimension_setpoint_is_rejected(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # An EFFORT setpoint whose length != spatial_dim cannot be evaluated against the 3-D plant.
    # Zero-padding it would be a silent guess about the missing axis → reject (fail-closed).
    two_d = make_effort_action("rover", [1.0, 2.0])
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse, ActionBatch(actions=[two_d]), sink=sink, **_safe_kwargs()
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert out.actions[0].actuator is not None
    assert out.actions[0].actuator.setpoint != [1.0, 2.0]
    assert sink.verdicts[0].reason == "not_certifiable"


def test_rejected_action_is_answered_in_the_configured_channel(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # A velocity-tracking plant configures the fallback channel, so a rejected *directive* comes
    # back as a velocity command its actuator actually reads (an EFFORT brake would be silently
    # ignored).
    policy = ActionPolicy(fallback_control_mode=ControlMode.VELOCITY, fallback_target="wheels")
    mode = Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="drill_hard"))
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[mode]),
        **_safe_kwargs(core_config=CoreConfig(action_policy=policy)),
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    actuator = out.actions[0].actuator
    assert actuator is not None
    assert actuator.control_mode == ControlMode.VELOCITY
    assert actuator.target == "wheels"
    assert actuator.unit == "m/s"
    assert actuator.setpoint == [0.0, 0.0, 0.0]  # the kinematic safe floor: stop


def test_fallback_channel_must_be_a_modelled_control_mode() -> None:
    # Configuring the shield to answer in a channel the TCB cannot model is a construction-time
    # error — a misconfigured Guard fails loud, never mid-episode.
    with pytest.raises(ValueError, match="no plant model"):
        ActionPolicy(fallback_control_mode=ControlMode.IMPEDANCE)


# --- the newly-shielded kinematic modes (RM-P1-GUARD-03) -------------------------------------


def _velocity_action(agent_id: str, setpoint: list[float], *, target: str = "wheels") -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target=target, control_mode=ControlMode.VELOCITY, setpoint=setpoint, unit="m/s"
        ),
    )


def _position_action(agent_id: str, setpoint: list[float]) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.POSITION, setpoint=setpoint, unit="m"
        ),
    )


def test_velocity_setpoint_is_capped_by_the_reviewed_kinematic_limit(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # The anchor's reviewed `c_traverse_speed` bounds the traverse to 0.1 m/s — so *that* is the
    # ceiling the shield projects a commanded velocity onto, not a runtime knob.
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[_velocity_action("rover", [3.0, 4.0, 0.0])]),  # ‖w‖ = 5 m/s
        sink=sink,
        **_safe_kwargs(),
    )
    out = shield.decide(
        {"rover": make_observation("rover", position=(500.0, 500.0, 500.0))}, DecisionContext()
    )
    actuator = out.actions[0].actuator
    assert actuator is not None
    assert actuator.control_mode == ControlMode.VELOCITY  # the channel is preserved
    speed = math.sqrt(sum(x * x for x in actuator.setpoint))
    assert speed == pytest.approx(0.1, abs=1e-9)
    assert sink.verdicts[0].intervention == "modified"


def test_velocity_setpoint_into_the_keepout_is_projected(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # Driving straight at the lander-zone sphere (r = 30, margin = 3) from just outside it. The
    # certificate is not "the x component is non-negative" — the shield is free to *redirect* the
    # command — it is that the **realised next pose** stays outside the keep-out. Check that.
    start = (33.05, 0.0, 10.0)
    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[_velocity_action("rover", [-0.1, 0.0, 0.0])]),
        sink=sink,
        **_safe_kwargs(),
    )
    out = shield.decide({"rover": make_observation("rover", position=start)}, DecisionContext())
    actuator = out.actions[0].actuator
    assert actuator is not None
    assert sink.verdicts[0].intervention == "modified"  # the proposal really was unsafe
    dt = COARSE_PERIOD_S
    nxt = [p + w * dt for p, w in zip(start, actuator.setpoint, strict=True)]
    assert math.sqrt(sum(x * x for x in nxt)) >= 33.0  # outside r + margin
    # …and the certified command still respects the reviewed speed ceiling.
    assert math.sqrt(sum(x * x for x in actuator.setpoint)) <= 0.1 + 1e-9


def test_position_setpoint_is_step_capped_and_safe(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # A commanded teleport into the lander zone: capped to one reviewed step (v_max · dt) and kept
    # outside the keep-out.
    start = (60.0, 0.0, 10.0)
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[_position_action("rover", [0.0, 0.0, 10.0])]),  # the sphere centre
        **_safe_kwargs(),
    )
    out = shield.decide({"rover": make_observation("rover", position=start)}, DecisionContext())
    actuator = out.actions[0].actuator
    assert actuator is not None
    target = actuator.setpoint
    # Outside the lander keep-out (radius 30 + margin 3) …
    assert math.sqrt(target[0] ** 2 + target[1] ** 2 + target[2] ** 2) >= 33.0 - 1e-6
    # … and no further than the reviewed step from where the rover actually is.
    step = math.dist(target, start)
    assert step <= 0.1 * COARSE_PERIOD_S + 1e-6


def test_shield_exposes_the_enforced_speed_ceiling(
    anchor_compiled_coarse: CompiledSafetyModel,
) -> None:
    # The reviewed spec (0.1 m/s) tightens the configured default (2.0 m/s) — configuration can
    # never loosen the reviewed contract.
    shield = _shield(anchor_compiled_coarse, ActionBatch(), **_safe_kwargs())
    assert shield.v_max == pytest.approx(0.1)
    loose = _shield(
        anchor_compiled_coarse, ActionBatch(), **_safe_kwargs(core_config=CoreConfig(v_max=99.0))
    )
    assert loose.v_max == pytest.approx(0.1)


# --- per-agent cores, ordering, properties ---------------------------------------------------


def test_one_core_per_agent(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    batch = ActionBatch(
        actions=[make_effort_action("a", [0.0, 0.0, 0.0]), make_effort_action("b", [0.0, 0.0, 0.0])]
    )
    shield = _shield(
        anchor_compiled_coarse, batch, signal_resolver=MappingSignalResolver(SAFE_SIGNALS)
    )
    observations = {"a": make_observation("a"), "b": make_observation("b")}
    out = shield.decide(observations, DecisionContext())
    assert [a.agent_id for a in out.actions] == ["a", "b"]  # batch order preserved
    assert set(shield._cores) == {"a", "b"}  # one core per agent


def test_properties(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    wrapped = StubPolicy(ActionBatch())
    shield = PolicyShield(wrapped, anchor_compiled_coarse)
    assert shield.wrapped is wrapped
    assert shield.spatial_dim == 3
    assert shield.spec_id == "anchor-lunar-polar-v0"
    assert shield.spec_content_hash.startswith("sha256:")
    assert shield.compiled_content_hash == anchor_compiled_coarse.content_hash()


def test_construction_fails_loud_on_unbounded_history(anchor_compiled: CompiledSafetyModel) -> None:
    # The anchor at the default 1 s period needs a 1.2M-sample monitor history — over the default
    # cap. Construction fails loudly (fail-safe at build time), not mid-episode.
    with pytest.raises(ValueError, match="resource bound"):
        PolicyShield(StubPolicy(ActionBatch()), anchor_compiled)


def test_custom_core_config_raises_cap(anchor_compiled: CompiledSafetyModel) -> None:
    # Raising max_history_cap lets the default-period anchor construct.
    shield = PolicyShield(
        StubPolicy(ActionBatch()),
        anchor_compiled,
        core_config=CoreConfig(max_history_cap=2_000_000),
    )
    assert shield.spatial_dim == 3


def test_watchdog_path_runs(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
        watchdog=True,
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert isinstance(out, ActionBatch)


# --- best-effort audit -----------------------------------------------------------------------


def test_best_effort_sink_never_raises(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    class Boom:
        def write_verdict(self, verdict: object) -> None:
            raise RuntimeError("telemetry backend down")

    proposed = [0.0, 0.0, 0.0]
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", proposed)]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
        sink=Boom(),
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())  # must not raise
    assert out.actions[0].actuator.setpoint == proposed  # type: ignore[union-attr]


def test_no_sink_skips_emission(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])]),
        signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
    )
    assert isinstance(
        shield.decide({"rover": make_observation("rover")}, DecisionContext()), ActionBatch
    )


# --- codec & resolvers -----------------------------------------------------------------------


def test_action_codec_write_preserves_other_fields() -> None:
    codec = ActionCodec()
    action = make_effort_action("rover", [1.0, 2.0, 3.0], target="thruster")
    action.actuator.unit = "m/s^2"  # type: ignore[union-attr]
    out = codec.write_certified_action(action, [9.0, 8.0, 7.0])
    assert out.actuator.setpoint == [9.0, 8.0, 7.0]  # type: ignore[union-attr]
    assert out.actuator.target == "thruster"  # type: ignore[union-attr]
    assert out.actuator.unit == "m/s^2"  # type: ignore[union-attr]


def test_action_codec_reads_position_and_velocity() -> None:
    codec = ActionCodec()
    obs = make_observation("rover", position=(1.0, 2.0, 3.0), velocity=(0.4, 0.5, 0.6))
    assert codec.read_position(obs, 3) == [1.0, 2.0, 3.0]
    assert codec.read_velocity(obs, 3) == [0.4, 0.5, 0.6]
    # No observation / no velocity → zeros.
    assert codec.read_position(None, 3) == [0.0, 0.0, 0.0]
    assert codec.read_velocity(make_observation("rover"), 3) == [0.0, 0.0, 0.0]


def test_default_resolver_from_sensor_and_state_scalar() -> None:
    resolver = DefaultSignalResolver()
    obs = make_observation("rover", signals={"chassis_temp_k": 250.0})
    obs.self_state.battery_soc_j = 400_000.0  # a known StateSample scalar, not a sensor
    values = resolver.resolve(["chassis_temp_k", "battery_soc_j", "unknown_sadf_signal"], obs)
    assert values[0] == 250.0  # from the sensor
    assert values[1] == 400_000.0  # from the state scalar
    assert math.isnan(values[2])  # unresolvable → NaN (fail-safe)
    assert resolver.resolve(["x"], None) == pytest.approx([math.nan], nan_ok=True)


def test_mapping_resolver_unknown_is_nan() -> None:
    resolver = MappingSignalResolver({"a": 1.0})
    values = resolver.resolve(["a", "b"], None)
    assert values[0] == 1.0
    assert math.isnan(values[1])
    assert isinstance(resolver, SignalResolver)


def test_short_resolver_output_fails_safe(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    # A misbehaving resolver returning too few values is padded with NaN → core fails closed.
    class ShortResolver:
        def resolve(self, signal_keys: object, observation: object) -> list[float]:
            return []  # wrong length

    sink = CollectingSink()
    shield = _shield(
        anchor_compiled_coarse,
        ActionBatch(actions=[make_effort_action("rover", [0.0, 0.0, 0.0])]),
        signal_resolver=ShortResolver(),
        sink=sink,
    )  # type: ignore[arg-type]
    shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert sink.verdicts[0].intervention == "fallback"


def test_accel_control_mode_is_effort() -> None:
    assert ACCEL_CONTROL_MODE == ControlMode.EFFORT


def test_modelled_control_modes_are_the_shields_coverage() -> None:
    # The three Core control modes the TCB has a plant model for. Everything else — IMPEDANCE,
    # TRAJECTORY — is classified `opaque` and rejected, so this dict *is* the shield's coverage and
    # the boundary of what can be certified at all.
    assert set(MODELLED_CONTROL_MODES) == {
        ControlMode.EFFORT,
        ControlMode.VELOCITY,
        ControlMode.POSITION,
    }
    assert ControlMode.IMPEDANCE not in MODELLED_CONTROL_MODES
    assert ControlMode.TRAJECTORY not in MODELLED_CONTROL_MODES


def test_codec_classifies_every_action_kind(anchor_compiled_coarse: CompiledSafetyModel) -> None:
    codec = ActionCodec()
    assert codec.classify(make_effort_action("r", [1.0, 2.0, 3.0]), 3)[0] == "effort"
    assert codec.classify(_velocity_action("r", [1.0, 2.0, 3.0]), 3)[0] == "velocity"
    assert codec.classify(_position_action("r", [1.0, 2.0, 3.0]), 3)[0] == "position"
    # A malformed union arm (kind says ACTUATOR, no actuator payload) is `opaque` — the *core*
    # decides what to do about it, and it decides closed.
    malformed = Action(agent_id="r", kind=ActionKind.ACTUATOR)
    assert codec.classify(malformed, 3) == ("opaque", [], None)
    # A non-spatial model can evaluate no spatial setpoint at all.
    assert codec.classify(make_effort_action("r", [1.0, 2.0, 3.0]), 0)[0] == "opaque"


# --- the effective allowlist is `spec ∩ config` (RFC-0004 Amendment 2) -------------------------
#
# The MODE/TASK allowlist is part of the *reviewed, content-addressed, signed* SafetySpec, not of
# deployment configuration. CoreConfig.action_policy survives as a NARROWING-only knob: a directive
# is certifiable iff the reviewed spec admits it AND the configuration admits it. These tests pin
# that end-to-end, through the real trusted core.


def _compiled_granting(
    anchor_document: SafetyDocument, grant: AdmissibleDirectives | None
) -> CompiledSafetyModel:
    """The anchor contract re-authored with ``grant`` as its allowlist (``None`` = silent)."""
    spec = anchor_document.safety.model_copy(update={"admissible_directives": grant})
    doc = anchor_document.model_copy(update={"safety": spec})
    return compile_spec(doc, sample_period_s=COARSE_PERIOD_S)


def _decide_directive(compiled: CompiledSafetyModel, action: Action, policy: ActionPolicy) -> str:
    """Run one directive through a real shield and return the verdict's ``reason``."""
    sink = CollectingSink()
    shield = _shield(
        compiled,
        ActionBatch(actions=[action]),
        sink=sink,
        **_safe_kwargs(core_config=CoreConfig(action_policy=policy)),
    )
    shield.decide({"rover": make_observation("rover")}, DecisionContext())
    return sink.verdicts[0].reason


def _mode(name: str) -> Action:
    return Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode=name))


def _task(kind: TaskKind) -> Action:
    return Action(agent_id="rover", kind=ActionKind.TASK, task=TaskDirective(task_kind=kind))


def test_a_spec_silent_contract_certifies_nothing_however_permissive_the_config(
    anchor_document: SafetyDocument,
) -> None:
    """**The headline invariant.** Silence in the reviewed contract grants NOTHING.

    The configuration below allowlists every directive by name, and the trusted core certifies none
    of them, because the contract authored no grant. Before Amendment 2 this certified all of them —
    which made an unsigned ``params`` dict, absent from the SafetyVerdict record, the only thing
    standing between an untrusted policy and an uncertified passthrough to an actuator.
    """
    silent = _compiled_granting(anchor_document, None)
    permissive = ActionPolicy(
        certified_modes=frozenset({"safe_hold", "velocity", "drill_hard"}),
        certified_tasks=frozenset({t.value for t in TaskKind}),
    )
    for action in (
        _mode("safe_hold"),
        _mode("velocity"),
        _mode("drill_hard"),
        _task(TaskKind.STANDBY),
        _task(TaskKind.EXCAVATE),
    ):
        assert _decide_directive(silent, action, permissive) == "not_certifiable"


def test_configuration_can_only_narrow_the_authored_grant(
    anchor_document: SafetyDocument,
) -> None:
    # Contract admits {safe_hold, drill}; deployment is configured for {drill, teleop}. Only the
    # intersection — `drill` — is certifiable. `safe_hold` is authored-but-unconfigured (a
    # deployment running stricter than its contract: legitimate, and the reason the knob survives);
    # `teleop` is configured-but-unauthored (the fail-open this change removes).
    compiled = _compiled_granting(
        anchor_document, AdmissibleDirectives(modes=["safe_hold", "drill"])
    )
    policy = ActionPolicy(certified_modes=frozenset({"drill", "teleop"}))
    assert _decide_directive(compiled, _mode("drill"), policy) == "certified"
    assert _decide_directive(compiled, _mode("safe_hold"), policy) == "not_certifiable"
    assert _decide_directive(compiled, _mode("teleop"), policy) == "not_certifiable"


def test_an_empty_config_admits_nothing_even_when_the_spec_grants(
    anchor_document: SafetyDocument,
) -> None:
    # The dual: an empty configured allowlist is a total revocation. `min`/`∩` cuts both ways —
    # that is what makes it a *narrowing* knob rather than a no-op.
    compiled = _compiled_granting(
        anchor_document,
        AdmissibleDirectives(modes=["safe_hold"], tasks=[TaskKind.STANDBY]),
    )
    empty = ActionPolicy()
    assert _decide_directive(compiled, _mode("safe_hold"), empty) == "not_certifiable"
    assert _decide_directive(compiled, _task(TaskKind.STANDBY), empty) == "not_certifiable"


def test_an_authored_mode_grant_never_certifies_a_task(
    anchor_document: SafetyDocument,
) -> None:
    # Re-asserted through the new path: modes and tasks are separate permission sets on *both* sides
    # of the intersection, so a MODE grant cannot certify a TASK — even when the configuration lists
    # the same string in both allowlists.
    compiled = _compiled_granting(anchor_document, AdmissibleDirectives(modes=["dock"]))
    policy = ActionPolicy(
        certified_modes=frozenset({"dock"}),
        certified_tasks=frozenset({TaskKind.DOCK.value}),
    )
    assert _decide_directive(compiled, _mode("dock"), policy) == "certified"
    assert _decide_directive(compiled, _task(TaskKind.DOCK), policy) == "not_certifiable"


def test_a_certified_directive_is_re_emitted_untouched_which_is_why_the_grant_is_reviewed(
    anchor_document: SafetyDocument,
) -> None:
    # The fact that motivates the whole amendment: an admitted directive is NOT projected, corrected
    # or rewritten — the marshal layer hands the policy's own proposal straight through, with an
    # empty certified_action from the core. That is a passthrough, and the only thing gating it is
    # the allowlist. So the allowlist is a safety decision and belongs in the reviewed contract.
    compiled = _compiled_granting(anchor_document, AdmissibleDirectives(modes=["safe_hold"]))
    action = _mode("safe_hold")
    sink = CollectingSink()
    shield = _shield(
        compiled,
        ActionBatch(actions=[action]),
        sink=sink,
        **_safe_kwargs(
            core_config=CoreConfig(
                action_policy=ActionPolicy(certified_modes=frozenset({"safe_hold"}))
            )
        ),
    )
    out = shield.decide({"rover": make_observation("rover")}, DecisionContext())
    assert out.actions[0] is action  # byte-for-byte the untrusted policy's proposal
    assert sink.verdicts[0].certified_action == []  # the core certified no numeric command
