"""Deterministic PolicyShield run fixture — shared by the golden gate and its generator.

A fixed observation sequence driven through a ``PolicyShield`` over the anchor SafetySpec, hitting
all three assurance layers (primary certify / shield correct / backup fall-back) and the
:class:`~astro_mine.guard.wrap.DefaultSignalResolver` (signals ride as
:class:`~astro_mine.core.messages.SensorReading`\\ s). Both ``scripts/gen_shield_golden.py`` (which
pins the golden) and ``tests/test_shield_golden.py`` (which re-runs it) import this one builder, so
there is a single source of truth for the reproducibility gate (RM-P1-GUARD-03/-06 determinism).
"""

from __future__ import annotations

from importlib import resources

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    Observation,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.guard.audit.model import SafetyVerdict
from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.spec import compile_spec, load_safety_spec
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.wrap import DefaultSignalResolver, PolicyShield

ANCHOR = resources.files("astro_mine.guard.reference").joinpath(
    "safety_specs", "anchor.safety.yaml"
)
AGENT = "rover"
#: A coarse sample period keeps the two 14-day survival monitors' history windows tiny for the run.
SAMPLE_PERIOD_S = 120_960.0

_SAFE_SIGNALS = {
    "anchor_torque_nm": 10.0,
    "battery_soc_j": 500_000.0,
    "charging_window_active": 1.0,
    "chassis_temp_k": 250.0,
    "power_available_w": 50.0,
    "traverse_speed_mps": 0.05,
}


def _sensors(overrides: dict[str, float] | None = None) -> list[SensorReading]:
    values = {**_SAFE_SIGNALS, **(overrides or {})}
    return [SensorReading(sensor=k, values=[v]) for k, v in values.items()]


def _observation(
    tick: int, position: tuple[float, float, float], **overrides: float
) -> Observation:
    frame = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")
    return Observation(
        tick=tick,
        sim_time_s=float(tick),
        agent_id=AGENT,
        self_state=StateSample(
            agent_id=AGENT,
            frame=frame,
            pose=Transform(
                translation_m=Vec3(x=position[0], y=position[1], z=position[2]),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            linear_velocity_mps=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        sensors=_sensors(overrides or None),
    )


def _action(setpoint: list[float]) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id=AGENT,
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="body", control_mode=ControlMode.EFFORT, setpoint=setpoint
                ),
            )
        ]
    )


class _ScriptedPolicy:
    """A wrapped policy replaying a fixed per-tick proposed action (adversarial to the shield)."""

    def __init__(self, batches: list[ActionBatch]) -> None:
        self._batches = batches
        self._i = 0

    def decide(self, observations: object, context: object) -> ActionBatch:
        batch = self._batches[self._i]
        self._i += 1
        return batch


def compiled_anchor() -> CompiledSafetyModel:
    """The anchor SafetySpec compiled at the fixture's coarse sample period."""
    return compile_spec(
        load_safety_spec(ANCHOR.read_text(encoding="utf-8")), sample_period_s=SAMPLE_PERIOD_S
    )


#: The three fixed (observation, proposed action) steps: primary certify, shield correct, backup.
_STEPS: list[tuple[Observation, ActionBatch]] = [
    (_observation(0, (100.0, 100.0, 100.0)), _action([0.0, 0.0, 0.0])),  # safe → certified
    (_observation(1, (31.0, 0.0, 0.0)), _action([-5.0, 0.0, 0.0])),  # into lander sphere → shield
    (
        _observation(2, (100.0, 100.0, 100.0), anchor_torque_nm=100.0),
        _action([0.0, 0.0, 0.0]),
    ),  # torque > 40 → backup
]


def run_shield() -> list[SafetyVerdict]:
    """Drive the fixed sequence through a fresh ``PolicyShield`` and return the collected verdicts.

    Deterministic and independent per call (fresh cores, no shared state, ``watchdog=False``) — the
    reproducibility contract the golden gate pins."""
    sink = CollectingSink()
    shield = PolicyShield(
        _ScriptedPolicy([batch for _, batch in _STEPS]),
        compiled_anchor(),
        signal_resolver=DefaultSignalResolver(),
        sink=sink,
        watchdog=False,
    )
    for observation, _ in _STEPS:
        shield.decide({AGENT: observation}, DecisionContext(sim_time_s=observation.sim_time_s))
    return sink.verdicts
