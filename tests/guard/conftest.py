"""Shared fixtures for the SafetySpec tests."""

from __future__ import annotations

from importlib import resources

import pytest

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
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.guard.spec import (
    CompiledSafetyModel,
    SafetyDocument,
    compile_spec,
    load_safety_spec,
)

#: The reviewed anchor SafetySpec, resolved from Guard's package data (the shipped reference spec).
ANCHOR_PATH = resources.files("astro_mine.guard.reference").joinpath(
    "safety_specs", "anchor.safety.yaml"
)

#: A coarse sample period keeps the anchor's two 14-day survival monitors' history windows tiny, so
#: a per-agent SafetyCore constructs without a huge pre-allocated ring buffer (per test_rust_core).
COARSE_SAMPLE_PERIOD_S = 120_960.0

#: A signal map satisfying every anchor scalar bound and temporal clause (the certified path).
SAFE_SIGNALS: dict[str, float] = {
    "anchor_torque_nm": 10.0,
    "battery_soc_j": 500_000.0,
    "charging_window_active": 1.0,
    "chassis_temp_k": 250.0,
    "power_available_w": 50.0,
    "traverse_speed_mps": 0.05,
}


@pytest.fixture
def anchor_document() -> SafetyDocument:
    """The flagship lunar-polar anchor SafetySpec, loaded and validated."""
    return load_safety_spec(ANCHOR_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def anchor_compiled(anchor_document: SafetyDocument) -> CompiledSafetyModel:
    """The anchor SafetySpec compiled to its IR (default sample period)."""
    return compile_spec(anchor_document)


@pytest.fixture
def anchor_compiled_coarse(anchor_document: SafetyDocument) -> CompiledSafetyModel:
    """The anchor SafetySpec compiled at the coarse sample period (shieldable, small cap)."""
    return compile_spec(anchor_document, sample_period_s=COARSE_SAMPLE_PERIOD_S)


def make_observation(
    agent_id: str = "rover",
    *,
    tick: int = 0,
    sim_time_s: float = 0.0,
    position: tuple[float, float, float] = (100.0, 100.0, 100.0),
    velocity: tuple[float, float, float] | None = None,
    signals: dict[str, float] | None = None,
) -> Observation:
    """Build a per-agent Observation; ``signals`` ride as one-value SensorReadings."""
    frame = ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center="MOON")
    return Observation(
        tick=tick,
        sim_time_s=sim_time_s,
        agent_id=agent_id,
        self_state=StateSample(
            agent_id=agent_id,
            frame=frame,
            pose=Transform(
                translation_m=Vec3(x=position[0], y=position[1], z=position[2]),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            linear_velocity_mps=(
                None if velocity is None else Vec3(x=velocity[0], y=velocity[1], z=velocity[2])
            ),
        ),
        sensors=[SensorReading(sensor=k, values=[v]) for k, v in (signals or {}).items()],
    )


def make_effort_action(agent_id: str, setpoint: list[float], *, target: str = "body") -> Action:
    """An ACTUATOR/EFFORT action carrying a commanded-acceleration setpoint (shieldable case)."""
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(target=target, control_mode=ControlMode.EFFORT, setpoint=setpoint),
    )


class StubPolicy:
    """A wrapped policy returning a fixed :class:`ActionBatch` (adversarial input to the shield)."""

    def __init__(self, batch: ActionBatch) -> None:
        self.batch = batch

    def decide(self, observations: object, context: object) -> ActionBatch:
        return self.batch


def make_verdict(**overrides: object):
    """Build a valid :class:`~astro_mine.guard.audit.model.SafetyVerdict` with sane defaults."""
    from astro_mine.guard.audit.model import SafetyVerdict

    fields: dict[str, object] = {
        "verdict_version": "0.1",
        "agent_id": "rover",
        "tick": 0,
        "sim_time_s": 0.0,
        "spec_id": "anchor-lunar-polar-v0",
        "spec_content_hash": "sha256:" + "a" * 64,
        "compiled_content_hash": "sha256:" + "b" * 64,
        "guard_code_version": "0.1.0",
        "layer": "primary",
        "intervention": "none",
        "reason": "certified",
        "backup_kind": None,
        "constraint_ids": [],
        "certified_action": [0.0, 0.0, 0.0],
        "min_barrier_margin": 1.5,
        "action_divergence": 0.0,
        "inputs_content_hash": "sha256:" + "c" * 64,
        "shield_latency_us": 3.0,
    }
    fields.update(overrides)
    return SafetyVerdict(**fields)  # type: ignore[arg-type]
