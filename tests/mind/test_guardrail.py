"""Mandatory Guard-wrapping — the single, enforcing, provenance-carrying egress (RM-P1-MIND-05)."""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.registry import UnsignedManifest
from astro_mine.mind.compose import compose
from astro_mine.mind.exec import Executive
from astro_mine.mind.guardrail import InterventionKind, ReportingShield, ShieldReport, shield_egress
from astro_mine.mind.reference import ConstraintShield
from astro_mine.mind.registry import TierRegistry
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import ShieldBinding, StackSpec, StackSpecDocument, TierBinding
from tests.mind.support.harness import policy_plugin, reference_registry, run_stack
from tests.mind.support.toy_env import ToyProspectingEnv


def _velocity(agent_id: str, vx: float, vy: float) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=[vx, vy], unit="m/s"
        ),
    )


def test_constraint_shield_clamps_and_reports() -> None:
    shield = ConstraintShield(max_speed_mps=1.5)
    proposed = ActionBatch(actions=[_velocity("r0", 3.0, 4.0)])  # speed 5.0 > 1.5
    emitted = shield.decide({}, DecisionContext(upstream=proposed))
    speed = math.hypot(*emitted.actions[0].actuator.setpoint)
    assert speed == pytest.approx(1.5)
    report = shield.report()
    assert report == ShieldReport(
        intervened=True, kind=InterventionKind.SHIELD_EDIT, clauses=(ConstraintShield.CLAUSE,)
    )
    assert isinstance(shield, ReportingShield)


def test_constraint_shield_passes_safe_actions_unchanged() -> None:
    shield = ConstraintShield(max_speed_mps=2.0)
    proposed = ActionBatch(actions=[_velocity("r0", 0.5, 0.5)])  # under ceiling
    emitted = shield.decide({}, DecisionContext(upstream=proposed))
    assert emitted == proposed
    assert shield.report() == ShieldReport(intervened=False)


def test_shield_egress_records_report_provenance() -> None:
    shield = ConstraintShield(max_speed_mps=1.0)
    proposed = ActionBatch(actions=[_velocity("r0", 2.0, 0.0)])
    emitted, record = shield_egress(shield, {}, DecisionContext(), proposed, shield_name="ref")
    assert record.intervened is True
    assert record.kind == InterventionKind.SHIELD_EDIT.value
    assert record.clauses == (ConstraintShield.CLAUSE,)
    assert emitted.actions[0].actuator.setpoint[0] == pytest.approx(1.0)


def test_shield_egress_falls_back_to_diff_detection() -> None:
    # A non-reporting shield: the record infers intervention from a batch change.
    class _EditingShield:
        def decide(self, observations, context):  # type: ignore[no-untyped-def]
            return ActionBatch()  # drops all actions

    proposed = ActionBatch(actions=[_velocity("r0", 0.1, 0.0)])
    _, record = shield_egress(_EditingShield(), {}, DecisionContext(), proposed, shield_name="x")
    assert record.intervened is True
    assert record.kind is None and record.clauses == ()


def test_trace_carries_guard_clause_provenance() -> None:
    # The backend stack binds the enforcing constraint shield; its interventions land in the
    # trace with the invoked clause id.
    result = run_stack("lunar_prospecting_backends.yaml", horizon=5, max_ticks=5)
    intervened = [t.shield for t in result.trace.ticks if t.shield.intervened]
    assert intervened, "expected the constraint shield to clamp at least one tick"
    assert all(s.clauses == (ConstraintShield.CLAUSE,) for s in intervened)


def test_guard_enforces_independently_of_an_adversarial_tier() -> None:
    # A control tier that tries to command a reckless over-speed cannot get it past the shield:
    # the shield's ceiling is enforced independently of the tier (a learned/classical controller
    # cannot raise its own limit), and the shield's output is the ONLY thing that leaves Mind.
    class _RecklessController:
        def decide(self, observations, context):  # type: ignore[no-untyped-def]
            return ActionBatch(actions=[_velocity(a, 99.0, 99.0) for a in sorted(observations)])

    registry = reference_registry()
    registry.register(
        policy_plugin("test.reckless", lambda params: _RecklessController(), tier="control")
    )
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="adversarial",
            name="adversarial",
            tiers=[TierBinding(role=TierRole.CONTROL, plugin="test.reckless")],
            shield=ShieldBinding(
                plugin="mind.reference.constraint_shield", params={"max_speed_mps": 1.5}
            ),
        ),
    )
    result = Executive(compose(doc, registry, seed=7)).run(
        ToyProspectingEnv(horizon=3), max_ticks=3, seed=7
    )
    speeds = [
        math.hypot(*a.actuator.setpoint)
        for tick in result.trace.ticks
        for a in tick.action_batch.actions
        if a.actuator is not None
    ]
    assert speeds and all(s <= 1.5 + 1e-9 for s in speeds)  # nothing reckless reached the env
    assert all(t.shield.intervened for t in result.trace.ticks)


def test_signature_gate_rejects_unsigned_shield_in_production() -> None:
    # Production posture: a signature-requiring registry refuses the unsigned reference plugins.
    with pytest.raises(UnsignedManifest):
        TierRegistry.from_entry_points(require_signature=True)
