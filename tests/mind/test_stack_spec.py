"""Stack-spec schema + loader validation (RM-P1-MIND-01)."""

from __future__ import annotations

import pytest

from astro_mine.mind.spec import (
    StackSpecValidationError,
    load_stack_spec,
    validate_stack_spec,
)
from astro_mine.mind.spec.enums import ReplanTriggerKind, TierRole

_VALID = """
stack_spec_version: "0.1"
stack_spec:
  id: s
  name: Stack
  tiers:
    - role: mission
      plugin: p-mission
      validity_horizon_s: 5.0
      replan_triggers:
        - kind: plan_expired
    - role: tamp
      plugin: p-tamp
      replan_triggers:
        - kind: periodic
          every_ticks: 3
    - role: control
      plugin: p-control
      fallback:
        plugin: p-control-safe
  shield:
    plugin: p-shield
"""


def test_valid_stack_spec_round_trips() -> None:
    doc = load_stack_spec(_VALID)
    spec = doc.stack_spec
    assert doc.stack_spec_version == "0.1"
    assert [t.role for t in spec.tiers] == [TierRole.MISSION, TierRole.TAMP, TierRole.CONTROL]
    assert spec.tiers[0].validity_horizon_s == 5.0
    assert spec.tiers[0].replan_triggers[0].kind is ReplanTriggerKind.PLAN_EXPIRED
    assert spec.tiers[1].replan_triggers[0].every_ticks == 3
    assert spec.tiers[2].fallback is not None
    assert spec.tiers[2].fallback.plugin == "p-control-safe"
    assert spec.shield.plugin == "p-shield"
    # defaults reserved for later waves
    assert spec.execution.kind.value == "composition"
    assert spec.coordination.kind.value == "centralized"
    validate_stack_spec(doc)  # a typed doc re-validates cleanly


def test_single_tier_collapse_is_valid() -> None:
    # principle 3: a stack MAY collapse to one end-to-end tier.
    doc = load_stack_spec(
        """
        stack_spec_version: "0.1"
        stack_spec:
          id: collapsed
          name: One tier
          tiers:
            - {role: control, plugin: e2e}
          shield: {plugin: p-shield}
        """
    )
    assert [t.role for t in doc.stack_spec.tiers] == [TierRole.CONTROL]


@pytest.mark.parametrize(
    ("why", "source"),
    [
        (
            "duplicate role",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a}, {role: mission, plugin: b}]
              shield: {plugin: s}
            """,
        ),
        (
            "periodic without every_ticks",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: tamp, plugin: a, replan_triggers: [{kind: periodic}]}]
              shield: {plugin: s}
            """,
        ),
        (
            "plan_expired without horizon",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a, replan_triggers: [{kind: plan_expired}]}]
              shield: {plugin: s}
            """,
        ),
        (
            "missing shield",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a}]
            """,
        ),
        (
            "empty tiers",
            """
            stack_spec_version: "0.1"
            stack_spec: {id: s, name: s, tiers: [], shield: {plugin: s}}
            """,
        ),
        (
            "unknown role",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: bogus, plugin: a}]
              shield: {plugin: s}
            """,
        ),
        (
            "typo'd field",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a, plugn: oops}]
              shield: {plugin: s}
            """,
        ),
        (
            "wrong version",
            """
            stack_spec_version: "9.9"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a}]
              shield: {plugin: s}
            """,
        ),
        (
            "non-positive horizon",
            """
            stack_spec_version: "0.1"
            stack_spec:
              id: s
              name: s
              tiers: [{role: mission, plugin: a, validity_horizon_s: 0}]
              shield: {plugin: s}
            """,
        ),
    ],
)
def test_invalid_stack_specs_are_rejected(why: str, source: str) -> None:
    with pytest.raises(StackSpecValidationError):
        load_stack_spec(source)
