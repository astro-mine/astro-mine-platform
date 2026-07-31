"""The RFC-0006 sibling binding: Mind's registry discovers and drives the **real** Guard TCB.

The consumer-driven contract test for the ``[mind]`` extra (RM-P1-GUARD-03; RFC-0006; mind.md §7;
conventions.md §7). Until this existed, Mind's mandatory Guard-wrapping (RM-P1-MIND-05) was proven
only against Mind's in-repo ``ConstraintShield`` stand-in — a reference clamp, not the Rust TCB.
What is asserted here is the *binding*, from the consumer's side:

1. Mind's :class:`~astro_mine.mind.registry.registry.TierRegistry` **discovers** ``guard.shield``
   through the ``astro_mine.mind.tier_plugins`` entry point Guard's ``[mind]`` extra publishes, and
   the manifest passes Core's registry gates.
2. ``instantiate`` yields a **working** :class:`PolicyShield` — a Core ``Policy`` *and* a Mind
   :class:`~astro_mine.core.policy.guardrail.ReportingShield` — enforcing the real,
   content-addressed anchor ``SafetySpec``.
3. Driven through Mind's own egress (:func:`~astro_mine.mind.guardrail.shield.shield_egress`), it
   **actually intervenes**: an unsafe proposal is corrected by the Rust core, and the intervention
   surfaces in Mind's :class:`ShieldReport` with the invoked ``SafetySpec`` clause ids.
4. A stack that binds the shield but authors **no contract** fails loud — a shield with no spec is
   not a shield, and must never degrade to a permissive default.

There is no ``mind → guard`` import anywhere in base Guard: the edge exists only inside the extra
(and this test), which is the whole point of the convention.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.enums import PluginKind
from tests.guard.conftest import ANCHOR_PATH, COARSE_SAMPLE_PERIOD_S, SAFE_SIGNALS, make_observation

pytest.importorskip("astro_mine.guard._core", reason="Rust safety core not built (run `uv sync`)")
pytest.importorskip("astro_mine.mind", reason="the [mind] extra is not installed")

from astro_mine.core.policy.guardrail import InterventionKind, ReportingShield
from astro_mine.guard.mind import PLUGIN_NAME, GuardShield
from astro_mine.guard.wrap import MappingSignalResolver
from astro_mine.mind.guardrail.shield import shield_egress
from astro_mine.mind.registry.registry import TierRegistry

_AGENT = "rover"


def _params(**extra: object) -> dict[str, object]:
    return {
        "spec_path": str(ANCHOR_PATH),
        "sample_period_s": COARSE_SAMPLE_PERIOD_S,
        **extra,
    }


def _registry() -> TierRegistry:
    """Mind's own discovery mechanism — nothing Guard-specific about the call."""
    return TierRegistry.from_entry_points()


def _shield(**extra: object) -> GuardShield:
    shield = _registry().instantiate(PLUGIN_NAME, _params(**extra))
    assert isinstance(shield, GuardShield)
    # Resolve the anchor's SADF/Worlds signals so the *spatial* shield (not the detect layer) is
    # what this test exercises; GUARD-04's resolver is the production path.
    shield.shield._resolver = MappingSignalResolver(SAFE_SIGNALS)
    return shield


def _proposal(setpoint: list[float], mode: ControlMode = ControlMode.VELOCITY) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id=_AGENT,
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(target="base", control_mode=mode, setpoint=setpoint),
            )
        ]
    )


# --- 1. discovery through Mind's entry-point group -------------------------------------------


def test_mind_registry_discovers_the_real_guard_shield() -> None:
    registry = _registry()
    assert PLUGIN_NAME in registry
    manifest = registry.manifest(PLUGIN_NAME)
    # It passed Core's registry gates (validity → interface negotiation → signature) as a policy.
    assert manifest.kind is PluginKind.POLICY
    assert manifest.attributes["tier"] == "shield"
    assert manifest.core_interfaces["policy"] == "0.1.0"


# --- 2. instantiation yields a working PolicyShield -------------------------------------------


def test_entry_point_resolves_to_a_working_policy_shield() -> None:
    shield = _shield()
    # A Core Policy (Mind's executive binds it as the shield stage) …
    assert isinstance(shield, Policy)
    # … and a Mind ReportingShield (so its interventions land in the decision trace).
    assert isinstance(shield, ReportingShield)
    # It is enforcing the *real*, content-addressed anchor contract — not a stand-in clamp.
    assert shield.shield.spec_id == "anchor-lunar-polar-v0"
    assert shield.shield.spec_content_hash.startswith("sha256:")
    # …including the anchor's reviewed 0.1 m/s traverse limit as the commanded-speed ceiling.
    assert shield.shield.v_max == pytest.approx(0.1)


# --- 3. it really enforces, through Mind's own egress ------------------------------------------


def test_the_real_tcb_intervenes_through_mind_egress() -> None:
    shield = _shield()
    # An unsafe proposal: 5 m/s straight at the lander-zone keep-out — 50x the reviewed limit.
    proposed = _proposal([-3.0, -4.0, 0.0])
    observations = {_AGENT: make_observation(_AGENT, position=(40.0, 0.0, 10.0))}
    emitted, record = shield_egress(
        shield,
        observations,
        DecisionContext(upstream=proposed),
        proposed,
        shield_name=PLUGIN_NAME,
    )
    actuator = emitted.actions[0].actuator
    assert actuator is not None
    # The Rust core corrected it: the certified speed respects the reviewed ceiling.
    speed = math.hypot(*actuator.setpoint[:2])
    assert speed <= 0.1 + 1e-9
    assert actuator.setpoint != proposed.actions[0].actuator.setpoint  # type: ignore[union-attr]
    # …and Mind's decision trace records the intervention through the ReportingShield seam.
    assert record.intervened
    report = shield.report()
    assert report is not None and report.intervened
    assert report.kind is InterventionKind.SHIELD_EDIT
    assert report.certificate is not None and report.certificate.startswith("sha256:")


def test_a_safe_proposal_is_certified_unchanged() -> None:
    shield = _shield()
    proposed = _proposal([0.05, 0.0, 0.0])  # inside the reviewed envelope, far from the keep-out
    observations = {_AGENT: make_observation(_AGENT, position=(400.0, 400.0, 400.0))}
    emitted, record = shield_egress(
        shield,
        observations,
        DecisionContext(upstream=proposed),
        proposed,
        shield_name=PLUGIN_NAME,
    )
    assert emitted.actions[0].actuator.setpoint == pytest.approx([0.05, 0.0, 0.0])  # type: ignore[union-attr]
    assert not record.intervened
    report = shield.report()
    assert report is not None and not report.intervened


def test_the_report_carries_the_invoked_safetyspec_clauses() -> None:
    # A detect-layer breach (torque over the anchor's 40 N·m ceiling) hands control to the verified
    # backup, and the clause id travels into Mind's trace — provenance the ConstraintShield
    # stand-in could never supply.
    shield = _shield()
    shield.shield._resolver = MappingSignalResolver({**SAFE_SIGNALS, "anchor_torque_nm": 100.0})
    proposed = _proposal([0.0, 0.0, 0.0])
    shield_egress(
        shield,
        {_AGENT: make_observation(_AGENT)},
        DecisionContext(upstream=proposed),
        proposed,
        shield_name=PLUGIN_NAME,
    )
    report = shield.report()
    assert report is not None
    assert report.intervened
    assert report.kind is InterventionKind.BACKUP_ACTIVATION
    assert report.clauses == ("c_anchor_torque",)


# --- 4. fail-closed construction --------------------------------------------------------------


def test_a_shield_with_no_contract_fails_loud() -> None:
    # Binding guard.shield without a SafetySpec must raise, never yield a permissive shield.
    with pytest.raises(ValueError, match="requires a SafetySpec"):
        _registry().instantiate(PLUGIN_NAME, {})


def test_the_action_gate_is_configurable_from_a_mind_stack_spec() -> None:
    # The stack spec's `params` reach the action gate, so a Mind stack declares which discrete
    # directives its Guard certifies — fail-closed by default (nothing certified).
    shield = _shield(certified_tasks=["standby"], fallback_control_mode="velocity")
    assert shield.shield.action_policy.certified_tasks == frozenset({"standby"})
    assert shield.shield.action_policy.fallback_control_mode is ControlMode.VELOCITY
