"""Contract smoke tests: each Core subpackage exists and its entry points are wired.

These tests pin the public surface and assert each entry point is implemented (every
RM-P0-CORE-* item has landed): the loaders/validators reject malformed input loudly and
the Environment/Policy conformance utilities refuse a non-conforming object, rather than
raising ``NotImplementedError``. Full behavioural coverage lives in the per-module tests.
"""

from __future__ import annotations

import pytest

from astro_mine.core import compat, env, messages, objective, policy, registry, sadf, units


def test_sadf_loader_is_implemented() -> None:
    # RM-P0-CORE-01 has landed: the loader validates instead of raising
    # NotImplementedError. An empty document fails validation loudly.
    with pytest.raises(sadf.SadfValidationError):
        sadf.load_sadf("{}")


def test_objective_loader_is_implemented() -> None:
    # RM-P0-CORE-04 has landed: the objective loader validates loudly.
    with pytest.raises(objective.ObjectiveValidationError):
        objective.load_objective("{}")


def test_messages_validators_are_implemented() -> None:
    # RM-P0-CORE-04 has landed: the message validators reject malformed input loudly.
    with pytest.raises(messages.MessagesValidationError):
        messages.validate_action_batch({"actions": [{"kind": "warp"}]})


def test_registry_loader_is_implemented() -> None:
    # RM-P0-CORE-05 has landed: the manifest loader validates loudly instead of raising
    # NotImplementedError. An empty document fails validation (see test_registry.py).
    with pytest.raises(registry.ManifestValidationError):
        registry.load_manifest("{}")


def test_compat_is_implemented() -> None:
    # RM-P0-CORE-07 has landed: version negotiation is real (see test_compat.py).
    assert compat.check_compatible("1.0.0", "1.0.0") is True
    assert compat.check_compatible("0.1.0", "0.2.0") is False  # 0.y minor is breaking


def test_units_is_implemented() -> None:
    # RM-P0-CORE-06 has landed: the frame/CRS/epoch primitives validate loudly and
    # spatial data without an explicit CRS is rejected at ingest.
    with pytest.raises(units.UnitsValidationError):
        units.require_crs(None)
    assert units.require_si_unit("m") == "m"


def test_env_is_implemented() -> None:
    # RM-P0-CORE-02 has landed: the Environment contract + conformance utility reject a
    # non-conforming object loudly (see test_env.py for the full contract coverage).
    with pytest.raises(env.EnvironmentContractError):
        env.check_environment(object())  # type: ignore[arg-type]


def test_policy_is_implemented() -> None:
    # RM-P0-CORE-03 has landed: the Policy/Planner contract + conformance utility reject
    # a non-conforming object loudly (see test_policy.py for full contract coverage).
    with pytest.raises(policy.PolicyContractError):
        policy.check_policy(object(), {}, policy.DecisionContext())  # type: ignore[arg-type]


def test_protocols_are_exported() -> None:
    assert hasattr(env, "Environment")
    assert hasattr(policy, "Policy")
