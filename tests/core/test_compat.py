"""Interface version negotiation + contract-test utilities (RM-P0-CORE-07).

Covers the acceptance criterion "a downstream component passes a contract test
asserting Core-version compatibility": the last test plays the role of a downstream
consumer (Fleet / Sim) declaring the Core interface versions it builds against.
"""

from __future__ import annotations

import pytest

from astro_mine.core import compat


def test_published_interface_versions_are_well_formed() -> None:
    assert compat.CORE_INTERFACE_VERSIONS, "Core must publish at least one interface version"
    for name, version in compat.CORE_INTERFACE_VERSIONS.items():
        assert isinstance(name, str) and name
        # every published version parses as MAJOR.MINOR.PATCH
        compat.parse_version(version)


@pytest.mark.parametrize("bad", ["1.0", "1", "1.0.0.0", "1.x.0", "v1.0.0", ""])
def test_parse_version_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        compat.parse_version(bad)


@pytest.mark.parametrize(
    ("required", "provided", "ok"),
    [
        # exact match always compatible
        ("1.2.3", "1.2.3", True),
        # >=1.0: same major, provided minor >= required minor (additive)
        ("1.2.0", "1.3.0", True),
        ("1.3.0", "1.2.0", False),
        ("1.0.0", "2.0.0", False),  # major bump breaks
        ("2.0.0", "1.9.0", False),
        # patch is ignored
        ("1.2.0", "1.2.9", True),
        ("1.2.9", "1.2.0", True),
        # 0.y: minor is breaking, so it must match exactly
        ("0.1.0", "0.1.5", True),
        ("0.1.0", "0.2.0", False),
        ("0.2.0", "0.1.0", False),
        ("0.1.0", "1.1.0", False),  # 0.x vs 1.x major mismatch
    ],
)
def test_check_compatible_rules(required: str, provided: str, ok: bool) -> None:
    assert compat.check_compatible(required, provided) is ok


def test_check_compatible_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        compat.check_compatible("1.0", "1.0.0")


def test_assert_core_compatible_passes_for_satisfied_claims() -> None:
    # Claim a subset of what this Core publishes, at the published versions.
    claimed = {k: compat.CORE_INTERFACE_VERSIONS[k] for k in ("sadf", "messages", "objective")}
    compat.assert_core_compatible(claimed)  # must not raise


def test_assert_core_compatible_reports_unknown_interface() -> None:
    with pytest.raises(compat.IncompatibleCoreInterface, match="unknown Core interface 'nope'"):
        compat.assert_core_compatible({"nope": "0.1.0"})


def test_assert_core_compatible_reports_version_mismatch() -> None:
    with pytest.raises(compat.IncompatibleCoreInterface, match="sadf"):
        # require a 0.y minor Core does not provide
        compat.assert_core_compatible({"sadf": "0.99.0"})


def test_assert_core_compatible_collects_all_problems() -> None:
    with pytest.raises(compat.IncompatibleCoreInterface) as exc:
        compat.assert_core_compatible({"nope": "0.1.0", "sadf": "0.99.0"})
    message = str(exc.value)
    assert "nope" in message and "sadf" in message


def test_assert_core_compatible_accepts_explicit_provided() -> None:
    # Negotiate against a hypothetical future Core (provided override).
    future = {"sadf": "1.4.0"}
    compat.assert_core_compatible({"sadf": "1.2.0"}, provided=future)  # additive: ok
    with pytest.raises(compat.IncompatibleCoreInterface):
        compat.assert_core_compatible({"sadf": "1.5.0"}, provided=future)


def test_downstream_component_contract() -> None:
    """A stand-in for a downstream component (e.g. Fleet/Sim) proving, in its own CI,
    that it honors the Core interface versions it claims (RM-P0-CORE-07 acceptance;
    Fleet#1 / Sim#1 carry the matching AC)."""
    # What a Fleet-like consumer is built against:
    fleet_claims = {"sadf": "0.1.0", "registry": "0.1.0"}
    compat.assert_core_compatible(fleet_claims)
    # What a Sim-like consumer is built against:
    sim_claims = {"env": "0.1.0", "messages": "0.1.0", "objective": "0.1.0"}
    compat.assert_core_compatible(sim_claims)
