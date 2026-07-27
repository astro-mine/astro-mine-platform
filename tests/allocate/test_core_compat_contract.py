"""Core-version compatibility contract test (RM-P0-CORE-07; prerequisite for RM-P1-ALLOC-*).

Proves Allocate honors the Core interface versions it builds against, using Core's own
contract-test utilities (``astro_mine.core.compat``). ``assert_core_compatible`` raises if
the installed Core does not satisfy ``CORE_INTERFACES`` — the consumer-driven contract test
the acceptance criteria require.
"""

from __future__ import annotations

import pytest

import astro_mine.allocate as allocate
from astro_mine.allocate.api._core import CORE_INTERFACES, assert_core_compatible
from astro_mine.core import compat


def test_allocate_declares_the_interfaces_it_builds_against() -> None:
    # Allocate implements the `policy` (Policy/Planner) contract as the allocation
    # sub-interface and builds a PluginManifest against the `registry` contract
    # (RM-P1-ALLOC-01). Versions are pinned to this Core, not invented.
    assert CORE_INTERFACES == {"policy": "0.1.0", "registry": "0.1.0"}
    assert allocate.CORE_INTERFACES is CORE_INTERFACES


def test_installed_core_satisfies_allocates_claimed_interfaces() -> None:
    assert_core_compatible()  # must not raise
    for interface, version in CORE_INTERFACES.items():
        assert compat.check_compatible(version, compat.CORE_INTERFACE_VERSIONS[interface])


def test_incompatible_interface_is_rejected() -> None:
    # Guard the negotiation actually bites: a future-minor claim must fail at 0.1.0.
    with pytest.raises(compat.IncompatibleCoreInterface):
        compat.assert_core_compatible({"policy": "0.2.0"})
