"""Core-version compatibility contract test (RM-P0-CORE-07; prerequisite for RM-P1-SURR-*).

Proves Surrogate honors the Core interface versions it builds against, using Core's own
contract-test utilities (``astro_mine.core.compat``). ``assert_core_compatible`` raises if
the installed Core does not satisfy ``CORE_INTERFACES`` — the consumer-driven contract
test the scaffold acceptance requires.
"""

from __future__ import annotations

import pytest

import astro_mine.surrogate as surrogate
from astro_mine.core import compat
from astro_mine.surrogate._core import CORE_INTERFACES, assert_core_compatible


def test_surrogate_declares_the_interfaces_it_builds_against() -> None:
    # A surrogate is a fidelity tier behind the Core Environment (`env`) contract, and it
    # builds a PluginManifest against the `registry` contract (RM-P1-SURR-01). Versions are
    # pinned to this Core, not invented.
    assert CORE_INTERFACES == {"env": "0.1.0", "registry": "0.1.0"}
    assert surrogate.CORE_INTERFACES is CORE_INTERFACES


def test_installed_core_satisfies_surrogates_claimed_interfaces() -> None:
    # The prerequisite: the versions Surrogate builds against are satisfied by this Core.
    assert_core_compatible()  # must not raise
    for interface, version in CORE_INTERFACES.items():
        assert compat.check_compatible(version, compat.CORE_INTERFACE_VERSIONS[interface])


def test_incompatible_interface_is_rejected() -> None:
    # Guard the negotiation actually bites: a future-minor claim must fail at 0.1.0.
    with pytest.raises(compat.IncompatibleCoreInterface):
        compat.assert_core_compatible({"env": "0.2.0"})
