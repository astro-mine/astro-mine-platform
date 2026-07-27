"""Core-version compatibility contract test (issue #1 acceptance; RM-P0-CORE-07).

Proves Fleet honors the Core interface versions it builds against, using Core's own
contract-test utilities (``astro_mine.core.compat``). This is the consumer-driven
contract test the acceptance criteria require: ``assert_core_compatible`` raises if the
installed Core does not satisfy ``CORE_INTERFACES``.
"""

from __future__ import annotations

import pytest

import astro_mine.fleet as fleet
from astro_mine.core import compat
from astro_mine.fleet._core import CORE_INTERFACES, assert_core_compatible


def test_fleet_declares_the_sadf_interface() -> None:
    # Fleet authors content against SADF; the version is pinned, not invented.
    assert CORE_INTERFACES == {"sadf": "0.1.0"}
    assert fleet.CORE_INTERFACES is CORE_INTERFACES


def test_installed_core_satisfies_fleets_claimed_interfaces() -> None:
    # The AC: the versions Fleet builds against are satisfied by this Core.
    assert_core_compatible()  # must not raise
    for interface, version in CORE_INTERFACES.items():
        assert compat.check_compatible(version, compat.CORE_INTERFACE_VERSIONS[interface])


def test_incompatible_interface_is_rejected() -> None:
    # Guard the negotiation actually bites: a future-minor claim must fail at 0.1.0.
    with pytest.raises(compat.IncompatibleCoreInterface):
        compat.assert_core_compatible({"sadf": "0.2.0"})
