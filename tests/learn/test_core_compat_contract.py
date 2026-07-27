"""Core-version compatibility contract test (RM-P0-CORE-07).

Proves Learn honors the Core interface versions it builds against, using Core's own
contract-test utilities (``astro_mine.core.compat``). This is the consumer-driven
contract-test hook the scaffold establishes: ``assert_core_compatible`` raises if the
installed Core does not satisfy ``CORE_INTERFACES``. Extend ``CORE_INTERFACES`` as the
RM-P1-LEARN-* feature work consumes more of the waist.
"""

from __future__ import annotations

import pytest

import astro_mine.learn as learn
from astro_mine.core import compat
from astro_mine.learn._core import CORE_INTERFACES, assert_core_compatible


def test_learn_declares_the_interfaces_it_consumes() -> None:
    # Learn wraps the world through the Environment API and — with the SwarmEnv adapter
    # (RM-P1-LEARN-01) — genuinely consumes the message catalog (Observation/Action
    # encode/decode) and SADF (capability-keyed spaces); policies export against the
    # Policy API. The versions are pinned, not invented.
    assert CORE_INTERFACES == {
        "env": "0.1.0",
        "messages": "0.1.0",
        "sadf": "0.1.0",
        "policy": "0.1.0",
    }
    assert learn.CORE_INTERFACES is CORE_INTERFACES


def test_installed_core_satisfies_learns_claimed_interfaces() -> None:
    # The AC: the versions Learn builds against are satisfied by this Core.
    assert_core_compatible()  # must not raise
    for interface, version in CORE_INTERFACES.items():
        assert compat.check_compatible(version, compat.CORE_INTERFACE_VERSIONS[interface])


def test_incompatible_interface_is_rejected() -> None:
    # Guard the negotiation actually bites: a future-minor claim must fail at 0.1.0.
    with pytest.raises(compat.IncompatibleCoreInterface):
        compat.assert_core_compatible({"env": "0.2.0"})
