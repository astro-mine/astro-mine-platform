"""RM-P0-SIM-01 acceptance criterion 3 — the Core-version compatibility contract.

Sim runs ``assert_core_compatible`` in its own CI to prove it honors the Core interface
versions it is built against (RM-P0-CORE-07). SIM-01 builds against the Environment API
and the message catalog; units/frames/time are shared *primitives*, not version-negotiated
interfaces, so they are deliberately not claimed here.
"""

from __future__ import annotations

import pytest

from astro_mine.core.compat import IncompatibleCoreInterface, assert_core_compatible
from astro_mine.sim.runtime import CORE_INTERFACES


def test_sim_is_core_compatible() -> None:
    # Raises IncompatibleCoreInterface if any claimed interface is unsatisfied. CORE_INTERFACES
    # is the single declared set Sim is built against — also stamped into Trace provenance.
    assert_core_compatible(CORE_INTERFACES)


def test_unknown_interface_is_rejected() -> None:
    with pytest.raises(IncompatibleCoreInterface, match="unknown Core interface"):
        assert_core_compatible({"environment": "0.1.0"})  # misspelled -> unknown
