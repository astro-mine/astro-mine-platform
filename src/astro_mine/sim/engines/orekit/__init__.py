"""The Orekit higher-fidelity orbital tier (RM-P0-SIM-03) — the flight-grade orbital backend.

Behind the *same* ``RegimeEngine`` waist as the reduced-order RK4 two-body engine, so routing it is
**configuration** (a scenario's ``dynamics.kind``), never a Sim code change, and no Orekit type
leaks through the Core Environment API (sim.md §2 principle 1). The reduced-order tier remains the
always-works local fallback (CX-LOCAL); this is the tier you select when the two-body approximation
is not good enough — it integrates with an error-controlled adaptive Dormand-Prince 8(5,3) scheme
and carries the central body's **J2 oblateness** perturbation, which pure two-body motion cannot
express.

**JVM-free package surface by design.** This module exposes only the engine's
:data:`OREKIT_ORBITAL_ENGINE_DESCRIPTOR` (registered in ``engines/builtins.py``) and a factory whose
body imports the Orekit binding *lazily* — so importing the engine set, and registering the Orekit
engine's manifest, boot no JVM. Orekit + its bundled JVM arrive with the ``[orekit]`` extra; a
scenario that actually selects the tier calls the factory, which then requires them and otherwise
raises a clean :class:`ModuleNotFoundError` naming the extra.

**Nothing to download.** The Cartesian/Keplerian propagation path touches no leap-second table, no
Earth-orientation history and no gravity-field file, so the tier needs **no ``orekit-data`` bundle**
and runs fully offline.

(RM-P0-SIM-03 names *Basilisk or Orekit*. Basilisk is not distributed on PyPI — it requires a
from-source CMake build — so Orekit is the integrated backend; Basilisk can join later behind this
same descriptor pattern.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.sim.engines.orekit._descriptor import OREKIT_ORBITAL_ENGINE_DESCRIPTOR

if TYPE_CHECKING:
    from astro_mine.sim.engines.adapter import RegimeEngine
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = [
    "OREKIT_ORBITAL_ENGINE_DESCRIPTOR",
    "orekit_orbital_engine_factory",
]

_OREKIT_HINT = (
    "the Orekit orbital tier requires the Orekit JPype binding and a JVM "
    "(orekit-jpype, jdk4py); install it with: pip install 'astro-mine-sim[orekit]'"
)


def orekit_orbital_engine_factory(scenario: Scenario, rng: RngStreams) -> RegimeEngine:
    """Build the Orekit orbital engine for a scenario's ``orekit_orbital`` agents (``[orekit]``).

    Lazy-imports the Orekit-backed engine (and boots the JVM) so the engine set stays importable —
    and the manifest registrable — with no JVM present. Raises a clear :class:`ModuleNotFoundError`
    naming ``astro-mine-sim[orekit]`` only when a scenario actually selects the tier without it."""
    try:
        from astro_mine.sim.engines.orekit._engine import build_orekit_orbital_engine
    except ModuleNotFoundError as exc:  # orekit_jpype / jdk4py absent
        raise ModuleNotFoundError(_OREKIT_HINT) from exc
    return build_orekit_orbital_engine(scenario, rng)
