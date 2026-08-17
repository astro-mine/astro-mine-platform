# SPDX-License-Identifier: Apache-2.0
"""The Sim-side engine registry — the plugin seam over Core's registry (RM-P0-SIM-02).

:class:`EngineRegistry` is where a host (the stepping core, RM-P0-SIM-03's engine set,
Bench) registers and resolves :class:`~astro_mine.sim.engines.adapter.RegimeEngine`
factories. Registration renders the engine's
:class:`~astro_mine.sim.engines.adapter.EngineDescriptor` as a Core plugin manifest and
runs it through Core's :class:`~astro_mine.core.registry.PluginRegistry`, so every engine
load is **gated** exactly like any other plugin (RM-P0-CORE-05): the manifest is validated
(structural + the reserved/gated capability-tag gate), version-negotiated against *this*
Core, and signature-checked. Core resolves the *manifest*; the host instantiates the
engine — the registry never executes engine code beyond the factory the host supplied
(core.md §9).

Signing is **off by default** here (``require_signature=False``): the reference engine and
the Phase-0 local tier must always run with no key material (CX-LOCAL). A hardened host
flips it on and supplies a ``verifier`` (Cloud/Fleet, P1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from astro_mine.core.registry import PluginManifest, PluginRegistry
from astro_mine.sim.engines.adapter import EngineDescriptor, RegimeEngine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astro_mine.core.registry import Signature, Verifier
    from astro_mine.sim.runtime.rng import RngStreams
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["EngineFactory", "EngineRegistry"]

#: Builds a fresh engine for a scenario, seeded by the stepping core's RNG streams. The
#: scenario/RNG forward-refs keep ``engines`` from importing ``runtime`` at module load —
#: ``runtime`` drives ``engines``, not the reverse.
EngineFactory = Callable[["Scenario", "RngStreams"], RegimeEngine]


class EngineRegistry:
    """An in-memory registry of regime-engine factories, gated by Core's registry.

    ``provided`` overrides the Core interface versions negotiated against (defaults to this
    build's set — useful for tests). ``require_signature`` / ``verifier`` are handed
    straight to the backing :class:`~astro_mine.core.registry.PluginRegistry`."""

    def __init__(
        self,
        *,
        provided: Mapping[str, str] | None = None,
        require_signature: bool = False,
        verifier: Verifier | None = None,
    ) -> None:
        self._core = PluginRegistry(
            provided=provided,
            require_signature=require_signature,
            verifier=verifier,
        )
        self._factories: dict[str, EngineFactory] = {}

    def register(
        self,
        descriptor: EngineDescriptor,
        factory: EngineFactory,
        *,
        signature: Signature | None = None,
    ) -> PluginManifest:
        """Gate and register an engine.

        Renders ``descriptor`` to a Core manifest, runs it through every Core load gate
        (validity, version negotiation, signature policy), then stores ``factory`` under
        the engine name. The factory is recorded only if the manifest passes, so a
        rejected engine leaves no partial registration. Raises the Core registry's errors
        (e.g. a gated capability tag, an incompatible Core version, a missing signature, or
        a duplicate name)."""
        manifest = self._core.register(descriptor.to_manifest(signature=signature))
        self._factories[descriptor.name] = factory
        return manifest

    def create(self, name: str, scenario: Scenario, rng: RngStreams) -> RegimeEngine:
        """Resolve a registered engine by name and instantiate it for ``scenario``.

        Raises :class:`~astro_mine.core.registry.RegistryError` if no engine is registered
        under ``name``."""
        self._core.resolve(name)
        return self._factories[name](scenario, rng)

    def manifest(self, name: str) -> PluginManifest:
        """The gated Core manifest registered under ``name``."""
        return self._core.resolve(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered engine names, in registration order."""
        return tuple(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories
