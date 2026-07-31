"""Tier-plugin discovery, gating, and instantiation (RM-P1-MIND-01).

Core's :class:`~astro_mine.core.registry.PluginRegistry` resolves plugin **manifests** and
runs the three load gates (validity → Core-interface-version negotiation → signature) — but
it deliberately never instantiates plugin code (core.md §9). :class:`TierRegistry` is the
Mind-side host that adds exactly that missing half: a manifest → concrete
:class:`~astro_mine.core.policy.protocol.Policy` **factory** map, discovered via Python
entry points (conventions.md §7's in-process plugin mechanism).

A tier plugin declares itself with a Core :class:`~astro_mine.core.registry.TierPlugin` — its
manifest (the self-declaration Core gates) plus a factory that builds the tier's ``Policy``.
The declaration is Core's because Allocate and Guard both implement it from outside Mind
(conventions.md §3.3); the gating, discovery and instantiation below are Mind's, and stayed.
Registration runs the manifest through the Core gates and requires ``kind == policy`` (all three
tiers and the Guard shield are ``PluginKind.POLICY``; RFC-0004); instantiation calls the factory
and re-checks the result satisfies the ``Policy`` contract. The out-of-process
(gRPC + sandbox) transport for untrusted/non-Python plugins (conventions.md §7, §9) is a
reserved seam — not implemented in v0.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.loader import RegistryError
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.registry import PluginRegistry, Verifier
from astro_mine.core.registry.tier import TierFactory, TierPlugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "NotAPolicyPlugin",
    "PluginNotRegistered",
    "TierRegistry",
    "TierRegistryError",
]

#: The entry-point group Mind tier/shield plugins register under. Each entry point resolves
#: to a zero-argument provider callable returning a Core
#: :class:`~astro_mine.core.registry.TierPlugin`.
ENTRY_POINT_GROUP = "astro_mine.mind.tier_plugins"


class TierRegistryError(Exception):
    """Base class for tier-registry errors."""


class PluginNotRegistered(TierRegistryError):
    """Raised when a stack spec names a plugin no registered manifest provides."""


class NotAPolicyPlugin(TierRegistryError):
    """Raised when a plugin's manifest is not ``kind == policy`` or its factory returns a
    value that does not satisfy the Core Policy/Planner contract."""


class TierRegistry:
    """Registers, gates, and instantiates Mind tier/shield plugins.

    Wraps a Core :class:`PluginRegistry` for the manifest gates and keeps the factory map
    Core does not. ``require_signature`` defaults to ``False`` for local/dev use (the
    reference plugins ship ``unsigned`` manifests, and Tier-1 local MUST work,
    conventions.md §7); production and RM-P1-MIND-05 flip it on. ``provided`` overrides the
    Core interface versions negotiated against (defaults to this Core build).
    """

    def __init__(
        self,
        *,
        require_signature: bool = False,
        provided: Mapping[str, str] | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self._core = PluginRegistry(
            provided=provided, require_signature=require_signature, verifier=verifier
        )
        self._factories: dict[str, TierFactory] = {}

    def register(self, plugin: TierPlugin) -> PluginManifest:
        """Gate ``plugin``'s manifest through Core and store its factory; return the
        manifest. Raises :class:`NotAPolicyPlugin` for a non-``policy`` kind, or the Core
        registry errors (invalid / incompatible / unsigned / duplicate name) otherwise."""
        manifest = plugin.manifest
        if manifest.kind is not PluginKind.POLICY:
            raise NotAPolicyPlugin(
                f"plugin {manifest.name!r} has kind {manifest.kind.value!r}; Mind tiers and "
                f"shields must be {PluginKind.POLICY.value!r}"
            )
        self._core.register(manifest)
        self._factories[manifest.name] = plugin.factory
        return manifest

    @classmethod
    def from_entry_points(
        cls,
        *,
        require_signature: bool = False,
        provided: Mapping[str, str] | None = None,
        verifier: Verifier | None = None,
        group: str = ENTRY_POINT_GROUP,
    ) -> TierRegistry:
        """Discover and register every tier plugin advertised under ``group``. Each entry
        point loads to a zero-argument provider returning a :class:`TierPlugin`."""
        registry = cls(require_signature=require_signature, provided=provided, verifier=verifier)
        for entry_point in entry_points(group=group):
            provider = entry_point.load()
            plugin = provider()
            if not isinstance(plugin, TierPlugin):
                raise TierRegistryError(
                    f"entry point {entry_point.name!r} in group {group!r} returned "
                    f"{type(plugin).__name__}, expected a TierPlugin"
                )
            registry.register(plugin)
        return registry

    def manifest(self, name: str) -> PluginManifest:
        """Return the gated manifest registered under ``name``; raise
        :class:`PluginNotRegistered` if none is."""
        try:
            return self._core.resolve(name)
        except RegistryError as exc:
            raise PluginNotRegistered(str(exc)) from exc

    def instantiate(self, name: str, params: Mapping[str, Any] | None = None) -> Policy:
        """Build the plugin's :class:`Policy` from ``params``.

        The plugin must be registered (so its manifest passed the Core gates); the factory's
        result is re-checked against the ``Policy`` contract before it is returned. Raises
        :class:`PluginNotRegistered` or :class:`NotAPolicyPlugin`."""
        self.manifest(name)  # ensures registered + gated; raises PluginNotRegistered
        policy = self._factories[name](dict(params or {}))
        if not isinstance(policy, Policy):
            raise NotAPolicyPlugin(
                f"factory for plugin {name!r} returned {type(policy).__name__}, which does "
                f"not satisfy the Core Policy/Planner contract"
            )
        return policy

    def __contains__(self, name: object) -> bool:
        return name in self._core

    def __len__(self) -> int:
        return len(self._core)

    @property
    def manifests(self) -> tuple[PluginManifest, ...]:
        """All registered manifests, in registration order."""
        return self._core.manifests
