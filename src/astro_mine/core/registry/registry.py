# SPDX-License-Identifier: Apache-2.0
"""The plugin registry — discovery, resolution, and the load-time gates (RM-P0-CORE-05).

:class:`PluginRegistry` is the in-process point where a host component (Sim, Bench, …)
registers and resolves plugin manifests. Every load is **gated**, and a failing gate
refuses the load loudly — no silent partial registration:

1. **validity** — the manifest passes the full document contract (structural + the
   reserved/gated capability-tag gate; :mod:`astro_mine.core.registry.loader`);
2. **version negotiation** — every Core interface the manifest is built against is
   satisfied by *this* Core, via
   :func:`astro_mine.core.compat.assert_core_compatible`; an unsupported major (or a
   mismatched ``0.y`` minor, or an unknown interface) raises
   :class:`IncompatibleManifest` listing every problem;
3. **signature** — a signature is present and not the ``unsigned`` dev marker, then a
   pluggable ``verifier`` runs. Core enforces presence/shape; the cryptographic
   chain-of-trust (cosign/Rekor) is the verifier's job, injected by the host (P1).

Core never executes plugin code (core.md §9): the registry resolves *manifests*, and the
host instantiates/sandboxes the plugin behind them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from astro_mine.core.compat import IncompatibleCoreInterface, assert_core_compatible
from astro_mine.core.registry.enums import SignatureScheme
from astro_mine.core.registry.loader import RegistryError, validate_manifest
from astro_mine.core.registry.model import MANIFEST_VERSION, ManifestDocument, PluginManifest

__all__ = ["IncompatibleManifest", "PluginRegistry", "UnsignedManifest", "Verifier"]

#: A host-supplied cryptographic verifier. Receives a validated manifest and raises if
#: the signature does not verify; returns ``None`` on success. Core ships none — the
#: cosign/Rekor chain-of-trust is delegated to the host (Cloud/Fleet, Phase 1).
Verifier = Callable[[PluginManifest], None]


class IncompatibleManifest(RegistryError):
    """Raised when a manifest's declared Core interface versions are not satisfied."""


class UnsignedManifest(RegistryError):
    """Raised when a signature-requiring registry is handed an unsigned manifest."""


class PluginRegistry:
    """An in-memory registry of plugin manifests, keyed by plugin name.

    ``provided`` overrides the Core interface versions negotiated against (defaults to
    this build's :data:`~astro_mine.core.compat.CORE_INTERFACE_VERSIONS`) — useful for
    tests. ``require_signature`` (default ``True``) makes the signature gate mandatory,
    honoring "signature + Core-version checks gate every plugin load"; ``verifier`` adds
    cryptographic verification on top of the presence/shape check.
    """

    def __init__(
        self,
        *,
        provided: Mapping[str, str] | None = None,
        require_signature: bool = True,
        verifier: Verifier | None = None,
    ) -> None:
        self._provided = dict(provided) if provided is not None else None
        self._require_signature = require_signature
        self._verifier = verifier
        self._by_name: dict[str, PluginManifest] = {}

    def load(self, source: str | bytes) -> PluginManifest:
        """Parse + validate a manifest document from text/bytes, then register it."""
        from astro_mine.core.registry.loader import load_manifest

        return self.register(load_manifest(source))

    def register(self, manifest: PluginManifest | ManifestDocument) -> PluginManifest:
        """Validate, negotiate, signature-gate, and store a manifest; return it.

        Accepts a typed :class:`PluginManifest` or :class:`ManifestDocument` (e.g. one
        built in code or returned by :func:`~astro_mine.core.registry.loader.load_manifest`).
        Raises :class:`~astro_mine.core.registry.loader.ManifestValidationError`,
        :class:`IncompatibleManifest`, or :class:`UnsignedManifest`.
        """
        doc = (
            manifest
            if isinstance(manifest, ManifestDocument)
            else ManifestDocument(manifest_version=MANIFEST_VERSION, manifest=manifest)
        )
        validate_manifest(doc)
        spec = doc.manifest
        self._negotiate(spec)
        self._check_signature(spec)
        if spec.name in self._by_name:
            raise RegistryError(f"a plugin named {spec.name!r} is already registered")
        self._by_name[spec.name] = spec
        return spec

    def _negotiate(self, manifest: PluginManifest) -> None:
        try:
            assert_core_compatible(manifest.core_interfaces, provided=self._provided)
        except IncompatibleCoreInterface as exc:
            raise IncompatibleManifest(
                f"cannot load plugin {manifest.name!r} v{manifest.version}: {exc}"
            ) from exc

    def _check_signature(self, manifest: PluginManifest) -> None:
        if self._require_signature and (
            manifest.signature is None or manifest.signature.scheme == SignatureScheme.UNSIGNED
        ):
            raise UnsignedManifest(
                f"plugin {manifest.name!r} v{manifest.version} has no signature; this "
                f"registry requires a signed manifest (set require_signature=False for "
                f"local/dev use)"
            )
        if self._verifier is not None and manifest.signature is not None:
            self._verifier(manifest)

    def resolve(self, name: str) -> PluginManifest:
        """Return the registered manifest named ``name``; raise if none is registered."""
        try:
            return self._by_name[name]
        except KeyError:
            raise RegistryError(f"no plugin named {name!r} is registered") from None

    def by_kind(self, kind: str) -> list[PluginManifest]:
        """Return all registered manifests of the given :class:`PluginKind`."""
        return [m for m in self._by_name.values() if m.kind == kind]

    @property
    def manifests(self) -> tuple[PluginManifest, ...]:
        """All registered manifests, in registration order."""
        return tuple(self._by_name.values())

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)
