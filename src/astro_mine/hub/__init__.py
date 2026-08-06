"""Astro-Mine-Hub — the artifact registry for the commons.

The place a contributor *publishes* a plugin, world, SADF asset bundle, trained policy,
or surrogate model — and the place every other component, tool, and human *finds,
verifies, and pulls* those artifacts. Hub stores content as content-addressed OCI
artifacts, indexes each by its Core plugin manifest, discovers via faceted/semantic
search and capability negotiation, verifies supply-chain integrity (signature, SLSA,
SBOM), and gates downloads by license and export-control policy. It is a *consumer* of
the Core manifest schema, never its owner, and never executes the artifacts it serves.

All of that ships: the FastAPI registry service, the client SDK (``resolve``/``verify``/
``pull``/``publish``/``cache``), the ``astro-mine hub`` CLI, and the web front end — the
``/registry`` pages of the one application, in ``astro-mine-ui``. The tier-1 client works fully
offline against a local OCI-layout directory, with no server, no account, and no Docker. See
``docs/architecture/hub.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"
