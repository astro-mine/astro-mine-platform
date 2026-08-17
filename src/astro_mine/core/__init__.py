# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Core — the narrow waist.

The thin, stable contract layer every Astro-Mine package and third-party plugin
speaks to. It defines, and only defines: the Swarm Asset Description Format
(:mod:`astro_mine.core.sadf`), the Environment API (:mod:`astro_mine.core.env`),
the Policy/Planner API (:mod:`astro_mine.core.policy`), the world/terrain and
resource-field query Protocols (:mod:`astro_mine.core.world`,
:mod:`astro_mine.core.resource`), the message schemas
(:mod:`astro_mine.core.messages`, incl. :mod:`astro_mine.core.objective`), the
run-provenance vocabulary (:mod:`astro_mine.core.provenance`), and the plugin registry
(:mod:`astro_mine.core.registry`, :mod:`astro_mine.core.compat`).

Schema, validators, and lightweight helpers only — no physics, no solvers, no
learning, no UI. If it can live in a plugin, it must not live in Core. See
``docs/architecture/core.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from astro_mine.core import _schema_digest
from astro_mine.core.schemas import schema_registry

__all__ = ["SCHEMA_DIGEST", "__version__", "schema_registry"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"

#: Content address of the exact Core schema set this build carries — the identity a
#: ``ScenarioSpec`` pins so a benchmark reproduces byte-for-byte (``VERSIONING.md`` §4.2;
#: CX-REPRO). Equal to the published schema bundle's ``schema_digest`` for the same commit.
#:
#: While ``CORE_INTERFACE_VERSIONS`` is frozen at ``0.1.0`` (``VERSIONING.md`` §4), version
#: negotiation is a no-op and *this* is the value that actually distinguishes one Core schema
#: set from another. Prefer it over ``__version__`` when pinning the contract.
SCHEMA_DIGEST: str = _schema_digest.SCHEMA_DIGEST
