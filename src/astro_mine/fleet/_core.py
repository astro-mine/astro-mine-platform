# SPDX-License-Identifier: Apache-2.0
"""Fleet's binding to the Core narrow waist (RM-P0-FLEET-01).

The single place that declares which Core interface versions Fleet authors against,
and a thin compatibility check over them. Fleet *consumes* the waist — it never
re-defines SADF (CONTRIBUTING.md: "consume the waist, never widen it"). Every SADF
type, validator, and the wire form come from :mod:`astro_mine.core.sadf`; this module
only records the contract Fleet builds against so the CLI scaffolds, the packaging
manifest, and the contract test all cite one source of truth.
"""

from __future__ import annotations

import json

from astro_mine.core import compat
from astro_mine.core.sadf import SadfDocument

__all__ = ["CORE_INTERFACES", "assert_core_compatible", "canonical_json"]

#: Core interface versions Fleet is built against — the input to the registry's
#: version negotiation (mirrors a SADF asset's ``core_interface_versions`` and the
#: plugin manifest's ``core_interfaces``). Phase 0 uses only SADF; append as Fleet
#: grows to consume more of the waist (e.g. ``registry`` when Hub publish lands in P1).
CORE_INTERFACES: dict[str, str] = {"sadf": "0.1.0"}


def assert_core_compatible() -> None:
    """Assert the installed Core satisfies the interface versions Fleet is built against.

    Delegates to :func:`astro_mine.core.compat.assert_core_compatible`. This is the
    consumer-driven contract test Fleet runs in its own CI (issue #1 acceptance,
    RM-P0-CORE-07); raises :class:`~astro_mine.core.compat.IncompatibleCoreInterface`
    on any mismatch.
    """
    compat.assert_core_compatible(CORE_INTERFACES)


def canonical_json(doc: SadfDocument) -> str:
    """A deterministic, human-readable JSON projection of a SADF document.

    Sorted keys, two-space indent, unset optionals dropped — stable across runs so it
    is safe to diff, hash, and re-emit. Shared by ``fleet resolve`` (the canonical
    form) and the packaging manifest's JSON projection.
    """
    data = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
