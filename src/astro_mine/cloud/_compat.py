# SPDX-License-Identifier: Apache-2.0
"""Core interface-version admission for Cloud's provenance / job envelopes.

Cloud declares no new Core schema (``cloud.md`` §6): its :class:`RunContext` and
:class:`JobSpec` are Cloud-local pydantic models. But they carry a
``core_interface_version`` -- the Core interface generation a run/job is built against --
and that string must be one *this* Core can satisfy. This module is the single admission
point, validating the declared version against Core's own compatibility contract
(``astro_mine.core.compat``) rather than treating the field as a free string
(``VERSIONING.md`` §4-5; ``core.md`` §6).
"""

from __future__ import annotations

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS, check_compatible, parse_version

__all__ = ["validate_core_interface_version"]


def validate_core_interface_version(value: str | None) -> str | None:
    """Return *value* if this Core can satisfy it, else raise ``ValueError``.

    ``None`` (unset) is allowed. A set value must be a ``MAJOR.MINOR.PATCH`` string that
    is compatible -- by Core's own SemVer rule (:func:`check_compatible`) -- with every
    interface Core publishes. Core holds all interfaces in lockstep (``VERSIONING.md``
    §4), so during the frozen-``0.1.0`` period this admits exactly ``0.1.x`` and rejects
    a typo'd or incompatible generation up front.
    """
    if value is None:
        return value
    parse_version(value)  # reject a non-MAJOR.MINOR.PATCH string loudly
    provided = set(CORE_INTERFACE_VERSIONS.values())
    if not all(check_compatible(value, published) for published in provided):
        raise ValueError(
            f"core_interface_version {value!r} is not compatible with this Core "
            f"(Core publishes {sorted(provided)})"
        )
    return value
