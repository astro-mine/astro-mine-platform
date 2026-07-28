"""Astro-Mine-Fleet — SADF asset library and authoring toolchain.

The authoring commands (`astro-mine fleet …`, in astro-mine-cli),
:mod:`~astro_mine.fleet.importers`
(URDF/SDF + USD/glTF geometry), physical-plausibility :mod:`~astro_mine.fleet.lint`,
the reference asset :mod:`~astro_mine.fleet.library`, multi-fidelity
:mod:`~astro_mine.fleet.fidelity` profiles, and content-addressed
:mod:`~astro_mine.fleet.packaging`.

Fleet consumes the Core SADF; it never widens the waist. See
``docs/architecture/fleet.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"

# The Core interface versions Fleet is built against — advertised here so consumers and
# the contract test cite one source of truth (defined in :mod:`astro_mine.fleet._core`).
#
# ``canonical_json`` is public for the same reason: it is the *definition* of a SADF
# document's canonical form — sorted keys, two-space indent, unset optionals dropped —
# and three separate surfaces have to agree on it byte for byte. Fleet's own packaging
# manifest is one; `astro-mine fleet resolve` and `astro-mine fleet package`, which live
# in astro-mine-cli, are the others. A CLI that re-implemented the projection would
# produce documents that no longer hash equal to the ones Fleet packages, so the
# function is exported rather than copied (astro-mine-cli#12).
from astro_mine.fleet._core import CORE_INTERFACES, canonical_json

__all__ = ["CORE_INTERFACES", "__version__", "canonical_json"]
