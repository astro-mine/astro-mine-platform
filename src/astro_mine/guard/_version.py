# SPDX-License-Identifier: Apache-2.0
"""The installed distribution version, isolated so ``__init__`` can bind it before heavier imports.

``spec.catalog`` imports ``__version__`` from the package, so it must exist before the spec-tooling
re-exports in ``__init__`` run. Keeping it in its own module (which imports nothing from
``astro_mine.guard``) lets ``__init__`` import it first, in one sorted import block, with no cycle.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"
