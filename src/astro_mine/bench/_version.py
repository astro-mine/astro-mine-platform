# SPDX-License-Identifier: Apache-2.0
"""The package version, in its own module to keep the import graph acyclic.

Imported by the package ``__init__`` and by :mod:`astro_mine.bench.harness` (which stamps it
into a run's provenance). Living here rather than in ``__init__`` lets the harness read the
version without importing the package root, which now imports ``baseline`` (which imports the
harness) — a cycle this module breaks.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
