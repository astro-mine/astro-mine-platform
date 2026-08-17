# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Cloud — distributed orchestration (Phase 0: local-first discipline).

Container-first :mod:`~astro_mine.cloud.packaging`, the
:mod:`~astro_mine.cloud.submission` backend-equivalence contract and its local
:func:`~astro_mine.cloud.submission.submit`, and content-addressed
:mod:`~astro_mine.cloud.artifacts` I/O with a RunContext provenance envelope.

Phase 0 ships no cluster -- it ships the discipline so workloads scale out in Phase 1
without rework. See ``docs/architecture/cloud.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from astro_mine.cloud.submission import submit

__all__ = ["__version__", "submit"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
