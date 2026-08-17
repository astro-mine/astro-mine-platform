# SPDX-License-Identifier: Apache-2.0
"""SPICE kernel management — furnish, clear, fail loud (RFC-0002).

SPICE's kernel pool is process-global, so this is a thin, stateful wrapper over
``spiceypy.furnsh``/``kclear``. The load-bearing guarantee is **fail-loud**: a missing
kernel raises rather than letting geometry silently default to a guess — the shared
"degrade, don't lie" rule every consumer (Worlds illumination/PSR, Link LOS/contact
windows) depends on (link.md §2.9; conventions.md §1.x).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import spiceypy as sp

if TYPE_CHECKING:
    from astro_mine.core.units import EpochWindow

__all__ = ["SpiceKernelError", "clear_kernels", "kernel_pool", "load_metakernel"]


class SpiceKernelError(Exception):
    """Raised when a SPICE kernel cannot be loaded, or the furnished pool leaves an
    epoch window without coverage."""


def load_metakernel(path: str | Path, *, coverage: EpochWindow | None = None) -> None:
    """Furnish a SPICE meta-kernel (or any kernel) into the global pool, failing loudly.

    A missing path raises :class:`SpiceKernelError`; a malformed kernel raises the
    underlying SpiceyPy error. Either way the caller never proceeds on a silent default.

    When ``coverage`` is given, the furnished SPK pool is validated *up front* against
    that epoch window (``spice.md`` §10): a body with no SPK data spanning the window
    raises :class:`SpiceKernelError` now, rather than surfacing mid-rollout as a raw
    ``SPICE(SPKINSUFFDATA)`` at the first geometry query.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise SpiceKernelError(
            f"SPICE kernel not found: {resolved} — missing kernels must fail loudly, "
            "never default to assumed geometry"
        )
    sp.furnsh(str(resolved))
    if coverage is not None:
        _assert_spk_coverage(coverage)


def _assert_spk_coverage(window: EpochWindow) -> None:
    """Fail loudly unless every body in the furnished SPK pool spans ``window``.

    Lifts SPICE's mid-query ``SPICE(SPKINSUFFDATA)`` to furnish time so a coverage gap is
    caught at setup (``spice.md`` §10). Uses ``spkobj``/``spkcov`` to read each SPK's body
    set and covered intervals, and ``wnincd`` to test containment of the query window.
    """
    start = window.start.tdb_seconds
    end = window.end.tdb_seconds
    spk_count = sp.ktotal("SPK")
    if spk_count == 0:
        raise SpiceKernelError(
            f"no SPK kernel furnished to cover [{start}, {end}] ET; "
            "geometry queries would fail loudly at call time"
        )
    for i in range(spk_count):
        spk_file = sp.kdata(i, "SPK")[0]
        for body in sp.spkobj(spk_file, sp.cell_int(4096)):
            cover = sp.spkcov(spk_file, body, sp.cell_double(4096))
            if not sp.wnincd(start, end, cover):
                raise SpiceKernelError(
                    f"SPK {spk_file!r} does not cover body {body} over the query "
                    f"window [{start}, {end}] ET (coverage gap surfaced up front)"
                )


def clear_kernels() -> None:
    """Clear the entire SPICE kernel pool (``kclear``)."""
    sp.kclear()


@contextmanager
def kernel_pool(*paths: str | Path) -> Iterator[None]:
    """Furnish ``paths`` for the duration of the block, then clear the whole pool.

    A convenience for tests and one-shot jobs. Because ``kclear`` is global, do not nest
    pools or rely on kernels furnished outside the block surviving it.
    """
    for path in paths:
        load_metakernel(path)
    try:
        yield
    finally:
        clear_kernels()
