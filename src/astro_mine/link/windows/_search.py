"""The window-search mechanism: a visibility series reduced to contact intervals.

One generic search underlies both relay-orbiter and ground-station windows (LINK-02): sample
a boolean visibility predicate across an epoch window, then reduce the rise/set transitions
into :class:`~astro_mine.link.windows.ContactWindow` intervals. The boundary epoch of each
transition is optionally sharpened by **bisection** down to a ``refine_s`` tolerance.

**Geometry is ground truth.** The predicate is whatever decides visibility — LINK-01's
``compute_los`` (terrain-occluded relay LOS) or an Earth-station elevation mask — so the
search itself stays agnostic to *why* two nodes can see each other.

**Degrade loudly.** A predicate that raises (a missing kernel, a missing provider) propagates
unchanged; the search never folds a failed query into a silent "no contact" (link.md §2.9).

**Single-transition assumption.** Bisection assumes at most one rise/set between two adjacent
samples; ``step_s`` should be fine enough to resolve the shortest contact of interest. This
mirrors how a SPICE geometry-finder also works off a base step.
"""

from __future__ import annotations

from collections.abc import Callable

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.windows._errors import LinkWindowError
from astro_mine.link.windows._window import ContactWindow

__all__ = ["VisibilityPredicate", "search_windows"]

#: Decides whether an ordered node pair is connected at an epoch. The caller binds the
#: geometry (relay LOS, an elevation mask); the search only thresholds the boolean.
VisibilityPredicate = Callable[[Epoch], bool]


def _epoch(tdb_seconds: float) -> Epoch:
    return Epoch(tdb_seconds=tdb_seconds, scale=TimeScale.TDB)


def _grid(window: EpochWindow, step_s: float) -> list[float]:
    """Deterministic TDB sample times across the half-open ``[start, end)`` window.

    Always includes ``start`` and never reaches ``end``; ``i * step_s`` is added to the
    start (not accumulated) so the grid is drift-free and reproducible.
    """
    start = window.start.tdb_seconds
    end = window.end.tdb_seconds
    times: list[float] = []
    i = 0
    t = start
    while t < end:
        times.append(t)
        i += 1
        t = start + i * step_s
    return times


def _boundary(
    visible_at: VisibilityPredicate,
    lo_t: float,
    hi_t: float,
    lo_vis: bool,
    refine_s: float | None,
) -> float:
    """The transition epoch between ``lo_t`` and ``hi_t`` (which carry opposite visibility).

    Returns the ``hi``-side boundary — the first instant carrying the *opposite* of
    ``lo_vis`` (a rise → first visible epoch; a set → first occluded epoch). With
    ``refine_s`` set, the interval is bisected to within that tolerance; otherwise the
    coarse ``hi_t`` sample is returned.
    """
    if refine_s is None:
        return hi_t
    while hi_t - lo_t > refine_s:
        mid = (lo_t + hi_t) / 2.0
        if visible_at(_epoch(mid)) == lo_vis:
            lo_t = mid
        else:
            hi_t = mid
    return hi_t


def search_windows(
    pair: tuple[str, str],
    visible_at: VisibilityPredicate,
    window: EpochWindow,
    step_s: float,
    *,
    refine_s: float | None = None,
) -> list[ContactWindow]:
    """Reduce ``visible_at`` sampled over ``window`` into ``observer -> target`` contact intervals.

    ``pair`` is the ordered ``(observer, target)`` name pair stamped onto each window. The
    predicate is sampled every ``step_s`` seconds; a contact open at the window start (or
    still open at its end) is clamped to the window bound (boundaries outside the window
    cannot be refined). When ``refine_s`` is given, each interior rise/set epoch is bisected
    to within ``refine_s``.

    Raises :class:`LinkWindowError` for a non-positive ``step_s`` or a ``refine_s`` outside
    ``(0, step_s]``. Predicate exceptions propagate unchanged.
    """
    if step_s <= 0.0:
        raise LinkWindowError(f"step_s must be positive, got {step_s}")
    if refine_s is not None and not 0.0 < refine_s <= step_s:
        raise LinkWindowError(f"refine_s must be in (0, step_s={step_s}], got {refine_s}")

    observer, target = pair
    times = _grid(window, step_s)

    windows: list[ContactWindow] = []
    prev_t = times[0]
    prev_vis = visible_at(_epoch(prev_t))
    open_start: Epoch | None = window.start if prev_vis else None

    for t in times[1:]:
        vis = visible_at(_epoch(t))
        if vis and not prev_vis:
            open_start = _epoch(_boundary(visible_at, prev_t, t, prev_vis, refine_s))
        elif not vis and prev_vis:
            assert open_start is not None  # invariant: a visible run always has a start
            end = _epoch(_boundary(visible_at, prev_t, t, prev_vis, refine_s))
            windows.append(ContactWindow(observer, target, open_start, end))
            open_start = None
        prev_t = t
        prev_vis = vis

    if prev_vis and open_start is not None:
        windows.append(ContactWindow(observer, target, open_start, window.end))
    return windows
