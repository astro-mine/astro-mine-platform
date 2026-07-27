"""Oracle cross-check for contact pass times (RM-P0-LINK-05).

The dependency-free comparator behind Link's external-oracle validation (link.md §10): reduce
two sets of contact windows — Link's computed passes and a reference from GMAT/STK/Skyfield —
to the worst rise/set disagreement and check it against an explicit tolerance budget. Pass
times are a load-bearing correctness claim (a comms blackout Link reports wrongly could let a
planner assume an unsendable command will arrive — link.md §9), so they are validated, never
assumed.

The comparator is pure data: the *live* GMAT run that produces the reference lives in the test
suite (``tests/test_oracle_gmat.py``, driven by ``gmat-run`` against a provisioned GMAT), and
uses this same budget check — mirroring the Sim orbital regression (RM-P0-SIM-10).

Backlog: RM-P0-LINK-05 -- https://github.com/astro-mine/astro-mine-link/issues/5
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from astro_mine.link.cache._errors import LinkCacheError, PassTimeBudgetError

__all__ = ["PassTimeReport", "assert_within_budget", "cross_check_pass_times"]

_Interval = tuple[float, float]


def _interval(window: Any) -> _Interval:
    """Read a ``(start_tdb_s, end_tdb_s)`` pair from a window-like value.

    Accepts a Link :class:`~astro_mine.link.windows.ContactWindow` (``start``/``end`` epochs),
    a Core :class:`~astro_mine.core.messages.ContactInterval` (``start_tdb_s``/``end_tdb_s``),
    or a bare ``(start, end)`` tuple — so a caller can compare Link output against an oracle
    supplied in whatever form. Duck-typed to keep this comparator free of heavy imports.
    """
    if isinstance(window, tuple):
        start, end = window
        return (float(start), float(end))
    start = getattr(window, "start", None)
    end = getattr(window, "end", None)
    if start is not None and end is not None and hasattr(start, "tdb_seconds"):
        return (float(start.tdb_seconds), float(end.tdb_seconds))
    start_tdb = getattr(window, "start_tdb_s", None)
    end_tdb = getattr(window, "end_tdb_s", None)
    if start_tdb is not None and end_tdb is not None:
        return (float(start_tdb), float(end_tdb))
    raise LinkCacheError(f"cannot read a pass interval from {type(window).__name__}")


@dataclass(frozen=True, slots=True)
class PassTimeReport:
    """The verdict of comparing computed pass times against a reference within a budget.

    ``max_rise_error_s`` / ``max_set_error_s`` are the worst absolute rise/set deltas over the
    matched passes; ``computed_passes`` / ``reference_passes`` let a count mismatch (Link found
    a pass the oracle did not, or vice versa) fail the gate even when the matched deltas are
    small.
    """

    name: str
    max_rise_error_s: float
    max_set_error_s: float
    tolerance_s: float
    computed_passes: int
    reference_passes: int
    detail: str = ""

    @property
    def max_error_s(self) -> float:
        """The worst of the rise and set deltas."""
        return max(self.max_rise_error_s, self.max_set_error_s)

    @property
    def counts_match(self) -> bool:
        """Whether Link and the oracle found the same number of passes."""
        return self.computed_passes == self.reference_passes

    @property
    def within_budget(self) -> bool:
        """Whether pass counts agree and every rise/set delta stayed within the tolerance."""
        return self.counts_match and self.max_error_s <= self.tolerance_s


def cross_check_pass_times(
    computed: Iterable[Any],
    reference: Iterable[Any],
    *,
    tolerance_s: float,
    name: str = "pass-times",
) -> PassTimeReport:
    """Compare ``computed`` contact passes against ``reference`` within ``tolerance_s``.

    Both are iterables of window-like values (see :func:`_interval`); each side is sorted by
    start time and matched positionally, and the worst rise/set delta is reported against the
    budget. Raises :class:`LinkCacheError` for a negative tolerance. This never raises on a
    disagreement — inspect :attr:`PassTimeReport.within_budget` or call
    :func:`assert_within_budget` to make it a hard gate.
    """
    if tolerance_s < 0.0:
        raise LinkCacheError(f"tolerance_s must be non-negative, got {tolerance_s}")
    computed_intervals = sorted(_interval(window) for window in computed)
    reference_intervals = sorted(_interval(window) for window in reference)
    matched = min(len(computed_intervals), len(reference_intervals))
    max_rise = max(
        (abs(computed_intervals[i][0] - reference_intervals[i][0]) for i in range(matched)),
        default=0.0,
    )
    max_set = max(
        (abs(computed_intervals[i][1] - reference_intervals[i][1]) for i in range(matched)),
        default=0.0,
    )
    return PassTimeReport(
        name=name,
        max_rise_error_s=max_rise,
        max_set_error_s=max_set,
        tolerance_s=tolerance_s,
        computed_passes=len(computed_intervals),
        reference_passes=len(reference_intervals),
        detail=f"{matched} matched of {len(computed_intervals)} computed / "
        f"{len(reference_intervals)} reference",
    )


def assert_within_budget(report: PassTimeReport) -> PassTimeReport:
    """Return ``report`` if it is within budget, else raise :class:`PassTimeBudgetError`."""
    if not report.within_budget:
        raise PassTimeBudgetError(
            f"{report.name}: {report.computed_passes} computed vs {report.reference_passes} "
            f"reference passes; max rise Δ={report.max_rise_error_s:.3f}s, "
            f"set Δ={report.max_set_error_s:.3f}s vs tolerance {report.tolerance_s:.3f}s"
        )
    return report
