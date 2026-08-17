# SPDX-License-Identifier: Apache-2.0
"""Errors for the Link budget layer."""

from __future__ import annotations

__all__ = ["LinkBudgetError", "ModCodError"]


class LinkBudgetError(RuntimeError):
    """A link budget cannot be computed from the given radios and geometry.

    Raised when an input needed to close the budget is missing or inconsistent — no EIRP
    (and no power+gain to derive it), no receiver G/T, a non-positive range/frequency, a
    band mismatch with no explicit frequency, or two radios that share no mod/cod. A merely
    *weak* geometry (a link that closes on geometry but is too faint for any supported
    mod/cod) is **not** an error — that returns ``feasible=False``, which a planner needs.
    """


class ModCodError(LinkBudgetError):
    """A referenced modulation/coding scheme is not present in the mod/cod table.

    A subclass of :class:`LinkBudgetError` so callers may catch either; raised when a radio
    declares a mod/cod the active table does not define, rather than silently skipping it.
    """
