"""The executive — ticks a composed hierarchy, enforcing the shielded single output path.

See :mod:`astro_mine.mind.exec.executive` for the :class:`Executive` run loop and
:func:`~astro_mine.mind.exec.executive.build_strategy`, :mod:`astro_mine.mind.exec.strategy` for
the default composition execution strategy (per-tier validity horizons, replan triggers, and
fallbacks), :mod:`astro_mine.mind.exec.degrade` for the comms-aware degrade-not-collapse
strategy (RM-P1-MIND-06), :mod:`astro_mine.mind.exec.plan` for the behavior Mind layers over
Core's ``Plan``/``ContingentPlan`` schema (RFC-0006), and :mod:`astro_mine.mind.exec.policy` for
:class:`~astro_mine.mind.exec.policy.StackPolicy` — the composed stack behind the Core ``Policy``
contract, for the runtimes (Bench, Sim) that own the loop themselves.
"""

from __future__ import annotations

from astro_mine.mind.exec.degrade import DecentralizedStrategy
from astro_mine.mind.exec.executive import Executive, RunResult, build_strategy
from astro_mine.mind.exec.plan import (
    PLAN_VERSION,
    branch_for,
    build_contingent_plan,
    expires_at_s,
    is_stale,
    is_valid,
    plan_document,
)
from astro_mine.mind.exec.policy import StackPolicy
from astro_mine.mind.exec.strategy import CompositionStrategy, Strategy

__all__ = [
    "PLAN_VERSION",
    "CompositionStrategy",
    "DecentralizedStrategy",
    "Executive",
    "RunResult",
    "StackPolicy",
    "Strategy",
    "branch_for",
    "build_contingent_plan",
    "build_strategy",
    "expires_at_s",
    "is_stale",
    "is_valid",
    "plan_document",
]
