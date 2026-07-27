"""Native motion-planning backends (RM-P1-MIND-03) — the ``[native]`` extra.

The real sampling-based motion planner behind
:class:`~astro_mine.mind.tamp.motion.protocol.MotionPlanner`:
:class:`~astro_mine.mind.tamp.motion.native.ompl.OmplMotionPlanner` plans with **OMPL**
(RRT*/PRM*/BIT*, the algorithms mind.md §4 names) and checks collision with **FCL** — the
native C++ stack the pure-Python reference RRT stands in for.

Heavy, native-toolchain dependencies: every ``ompl``/``fcl`` import is **deferred into the call**
so the base wheel stays importable and the entry-point provider costs nothing to discover. The
adapter's tests are marker-gated (``native``) and deselected in CI; the reference RRT remains the
CI-tested, bit-exact default.
"""

from __future__ import annotations

from astro_mine.mind.tamp.motion.native.ompl import OmplMotionPlanner, ompl_tamp_plugin

__all__ = ["OmplMotionPlanner", "ompl_tamp_plugin"]
