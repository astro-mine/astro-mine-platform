# SPDX-License-Identifier: Apache-2.0
"""Sampling-based motion feasibility (RM-P1-MIND-03).

The geometric half of TAMP behind the small
:class:`~astro_mine.mind.tamp.motion.protocol.MotionPlanner` contract: given a start, a goal, and
keep-out obstacles, return a collision-free path (or the best effort). Two backends fill it — the
deterministic pure-Python RRT (:mod:`~astro_mine.mind.tamp.motion.reference`, the CI-tested
bit-exact default) and the native OMPL (RRT*/PRM*/BIT*) + FCL planner
(:mod:`~astro_mine.mind.tamp.motion.native`, the ``[native]`` extra) — and the TAMP tier holds
either without knowing which.

``native`` is deliberately **not** re-exported here: importing it pulls the heavy OMPL/FCL
libraries, which the base wheel does not ship. Import it explicitly (or bind the
``mind.tamp.ompl`` plugin, whose factory defers the import).
"""

from __future__ import annotations

from astro_mine.mind.tamp.motion.protocol import MotionPlanner
from astro_mine.mind.tamp.motion.reference import Obstacle, Point, ReferenceMotionPlanner

__all__ = ["MotionPlanner", "Obstacle", "Point", "ReferenceMotionPlanner"]
