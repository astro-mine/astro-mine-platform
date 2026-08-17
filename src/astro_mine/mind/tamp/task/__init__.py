# SPDX-License-Identifier: Apache-2.0
"""Symbolic task selection (RM-P1-MIND-03).

The symbolic half of TAMP: turn the mission tier's decomposition (assigned prospect regions)
into per-agent tactical tasks (a GOTO to the region centre), handed to the motion planner for
feasibility. :mod:`~astro_mine.mind.tamp.task.reference` is the deterministic reference.
"""

from __future__ import annotations
