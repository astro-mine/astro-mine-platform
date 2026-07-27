"""The tactical task-and-motion tier (mind.md §3, §4).

``tamp/task/`` selects the symbolic task (which GOTO/prospect step) from the mission tier's
decomposition; ``tamp/motion/`` checks geometric feasibility with a sampling-based planner,
interleaved PDDLStream-style (RM-P1-MIND-03). :mod:`~astro_mine.mind.tamp.reference` composes the
two into the default backend behind Core's
:class:`~astro_mine.core.policy.protocol.TaskMotionPlanner` sub-interface. The reference motion
planner is a pure-Python RRT; OMPL (RRT*/PRM*/BIT*) + FCL collision drop in behind the same
motion contract via the ``[native]`` extra — the framework commits to the interface (principle 2).
"""

from __future__ import annotations
