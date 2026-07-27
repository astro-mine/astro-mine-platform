"""The pluggable mission-planner backend (RM-P1-MIND-03).

The strategic tier behind Core's :class:`~astro_mine.core.policy.protocol.MissionPlanner`
sub-interface: a PDDL/temporal default (mind.md §4, §11). The ``pddl`` module
generates the problem from the belief; the ``reference`` module is the
deterministic pure-Python reference planner (the CI-tested default). A ``unified-planning``
façade over Fast Downward / OPTIC / ENHSP drops in behind the same sub-interface via the
``[pddl]`` extra — the framework commits to the interface, not the backend (principle 2).
"""

from __future__ import annotations
