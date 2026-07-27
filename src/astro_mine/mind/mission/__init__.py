"""The strategic mission tier — global decomposition + Allocate delegation (mind.md §3).

``mission/planner/`` holds the pluggable mission-planner backend (RM-P1-MIND-03): the
PDDL/temporal default, generating a problem from the belief view each replan and producing a
global decomposition (which regions to prospect). ``mission/allocate/`` (RM-P1-MIND-04) is the
thin adapter that delegates the coupled who-does-what-when-where assignment to
Astro-Mine-Allocate — Mind owns decomposition and execution, never the combinatorics.
"""

from __future__ import annotations
