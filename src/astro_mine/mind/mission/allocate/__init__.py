# SPDX-License-Identifier: Apache-2.0
"""Allocate delegation (RM-P1-MIND-04) — Mind's side of it.

Mind owns *decomposition and execution*; Allocate owns *who does what, when, where* — Mind embeds
**no** combinatorial solver (the boundary, charter §5.4; mind.md §6).

The delegation crosses the Core :class:`~astro_mine.core.policy.protocol.Allocator` sub-interface,
and the whole contract for that crossing now lives at the waist, in
:mod:`astro_mine.core.plan.allocation`: :class:`~astro_mine.core.plan.allocation.AllocationAdapter`
assembles an :class:`~astro_mine.core.plan.allocation.AllocationRequest` from the mission
decomposition, publishes it under the shared ``allocation.request`` extras key, and delegates to an
injected solver policy — resolved through the registry, never a sibling import. It moved because
three parties construct it and one of them, Allocate, had to import Mind to do so (conventions.md
§3.3); the RFC-0006 note that used to sit in this docstring anticipated exactly that.

What remains here is Mind's own contribution: the deterministic
:class:`~astro_mine.mind.mission.allocate.reference.GreedyReferenceAllocator` stand-in that base
Mind delegates to when no real solver is bound (as MIND-01 stood in for Guard with the
pass-through shield). The real ``astro_mine.allocate.AllocationPlanner`` binds via its own
entry-point shim behind the same adapter.
"""

from __future__ import annotations

from astro_mine.mind.mission.allocate.reference import (
    GreedyReferenceAllocator,
    greedy_allocator_plugin,
)

__all__ = ["GreedyReferenceAllocator", "greedy_allocator_plugin"]
