"""The allocation sub-interface: request/response types, Core manifest, and the planner.

``api/`` holds the Core Policy/Planner allocation sub-interface implementation and the
Core-typed request/response types (allocate.md §3): :mod:`.model` (the
``AllocationRequest``/``Allocation`` Pydantic types + the ``ConstraintContext`` handle
container), :mod:`.manifest` (the Core ``PluginManifest`` Allocate publishes itself as),
:mod:`.planner` (the :class:`~astro_mine.core.policy.protocol.Allocator` impl), and
:mod:`._core` (the Core interface versions Allocate builds against).
"""

from __future__ import annotations
