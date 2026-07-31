"""The Mind binding — Guard's ``PolicyShield`` as a Mind tier/shield plugin (RFC-0006).

The **only** place in Guard that knows Mind exists, and it exists only inside the optional
``[mind]`` extra. RFC-0006's sibling-binding convention: Mind's :class:`TierRegistry` discovers
tier/shield plugins through the ``astro_mine.mind.tier_plugins`` Python entry-point group
(conventions.md §7), and each real sibling registers there **from its own side** — so installing
``astro-mine-platform[guard-mind]`` wires Mind's mandatory shield stage (RM-P1-MIND-05) to the
*real* Rust TCB, with **no** ``mind → guard`` dependency and no edit to either component's core.
Base Guard imports nothing from Mind; base Mind ships its own reference ``ConstraintShield``
stand-in.

The direction of the edge is the point (mind.md §2 principle 2, §6, §7): the framework commits to
the Core contract, not to a backend, and the backend binds itself through the registry.

Only Mind's own registry ever loads the entry point, so this module is reached by construction
rather than by import.
"""

from __future__ import annotations

from astro_mine.guard.mind.plugin import (
    PLUGIN_NAME,
    GuardShield,
    guard_shield_plugin,
    load_manifest,
)

__all__ = ["PLUGIN_NAME", "GuardShield", "guard_shield_plugin", "load_manifest"]
