"""Astro-Mine-Mind — the hierarchical autonomy composition framework.

Turns a stated objective into actuator-level commands across a heterogeneous swarm through
three pluggable, swappable tiers wired over Core's Policy/Planner contract: a strategic
**mission planner**, per-agent **task-and-motion planners (TAMP)**, and reactive **local
controllers**, with every emitted action wrapped by a mandatory Guard shield. Mind
*implements and orchestrates* the Core interfaces; it does not own them. See
``docs/architecture/mind.md``.

The RM-P1-MIND-01 spine (this release): author a stack spec, discover tier plugins through
the Core manifest/registry, :func:`compose` them into a runnable hierarchy, and step it
against a Core :class:`~astro_mine.core.env.protocol.Environment` with the
:class:`Executive` — Guard-shielded output as the only egress and a deterministic decision
trace. The behavior-tree scaffold (RM-P1-MIND-02), heavyweight backends (RM-P1-MIND-03),
Allocate delegation (RM-P1-MIND-04), and the real Guard shield (RM-P1-MIND-05) plug in
through the same seams.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from astro_mine.mind.compose import ComposeError, HierarchyGraph, compose
from astro_mine.mind.exec import Executive, RunResult
from astro_mine.mind.registry import TierRegistry
from astro_mine.mind.spec import StackSpec, StackSpecDocument, load_stack_spec
from astro_mine.mind.trace import DecisionTrace, to_canonical_json

__all__ = [
    "ComposeError",
    "DecisionTrace",
    "Executive",
    "HierarchyGraph",
    "RunResult",
    "StackSpec",
    "StackSpecDocument",
    "TierRegistry",
    "__version__",
    "compose",
    "load_stack_spec",
    "to_canonical_json",
]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"
