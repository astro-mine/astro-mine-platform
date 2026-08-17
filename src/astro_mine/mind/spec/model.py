# SPDX-License-Identifier: Apache-2.0
"""Stack spec v0.1 — typed Pydantic models (RM-P1-MIND-01).

The **stack spec** is Mind's authored artifact: a declarative description of an autonomy
stack — which registry plugin fills each tier, how the tiers wire, the mandatory Guard
shield, per-tier validity horizons, replan triggers, and fallbacks. Studio authors it and
Bench pins it (mind.md §3). It follows Core's "Config & scenario spec" schema stack
(conventions.md §3): human-authored **YAML/JSON**, validated by a canonical **JSON
Schema** (``schema/stack_spec.schema.json``, the source of truth), mirrored by these
**Pydantic v2** models. Every model sets ``extra="forbid"`` (reject typo'd fields loudly);
all durations are SI seconds.

These models are **purely structural**; the trigger-consistency semantic checks live in
:mod:`astro_mine.mind.spec.loader`, and graph-validity (roles, ordering, plugin
resolution, shield presence) is the composer's job
(:mod:`astro_mine.mind.compose.composer`) — the same split Core uses between its manifest
loader and its registry.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.mind.spec.enums import (
    CoordinationKind,
    ExecutionKind,
    ReplanTriggerKind,
    TierRole,
)

__all__ = [
    "CoordinationSpec",
    "ExecutionSpec",
    "FallbackBinding",
    "ReplanTrigger",
    "ShieldBinding",
    "SpecProvenance",
    "StackSpec",
    "StackSpecDocument",
    "TierBinding",
]

STACK_SPEC_VERSION: Literal["0.1"] = "0.1"


class _Model(BaseModel):
    """Base for every stack-spec model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class SpecProvenance(_Model):
    """Reproducibility provenance of the stack spec (conventions.md §5), mirroring Core's
    provenance blocks: the content hashes of its inputs, the producing code/toolchain
    versions, the environment lockfile, and the seed."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


class ReplanTrigger(_Model):
    """A condition that makes the executive re-invoke a tier (mind.md §3, principle 5).

    ``every_ticks`` is required for (and only meaningful to) the ``periodic`` kind — the
    tier re-decides every N ticks; the loader enforces that consistency."""

    kind: ReplanTriggerKind
    every_ticks: int | None = Field(default=None, gt=0)


class FallbackBinding(_Model):
    """The plugin that takes over when a tier fails or has no fresh input — the explicit,
    reachable degradation path of principle 4. Itself a registry plugin resolved and
    instantiated exactly like a tier."""

    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


class TierBinding(_Model):
    """One tier of the hierarchy: which registry ``plugin`` fills ``role``, how long its
    decision stays valid (``validity_horizon_s``), what forces a re-decision
    (``replan_triggers``), and where it degrades to (``fallback``). ``params`` are passed
    to the plugin factory at instantiation. A tier with no horizon and no triggers
    re-decides every tick (the reactive default)."""

    role: TierRole
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)
    validity_horizon_s: float | None = Field(default=None, gt=0)
    replan_triggers: list[ReplanTrigger] = Field(default_factory=list)
    fallback: FallbackBinding | None = None


class ShieldBinding(_Model):
    """The mandatory Guard shield binding — the single output path (principle 7). Every
    action the executive emits passes through this shield before it becomes an Environment
    API action; the composer refuses to build a stack without it. The shield is a Core
    Policy/Planner (Guard's ``PolicyShield`` *is* a policy; RFC-0004), resolved and
    instantiated through the registry like any tier. The real Guard shield arrives via the
    registry in **RM-P1-MIND-05**; a reference pass-through shield ships for local dev."""

    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionSpec(_Model):
    """How the executive executes the hierarchy each tick (reserved switch point). Defaults
    to ``composition`` (direct tier composition); ``behavior_tree_ref`` names a
    Groot-compatible BT document once **RM-P1-MIND-02** adds the ``behavior_tree`` kind."""

    kind: ExecutionKind = ExecutionKind.COMPOSITION
    behavior_tree_ref: str | None = None


class CoordinationSpec(_Model):
    """The cross-agent coordination posture (reserved switch point). Defaults to
    ``centralized``; decentralized `coord/` strategies arrive with **RM-P1-MIND-06**."""

    kind: CoordinationKind = CoordinationKind.CENTRALIZED


class StackSpec(_Model):
    """A declarative autonomy stack (mind.md §3): the tiers, their wiring, the mandatory
    shield, and the execution/coordination posture. ``tiers`` must be non-empty; the
    composer validates that roles are unique and composes them in canonical order."""

    id: str
    name: str
    description: str | None = None
    scenario_ref: str | None = None
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    coordination: CoordinationSpec = Field(default_factory=CoordinationSpec)
    tiers: list[TierBinding] = Field(min_length=1)
    shield: ShieldBinding
    provenance: SpecProvenance | None = None

    @model_validator(mode="after")
    def _reject_duplicate_roles(self) -> StackSpec:
        roles = [t.role for t in self.tiers]
        if len(roles) != len(set(roles)):
            dupes = sorted({str(r) for r in roles if roles.count(r) > 1})
            raise ValueError(f"stack spec {self.id!r}: duplicate tier role(s): {', '.join(dupes)}")
        return self


class StackSpecDocument(_Model):
    """Top-level stack-spec document. ``stack_spec_version`` pins the schema minor."""

    stack_spec_version: Literal["0.1"]
    stack_spec: StackSpec
