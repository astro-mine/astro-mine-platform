# SPDX-License-Identifier: Apache-2.0
"""Plan / ContingentPlan v0.1 — typed Pydantic models (the delay-tolerant plan contract).

A first-class Core message schema (RFC-0006; mind.md §5): the time-stamped, validity-horizoned
plan artifact the autonomy stack acts on — Mind composes it, Ops replays/supervises it, View
renders it, Bench scores it — so every component reads the *same* plan vocabulary rather than
re-deriving it. A :class:`Plan` carries its validity horizon and the assumptions it rests on; a
:class:`ContingentPlan` adds explicit contingency branch points so an agent knows what to do when
a trigger fires (comms loss, plan expiry) — act on the base plan while valid, branch when not,
reconcile on recovery (principle 5, delay-tolerant by construction).

The **canonical** schema is the hand-authored JSON Schema in ``schema/plan.schema.json`` (shipped
in-package); these models mirror it, and a consistency test asserts the two agree. All durations
are SI seconds. The Protobuf wire form is deferred to the first cross-process consumer (RFC-0006,
P2) — no Phase-1 consumer crosses a process boundary (Mind hosts plans in-process).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.messages.model import ActionBatch

__all__ = [
    "Assumption",
    "ContingencyBranch",
    "ContingentPlan",
    "Plan",
    "PlanDocument",
    "PlanProvenance",
    "PlanValidity",
]

PLAN_VERSION = "0.1"


class _Model(BaseModel):
    """Base for every plan model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class PlanValidity(_Model):
    """When a plan was issued and how long it stays valid. ``horizon_s`` of ``None`` means a
    standing plan (no expiry). An agent MAY keep acting on a stale plan under comms loss
    (act-while-stale); the decision trace records that it did (RM-P1-MIND-06/07)."""

    issued_at_s: float
    horizon_s: float | None = Field(default=None, gt=0.0)


class Assumption(_Model):
    """A precondition the plan was built under (e.g. ``earth_contact`` available). ``holds`` is
    the last known truth (``None`` when unchecked); a violated assumption is a reason to take a
    contingency branch."""

    key: str
    description: str = ""
    holds: bool | None = None


class ContingencyBranch(_Model):
    """What to do when ``trigger`` fires. ``trigger`` aligns with the Mind stack-spec replan
    vocabulary (``plan_expired`` / ``periodic`` / ``on_fallback`` / ``comms_lost``); ``action``
    is a small append-only set (``hold_cached`` / ``reconcile`` / ``safe_idle`` / ``coordinate``).
    """

    trigger: str
    action: str
    description: str = ""


class PlanProvenance(_Model):
    """Reproducibility provenance (conventions.md §5): the content-addressed identities of the
    plan's inputs (SADF, belief snapshot, comms model, policy artifacts), the producing code
    version and environment lockfile, and the seed — so a plan reproduces exactly."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


class Plan(_Model):
    """A time-stamped, validity-horizoned plan for one tier: the ``actions`` it proposes (a Core
    :class:`~astro_mine.core.messages.model.ActionBatch`), when they are valid, the assumptions
    they rest on, and optional reproducibility provenance."""

    plan_id: str
    tier: str
    validity: PlanValidity
    actions: ActionBatch = Field(default_factory=ActionBatch)
    assumptions: list[Assumption] = Field(default_factory=list)
    provenance: PlanProvenance | None = None


class ContingentPlan(_Model):
    """A :class:`Plan` plus explicit contingency ``branches`` — the degrade-not-collapse artifact:
    idempotent, act-while-stale, reconciled on recovery (principle 5). A bare plan is a contingent
    plan with no branches."""

    base: Plan
    branches: list[ContingencyBranch] = Field(default_factory=list)


class PlanDocument(_Model):
    """Top-level plan document. ``plan_version`` pins the schema minor."""

    plan_version: Literal["0.1"]
    plan: ContingentPlan
