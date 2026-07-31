"""Policy / Planner API — how decisions are computed and composed (RM-P0-CORE-03).

One uniform "observations + context -> actions/assignments" contract: a :class:`Policy`
maps the per-agent :class:`~astro_mine.core.messages.Observation`\\ s the Environment API
yields to an :class:`~astro_mine.core.messages.ActionBatch` it consumes — so
``env.step(policy.decide(observations, context))`` closes the design/operations loop. The
four tiers (:class:`MissionPlanner`, :class:`TaskMotionPlanner`, :class:`Allocator`,
:class:`Controller`) are composable sub-interfaces of this one contract; implemented by
Mind/Allocate/Learn and wrapped by Guard (all Phase 1).

A learned policy crosses the waist as a :class:`PolicyPackage` — an ONNX graph plus this
typed metadata sidecar. :class:`OnnxPolicy` binds a package + a host-supplied inference
callable to the Policy contract (Core defines the shape; the ONNX runtime stays at the
edge), so a Learn-exported policy drops into a composed stack, is wrapped by Guard, and is
scored by Bench through the standard interface (RM-P1-CORE-01).

Public API:

- the contract — :class:`Policy` and the tier sub-interfaces;
- the context — :class:`DecisionContext` (and :data:`AgentId`);
- composition — :class:`ComposedPolicy` (the seam an allocator+controller compose through);
- the exported-policy sidecar — :class:`PolicyPackage` (+ :class:`ModelRef`,
  :class:`IoSignature`, :class:`TensorSpec`, :class:`PolicyAssumptions`) and its
  :func:`load_policy_package` / :func:`validate_policy_package`; the adapter
  :class:`OnnxPolicy` (+ :data:`InferenceFn`);
- conformance — :func:`check_policy`, :func:`check_composition`, :class:`PolicyContractError`;
- the shield-report side-channel — :class:`ShieldReport` (+ :class:`InterventionKind`) and the
  optional :class:`ReportingShield` protocol. Guard produces the report and Mind's executive reads
  it, so by §3.3 the contract is Core's rather than either component's.

Backlog: RM-P0-CORE-03 — https://github.com/astro-mine/astro-mine-core/issues/3
"""

from __future__ import annotations

from astro_mine.core.env.model import AgentId
from astro_mine.core.policy import (
    compose,
    conformance,
    enums,
    guardrail,
    loader,
    model,
    onnx,
    protocol,
)
from astro_mine.core.policy.compose import ComposedPolicy
from astro_mine.core.policy.conformance import (
    PolicyContractError,
    check_composition,
    check_policy,
)
from astro_mine.core.policy.enums import TensorDType
from astro_mine.core.policy.guardrail import (
    InterventionKind,
    ReportingShield,
    ShieldReport,
)
from astro_mine.core.policy.loader import (
    PolicyPackageError,
    PolicyPackageValidationError,
    load_policy_package,
    load_schema,
    validate_policy_package,
)
from astro_mine.core.policy.model import (
    DecisionContext,
    IoSignature,
    ModelRef,
    PolicyAssumptions,
    PolicyPackage,
    PolicyPackageDocument,
    TensorSpec,
)
from astro_mine.core.policy.onnx import InferenceFn, OnnxPolicy
from astro_mine.core.policy.protocol import (
    Allocator,
    Controller,
    MissionPlanner,
    Policy,
    TaskMotionPlanner,
)

__all__ = [
    "AgentId",
    "Allocator",
    "ComposedPolicy",
    "Controller",
    "DecisionContext",
    "InferenceFn",
    "InterventionKind",
    "IoSignature",
    "MissionPlanner",
    "ModelRef",
    "OnnxPolicy",
    "Policy",
    "PolicyAssumptions",
    "PolicyContractError",
    "PolicyPackage",
    "PolicyPackageDocument",
    "PolicyPackageError",
    "PolicyPackageValidationError",
    "ReportingShield",
    "ShieldReport",
    "TaskMotionPlanner",
    "TensorDType",
    "TensorSpec",
    "check_composition",
    "check_policy",
    "compose",
    "conformance",
    "enums",
    "guardrail",
    "load_policy_package",
    "load_schema",
    "loader",
    "model",
    "onnx",
    "protocol",
    "validate_policy_package",
]
