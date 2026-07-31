"""Policy/Planner API v0.1 — the decision context (RM-P0-CORE-03).

The non-observation inputs a policy decides against, assembled by the runtime and
passed to :meth:`Policy.decide` alongside the per-agent observations. A lightweight,
frozen dataclass (like the Environment API's result containers), **not** a wire
document — the serializable types in the loop are the ``messages`` (Observation in,
ActionBatch out).

Guard the waist: Core schematizes only what it owns — the ``objective`` the policy
optimizes toward, the ``seed`` that makes it reproducible, and ``upstream`` (the prior
tier's output, the composition seam). Everything else a planner needs — belief /
information-gain handles (Prospect), per-agent capability tags (Fleet), comms/contact
state (Link) — rides in the open ``extras`` map as downstream-owned values, never new
Core schema. (Distinct from Cloud's ``RunContext`` provenance envelope — do not conflate.)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core import compat
from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.objective.model import ObjectiveSpec
from astro_mine.core.policy.enums import TensorDType

__all__ = [
    "POLICY_PACKAGE_VERSION",
    "DecisionContext",
    "IoSignature",
    "ModelRef",
    "PolicyAssumptions",
    "PolicyPackage",
    "PolicyPackageDocument",
    "Provenance",
    "TensorSpec",
]


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """The context a policy decides against (see module docstring).

    ``upstream`` carries the previous composition stage's decision (e.g. an allocator's
    assignments handed to a controller); ``seed`` lets a stochastic policy derive a
    reproducible RNG — the runtime owns the clock/RNG and seeds the policy. ``extras``
    is the open, Core-untyped channel for belief/info-gain handles, capability tags, and
    comms state.
    """

    sim_time_s: float = 0.0
    objective: ObjectiveSpec | None = None
    upstream: ActionBatch | None = None
    seed: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# --- PolicyPackage: the exported-ONNX-policy metadata sidecar (RM-P1-CORE-01) -----
#
# Unlike DecisionContext (an in-memory loop container), the PolicyPackage sidecar is an
# authored, content-addressed *wire artifact* that crosses component boundaries (Learn
# exports it; Mind/Guard/Bench consume it), so it is a canonical JSON-Schema + Pydantic
# document (schema/policy_package.schema.json) — a metadata record with no Protobuf wire
# form yet, mirroring the plugin manifest and run-provenance schemas.

POLICY_PACKAGE_VERSION = "0.1"


class _Model(BaseModel):
    """Base for the PolicyPackage models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


class Provenance(_Model):
    """Reproducibility provenance of the exported policy (conventions.md §5). The ONNX
    graph's own content digest is carried on :class:`ModelRef`, not here."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


class TensorSpec(_Model):
    """One ONNX-graph input/output tensor — its name, element type, and shape (a ``-1``
    dimension marks a dynamic axis, e.g. a variable agent count)."""

    name: str
    dtype: TensorDType
    shape: list[int] = Field(default_factory=list)


class ModelRef(_Model):
    """A reference to the exported ONNX graph. ``digest`` is its content hash — the
    artifact's identity (a PolicyPackage is content-addressed by its graph); ``uri`` is an
    optional locator (OCI/file) and ``opset`` the ONNX opset the graph targets."""

    digest: str
    uri: str | None = None
    opset: int | None = None


class IoSignature(_Model):
    """The policy's observation/action interface. ``inputs``/``outputs`` are the concrete
    ONNX-graph tensors a host binds an ONNX-Runtime session to; ``observation_space`` /
    ``action_space`` carry the semantic (Gymnasium-style, SADF-capability-keyed) space
    descriptors Guard/Bench read — kept as open maps so Core does not over-constrain the
    per-asset space dicts."""

    inputs: list[TensorSpec] = Field(default_factory=list)
    outputs: list[TensorSpec] = Field(default_factory=list)
    observation_space: dict[str, Any] = Field(default_factory=dict)
    action_space: dict[str, Any] = Field(default_factory=dict)


class PolicyAssumptions(_Model):
    """The honest provenance a shield needs (learn.md; guard.md): the comms/observability
    regime the policy was trained under, its declared action bounds, any surrogate-fidelity
    caveats, and whether inference is deterministic given a seed. Guard treats a wrapped
    policy as adversarial input, so these are *declarations to check against*, never trusted
    blindly."""

    comms_observability: str | None = None
    action_bounds: dict[str, Any] = Field(default_factory=dict)
    surrogate_fidelity_caveats: list[str] = Field(default_factory=list)
    deterministic: bool = True


class PolicyPackage(_Model):
    """The typed metadata sidecar of an exported ONNX policy — the portable policy artifact
    Learn exports, Mind hosts (ONNX Runtime), Guard wraps, and Bench scores. Core owns the
    *shape* so the artifact loads and satisfies the Policy/Planner contract uniformly; it
    never runs the graph (zero heavy deps, core.md §2 principle 3). ``core_interfaces`` maps
    a Core interface name to the version the policy was built against — the input to the same
    load-time negotiation the registry applies (see :meth:`assert_core_compatible`)."""

    name: str
    version: str
    onnx_model: ModelRef
    io_signature: IoSignature | None = None
    core_interfaces: dict[str, str] = Field(default_factory=dict)
    assumptions: PolicyAssumptions | None = None
    provenance: Provenance | None = None

    def assert_core_compatible(self, provided: Mapping[str, str] | None = None) -> None:
        """Assert this package's :attr:`core_interfaces` are satisfied by this Core.

        The bridge from an exported policy to interface-version negotiation — the same rule
        the plugin registry applies to a manifest and a SADF asset applies to itself
        (:func:`astro_mine.core.compat.assert_core_compatible`). ``provided`` overrides the
        Core interface versions negotiated against (defaults to this build's
        :data:`~astro_mine.core.compat.CORE_INTERFACE_VERSIONS`). Raises
        :class:`~astro_mine.core.compat.IncompatibleCoreInterface` on any mismatch."""
        compat.assert_core_compatible(self.core_interfaces, provided=provided)


class PolicyPackageDocument(_Model):
    """Top-level PolicyPackage document. ``policy_package_version`` pins the schema minor."""

    policy_package_version: Literal["0.1"]
    policy_package: PolicyPackage
