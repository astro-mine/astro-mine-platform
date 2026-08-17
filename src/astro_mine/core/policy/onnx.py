# SPDX-License-Identifier: Apache-2.0
"""The ONNX-PolicyPackage → Policy adapter (RM-P1-CORE-01).

Guarantees a Learn-exported ONNX policy is *loadable and callable as a plain
:class:`~astro_mine.core.policy.protocol.Policy`* — so it drops into a composed stack (as
a controller), is wrapped by Guard, and is scored by Bench through the one standard
interface.

Core defines the adapter *shape*; it never runs ONNX (zero heavy dependencies —
onnxruntime/torch stay out of the waist, core.md §2 principle 3). The host — Mind — supplies
the ``infer`` callable that maps observations to an :class:`~astro_mine.core.messages.ActionBatch`
using the package's declared :class:`~astro_mine.core.policy.model.IoSignature`. This keeps
the swappable inference engine at the edge while the contract stays in Core.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext, PolicyPackage

__all__ = ["InferenceFn", "OnnxPolicy"]

#: A host-supplied inference function: maps the per-agent observations + context to an
#: action batch, running the ONNX graph (e.g. an ONNX-Runtime session in Mind). Core ships
#: none — the runtime is the host's, injected here.
InferenceFn = Callable[[Mapping[AgentId, Observation], DecisionContext], ActionBatch]


class OnnxPolicy:
    """Binds an exported :class:`PolicyPackage` + a host ``infer`` callable to the Policy
    contract, so a learned ONNX policy satisfies :class:`~astro_mine.core.policy.protocol.Policy`
    unchanged.

    Construction negotiates the package's declared ``core_interfaces`` against this Core
    (the same rule the registry applies to a plugin manifest), so an incompatible policy
    fails **loudly at load**, not mid-episode. ``provided`` overrides the Core interface
    versions negotiated against (for tests / a hypothetical future Core).
    """

    def __init__(
        self,
        package: PolicyPackage,
        infer: InferenceFn,
        *,
        provided: Mapping[str, str] | None = None,
    ) -> None:
        package.assert_core_compatible(provided=provided)
        self._package = package
        self._infer = infer

    @property
    def package(self) -> PolicyPackage:
        """The exported policy's metadata sidecar."""
        return self._package

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Run the exported policy for this decision step via the host inference function."""
        return self._infer(observations, context)
