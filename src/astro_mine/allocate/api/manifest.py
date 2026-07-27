"""``build_allocation_manifest`` — the Core plugin manifest Allocate publishes itself as.

Allocate **consumes** Core's manifest schema, it does not invent one (allocate.md §6;
core.md §9; the Surrogate/Prospect precedent). There is no ``AllocationManifest`` subclass:
Core's :class:`~astro_mine.core.registry.PluginManifest` is ``extra="forbid"`` with a closed
JSON Schema, so allocation-specific facets ride in its open ``attributes`` map
(:class:`AllocationAttributes`) while identity, the Core interfaces it implements,
provenance, and signature use the manifest's own typed fields.

Allocate registers as a **Core Policy/Planner allocation sub-interface** plugin: it reuses
the existing :class:`~astro_mine.core.registry.PluginKind.POLICY` kind — the allocation tier
is a *sub-interface* of the one Policy/Planner contract Core owns (core.md
``policy/protocol.py``: the four tiers share ``decide``), so no new ``PluginKind`` and no
Core RFC are needed (allocate.md §6). It declares **no capability tags**: the open-commons
scope is scientific/ISRU coordination and no gated tag applies (allocate.md §9); a
sensitive operational-targeting deployment would add the gated Core tag at that boundary,
not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.allocate.api._core import CORE_INTERFACES
from astro_mine.allocate.model.ir.model import IR_VERSION
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.registry import PluginKind, PluginManifest, Provenance

__all__ = ["MANIFEST_CORE_INTERFACES", "AllocationAttributes", "build_allocation_manifest"]

#: The Core interfaces the allocation plugin declares it is built against — the ``policy``
#: sub-interface it implements and the ``registry`` contract it is manifested through
#: (:data:`CORE_INTERFACES`), plus the ``messages`` (TaskDirective/ActionBatch it emits) and
#: ``objective`` (ObjectiveSpec it optimizes toward) contracts it consumes. Every entry is in
#: Core's ``CORE_INTERFACE_VERSIONS``, so the registry's negotiation gate passes.
MANIFEST_CORE_INTERFACES: dict[str, str] = {
    **CORE_INTERFACES,
    "messages": "0.1.0",
    "objective": "0.1.0",
}

#: The default task vocabulary the CP-SAT/skeleton allocator supports — the schedulable Core
#: task kinds (paramless ``deploy``/``standby`` and free-form ``custom`` are handled but not
#: advertised as first-class supported kinds).
_DEFAULT_SUPPORTED_TASK_KINDS: tuple[TaskKind, ...] = (
    TaskKind.GOTO,
    TaskKind.SAMPLE,
    TaskKind.EXCAVATE,
    TaskKind.HAUL,
    TaskKind.DOCK,
    TaskKind.HOP,
    TaskKind.CHARGE,
    TaskKind.PROSPECT,
)


class AllocationAttributes(BaseModel):
    """The allocation-specific facets carried in ``PluginManifest.attributes``.

    Typed here (frozen, ``extra="forbid"``) and folded into the manifest's open ``attributes``
    object via :meth:`model_dump`, so the facets are schema-checked and JSON-Schema-exportable
    without widening Core's manifest. They let a consumer (Mind, Hub's resolver) decide
    whether this allocator fits a problem from the **Core manifest alone**: the ``ir_version``
    it compiles to, the Core ``supported_task_kinds`` it can schedule, the solver ``backends``
    behind its strategy interface, and whether solves are ``deterministic``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: str
    supported_task_kinds: list[TaskKind] = Field(min_length=1)
    backends: list[str] = Field(min_length=1)
    deterministic: bool = True


def build_allocation_manifest(
    *,
    name: str,
    version: str,
    artifact_digest: str,
    backends: Sequence[str] = ("trivial-stub",),
    supported_task_kinds: Iterable[TaskKind] | None = None,
    deterministic: bool = True,
    code_version: str | None = None,
    toolchain_version: str | None = None,
    seed: int | None = None,
    input_hashes: Sequence[str] | None = None,
    env_lockfile: str | None = None,
) -> PluginManifest:
    """Build the Core ``PluginManifest`` for the allocation plugin (unsigned; caller attaches).

    The manifest declares ``kind=policy`` (the allocation sub-interface), the Core interfaces
    it is built against (:data:`MANIFEST_CORE_INTERFACES`), **no capability tags**, and carries
    the plugin's own ``sha256:<hex>`` content address as ``provenance.digest`` (the identity a
    signature binds to). Allocation facets — the IR version, supported task kinds, and solver
    backends — fold into ``attributes`` via :class:`AllocationAttributes`. The caller attaches
    a signature before a signature-requiring registry will load it.
    """
    kinds = (
        list(supported_task_kinds)
        if supported_task_kinds is not None
        else list(_DEFAULT_SUPPORTED_TASK_KINDS)
    )
    attrs = AllocationAttributes(
        ir_version=IR_VERSION,
        supported_task_kinds=kinds,
        backends=list(backends),
        deterministic=deterministic,
    )
    # Imported lazily to avoid a package-init import cycle (manifest is imported from
    # astro_mine.allocate.__init__, which defines __version__ before importing this module).
    from astro_mine.allocate import __version__ as _ALLOCATE_VERSION

    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.POLICY,
        core_interfaces=dict(MANIFEST_CORE_INTERFACES),
        capability_tags=[],
        license="Apache-2.0",
        description=(
            f"Allocation/scheduling sub-interface plugin {name!r} — compiles an "
            f"AllocationRequest to the solver-neutral Allocation IR v{IR_VERSION} and returns "
            "a feasible-by-construction, Guard-recheckable plan (allocate.md §3)."
        ),
        provenance=Provenance(
            digest=artifact_digest,
            code_version=code_version or version,
            toolchain_version=toolchain_version or f"astro-mine-allocate {_ALLOCATE_VERSION}",
            input_hashes=list(input_hashes or []),
            env_lockfile=env_lockfile,
            seed=seed,
        ),
        attributes=attrs.model_dump(mode="json"),
    )
