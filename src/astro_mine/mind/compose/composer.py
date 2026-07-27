"""The composer — stack spec → validated, runnable hierarchy graph (RM-P1-MIND-01).

:func:`compose` turns a loaded :class:`StackSpecDocument` into a :class:`HierarchyGraph`,
resolving and instantiating each tier's (and the shield's) plugin through the
:class:`TierRegistry` and validating the wiring the loader could not (it has no registry):

- every tier's ``plugin`` resolves and its factory yields a ``Policy`` of ``kind == policy``
  (enforced by the registry);
- a tier whose manifest advertises an ``attributes.tier`` matches the role it is bound to
  (a wiring cross-check via the manifest's open ``attributes`` map);
- the mandatory **shield** is present and instantiable — the composer refuses to build an
  un-shielded graph, so the executive's only egress is a real shield (principle 7);
- the requested ``execution.kind`` is one this build implements.

Tiers are ordered canonically (mission → tamp → control), independent of authoring order,
so the executive threads ``upstream`` in the right direction. Assignment delegation
(RM-P1-MIND-04) and the behavior-tree execution kind (RM-P1-MIND-02) slot in here additively.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astro_mine.core.compat import CORE_INTERFACE_VERSIONS
from astro_mine.core.policy.protocol import Policy
from astro_mine.mind.bt.model import BehaviorTree
from astro_mine.mind.compose.graph import HierarchyGraph, ShieldNode, TierNode
from astro_mine.mind.registry.registry import (
    NotAPolicyPlugin,
    PluginNotRegistered,
    TierRegistry,
)
from astro_mine.mind.spec.enums import TIER_ORDER, ExecutionKind
from astro_mine.mind.spec.model import ShieldBinding, StackSpecDocument, TierBinding
from astro_mine.mind.trace.model import DecisionProvenance

__all__ = ["ComposeError", "compose"]

#: Manifest ``attributes`` key a plugin MAY use to advertise the tier it implements, so the
#: composer can cross-check it against the role it is bound to (advisory; absent is fine).
TIER_ATTRIBUTE = "tier"


class ComposeError(Exception):
    """Raised when a stack spec cannot be composed into a runnable hierarchy."""


def compose(
    document: StackSpecDocument,
    registry: TierRegistry,
    *,
    seed: int | None = None,
    provided: Mapping[str, str] | None = None,
    behavior_tree: BehaviorTree | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> HierarchyGraph:
    """Compose ``document`` into a runnable :class:`HierarchyGraph` using ``registry``.

    ``seed`` and ``provided`` (the Core interface versions the stack composed against,
    defaulting to this Core build) are recorded in the graph's provenance, along with
    ``input_hashes`` — the content-addressed identities of the stack's inputs (SADF, belief
    snapshot, comms model, ONNX artifacts; RM-P1-MIND-07). When the stack selects
    ``behavior_tree`` execution, the parsed ``behavior_tree`` (loaded from the
    ``execution.behavior_tree_ref``) MUST be supplied — the composer validates that its
    planner/policy leaves resolve to composed tiers and attaches it. Raises
    :class:`ComposeError` on any resolution/wiring failure.
    """
    spec = document.stack_spec

    if spec.execution.kind not in (ExecutionKind.COMPOSITION, ExecutionKind.BEHAVIOR_TREE):
        raise ComposeError(
            f"stack {spec.id!r}: execution kind {spec.execution.kind.value!r} is not "
            f"implemented in this build"
        )

    plugin_versions: dict[str, str] = {}
    tiers_by_role = {tier.role: tier for tier in spec.tiers}  # duplicates rejected at load
    ordered = [tiers_by_role[role] for role in TIER_ORDER if role in tiers_by_role]

    nodes: list[TierNode] = []
    for binding in ordered:
        policy = _instantiate(registry, binding.plugin, binding.params, what=binding.role.value)
        manifest = registry.manifest(binding.plugin)
        _check_tier_attribute(manifest.attributes, binding)
        plugin_versions[binding.plugin] = manifest.version

        fallback_policy = None
        fallback_name = None
        if binding.fallback is not None:
            fallback_name = binding.fallback.plugin
            fallback_policy = _instantiate(
                registry,
                fallback_name,
                binding.fallback.params,
                what=f"{binding.role.value} fallback",
            )
            plugin_versions[fallback_name] = registry.manifest(fallback_name).version

        nodes.append(
            TierNode(
                role=binding.role,
                plugin_name=binding.plugin,
                policy=policy,
                validity_horizon_s=binding.validity_horizon_s,
                replan_triggers=tuple(binding.replan_triggers),
                fallback=fallback_policy,
                fallback_name=fallback_name,
                manifest=manifest,
            )
        )

    shield = _compose_shield(registry, spec.shield, plugin_versions)

    tree = None
    if spec.execution.kind is ExecutionKind.BEHAVIOR_TREE:
        tree = _resolve_behavior_tree(spec.id, spec.execution, behavior_tree, nodes)

    provenance = DecisionProvenance(
        plugin_versions=plugin_versions,
        core_interface_versions=dict(provided)
        if provided is not None
        else dict(CORE_INTERFACE_VERSIONS),
        seed=seed,
        input_hashes=dict(input_hashes) if input_hashes is not None else {},
    )
    return HierarchyGraph(
        stack_id=spec.id,
        tiers=tuple(nodes),
        shield=shield,
        execution=spec.execution,
        coordination=spec.coordination,
        provenance=provenance,
        behavior_tree=tree,
    )


def _resolve_behavior_tree(
    stack_id: str, execution: object, tree: BehaviorTree | None, nodes: list[TierNode]
) -> BehaviorTree:
    """Validate the behavior tree against the composed tiers and return it."""
    if tree is None:
        raise ComposeError(
            f"stack {stack_id!r}: behavior_tree execution requires the parsed BehaviorTree "
            f"(from execution.behavior_tree_ref) to be passed to compose()"
        )
    ref = getattr(execution, "behavior_tree_ref", None)
    if not ref:
        raise ComposeError(
            f"stack {stack_id!r}: behavior_tree execution requires execution.behavior_tree_ref "
            f"to record which tree was composed (provenance)"
        )
    roles = {node.role.value for node in nodes}
    missing = sorted(tree.tier_refs() - roles)
    if missing:
        raise ComposeError(
            f"stack {stack_id!r}: behavior tree {tree.tree_id!r} invokes tier(s) "
            f"{', '.join(missing)!r} the stack does not compose"
        )
    return tree


def _instantiate(
    registry: TierRegistry, plugin: str, params: Mapping[str, Any], *, what: str
) -> Policy:
    try:
        return registry.instantiate(plugin, params)
    except (PluginNotRegistered, NotAPolicyPlugin) as exc:
        raise ComposeError(f"{what}: cannot instantiate plugin {plugin!r}: {exc}") from exc


def _compose_shield(
    registry: TierRegistry, binding: ShieldBinding, plugin_versions: dict[str, str]
) -> ShieldNode:
    policy = _instantiate(registry, binding.plugin, binding.params, what="shield")
    manifest = registry.manifest(binding.plugin)
    plugin_versions[binding.plugin] = manifest.version
    return ShieldNode(plugin_name=binding.plugin, policy=policy, manifest=manifest)


def _check_tier_attribute(attributes: Mapping[str, object], binding: TierBinding) -> None:
    declared = attributes.get(TIER_ATTRIBUTE)
    if declared is not None and declared != binding.role.value:
        raise ComposeError(
            f"plugin {binding.plugin!r} advertises tier {declared!r} but is bound to role "
            f"{binding.role.value!r} in the stack spec"
        )
