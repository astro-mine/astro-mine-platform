"""Reusable test harness for the Mind spine.

The two cross-cutting invariants every Mind issue re-asserts — *Guard-wrapped output is the
only output* and *determinism on demand* — live here as shared helpers so RM-P1-MIND-02/05/06/07
import them rather than re-inventing them, alongside factories for composing the reference
stack and for registering ad-hoc test policies (a tagging shield, a raising tier).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.enums import DeterminismClass, PluginKind, SignatureScheme
from astro_mine.core.registry.model import PluginManifest, Signature
from astro_mine.mind.bt.model import BehaviorTree
from astro_mine.mind.compose import compose
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.exec import Executive, RunResult
from astro_mine.mind.reference import load_reference_bt, load_reference_stack, load_stack_resource
from astro_mine.mind.registry import TierPlugin, TierRegistry
from astro_mine.mind.spec.model import ShieldBinding, StackSpecDocument
from astro_mine.mind.trace import to_canonical_json
from tests.mind.support.toy_env import ToyProspectingEnv

REFERENCE_SEED = 7
#: Sentinel the tagging shield stamps onto every action it emits, so a test can prove every
#: emitted action passed through the shield.
TAGGING_SENTINEL = -999.0


def reference_registry(*, require_signature: bool = False) -> TierRegistry:
    """A registry with the reference tier/shield plugins discovered via entry points."""
    return TierRegistry.from_entry_points(require_signature=require_signature)


def compose_reference(
    registry: TierRegistry | None = None, *, seed: int = REFERENCE_SEED
) -> HierarchyGraph:
    """Compose the shipped reference stack."""
    return compose(load_reference_stack(), registry or reference_registry(), seed=seed)


def run_reference(*, horizon: int = 8, max_ticks: int = 8, seed: int = REFERENCE_SEED) -> RunResult:
    """Run the reference stack against a fresh toy env."""
    graph = compose_reference(seed=seed)
    return Executive(graph).run(ToyProspectingEnv(horizon=horizon), max_ticks=max_ticks, seed=seed)


def compose_stack(
    resource: str,
    registry: TierRegistry | None = None,
    *,
    seed: int = REFERENCE_SEED,
    behavior_tree: BehaviorTree | None = None,
) -> HierarchyGraph:
    """Compose a shipped reference stack spec (by ``stacks/`` name)."""
    return compose(
        load_stack_resource(resource),
        registry or reference_registry(),
        seed=seed,
        behavior_tree=behavior_tree,
    )


def run_stack(
    resource: str,
    *,
    horizon: int = 8,
    max_ticks: int = 8,
    seed: int = REFERENCE_SEED,
    comms_denied_ticks: tuple[int, ...] = (),
    behavior_tree: BehaviorTree | None = None,
) -> RunResult:
    """Compose and run a shipped reference stack against a fresh toy env."""
    graph = compose_stack(resource, seed=seed, behavior_tree=behavior_tree)
    env = ToyProspectingEnv(horizon=horizon, comms_denied_ticks=comms_denied_ticks)
    return Executive(graph).run(env, max_ticks=max_ticks, seed=seed)


def compose_reference_bt(seed: int = REFERENCE_SEED) -> HierarchyGraph:
    """Compose the reference behavior-tree stack with its parsed tree."""
    return compose_stack("lunar_prospecting_bt.yaml", seed=seed, behavior_tree=load_reference_bt())


def assert_deterministic_trace(make_run: Callable[[], RunResult], *, runs: int = 2) -> str:
    """Assert repeated runs produce byte-identical canonical traces; return the trace."""
    traces = {to_canonical_json(make_run().trace) for _ in range(runs)}
    assert len(traces) == 1, "runs produced diverging decision traces (non-determinism)"
    return traces.pop()


def assert_shielded_egress(result: RunResult, *, sentinel: float = TAGGING_SENTINEL) -> None:
    """Assert every emitted action bears the shield's ``sentinel`` — i.e. nothing reached
    the environment without passing through the shield (the only output path)."""
    emitted = [action for tick in result.trace.ticks for action in tick.action_batch.actions]
    assert emitted, "no actions were emitted"
    assert all(action.sim_time_s == sentinel for action in emitted), (
        "an emitted action did not pass through the shield"
    )


def policy_plugin(
    name: str, factory: Callable[[Mapping[str, Any]], Policy], *, tier: str | None = None
) -> TierPlugin:
    """A minimal signed-as-unsigned ``policy`` plugin wrapping ``factory`` for tests."""
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        kind=PluginKind.POLICY,
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
        determinism_class=DeterminismClass.BIT_EXACT,
        signature=Signature(scheme=SignatureScheme.UNSIGNED),
        attributes={"tier": tier} if tier is not None else {},
    )
    return TierPlugin(manifest=manifest, factory=factory)


def reference_doc_with_shield(shield_plugin: str) -> StackSpecDocument:
    """The reference stack spec with its shield binding swapped for ``shield_plugin``."""
    doc = load_reference_stack()
    spec = doc.stack_spec.model_copy(update={"shield": ShieldBinding(plugin=shield_plugin)})
    return StackSpecDocument(stack_spec_version="0.1", stack_spec=spec)


class TaggingShield:
    """A shield that stamps :data:`TAGGING_SENTINEL` onto every action it emits — the probe
    :func:`assert_shielded_egress` uses to prove the shield is the only egress."""

    def decide(self, observations: Mapping[str, Any], context: DecisionContext) -> ActionBatch:
        base = context.upstream if context.upstream is not None else ActionBatch()
        return ActionBatch(
            actions=[a.model_copy(update={"sim_time_s": TAGGING_SENTINEL}) for a in base.actions]
        )


class RaisingPolicy:
    """A tier that always fails — used to exercise fallback activation."""

    def decide(self, observations: Mapping[str, Any], context: DecisionContext) -> ActionBatch:
        raise RuntimeError("intentional tier failure")
