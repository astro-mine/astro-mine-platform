"""Degrade-not-collapse — plans, coord, and the injected-blackout stress harness (RM-P1-MIND-06)."""

from __future__ import annotations

from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ProspectTask,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.core.plan import PLAN_VERSION as CORE_PLAN_VERSION
from astro_mine.core.plan import ContingencyBranch, ContingentPlan, Plan, PlanValidity
from astro_mine.core.plan.loader import validate_plan
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.compose import compose
from astro_mine.mind.coord import GossipCoordinator, Intent
from astro_mine.mind.exec import (
    PLAN_VERSION,
    CompositionStrategy,
    DecentralizedStrategy,
    Executive,
    branch_for,
    build_strategy,
    expires_at_s,
    is_stale,
    is_valid,
    plan_document,
)
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import (
    CoordinationSpec,
    ReplanTrigger,
    ShieldBinding,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)
from astro_mine.mind.trace import to_canonical_json
from tests.mind.support.harness import (
    compose_reference,
    compose_stack,
    policy_plugin,
    reference_registry,
)
from tests.mind.support.toy_env import ToyProspectingEnv

_BLACKOUT = (3, 4, 5, 6, 7)


# --- plan artifacts (Core-owned schema, RFC-0006) --------------------------------------


def _contingent(horizon_s: float | None = 3.0) -> ContingentPlan:
    plan = Plan(
        plan_id="p",
        tier="mission",
        validity=PlanValidity(issued_at_s=0.0, horizon_s=horizon_s),
        actions=ActionBatch(),
    )
    return ContingentPlan(
        base=plan,
        branches=[
            ContingencyBranch(trigger="comms_lost", action="hold_cached"),
            ContingencyBranch(trigger="plan_expired", action="reconcile"),
        ],
    )


def test_plan_validity_horizon() -> None:
    validity = PlanValidity(issued_at_s=1.0, horizon_s=2.0)
    assert expires_at_s(validity) == 3.0
    assert is_valid(validity, 2.9) and not is_valid(validity, 3.0)
    assert is_stale(validity, 3.0)


def test_standing_plan_never_expires() -> None:
    validity = PlanValidity(issued_at_s=0.0, horizon_s=None)
    assert expires_at_s(validity) is None and is_valid(validity, 1e9)


def test_contingent_plan_branches() -> None:
    contingent = _contingent()
    resolved = branch_for(contingent, "comms_lost")
    assert resolved is not None and resolved.action == "hold_cached"
    assert branch_for(contingent, "nope") is None
    assert contingent.model_dump()["base"]["tier"] == "mission"


def test_issued_plan_validates_against_the_core_schema() -> None:
    # The migration's contract (RFC-0006): the plan Mind issues is a Core message, so it
    # round-trips through Core's canonical plan.schema.json — structural *and* semantic gates.
    validate_plan(plan_document(_contingent()))


def test_plan_version_tracks_core() -> None:
    # Mind re-declares the envelope version as a Literal (Core's widens to str); if Core bumps
    # the plan-schema minor, this fails loudly rather than emitting a stale envelope.
    assert PLAN_VERSION == CORE_PLAN_VERSION


# --- decentralized coordination -------------------------------------------------------


def test_coord_no_conflict_keeps_all_claims() -> None:
    resolution = GossipCoordinator().resolve([Intent("a", "t0"), Intent("b", "t1")])
    assert resolution.assignments == {"a": "t0", "b": "t1"}
    assert resolution.conflicts_resolved == 0 and resolution.yielded == ()


def test_coord_resolves_conflict_lowest_id_wins() -> None:
    resolution = GossipCoordinator().resolve([Intent("r1", "t"), Intent("r0", "t")])
    assert resolution.assignments == {"r0": "t", "r1": None}
    assert resolution.conflicts_resolved == 1 and resolution.yielded == ("r1",)


# --- strategy selection ---------------------------------------------------------------


def test_build_strategy_selects_by_coordination_posture() -> None:
    assert isinstance(build_strategy(compose_reference()), CompositionStrategy)
    assert isinstance(
        build_strategy(compose_stack("lunar_prospecting_degrade.yaml")), DecentralizedStrategy
    )


# --- injected-blackout stress harness -------------------------------------------------


def _run_degrade(comms_denied_ticks=_BLACKOUT):  # type: ignore[no-untyped-def]
    graph = compose_stack("lunar_prospecting_degrade.yaml", seed=7)
    env = ToyProspectingEnv(horizon=10, comms_denied_ticks=comms_denied_ticks)
    return Executive(graph).run(env, max_ticks=10, seed=7)


def test_degrade_not_collapse_is_a_safety_property() -> None:
    # Under injected comms loss, every agent stays in a DEFINED safe-productive state every tick
    # (a full action set, never undefined/empty), acting on cached intent and reconciling on
    # recovery — never collapsing.
    result = _run_degrade()
    assert all(len(t.action_batch.actions) == 2 for t in result.trace.ticks)
    notes = [next(r.note for r in t.tiers if r.role == "mission") for t in result.trace.ticks]
    assert "comms_stale_hold" in notes  # acted on cached intent through the blackout
    assert "comms_recovered" in notes  # reconciled on recovery
    # comms_lost trigger fired at the drop edge
    assert any(
        r.role == "mission" and r.trigger == "comms_lost"
        for t in result.trace.ticks
        for r in t.tiers
    )


def test_degrade_run_is_deterministic_including_blackout() -> None:
    assert to_canonical_json(_run_degrade().trace) == to_canonical_json(_run_degrade().trace)


def test_contingent_plan_issued_for_mission() -> None:
    graph = compose_stack("lunar_prospecting_degrade.yaml", seed=7)
    strategy = build_strategy(graph)
    # drive a couple of ticks so a mission ContingentPlan is issued
    env = ToyProspectingEnv(horizon=3)
    reset = env.reset(seed=7)
    from astro_mine.mind.belief.view import BELIEF_EXTRAS_KEY, assemble_belief

    belief = assemble_belief(reset.observations, tick=0, sim_time_s=0.0)
    strategy.decide(
        reset.observations,
        DecisionContext(seed=7, extras={BELIEF_EXTRAS_KEY: belief}),
        tick=0,
        sim_time_s=0.0,
    )
    assert isinstance(strategy, DecentralizedStrategy)
    contingent = strategy.contingent_plan(TierRole.MISSION)
    assert contingent is not None
    assert branch_for(contingent, "comms_lost") is not None
    assert contingent.base.assumptions[0].key == "earth_contact"
    validate_plan(plan_document(contingent))  # the issued plan is a valid Core plan document


def test_coord_yields_a_contended_region_under_blackout() -> None:
    # A mission planner that assigns BOTH agents the same region induces a conflict; under a
    # blackout the coord layer resolves it — the lower id keeps the region, the other yields
    # and holds (its action is dropped), annotated in the trace.
    region = Volume(
        frame="body", center_m=Vec3(x=10.0, y=0.0, z=0.0), dimensions_m=Vec3(x=2.0, y=2.0, z=2.0)
    )

    class _CollidingMission:
        def decide(self, observations, context):  # type: ignore[no-untyped-def]
            return ActionBatch(
                actions=[
                    Action(
                        agent_id=a,
                        kind=ActionKind.TASK,
                        task=TaskDirective(
                            task_kind=TaskKind.PROSPECT, prospect=ProspectTask(region=region)
                        ),
                    )
                    for a in sorted(observations)
                ]
            )

    registry = reference_registry()
    registry.register(
        policy_plugin("test.colliding-mission", lambda params: _CollidingMission(), tier="mission")
    )
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="collide",
            name="collide",
            coordination=CoordinationSpec(kind="decentralized"),
            tiers=[
                TierBinding(
                    role=TierRole.MISSION,
                    plugin="test.colliding-mission",
                    validity_horizon_s=2.0,
                    replan_triggers=[ReplanTrigger(kind="plan_expired")],
                ),
                TierBinding(role=TierRole.TAMP, plugin="mind.reference.tamp"),
                TierBinding(role=TierRole.CONTROL, plugin="mind.reference.control"),
            ],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    graph = compose(doc, registry, seed=7)
    result = Executive(graph).run(
        ToyProspectingEnv(horizon=6, comms_denied_ticks=(2, 3, 4)), max_ticks=6, seed=7
    )
    # during the blackout a control tick yields (one agent drops its action -> holds)
    yielded_ticks = [t for t in result.trace.ticks if any(r.note == "coord_yield" for r in t.tiers)]
    assert yielded_ticks
    assert any(len(t.action_batch.actions) == 1 for t in yielded_ticks)  # rover-1 yielded
