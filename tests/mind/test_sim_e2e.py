"""The Phase-1 Mind exit criterion, against the real thing (issue #14).

> *"a composed stack runs the anchor scenario against Sim, scored on Bench, with the
> degrade-not-collapse fallback validated under comms loss"* — roadmap, phase-1 Mind exit criteria;
> mind.md §12.

Every test here composes the anchor stack — **the real Guard PolicyShield** (over the Rust TCB) as
the single egress and **the real Allocate CP-SAT planner** in the allocator tier, both bound by
entry point — and runs it against **astro_mine.sim's own Simulator**: its stepping core, its
kinematic engine, its power/thermal evolution, and its ContactPlan-derived comms masking. The toy
env does not appear. Marked ``sim`` and run by the dedicated ``sim-e2e`` CI job, which installs the
``[sim]`` extra: a gate, not a skip.

**What makes these tests non-vacuous.** A shield that cannot certify a command does not error — it
substitutes a safe fallback (a zero-effort hold). A frozen swarm therefore still emits one action
per agent on every tick, and *"the swarm kept acting"* is satisfied by a swarm that did nothing at
all. That is not hypothetical: it is exactly what this stack did until the control tier was widened
to match the safety model's dimensionality. So every claim below is anchored on evidence a frozen
run cannot produce — agents that **moved**, commands that stayed in **VELOCITY** rather than
falling back to effort, and a shield that **edited** rather than **backed out**.
"""

from __future__ import annotations

import pytest

pytest.importorskip("astro_mine.sim", reason="the [sim] extra is not installed")
pytest.importorskip("astro_mine.bench", reason="the [sim] extra is not installed")
pytest.importorskip("astro_mine.guard", reason="the [sim] extra is not installed")
pytest.importorskip("astro_mine.allocate", reason="the [sim] extra is not installed")

from pathlib import Path

from astro_mine.bench.baseline import assert_score_reproducible
from astro_mine.bench.baseline import run as bench_run
from astro_mine.core.messages.enums import ControlMode
from astro_mine.mind.exec import Executive, StackPolicy
from astro_mine.mind.trace import to_canonical_json
from astro_mine.mind.trace.mcap import read_mcap_messages, write_mcap
from astro_mine.sim.bench import SimEpisodeRunner
from astro_mine.sim.runtime.episode import Simulator
from tests.mind.support.sim_anchor import (
    DT_S,
    SEED,
    agent_displacements,
    agent_ids,
    anchor_content,
    anchor_graph,
    anchor_spec,
    connectivity,
    keep_out_breaches,
    swarm_scenario,
)

pytestmark = pytest.mark.sim

#: mind.md §8 targets "tens to hundreds"; the anchor scenario's own fleet is ~12-25 heterogeneous
#: agents (scenarios/1-lunar-polar-ice-prospecting.md §6), and issue #14 asks for ≥20. 24 sits
#: inside the scenario's band and above the issue's floor.
SWARM_AGENTS = 24
_HORIZON = 16
#: Long enough to outlast the mission tier's 180 s (3-tick) validity horizon *after* the
#: `comms_lost` trigger re-arms it — otherwise act-while-stale is never reached. See the stack spec.
_BLACKOUT = (4, 5, 6, 7, 8, 9, 10, 11)


def _run(scenario, *, blackout=(), max_ticks=_HORIZON, seed=SEED):
    """Compose the anchor stack and run it against a real Sim episode."""
    graph = anchor_graph(seed=seed)
    env = Simulator(
        scenario,
        connectivity=connectivity(
            [a.agent_id for a in scenario.agents],
            blackout_ticks=blackout,
            horizon_steps=scenario.horizon_steps,
            dt_s=DT_S,
        ),
    )
    return graph, env, Executive(graph).run(env, max_ticks=max_ticks, seed=seed)


def _emitted_actions(result):
    return [a for tick in result.trace.ticks for a in tick.action_batch.actions]


# --- 1. the reference stack runs against real Sim ----------------------------------------


def test_the_anchor_stack_runs_against_real_sim() -> None:
    """The stack composes the real siblings and steps astro_mine.sim's own Simulator."""
    scenario = swarm_scenario(6, horizon_steps=_HORIZON)
    graph, _, result = _run(scenario)

    # The real siblings are bound — not Mind's in-repo stand-ins.
    assert graph.shield.plugin_name == "guard.shield"
    assert type(graph.shield.policy).__name__ == "GuardShield"
    assert [t.plugin_name for t in graph.tiers] == [
        "mind.mission.pddl",
        "allocate.planner",
        "mind.tamp.sampling",
        "mind.control.pid",
    ]

    assert result.ticks_run == _HORIZON
    # Every agent is commanded on every tick — a *defined* state, never undefined behaviour.
    assert all(len(t.action_batch.actions) == 6 for t in result.trace.ticks)

    # ...and the swarm actually PROSPECTED rather than being frozen by a fail-closed shield.
    moved = agent_displacements(scenario, result.final_observations)
    assert all(d > 1.0 for d in moved.values()), f"agents did not move: {moved}"
    assert not keep_out_breaches(result.final_observations)


def test_the_shield_certifies_rather_than_falls_back() -> None:
    """The real TCB *edits* the commands (a CBF projection); it does not refuse them.

    The distinction is the whole test. A refused command yields a zero-effort fallback, which looks
    identical to a working run in any assertion that only counts actions.
    """
    scenario = swarm_scenario(6, horizon_steps=_HORIZON)
    _, _, result = _run(scenario)

    emitted = _emitted_actions(result)
    assert emitted
    # Certified VELOCITY commands — NOT the action gate's EFFORT fallback.
    assert all(a.actuator.control_mode is ControlMode.VELOCITY for a in emitted), (
        "the shield fell back to an effort hold: the command was not certifiable"
    )
    kinds = {t.shield.kind for t in result.trace.ticks if t.shield.intervened}
    assert "backup_activation" not in kinds, f"the shield backed out of the run: {kinds}"


# --- 2. scored through Bench's metric path -----------------------------------------------


def test_the_run_is_scored_through_benchs_metric_path(tmp_path: Path) -> None:
    """`bench.baseline.run` scores the composed Mind stack on real Sim physics.

    Bench owns the loop here and calls ``policy.decide`` — so the stack is injected as a
    :class:`StackPolicy`, and Sim's ``SimEpisodeRunner`` supplies the real physics behind Bench's
    ``EpisodeRunner`` seam.
    """
    n_agents = 3
    content = anchor_content(tmp_path / "registry", n_agents=n_agents)
    spec = anchor_spec(content, n_agents=n_agents, horizon_steps=_HORIZON)
    runner = SimEpisodeRunner(
        store=content["store"],
        provider_factories=content["factories"],
        recording_dir=tmp_path / "mcap",
        connectivity=connectivity(
            agent_ids(n_agents),
            blackout_ticks=_BLACKOUT,
            horizon_steps=_HORIZON,
            dt_s=DT_S,
        ),
    )

    card = bench_run(spec, StackPolicy(anchor_graph()), runner=runner)

    assert card.scenario_id == spec.scenario_id
    assert {m.metric for m in card.metrics} == {m.name for m in spec.metrics}
    assert card.content_hash.startswith("sha256:")

    # comms_robustness is the scenario's definition of degrade-not-collapse ("goal attainment under
    # modeled relay-window/PSR-denial dropout"). It is scorable only because the run is really
    # masked -- a partial blackout, so neither 0 nor 1.
    comms = next(m for m in card.metrics if m.metric == "comms_robustness")
    assert comms.value is not None, "the run was comms-blind: nothing masked the observations"
    assert 0.0 < comms.value < 1.0, f"expected partial Earth contact, got {comms.value}"

    # Sim wrote a real MCAP per seed — the artifact boundary Bench and Sim meet at.
    assert set(runner.recordings) == set(spec.seeds.public)
    for path in runner.recordings.values():
        assert path.exists() and path.stat().st_size > 0


def test_the_score_reproduces(tmp_path: Path) -> None:
    """Bench's scoring determinism gate, on the composed stack: same inputs => same scorecard hash.

    This is also what proves ``StackPolicy`` resets between episodes — Bench reuses one policy
    object across seeds, and a stack that leaked cached tier decisions across them would score
    differently on the second run.
    """
    n_agents = 3
    content = anchor_content(tmp_path / "registry", n_agents=n_agents)
    spec = anchor_spec(content, n_agents=n_agents, horizon_steps=8, seeds=(1001, 1002))

    def runner() -> SimEpisodeRunner:
        return SimEpisodeRunner(
            store=content["store"],
            provider_factories=content["factories"],
            recording_dir=tmp_path / "mcap",
            connectivity=connectivity(
                agent_ids(n_agents), blackout_ticks=(3, 4, 5), horizon_steps=8, dt_s=DT_S
            ),
        )

    # A fresh policy per run would hide a leak; assert_score_reproducible reuses ONE, as Bench does.
    assert_score_reproducible(spec, StackPolicy(anchor_graph()), runner=runner(), runs=2)


# --- 3. swarm scale ----------------------------------------------------------------------


def test_the_stack_runs_at_swarm_scale() -> None:
    """≥20 agents through the composed hierarchy on real physics (mind.md §8; issue #14)."""
    scenario = swarm_scenario(SWARM_AGENTS, horizon_steps=_HORIZON)
    _, _, result = _run(scenario)

    assert len(scenario.agents) == SWARM_AGENTS
    assert result.ticks_run == _HORIZON
    # Every agent, every tick — the mission/allocator/TAMP/control tiers all fan out to the swarm.
    assert all(len(t.action_batch.actions) == SWARM_AGENTS for t in result.trace.ticks)

    moved = agent_displacements(scenario, result.final_observations)
    assert len(moved) == SWARM_AGENTS
    assert all(d > 1.0 for d in moved.values()), "agents did not move at swarm scale"
    assert not keep_out_breaches(result.final_observations)


# --- 4. degrade-not-collapse, under real Sim comms masking --------------------------------


def test_degrade_not_collapse_under_a_real_contact_plan() -> None:
    """LUNAR-FR-005 / RM-P1-MIND-06, against Sim's own comms masking rather than a synthetic mask.

    The blackout is the *absence of an open contact window* in a Core ContactPlan; Sim's sampler
    derives ``earth_contact`` from the contact graph and puts it on ``Observation.comms``. Mind's
    degrade path reads it through exactly the field a Link-produced plan would populate.

    Degrade-not-collapse is a **safety property** (mind.md §9): the swarm must reach a *defined*
    safe-productive state, never undefined behaviour. The docs specify no numeric floor, so this
    asserts the documented behavioural signature rather than inventing a threshold.
    """
    scenario = swarm_scenario(6, horizon_steps=_HORIZON)
    _, _, result = _run(scenario, blackout=_BLACKOUT)

    # The blackout really landed -- and it came from the contact plan, not from a tick set Mind was
    # handed. Without this the rest of the test would pass vacuously on a fully-lit run.
    denied = tuple(t.tick for t in result.trace.ticks if t.comms_denied)
    assert denied == _BLACKOUT, f"the ContactPlan did not deny comms as expected: {denied}"

    notes = {rec.note for tick in result.trace.ticks for rec in tick.tiers}
    # act-while-stale: the mission replan is suppressed, agents run on cached intent.
    assert "comms_stale_hold" in notes
    # reconcile-on-recovery: the tick the window reopens, the mission tier re-plans.
    assert "comms_recovered" in notes

    # DEGRADED, NOT COLLAPSED: every agent is still commanded on every dark tick...
    dark = [t for t in result.trace.ticks if t.comms_denied]
    assert all(len(t.action_batch.actions) == 6 for t in dark)
    # ...with certified VELOCITY commands, not a fail-closed hold...
    assert all(
        a.actuator.control_mode is ControlMode.VELOCITY
        for t in dark
        for a in t.action_batch.actions
    )
    # ...and the swarm kept working through the blackout rather than freezing.
    moved = agent_displacements(scenario, result.final_observations)
    assert all(d > 1.0 for d in moved.values()), f"the swarm froze under blackout: {moved}"
    # Guard held throughout: no hard-constraint violation, blackout or not (LUNAR-FR-006).
    assert not keep_out_breaches(result.final_observations)


def test_a_blackout_changes_the_decisions() -> None:
    """The comms mask is not inert: a blacked-out run decides differently from a lit one.

    Without this, every assertion above could hold on a stack that simply ignored ``comms``.
    """
    lit = _run(swarm_scenario(6, horizon_steps=_HORIZON))[2]
    dark = _run(swarm_scenario(6, horizon_steps=_HORIZON), blackout=_BLACKOUT)[2]

    assert not any(t.comms_denied for t in lit.trace.ticks)
    assert to_canonical_json(lit.trace) != to_canonical_json(dark.trace)


# --- 5. MCAP decision traces for the real-Sim run -----------------------------------------


def test_mcap_decision_trace_is_recorded_for_the_real_sim_run(tmp_path: Path) -> None:
    """RM-P1-MIND-07: the real-Sim run's decision trace serializes to MCAP, replayable in View."""
    scenario = swarm_scenario(6, horizon_steps=_HORIZON)
    _, _, result = _run(scenario, blackout=_BLACKOUT)

    path = tmp_path / "anchor.mcap"
    write_mcap(result.trace, path)
    assert path.exists() and path.stat().st_size > 0

    messages = read_mcap_messages(path)
    topics = {topic for topic, _ in messages}
    assert "mind/tier_decision" in topics
    assert "mind/plan_revision" in topics
    # The blackout's act-while-stale / reconcile notes ride the fallback-activation channel, so the
    # comms-degradation events are in the replayable artifact, not just in the in-memory trace.
    assert "mind/fallback_activation" in topics
    notes = {
        payload.get("note") for topic, payload in messages if topic == "mind/fallback_activation"
    }
    assert "comms_stale_hold" in notes


def test_the_real_sim_run_is_deterministic() -> None:
    """Same stack + same scenario + same seed => the identical decision trace (conventions.md §11).

    No golden file is committed for the real-Sim traces: they run real physics, and pinning their
    bytes across machines would gate CI on float reproducibility rather than on Mind's decisions.
    The property that matters -- a run reproduces -- is asserted directly.
    """
    first = _run(swarm_scenario(6, horizon_steps=8), blackout=(3, 4, 5), max_ticks=8)[2]
    second = _run(swarm_scenario(6, horizon_steps=8), blackout=(3, 4, 5), max_ticks=8)[2]

    assert to_canonical_json(first.trace) == to_canonical_json(second.trace)
