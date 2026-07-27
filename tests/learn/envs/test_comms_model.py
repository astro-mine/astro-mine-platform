"""``CommsModel`` — the comms-regime channel over the SwarmEnv stream (RM-P1-LEARN-02).

Exercises the four channel stages (gate / budget / drop / delay), the comms-budget
accounting, and — the headline acceptance criteria — that the realization is a pure
function of (config, seed): identical across algorithms and reproducible under a fixed seed
(learn.md §3; charter §8). The channel is unit-tested directly on hand-built Core
observations, then end-to-end through :func:`make_swarm_env`.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.core.messages.model import (
    CommsObservationMask,
    Observation,
    PeerLink,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.learn.envs import CommsModel, make_swarm_env
from astro_mine.learn.envs.comms import (
    BandwidthBudgetConfig,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
    RangeGateConfig,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _state(agent: str, x: float = 0.0) -> StateSample:
    return StateSample(
        agent_id=agent,
        frame=MOON_BODY_FIXED,
        pose=Transform(
            translation_m=Vec3(x=x, y=0.0, z=0.0),
            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _obs(tick: int, peers: dict[str, dict], *, agent: str = "a", pos_x: float = 0.0) -> Observation:
    """A one-agent observation whose comms mask carries the given peer links and whose
    neighbours are one peer each at the requested x offset."""
    links = [
        PeerLink(peer=p, reachable=True, **{k: v for k, v in attrs.items() if k != "_x"})
        for p, attrs in peers.items()
    ]
    neighbors = [_state(p, x=attrs.get("_x", 0.0)) for p, attrs in peers.items()]
    return Observation(
        tick=tick,
        sim_time_s=float(tick),
        agent_id=agent,
        self_state=_state(agent, x=pos_x),
        comms=CommsObservationMask(agent_id=agent, links=links, earth_contact=False),
        neighbors=neighbors,
    )


def _reachable(obs: Observation) -> set[str]:
    return {link.peer for link in (obs.comms.links if obs.comms else []) if link.reachable}


# --- identity / null-model invariant -------------------------------------------------


def test_identity_config_is_a_no_op() -> None:
    """An empty config degrades nothing — the wrapped rollout equals the un-wrapped one."""
    assert CommsModelConfig().is_identity

    def links(comms_model: CommsModel | None) -> list[dict[str, list]]:
        env = make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=comms_model)
        _, infos = env.reset(seed=1)
        frames = [{a: [link.peer for link in infos[a]["comms"]] for a in infos}]
        for _ in range(5):
            if not env.agents:
                break
            _, _, _, _, infos = env.step({a: {"kind": 0, "mode": 0} for a in env.agents})
            frames.append({a: [link.peer for link in infos[a]["comms"]] for a in infos})
        return frames

    assert links(None) == links(CommsModel(CommsModelConfig()))


# --- drop ----------------------------------------------------------------------------


def test_full_drop_delivers_nothing() -> None:
    model = CommsModel(CommsModelConfig(drop=DropConfig(probability=1.0)))
    model.reset(0)
    out = model.apply({"a": _obs(0, {"b": {}, "c": {}})})
    assert _reachable(out["a"]) == set()
    tally = model.ledger.tally("a")
    assert tally.offered == 2
    assert tally.loss_dropped == 2
    assert tally.delivered == 0


def test_non_degrading_channel_delivers_all_reachable() -> None:
    # A generous budget is non-identity (so accounting runs) but sheds nothing.
    model = CommsModel(
        CommsModelConfig(bandwidth=BandwidthBudgetConfig(per_agent_bits_per_tick=100.0))
    )
    model.reset(0)
    out = model.apply({"a": _obs(0, {"b": {}, "c": {}})})
    assert _reachable(out["a"]) == {"b", "c"}
    assert model.ledger.tally("a").delivered == 2


# --- gating --------------------------------------------------------------------------


def test_margin_gate_removes_weak_links() -> None:
    model = CommsModel(CommsModelConfig(range_gate=RangeGateConfig(min_margin_db=2.0)))
    model.reset(0)
    out = model.apply({"a": _obs(0, {"b": {"margin_db": 5.0}, "c": {"margin_db": 0.5}})})
    assert _reachable(out["a"]) == {"b"}
    assert model.ledger.tally("a").gated_out == 1


def test_range_gate_uses_neighbour_geometry() -> None:
    model = CommsModel(CommsModelConfig(range_gate=RangeGateConfig(max_range_m=50.0)))
    model.reset(0)
    obs = _obs(0, {"near": {"_x": 10.0}, "far": {"_x": 100.0}})
    out = model.apply({"a": obs})
    assert _reachable(out["a"]) == {"near"}


# --- bandwidth budget ----------------------------------------------------------------


def test_budget_admits_by_priority_and_counts_the_rest() -> None:
    cfg = CommsModelConfig(
        bandwidth=BandwidthBudgetConfig(
            per_agent_bits_per_tick=2.0, message_bits=1.0, priority="margin_db"
        )
    )
    model = CommsModel(cfg)
    model.reset(0)
    peers = {"hi": {"margin_db": 9.0}, "mid": {"margin_db": 4.0}, "lo": {"margin_db": 1.0}}
    out = model.apply({"a": _obs(0, peers)})
    assert _reachable(out["a"]) == {"hi", "mid"}  # budget = 2 messages, highest margin wins
    assert model.ledger.tally("a").budget_dropped == 1


# --- delay ---------------------------------------------------------------------------


def test_fixed_delay_defers_arrival_and_holds_last_known() -> None:
    model = CommsModel(CommsModelConfig(delay=DelayConfig(kind="fixed", ticks=2)))
    model.reset(0)
    arrivals = []
    for tick in range(5):
        out = model.apply({"a": _obs(tick, {"b": {"_x": float(tick)}})})
        neighbours = {n.agent_id for n in out["a"].neighbors}
        arrivals.append((_reachable(out["a"]), neighbours))
    # tick 0's message arrives at tick 2; before that nothing is reachable and no neighbour
    # is known; from tick 2 on the peer is reachable and its (stale) state is held.
    assert arrivals[0] == (set(), set())
    assert arrivals[1] == (set(), set())
    assert arrivals[2] == ({"b"}, {"b"})


# --- reproducibility / algorithm-agnosticism -----------------------------------------


def test_same_seed_same_config_is_reproducible() -> None:
    cfg = CommsModelConfig(
        drop=DropConfig(probability=0.5), delay=DelayConfig(kind="uniform", low=0, high=3)
    )

    def realize(seed: int) -> list[set[str]]:
        model = CommsModel(cfg)
        model.reset(seed)
        return [
            _reachable(model.apply({"a": _obs(t, {"b": {}, "c": {}, "d": {}})})["a"])
            for t in range(8)
        ]

    assert realize(123) == realize(123)


def test_realization_is_independent_of_actions() -> None:
    """Same config + seed ⇒ identical comms masks regardless of the actions taken — so
    comms-stress results are comparable across algorithms."""

    def comms(action_kind: int) -> list[dict[str, list]]:
        env = make_swarm_env(
            FakeSwarmWorld(),
            build_assets(),
            comms_model=CommsModel(CommsModelConfig(drop=DropConfig(probability=0.4))),
        )
        _, infos = env.reset(seed=99)
        frames = [{a: [link.peer for link in infos[a]["comms"]] for a in infos}]
        for _ in range(6):
            if not env.agents:
                break
            _, _, _, _, infos = env.step({a: {"kind": action_kind, "mode": 0} for a in env.agents})
            frames.append({a: [link.peer for link in infos[a]["comms"]] for a in infos})
        return frames

    assert comms(0) == comms(0)


# --- ledger invariant (property-based) -----------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    drop_p=st.floats(min_value=0.0, max_value=1.0),
    budget=st.sampled_from([None, 1.0, 2.0, 5.0]),
    seed=st.integers(min_value=0, max_value=10_000),
    n_peers=st.integers(min_value=0, max_value=6),
)
def test_every_offered_link_has_exactly_one_outcome(
    drop_p: float, budget: float | None, seed: int, n_peers: int
) -> None:
    cfg = CommsModelConfig(
        drop=DropConfig(probability=drop_p),
        bandwidth=BandwidthBudgetConfig(per_agent_bits_per_tick=budget, message_bits=1.0),
    )
    model = CommsModel(cfg)
    model.reset(seed)
    peers = {f"p{i}": {"margin_db": float(i)} for i in range(n_peers)}
    for tick in range(5):
        model.apply({"a": _obs(tick, peers)})
    t = model.ledger.tally("a")
    assert t.offered == t.delivered + t.gated_out + t.budget_dropped + t.loss_dropped
    assert 0.0 <= model.ledger.delivery_ratio() <= 1.0


# --- config: JSON Schema + round-trip + validation -----------------------------------


def test_config_round_trips_through_json() -> None:
    cfg = CommsModelConfig(
        range_gate=RangeGateConfig(max_range_m=1000.0, min_margin_db=3.0),
        drop=DropConfig(probability=0.25),
        delay=DelayConfig(kind="geometric", mean_ticks=2.0, max_ticks=8),
        bandwidth=BandwidthBudgetConfig(per_agent_bits_per_tick=8.0, message_bits=2.0),
    )
    assert CommsModelConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_config_json_schema_declares_the_stages() -> None:
    schema = CommsModelConfig.model_json_schema()
    assert set(schema["properties"]) >= {"range_gate", "bandwidth", "drop", "delay"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": {"probability": 1.5}},  # out of [0, 1]
        {"delay": {"kind": "uniform", "low": 5, "high": 2}},  # high < low
        {"bandwidth": {"message_bits": 0.0}},  # must be > 0
    ],
)
def test_invalid_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CommsModelConfig(**kwargs)


def test_provenance_records_the_declared_assumption() -> None:
    model = CommsModel(CommsModelConfig(drop=DropConfig(probability=0.3)))
    prov = model.provenance()
    assert prov["kind"] == "comms_model"
    assert prov["config"]["drop"]["probability"] == pytest.approx(0.3)


def test_delivery_ratio_defaults_to_one_when_nothing_offered() -> None:
    model = CommsModel(CommsModelConfig(drop=DropConfig(probability=0.5)))
    model.reset(0)
    assert model.ledger.delivery_ratio() == 1.0
