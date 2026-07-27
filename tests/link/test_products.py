"""ContactPlan + ConnectivitySampler + CommsObservationMask products (RM-P0-LINK-04).

Unit-level: the reduction from Link windows/budgets into the Core message catalog, and the
per-epoch sampler + comms masks built over a plan. No geometry layer is needed — windows and
budgets are constructed directly, so the tests exercise the product mapping and the
reachability/mask logic, not visibility correctness (that lives in geometry/windows).
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages import (
    CommsObservationMask,
    ContactInterval,
    ContactNode,
    ContactPlan,
)
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.sadf.enums import CommsBand
from astro_mine.core.sadf.model import Comms
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.budget import LinkBudget, compute_link_budget
from astro_mine.link.products import (
    ConnectivitySampler,
    LinkProductsError,
    LinkState,
    build_contact_plan,
)
from astro_mine.link.windows import ContactWindow


def _epoch(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


def _window(observer: str, target: str, start: float, end: float) -> ContactWindow:
    return ContactWindow(observer, target, _epoch(start), _epoch(end))


_ROVER = ContactNode(id="rover", role=NodeRole.SPACE, kind="surface_agent")
_RELAY = ContactNode(id="relay", role=NodeRole.SPACE, kind="relay_orbiter")
_DSS = ContactNode(id="dss", role=NodeRole.GROUND, kind="ground_station")
_NODES = [_ROVER, _RELAY, _DSS]


def _feasible_budget() -> LinkBudget:
    tx = Comms(name="tx", band=CommsBand.S_BAND, eirp_dbw=50.0, modcod_supported=["qpsk_r1_2"])
    rx = Comms(name="rx", band=CommsBand.S_BAND, gt_db_per_k=10.0, modcod_supported=["qpsk_r1_2"])
    return compute_link_budget(tx, rx, range_m=1.0e6, frequency_hz=2.2e9)


def _infeasible_budget() -> LinkBudget:
    # A rate floor the faint geometry cannot clear -> feasible=False (rate 0, no mod/cod).
    tx = Comms(
        name="tx",
        band=CommsBand.S_BAND,
        eirp_dbw=0.0,
        modcod_supported=["qpsk_r1_2"],
        min_rate_bps=1.0e3,
    )
    rx = Comms(
        name="rx",
        band=CommsBand.S_BAND,
        gt_db_per_k=-20.0,
        modcod_supported=["qpsk_r1_2"],
        min_rate_bps=1.0e3,
    )
    return compute_link_budget(tx, rx, range_m=3.8e8, frequency_hz=2.2e9)


# --- build_contact_plan -----------------------------------------------------------------


def test_build_maps_windows_to_intervals_with_epoch_span() -> None:
    windows = [_window("rover", "relay", 30.0, 60.0), _window("rover", "dss", 10.0, 25.0)]
    span = EpochWindow(start=_epoch(0.0), end=_epoch(100.0))
    plan = build_contact_plan(_NODES, windows, epoch_window=span)
    assert isinstance(plan, ContactPlan)
    assert [n.id for n in plan.nodes] == ["rover", "relay", "dss"]
    assert [(iv.node_a, iv.node_b, iv.start_tdb_s, iv.end_tdb_s) for iv in plan.intervals] == [
        ("rover", "relay", 30.0, 60.0),
        ("rover", "dss", 10.0, 25.0),
    ]
    assert (plan.epoch_start_tdb_s, plan.epoch_end_tdb_s) == (0.0, 100.0)
    # The additive typed wire fields are populated alongside the kept primitives (RFC-0007
    # Design §2: producers SHOULD populate typed; consumers MUST prefer it).
    assert plan.window == span
    assert plan.intervals[0].window == EpochWindow(start=_epoch(30.0), end=_epoch(60.0))
    assert plan.intervals[0].window is not None
    assert plan.intervals[0].window.start.scale is TimeScale.TDB


def test_build_without_budget_leaves_quality_unset() -> None:
    plan = build_contact_plan(_NODES, [_window("rover", "relay", 30.0, 60.0)])
    iv = plan.intervals[0]
    assert iv.max_rate_bps is None
    assert iv.min_latency_s is None
    assert iv.modcod is None
    assert iv.link_budget is None
    assert plan.epoch_start_tdb_s is None


def test_build_annotates_intervals_from_the_link_budget() -> None:
    budget = _feasible_budget()
    plan = build_contact_plan(
        _NODES,
        [_window("rover", "relay", 30.0, 60.0)],
        budgets={("rover", "relay"): budget},
    )
    iv = plan.intervals[0]
    assert iv.max_rate_bps == pytest.approx(budget.rate_bps)
    assert iv.min_latency_s == pytest.approx(budget.latency_s)
    assert iv.mean_latency_s == pytest.approx(budget.latency_s)
    assert iv.margin_db == pytest.approx(budget.margin_db)
    assert iv.modcod == budget.modcod
    assert iv.link_budget is not None
    assert iv.link_budget.eirp_dbw == pytest.approx(budget.eirp_dbw)
    assert iv.link_budget.path_loss_db == pytest.approx(budget.fspl_db)
    assert iv.link_budget.gt_db_per_k == pytest.approx(budget.gt_db_per_k)
    assert iv.link_budget.margin_db == pytest.approx(budget.margin_db)
    # required Eb/N0 is the achieved Eb/N0 minus the achieved margin.
    assert iv.link_budget.required_ebn0_db == pytest.approx(budget.ebn0_db - budget.margin_db)


def test_build_records_an_infeasible_budget_as_zero_rate_no_required_ebn0() -> None:
    budget = _infeasible_budget()
    assert not budget.feasible
    plan = build_contact_plan(
        _NODES,
        [_window("rover", "dss", 5.0, 15.0)],
        budgets={("rover", "dss"): budget},
    )
    iv = plan.intervals[0]
    assert iv.max_rate_bps == pytest.approx(0.0)
    assert iv.modcod is None
    assert iv.link_budget is not None
    assert iv.link_budget.required_ebn0_db is None  # no achieved Eb/N0 to derive it from
    assert iv.link_budget.eirp_dbw == pytest.approx(budget.eirp_dbw)


def test_build_only_annotates_the_matching_pair() -> None:
    plan = build_contact_plan(
        _NODES,
        [_window("rover", "relay", 30.0, 60.0), _window("rover", "dss", 10.0, 25.0)],
        budgets={("rover", "relay"): _feasible_budget()},
    )
    annotated, bare = plan.intervals
    assert annotated.max_rate_bps is not None
    assert bare.max_rate_bps is None  # no budget keyed for (rover, dss)


def test_build_rejects_a_window_referencing_an_undeclared_node() -> None:
    with pytest.raises(LinkProductsError, match="undeclared node"):
        build_contact_plan([_ROVER, _RELAY], [_window("rover", "ghost", 0.0, 10.0)])


def test_build_with_validate_false_still_produces_a_plan() -> None:
    plan = build_contact_plan(_NODES, [_window("rover", "relay", 30.0, 60.0)], validate=False)
    assert len(plan.intervals) == 1


# --- ConnectivitySampler ----------------------------------------------------------------


def _plan_one_pass() -> ContactPlan:
    return ContactPlan(
        nodes=_NODES,
        intervals=[
            ContactInterval(
                node_a="rover",
                node_b="relay",
                start_tdb_s=30.0,
                end_tdb_s=60.0,
                max_rate_bps=1.0e6,
                min_latency_s=1.3,
                mean_latency_s=1.5,
                margin_db=4.0,
            )
        ],
    )


def test_reachable_open_carries_link_quality() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    state = sampler.reachable("rover", "relay", _epoch(45.0))
    assert state == LinkState(reachable=True, rate_bps=1.0e6, latency_s=1.5, margin_db=4.0)


def test_reachable_is_symmetric_and_half_open() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    assert sampler.reachable("relay", "rover", _epoch(45.0)).reachable  # order-independent
    assert not sampler.reachable("rover", "relay", _epoch(10.0)).reachable  # before the window
    assert not sampler.reachable("rover", "relay", _epoch(60.0)).reachable  # end is exclusive


def test_reachable_uses_mean_latency_then_falls_back_to_min() -> None:
    plan = ContactPlan(
        nodes=_NODES,
        intervals=[
            ContactInterval(
                node_a="rover", node_b="relay", start_tdb_s=0.0, end_tdb_s=10.0, min_latency_s=2.0
            )
        ],
    )
    assert ConnectivitySampler(plan).reachable("rover", "relay", _epoch(5.0)).latency_s == 2.0


def test_reachable_unknown_pair_is_never_connected() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    assert not sampler.reachable("relay", "dss", _epoch(45.0)).reachable  # no interval for the pair


def test_reachable_across_disjoint_windows() -> None:
    plan = ContactPlan(
        nodes=[_ROVER, _RELAY],
        intervals=[
            ContactInterval(node_a="rover", node_b="relay", start_tdb_s=10.0, end_tdb_s=20.0),
            ContactInterval(node_a="rover", node_b="relay", start_tdb_s=40.0, end_tdb_s=50.0),
        ],
    )
    sampler = ConnectivitySampler(plan)
    assert not sampler.reachable("rover", "relay", _epoch(5.0)).reachable  # before all windows
    assert sampler.reachable("rover", "relay", _epoch(15.0)).reachable  # first window
    assert not sampler.reachable("rover", "relay", _epoch(30.0)).reachable  # gap between windows
    assert sampler.reachable("rover", "relay", _epoch(45.0)).reachable  # second window


def test_connectivity_reports_every_known_link() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    snapshot = sampler.connectivity(_epoch(45.0))
    assert set(snapshot) == {("relay", "rover")}  # canonical (sorted) pair key
    assert snapshot[("relay", "rover")].reachable
    assert sampler.nodes == ("rover", "relay", "dss")


def test_comms_mask_lists_peers_with_reachability_and_no_earth_contact() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    mask = sampler.comms_mask("rover", _epoch(45.0))
    assert isinstance(mask, CommsObservationMask)
    assert mask.agent_id == "rover"
    peers = {link.peer: link for link in mask.links}
    assert set(peers) == {"relay", "dss"}
    assert peers["relay"].reachable and peers["relay"].rate_bps == 1.0e6
    assert not peers["dss"].reachable  # no rover<->dss interval
    assert mask.earth_contact is False  # the only reachable peer is a space node


def test_comms_mask_earth_contact_true_when_a_ground_node_is_reachable() -> None:
    plan = ContactPlan(
        nodes=_NODES,
        intervals=[ContactInterval(node_a="rover", node_b="dss", start_tdb_s=30.0, end_tdb_s=60.0)],
    )
    mask = ConnectivitySampler(plan).comms_mask("rover", _epoch(45.0))
    assert mask.earth_contact is True  # dss is a GROUND node and is reachable this tick


def test_comms_mask_unknown_agent_raises() -> None:
    with pytest.raises(LinkProductsError, match="unknown agent"):
        ConnectivitySampler(_plan_one_pass()).comms_mask("ghost", _epoch(0.0))


def test_comms_masks_default_all_nodes_and_agent_subset() -> None:
    sampler = ConnectivitySampler(_plan_one_pass())
    everyone = sampler.comms_masks(_epoch(45.0))
    assert set(everyone) == {"rover", "relay", "dss"}
    only_rover = sampler.comms_masks(_epoch(45.0), agents=("rover",))
    assert set(only_rover) == {"rover"}
