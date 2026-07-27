"""CGR contact graph + store-and-forward delivery model (RM-P1-LINK-11).

The delay-tolerant behavior: a bundle held at a relay until its next contact opens is
delivered even when no *contemporaneous* end-to-end path ever exists — and the instantaneous
fidelity refuses exactly that. Built on small hand-made ContactPlans (no kernels).

Contact spans and the send/arrival instants are carried as typed Core ``Epoch``/``EpochWindow``
(RM-P1-LINK-14, RFC-0007), never bare TDB floats on the contract surface.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.network import Contact, ContactGraph, DeliveryModel, LinkNetworkError


def _epoch(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


def _win(start: float, end: float) -> EpochWindow:
    return EpochWindow(start=_epoch(start), end=_epoch(end))


def _plan(intervals: list[ContactInterval], nodes: list[str]) -> ContactPlan:
    return ContactPlan(
        nodes=[ContactNode(id=n, role=NodeRole.SPACE) for n in nodes],
        intervals=intervals,
    )


def _iv(
    a: str,
    b: str,
    start: float,
    end: float,
    *,
    rate: float | None = 1000.0,
    latency: float | None = 1.0,
) -> ContactInterval:
    return ContactInterval(
        node_a=a,
        node_b=b,
        start_tdb_s=start,
        end_tdb_s=end,
        max_rate_bps=rate,
        mean_latency_s=latency,
    )


# --- Contact / ContactGraph ------------------------------------------------------


def test_contact_other_and_open_at() -> None:
    contact = Contact("A", "B", _win(0.0, 10.0), rate_bps=100.0, latency_s=2.0)
    assert contact.other("A") == "B" and contact.other("B") == "A"
    assert contact.open_at(_epoch(0.0))
    assert contact.open_at(_epoch(9.999))
    assert not contact.open_at(_epoch(10.0))
    with pytest.raises(LinkNetworkError, match="not an endpoint"):
        contact.other("C")


def test_contact_carries_a_typed_epoch_window() -> None:
    # The contact span is a typed EpochWindow, not a pair of bare TDB floats (RFC-0007).
    contact = Contact("A", "B", _win(3.0, 8.0))
    assert isinstance(contact.window, EpochWindow)
    assert contact.window.start.scale is TimeScale.TDB
    assert (contact.window.start.tdb_seconds, contact.window.end.tdb_seconds) == (3.0, 8.0)


def test_contact_graph_from_plan_indexes_by_node() -> None:
    plan = _plan([_iv("S", "R", 0.0, 10.0), _iv("R", "G", 20.0, 30.0)], ["S", "R", "G"])
    graph = ContactGraph.from_plan(plan)
    assert graph.nodes == frozenset({"S", "R", "G"})
    assert graph.node_ids == frozenset({"S", "R", "G"})
    assert [c.other("R") for c in graph.contacts_from("R")] == ["S", "G"]  # sorted by start
    assert graph.contacts_from("absent") == []
    assert len(graph.contacts) == 2


def test_contact_graph_prefers_typed_interval_window() -> None:
    # A producer-populated typed ``window`` is preferred over the primitives (RFC-0007).
    interval = ContactInterval(
        node_a="A", node_b="B", start_tdb_s=0.0, end_tdb_s=5.0, window=_win(0.0, 5.0)
    )
    graph = ContactGraph.from_plan(_plan([interval], ["A", "B"]))
    assert graph.contacts[0].window == _win(0.0, 5.0)


def test_contact_graph_latency_falls_back_to_min_latency() -> None:
    interval = ContactInterval(
        node_a="A", node_b="B", start_tdb_s=0.0, end_tdb_s=5.0, min_latency_s=3.0
    )
    graph = ContactGraph.from_plan(_plan([interval], ["A", "B"]))
    assert graph.contacts[0].latency_s == 3.0


def test_contact_graph_declared_isolated_node_is_known() -> None:
    plan = _plan([_iv("A", "B", 0.0, 5.0)], ["A", "B", "ISOLATED"])
    graph = ContactGraph.from_plan(plan)
    assert "ISOLATED" in graph.node_ids and "ISOLATED" not in graph.nodes


# --- DeliveryModel ---------------------------------------------------------------


def _store_forward_graph() -> ContactGraph:
    # S can reach R only early [0,10); R can reach G only later [20,30). No overlap ⇒ the
    # only way to deliver S→G is to hold the bundle at R until its later contact opens.
    return ContactGraph.from_plan(
        _plan([_iv("S", "R", 0.0, 10.0), _iv("R", "G", 20.0, 30.0)], ["S", "R", "G"])
    )


def test_store_and_forward_delivers_across_disjoint_contacts() -> None:
    model = DeliveryModel(_store_forward_graph())
    result = model.deliver("S", "G", send=_epoch(0.0), fidelity="store_and_forward")
    assert result.delivered is True
    assert result.hops == ("S", "R", "G")
    # S→R arrives at t=1 (1 s light-time); held at R until the 20 s contact opens; the R→G
    # hop then arrives at 20 + 1 = 21 s.
    assert result.arrival is not None
    assert result.arrival.tdb_seconds == pytest.approx(21.0)
    assert result.arrival.scale is TimeScale.TDB
    assert result.latency_s == pytest.approx(21.0)
    assert result.route is not None and result.route.store_and_forward is True
    # The typed wire field is populated alongside the kept primitive (RFC-0007).
    assert result.route.earliest_delivery is not None
    assert result.route.earliest_delivery.tdb_seconds == pytest.approx(21.0)
    assert result.route.earliest_delivery_tdb_s == pytest.approx(21.0)


def test_instantaneous_refuses_when_no_contemporaneous_path() -> None:
    model = DeliveryModel(_store_forward_graph())
    result = model.deliver("S", "G", send=_epoch(0.0), fidelity="instantaneous")
    assert result.delivered is False and result.arrival is None and result.hops == ()


def test_instantaneous_delivers_over_a_live_path() -> None:
    graph = ContactGraph.from_plan(
        _plan([_iv("S", "R", 0.0, 30.0), _iv("R", "G", 0.0, 30.0)], ["S", "R", "G"])
    )
    result = DeliveryModel(graph).deliver("S", "G", send=_epoch(5.0), fidelity="instantaneous")
    assert result.delivered is True and result.hops == ("S", "R", "G")
    assert result.latency_s == pytest.approx(2.0)  # two 1 s light-time hops, no waiting


def test_source_equals_dest_delivers_immediately() -> None:
    result = DeliveryModel(_store_forward_graph()).deliver("R", "R", send=_epoch(7.0))
    assert result.delivered and result.hops == ("R",) and result.latency_s == 0.0
    assert result.arrival is not None and result.arrival.tdb_seconds == 7.0


def test_non_delivery_when_dest_isolated() -> None:
    plan = _plan([_iv("S", "R", 0.0, 10.0)], ["S", "R", "G"])
    result = DeliveryModel(ContactGraph.from_plan(plan)).deliver("S", "G", send=_epoch(0.0))
    assert result.delivered is False


def test_horizon_bounds_store_and_forward_delivery() -> None:
    model = DeliveryModel(_store_forward_graph())
    # Delivery would arrive ~22 s; a 15 s horizon makes it a non-delivery.
    result = model.deliver("S", "G", send=_epoch(0.0), horizon_s=15.0)
    assert result.delivered is False


def test_bundle_too_large_for_contact_window_is_dropped() -> None:
    # Contact windows are 10 s at 1000 bps ⇒ 10_000 bits capacity; a bigger bundle cannot fit.
    model = DeliveryModel(_store_forward_graph())
    small = model.deliver("S", "G", send=_epoch(0.0), size_bits=5_000.0)
    big = model.deliver("S", "G", send=_epoch(0.0), size_bits=50_000.0)
    assert small.delivered is True and big.delivered is False


def test_turnaround_adds_per_hop_delay() -> None:
    graph = ContactGraph.from_plan(_plan([_iv("S", "G", 0.0, 30.0, latency=0.0)], ["S", "G"]))
    result = DeliveryModel(graph, turnaround_s=2.0).deliver("S", "G", send=_epoch(0.0))
    assert result.latency_s == pytest.approx(2.0)


def test_unknown_endpoint_raises() -> None:
    model = DeliveryModel(_store_forward_graph())
    with pytest.raises(LinkNetworkError, match="unknown source"):
        model.deliver("ghost", "G", send=_epoch(0.0))
    with pytest.raises(LinkNetworkError, match="unknown dest"):
        model.deliver("S", "ghost", send=_epoch(0.0))


def test_negative_size_raises() -> None:
    with pytest.raises(LinkNetworkError, match="non-negative"):
        DeliveryModel(_store_forward_graph()).deliver("S", "G", send=_epoch(0.0), size_bits=-1.0)
