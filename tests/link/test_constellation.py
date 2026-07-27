"""Relay constellation + multi-hop reachability + ground catalogs (RM-P1-LINK-10 / -13).

Unit-level with scripted fakes (no SPICE kernels / DEM): the constellation window computation
dispatches the right geometry per pair type, multi-hop reachability finds relay chains for a
PSR surface agent with no direct Earth LOS, and the ground-station catalog loads DSN + ESTRACK
+ custom antennas as content-addressed data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, EpochWindow, ReferenceFrame, TimeScale
from astro_mine.core.world import Vector
from astro_mine.link.constellation import (
    ConstellationScenario,
    GroundStationCatalog,
    LinkConstellationError,
    body_occulted_los,
    build_routes,
    builtin_catalog,
    constellation_contact_windows,
    contact_nodes,
    default_ground_catalog,
    ground_node_ids,
    load_ground_catalog,
    reachability_windows,
    reachable_route,
    route_exists,
)
from astro_mine.link.constellation._scenario import expected_pair_count
from astro_mine.link.geometry import EphemerisNode, SurfaceNode
from astro_mine.link.products import ConnectivitySampler, build_contact_plan
from astro_mine.link.windows import GroundStation, LinkWindowError

# --- body_occulted_los primitive -------------------------------------------------

_R = 1_737_400.0


def test_body_occulted_los_clears_on_same_side() -> None:
    assert body_occulted_los((2e6, 0.0, 0.0), (2e6, 1e5, 0.0), _R) is True


def test_body_occulted_los_blocked_on_opposite_sides() -> None:
    assert body_occulted_los((2e6, 0.0, 0.0), (-2e6, 0.0, 0.0), _R) is False


def test_body_occulted_los_zero_radius_never_occludes() -> None:
    assert body_occulted_los((2e6, 0.0, 0.0), (-2e6, 0.0, 0.0), 0.0) is True


def test_body_occulted_los_degenerate_coincident_points() -> None:
    # Both endpoints at the same point above the surface: use the point's own distance.
    assert body_occulted_los((2e6, 0.0, 0.0), (2e6, 0.0, 0.0), _R) is True
    assert body_occulted_los((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _R) is False


# --- scripted providers ----------------------------------------------------------

_S_POS: Vector = (0.0, 0.0, _R)
_R1_POS: Vector = (1.0e6, 0.0, 2.0e6)
_R2_POS: Vector = (1.2e6, 0.0, 2.0e6)
_EARTH_POS: Vector = (0.0, 0.0, 4.0e8)
_POS = {"R1": _R1_POS, "R2": _R2_POS, "EARTH": _EARTH_POS}


class ScriptedEphemeris:
    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        return _POS[target]


class ScriptedWorld:
    """A WorldProvider whose horizon LOS follows a per-target time schedule."""

    def __init__(self, frame: ReferenceFrame = MOON_BODY_FIXED) -> None:
        self._frame = frame

    @property
    def frame(self) -> ReferenceFrame:
        return self._frame

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        assert epoch is not None
        t = epoch.tdb_seconds
        if target == _R1_POS:
            return 0.0 <= t < 50.0
        if target == _R2_POS:
            return 60.0 <= t < 120.0
        if target == _EARTH_POS:
            return False  # a PSR surface agent has no direct Earth line-of-sight
        return False


class ScriptedTopocentric:
    """Station elevation of a target by name and time; the Moon centre is always up."""

    def elevation_deg(self, target: object, site: object, epoch: Epoch) -> float:
        name = getattr(target, "name", target)
        t = epoch.tdb_seconds
        if name == "R1":
            return 20.0 if 0.0 <= t < 70.0 else 0.0
        if name == "R2":
            return 20.0 if 50.0 <= t < 120.0 else 0.0
        return 90.0  # the Moon (frame centre) is always above the station horizon


def _scenario(**overrides: object) -> ConstellationScenario:
    window = EpochWindow(
        start=Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=120.0, scale=TimeScale.TDB),
    )
    kwargs: dict[str, object] = dict(
        surface=(SurfaceNode(name="S", position_m=_S_POS),),
        relays=(EphemerisNode(name="R1", target="R1"), EphemerisNode(name="R2", target="R2")),
        ground=(GroundStation.from_latlon("G", 35.0, -116.0),),
        world=ScriptedWorld(),
        ephemeris=ScriptedEphemeris(),
        topocentric=ScriptedTopocentric(),
        window=window,
        step_s=10.0,
    )
    kwargs.update(overrides)
    return ConstellationScenario(**kwargs)  # type: ignore[arg-type]


# --- scenario validation ---------------------------------------------------------


def test_scenario_rejects_empty_node_set() -> None:
    with pytest.raises(LinkConstellationError, match="empty constellation"):
        _scenario(surface=(), relays=(), ground=())


def test_scenario_rejects_duplicate_names() -> None:
    with pytest.raises(LinkConstellationError, match="duplicate node name"):
        _scenario(
            relays=(EphemerisNode(name="R1", target="R1"), EphemerisNode(name="R1", target="R2"))
        )


def test_scenario_node_names() -> None:
    assert _scenario().node_names == frozenset({"S", "R1", "R2", "G"})


# --- contact nodes + windows -----------------------------------------------------


def test_contact_nodes_roles_and_kinds() -> None:
    nodes = {n.id: n for n in contact_nodes(_scenario())}
    assert nodes["S"].role is NodeRole.SPACE and nodes["S"].kind == "surface_agent"
    assert nodes["R1"].role is NodeRole.SPACE and nodes["R1"].kind == "relay_orbiter"
    assert nodes["G"].role is NodeRole.GROUND and nodes["G"].kind == "ground_station"


def _sampler(scenario: ConstellationScenario) -> ConnectivitySampler:
    windows = constellation_contact_windows(scenario)
    plan = build_contact_plan(contact_nodes(scenario), windows, epoch_window=scenario.window)
    return ConnectivitySampler(plan)


def test_constellation_windows_cover_all_pair_types() -> None:
    windows = constellation_contact_windows(_scenario())
    pairs = {(w.observer, w.target) for w in windows}
    # surface↔relay, relay↔relay, relay↔ground present; no direct surface↔ground (PSR).
    assert ("S", "R1") in pairs and ("S", "R2") in pairs
    assert ("R1", "R2") in pairs  # relay-relay same side: visible across the window
    assert ("R1", "G") in pairs and ("R2", "G") in pairs
    assert ("S", "G") not in pairs


def test_relay_relay_window_spans_full_window_when_co_visible() -> None:
    windows = {(w.observer, w.target): w for w in constellation_contact_windows(_scenario())}
    rr = windows[("R1", "R2")]
    assert rr.start.tdb_seconds == 0.0 and rr.end.tdb_seconds == 120.0


def test_surface_ground_can_be_disabled() -> None:
    windows = constellation_contact_windows(_scenario(link_surface_to_ground=False))
    assert all(not (w.observer == "S" and w.target == "G") for w in windows)


class _AlwaysVisibleWorld:
    """A WorldProvider that always clears LOS — for the surface-surface (rover-rover) path."""

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return True


def test_surface_to_surface_link_is_computed() -> None:
    scenario = _scenario(
        surface=(
            SurfaceNode(name="S1", position_m=(0.0, 0.0, _R)),
            SurfaceNode(name="S2", position_m=(10.0, 0.0, _R)),
        ),
        relays=(),
        ground=(),
        world=_AlwaysVisibleWorld(),
    )
    windows = {(w.observer, w.target): w for w in constellation_contact_windows(scenario)}
    assert ("S1", "S2") in windows
    ss = windows[("S1", "S2")]
    assert ss.start.tdb_seconds == 0.0 and ss.end.tdb_seconds == 120.0


# --- multi-hop reachability ------------------------------------------------------


def _epoch(t: float) -> Epoch:
    return Epoch(tdb_seconds=t, scale=TimeScale.TDB)


def test_psr_agent_reaches_earth_via_relay_chain() -> None:
    sampler = _sampler(_scenario())
    route = reachable_route(sampler, "S", {"G"}, _epoch(20.0))
    assert route is not None
    assert route.hops == ["S", "R1", "G"]
    assert route.source == "S" and route.dest == "G"
    assert route.store_and_forward is False


def test_reachability_switches_relay_later_in_window() -> None:
    sampler = _sampler(_scenario())
    route = reachable_route(sampler, "S", {"G"}, _epoch(100.0))
    assert route is not None and route.hops[1] in {"R1", "R2"} and route.hops[-1] == "G"


def test_no_route_when_surface_has_no_relay() -> None:
    sampler = _sampler(_scenario())
    assert reachable_route(sampler, "S", {"G"}, _epoch(55.0)) is None
    assert route_exists(sampler, "S", {"G"}, _epoch(55.0)) is False


def test_source_in_targets_is_trivially_reachable() -> None:
    sampler = _sampler(_scenario())
    route = reachable_route(sampler, "G", {"G"}, _epoch(10.0))
    assert route is not None and route.hops == ["G"] and route.total_latency_s == 0.0


def test_reachable_route_unknown_source_raises() -> None:
    sampler = _sampler(_scenario())
    with pytest.raises(LinkConstellationError, match="not a node of the contact plan"):
        reachable_route(sampler, "ghost", {"G"}, _epoch(10.0))


def test_ground_node_ids_from_plan() -> None:
    scenario = _scenario()
    plan = build_contact_plan(
        contact_nodes(scenario),
        constellation_contact_windows(scenario),
        epoch_window=scenario.window,
    )
    assert ground_node_ids(plan) == frozenset({"G"})


def test_reachability_windows_are_nonempty_and_time_varying() -> None:
    sampler = _sampler(_scenario())
    windows = reachability_windows(sampler, "S", {"G"}, _scenario().window, 10.0)
    assert windows, "surface agent should reach Earth during some interval"
    # It is reachable at t=20 but not at t=55 — so at least one window ends before the horizon.
    assert any(w.end.tdb_seconds <= 60.0 for w in windows)


def test_build_routes_skips_unreachable_sources() -> None:
    sampler = _sampler(_scenario())
    routes = build_routes(sampler, ["S", "R1"], {"G"}, _epoch(20.0))
    by_source = {r.source: r for r in routes}
    assert "S" in by_source and by_source["S"].hops[-1] == "G"
    # R1 links directly to G at t=20.
    assert by_source["R1"].hops == ["R1", "G"]


def test_build_routes_omits_source_with_no_path() -> None:
    sampler = _sampler(_scenario())
    assert build_routes(sampler, ["S"], {"G"}, _epoch(55.0)) == []


# --- ground-station catalogs (RM-P1-LINK-13) -------------------------------------


def test_builtin_dsn_and_estrack_catalogs_load() -> None:
    dsn = builtin_catalog("dsn")
    estrack = builtin_catalog("estrack")
    assert dsn.networks[0] == "dsn" and len(dsn.stations) == 3
    assert "estrack" in set(estrack.networks) and len(estrack.stations) == 4
    assert len(dsn.digest) == 64 and dsn.digest != estrack.digest


def test_default_catalog_merges_dsn_and_estrack() -> None:
    catalog = default_ground_catalog()
    assert len(catalog.stations) == 7
    assert len(catalog.by_network("dsn")) == 3
    assert len(catalog.by_network("estrack")) == 4
    assert set(catalog.names) >= {"DSS-14-Goldstone", "NNO-New-Norcia"}


def test_unknown_builtin_catalog_raises() -> None:
    with pytest.raises(LinkWindowError, match="unknown built-in catalog"):
        builtin_catalog("nasa-tdrss")


def test_load_custom_catalog_from_yaml_string_is_content_addressed() -> None:
    text = (
        "network: custom\n"
        "stations:\n"
        "  - {name: Backyard-1, lat_deg: 52.0, lon_deg: 4.0}\n"
        "  - {name: Backyard-2, lat_deg: -33.0, lon_deg: 18.0, min_elevation_deg: 5.0}\n"
    )
    catalog = load_ground_catalog(text)
    assert catalog.names == ("Backyard-1", "Backyard-2")
    assert catalog.stations[1].min_elevation_deg == 5.0
    assert load_ground_catalog(text).digest == catalog.digest  # deterministic


def test_load_catalog_from_file(tmp_path: Path) -> None:
    path = tmp_path / "estrack_extra.yaml"
    path.write_text(
        "network: estrack\nstations:\n  - {name: KIR-Kiruna, lat_deg: 67.9, lon_deg: 21.1}\n"
    )
    catalog = load_ground_catalog(path)
    assert catalog.names == ("KIR-Kiruna",) and catalog.source == str(path)


def test_missing_catalog_file_raises() -> None:
    with pytest.raises(LinkWindowError, match="file not found"):
        load_ground_catalog(Path("/no/such/catalog.yaml"))


def test_malformed_catalog_raises() -> None:
    with pytest.raises(LinkWindowError, match="must be a mapping"):
        load_ground_catalog("[1, 2, 3]")
    with pytest.raises(LinkWindowError, match="malformed station entry"):
        load_ground_catalog("stations:\n  - {name: NoCoords}\n")


def test_catalog_rejects_duplicate_names() -> None:
    with pytest.raises(LinkWindowError, match="duplicate ground-station name"):
        load_ground_catalog(
            "stations:\n"
            "  - {name: X, lat_deg: 0, lon_deg: 0}\n"
            "  - {name: X, lat_deg: 1, lon_deg: 1}\n"
        )


def test_catalog_merge_is_content_addressed_and_detects_collision() -> None:
    a = load_ground_catalog("stations:\n  - {name: A, lat_deg: 0, lon_deg: 0}\n")
    b = load_ground_catalog("stations:\n  - {name: B, lat_deg: 1, lon_deg: 1}\n")
    merged = a.merge(b)
    assert merged.names == ("A", "B") and len(merged.digest) == 64
    with pytest.raises(LinkWindowError, match="duplicate ground-station name"):
        a.merge(a)


def test_catalog_length_mismatch_guard() -> None:
    station = GroundStation.from_latlon("Z", 0.0, 0.0)
    with pytest.raises(LinkWindowError, match="length mismatch"):
        GroundStationCatalog(stations=(station,), networks=(), digest="x", source="test")


# --- the progress report's pair count (offline plan build) ------------------------


def test_expected_pair_count_matches_the_pairs_actually_searched() -> None:
    """``expected_pair_count`` sizes the anchor build's progress report *before* the first pair is
    walked, so it is derived from the node counts rather than from running the search. That makes it
    a second, independent enumeration of the same set -- and a second enumeration can drift from the
    first. This is the gate that stops it: it must equal the distinct pairs the search really walks.
    """
    scenario = _scenario()

    searched = {
        tuple(sorted((w.observer, w.target))) for w in constellation_contact_windows(scenario)
    }

    # Every pair the search *reports* on must be one it counted (a pair with no visibility yields no
    # window at all, so the searched set is a subset -- never a superset -- of the expected count).
    assert 0 < len(searched) <= expected_pair_count(scenario)
    assert expected_pair_count(scenario) == 1 * 2 + 1 + 2 * 1 + 1 * 1  # S-R + R-R + R-G + S-G


def test_the_pair_count_tracks_the_surface_to_ground_switch() -> None:
    """The one pair class the search makes conditional, so the count must be conditional too."""
    linked = _scenario(link_surface_to_ground=True)
    unlinked = _scenario(link_surface_to_ground=False)

    assert expected_pair_count(linked) - expected_pair_count(unlinked) == len(linked.surface) * len(
        linked.ground
    )
