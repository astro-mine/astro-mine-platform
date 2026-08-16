"""RM-P0-LINK-04 — the anchor scenario's comms model (lunar polar relay + DSN).

The anchor is the scenario that makes the benchmark comms-denied *for real* (``LUNAR-TR-003``;
link.md §12 "P0 exit"): a rover on the Shackleton PSR floor loses the relay behind the crater rim
and has no direct-to-Earth path, so it reaches Earth only through a relay pass.

These tests pin the *scenario* — the node set, the radios, the DSN complexes, the pinned config, and
the reduction into a Core ContactPlan — with **scripted providers** (no SPICE kernels, no DEM), so
what they assert is Link's contract, not the geometry backend's. Two properties matter:

- the anchor plan is a well-formed Core ContactPlan over the declared graph, budget-annotated,
  and it reopens as a ``ConnectivitySampler`` whose masks deny a PSR agent's Earth contact;
- it is **deterministic** — the same scenario yields the identical ``plan_digest`` and the identical
  published artifact digest, which is the property a Bench anchor pin rests on.

The full-fidelity plan (real DE440 kernels + the published Shackleton DEM) is built by
``scripts/build_anchor_contact_plan.py``; that is a maintainer path, not a CI gate, because the
CI carries neither a 120 MB kernel set nor a Worlds install — terrain reaches Link only via the
injected Core ``WorldProvider`` (link.md §2.2; conventions.md §1.1).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from astro_mine import spice
from astro_mine.core.messages import ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, EpochWindow, ReferenceFrame, TimeScale
from astro_mine.core.world import Vector
from astro_mine.hub.registry import Registry, open_registry
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.link.anchor import (
    ANCHOR_ARTIFACT_NAME,
    ANCHOR_ARTIFACT_VERSION,
    ANCHOR_EPOCH_WINDOW,
    ANCHOR_RELAY,
    ANCHOR_SCENARIO_ID,
    ANCHOR_SURFACE_SITES,
    anchor_budgets,
    anchor_config,
    anchor_ground_stations,
    anchor_node_ids,
    anchor_radios,
    anchor_scenario,
    build_anchor_contact_plan,
)
from astro_mine.link.cache import build_cache_key, plan_digest
from astro_mine.link.constellation import ConstellationScenario
from astro_mine.link.products import ConnectivitySampler
from astro_mine.link.registry import publish_contact_plan

# The contact-graph node ids are the Fleet SADF asset ids (astro-mine-sim#53): a Sim agent and a
# contact-plan node name the same robot, so Sim's exact-match binding masks the right observations.
_PSR = "prospecting-rover"
_RIDGE = "isru-plant"
_RELAY_PERIOD_S = 7200.0

# A 2 h slice of the anchor window keeps the scripted-provider sweep quick while still spanning a
# full relay rise/set — the structure the assertions need.
_WINDOW = EpochWindow(
    start=ANCHOR_EPOCH_WINDOW.start,
    end=Epoch(
        tdb_seconds=ANCHOR_EPOCH_WINDOW.start.tdb_seconds + _RELAY_PERIOD_S, scale=TimeScale.TDB
    ),
)


class ScriptedEphemeris:
    """The relay on a scripted polar arc; Earth fixed high over the near side.

    Stands in for a SPICE-resolved SPK: Link consumes ephemerides through the injected
    ``EphemerisProvider`` seam (link.md §2.2), so the scenario is testable without kernels.
    """

    def position_body_fixed(self, target: str, epoch: Epoch, *, frame: ReferenceFrame) -> Vector:
        if target == "EARTH":
            return (0.0, 0.0, 3.844e8)  # over the near side — on the horizon as seen from the pole
        t = epoch.tdb_seconds - ANCHOR_EPOCH_WINDOW.start.tdb_seconds
        phase = 2.0 * math.pi * t / _RELAY_PERIOD_S
        radius = spice.MOON_RADIUS_M + 1.0e6
        # A polar arc: overhead at the south pole at t = period/2, on the far side at t = 0.
        return (radius * math.sin(phase), 0.0, -radius * math.cos(phase))


class ScriptedWorld:
    """A crater-rim stand-in: terrain blocks any line of sight below a local elevation mask.

    The Core ``WorldProvider`` seam Link consumes (never an ``astro-mine-worlds`` import). A PSR
    floor node therefore sees the relay only around its high pass, and never sees Earth (which sits
    on the horizon from the pole) — the denial geometry the anchor exists to exercise.
    """

    def __init__(self, min_elevation_deg: float = 30.0) -> None:
        self._min = min_elevation_deg

    @property
    def frame(self) -> ReferenceFrame:
        return MOON_BODY_FIXED

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        return _elevation_deg(observer, target) >= self._min


class ScriptedTopocentric:
    """Earth stations that always see the Moon (elevation above every DSN mask)."""

    def elevation_deg(self, target: object, site: object, epoch: Epoch) -> float:
        return 45.0


def _elevation_deg(observer: Vector, target: Vector) -> float:
    """Elevation of ``target`` above ``observer``'s local horizontal (a spherical-body horizon)."""
    ox, oy, oz = observer
    up = math.sqrt(ox * ox + oy * oy + oz * oz)
    dx, dy, dz = target[0] - ox, target[1] - oy, target[2] - oz
    slant = math.sqrt(dx * dx + dy * dy + dz * dz)
    if up == 0.0 or slant == 0.0:
        return -90.0
    cos_zenith = (ox * dx + oy * dy + oz * dz) / (up * slant)
    return math.degrees(math.asin(max(-1.0, min(1.0, cos_zenith))))


def _scenario() -> ConstellationScenario:
    return anchor_scenario(
        world=ScriptedWorld(),  # type: ignore[arg-type]
        ephemeris=ScriptedEphemeris(),
        topocentric=ScriptedTopocentric(),  # type: ignore[arg-type]
        window=_WINDOW,
    )


@pytest.fixture(scope="module")
def anchor_plan() -> ContactPlan:
    return build_anchor_contact_plan(_scenario())


def _grid(n: int, step_s: float) -> list[Epoch]:
    start = _WINDOW.start.tdb_seconds
    return [Epoch(tdb_seconds=start + i * step_s, scale=TimeScale.TDB) for i in range(n)]


# --- the pinned scenario ----------------------------------------------------------------------


def test_anchor_declares_the_relay_and_the_three_dsn_complexes() -> None:
    stations = anchor_ground_stations()
    assert [s.name for s in stations] == ["DSS-14-Goldstone", "DSS-63-Madrid", "DSS-43-Canberra"]
    assert ANCHOR_RELAY.name == "relay-orbiter"
    assert set(anchor_node_ids()) == {site.name for site in ANCHOR_SURFACE_SITES} | {
        ANCHOR_RELAY.name,
        *(s.name for s in stations),
    }
    # Two of the four surface agents sit on a permanently shadowed floor — the denial the anchor is
    # built to exercise (scenarios/1 §3).
    assert sum(site.psr for site in ANCHOR_SURFACE_SITES) == 2


def test_anchor_radios_cover_every_node_and_yield_feasible_budgets() -> None:
    assert set(anchor_radios()) == set(anchor_node_ids())
    budgets = anchor_budgets(anchor_node_ids())
    surface_to_relay = budgets[(_PSR, ANCHOR_RELAY.name)]
    assert surface_to_relay.feasible
    assert surface_to_relay.rate_bps > 0.0
    # Light-time at the relay's slant range is milliseconds, not the Earth leg's ~1.3 s.
    assert 0.0 < surface_to_relay.latency_s < 0.1
    assert budgets[(ANCHOR_RELAY.name, "DSS-14-Goldstone")].latency_s > 1.0


def test_anchor_config_pins_the_search_grid() -> None:
    config = anchor_config()
    assert config["step_s"] == 60.0
    assert config["refine_s"] == 5.0
    assert config["link_surface_to_ground"] is True


def test_anchor_window_spans_the_pinned_mission() -> None:
    # The plan's window must cover the whole episode it is pinned into: Sim masks any epoch past the
    # plan as no-contact, so a window shorter than the episode silently reads un-modelled time as
    # denied and dilutes ``comms_robustness`` (astro-mine-link#34). The Bench anchor episode is
    # ``43_200 x 60 s = 2_592_000 s`` (30 days) from the same start epoch, so the window must too.
    mission_span_s = 43_200 * 60.0
    assert ANCHOR_EPOCH_WINDOW.start.tdb_seconds == 946_728_000.0
    span_s = ANCHOR_EPOCH_WINDOW.end.tdb_seconds - ANCHOR_EPOCH_WINDOW.start.tdb_seconds
    assert span_s == mission_span_s


# --- the plan ----------------------------------------------------------------------------------


def test_anchor_plan_is_a_well_formed_core_contact_plan(anchor_plan: ContactPlan) -> None:
    assert {node.id for node in anchor_plan.nodes} == set(anchor_node_ids())
    ground = {node.id for node in anchor_plan.nodes if node.role == NodeRole.GROUND}
    assert ground == {s.name for s in anchor_ground_stations()}
    assert anchor_plan.intervals
    # Every interval is budget-annotated (RM-P0-LINK-03 feeding RM-P0-LINK-04).
    assert all(iv.max_rate_bps is not None for iv in anchor_plan.intervals)
    assert anchor_plan.window == _WINDOW


def test_psr_agent_is_earth_denied_and_relay_intermittent(anchor_plan: ContactPlan) -> None:
    sampler = ConnectivitySampler(anchor_plan)
    masks = [sampler.comms_mask(_PSR, epoch) for epoch in _grid(60, 120.0)]

    # No direct Earth link from the PSR floor, ever — the property RM-P0-LINK-04 makes real.
    assert not any(mask.earth_contact for mask in masks)
    # The relay rises and sets: the PSR agent's only path to Earth is intermittent.
    relay_states = {
        peer.reachable for mask in masks for peer in mask.links if peer.peer == ANCHOR_RELAY.name
    }
    assert relay_states == {True, False}


def test_every_node_gets_a_mask(anchor_plan: ContactPlan) -> None:
    sampler = ConnectivitySampler(anchor_plan)
    epoch = _grid(1, 0.0)[0]
    masks = sampler.comms_masks(epoch)
    assert set(masks) == set(anchor_node_ids())
    assert _RIDGE in masks


# --- determinism: what a Bench anchor pin rests on --------------------------------------------


def test_anchor_plan_digest_is_deterministic() -> None:
    assert plan_digest(build_anchor_contact_plan(_scenario())) == plan_digest(
        build_anchor_contact_plan(_scenario())
    )


def test_anchor_publish_is_content_addressed(tmp_path: Path, anchor_plan: ContactPlan) -> None:
    key = build_cache_key(nodes=list(anchor_node_ids()), epoch=_WINDOW, config=anchor_config())
    private_pem, _ = generate_keypair()
    published = [
        publish_contact_plan(
            anchor_plan,
            registry=open_registry(str(tmp_path / name)),
            name=ANCHOR_ARTIFACT_NAME,
            version=ANCHOR_ARTIFACT_VERSION,
            scenario_id=ANCHOR_SCENARIO_ID,
            input_hashes={"nodes": key.nodes, "epoch": key.epoch, "config": key.config},
            private_key_pem=private_pem,
        )
        for name in ("a", "b")
    ]
    # Two clean publishes of the same anchor plan resolve the identical digest, so a Bench
    # ScenarioSpec can pin the comms model by content hash (bench#28).
    assert published[0].digest == published[1].digest
    assert Registry(tmp_path / "a").references() == [
        f"{ANCHOR_ARTIFACT_NAME}:{ANCHOR_ARTIFACT_VERSION}"
    ]
