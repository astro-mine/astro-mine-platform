"""The anchor baseline — a capability-aware mode policy (astro-mine-sim#61, G1.3).

Covers the mode derivation (from SADF capability tags + declared `loads_by_mode`), the honesty
constraint that keeps an ISRU plant out of an extraction mode, and the Core `Policy` conformance
Bench's scoring path requires.
"""

from __future__ import annotations

from typing import Any

import pytest

from astro_mine.core.policy import DecisionContext, Policy, check_policy
from astro_mine.core.sadf.model import Asset, Body, Inertia, Vec3
from astro_mine.sim.bench._policy import CapabilityModePolicy, mode_for_asset, mode_table
from astro_mine.sim.isru import DEFAULT_EXTRACTION_MODES

_BODY = Body(
    name="chassis",
    frame="body",
    mass_kg=100.0,
    center_of_mass_m=Vec3(x=0.0, y=0.0, z=0.0),
    inertia_kg_m2=Inertia(ixx=1.0, iyy=1.0, izz=1.0),
).model_dump(mode="json")


def _asset(
    asset_id: str,
    *,
    capabilities: list[str],
    modes: list[str],
    isru: dict[str, float] | None = None,
) -> Asset:
    payload: dict[str, Any] | None = None
    if isru is not None:
        payload = {"capacity_kg": 100.0, "isru": isru}
    return Asset.model_validate(
        {
            "identity": {"id": asset_id, "name": asset_id, "version": "0.1.0", "kind": "test"},
            "root_frame": "body",
            "bodies": [_BODY],
            "capabilities": capabilities,
            "power": {"loads_by_mode": [{"mode": m, "power_w": 10.0} for m in modes]},
            **({"payload": payload} if payload is not None else {}),
        }
    )


def test_a_prospector_prospects() -> None:
    asset = _asset(
        "rover", capabilities=["prospecting.neutron"], modes=["idle", "drive", "prospect"]
    )
    assert mode_for_asset(asset) == "prospect"


def test_an_excavator_excavates() -> None:
    asset = _asset("dig", capabilities=["excavation.bucket"], modes=["idle", "drive", "excavate"])
    assert mode_for_asset(asset) == "excavate"


def test_prospecting_outranks_excavation() -> None:
    """The anchor's rover declares both; its drill is an assay instrument, not its job."""
    asset = _asset(
        "rover",
        capabilities=["prospecting.neutron", "excavation.drill"],
        modes=["idle", "drive", "prospect", "drill"],
    )
    assert mode_for_asset(asset) == "prospect"


def test_a_relay_downlinks() -> None:
    asset = _asset("relay", capabilities=["comms.relay"], modes=["safe", "nominal", "downlink"])
    assert mode_for_asset(asset) == "downlink"


def test_a_hauler_drives_empty() -> None:
    asset = _asset(
        "hauler", capabilities=["mobility.wheeled"], modes=["idle", "drive_empty", "drive_loaded"]
    )
    assert mode_for_asset(asset) == "drive_empty"


def test_a_plant_is_now_commanded_into_the_extraction_mode_it_declares() -> None:
    """The inverse of what this test asserted before, and the point of #64.

    Under #61 a plant was deliberately held at `idle`: `IsruModel` was gated on the mode string
    alone, so commanding `extract` manufactured a `water_mass` that no digging and no haulage had
    earned, and the policy had to work around the physics.

    Extraction now consumes delivered feedstock, so the workaround is gone. Commanding the mode is
    both safe — an unfed plant still produces nothing — and *necessary*: an idle plant processes
    nothing however much regolith is hauled to it.
    """
    asset = _asset(
        "plant",
        capabilities=["isru.thermal_extraction", "isru.storage"],
        modes=["idle", "extract", "purify"],
        isru={"throughput_kg_hr": 10.0, "plant_power_w": 1500.0},
    )
    chosen = mode_for_asset(asset)
    assert chosen == "extract"
    assert chosen in DEFAULT_EXTRACTION_MODES


def test_an_asset_without_an_isru_model_may_use_an_extraction_mode() -> None:
    """The constraint is about *producing water*, not about the word: an excavator with no ISRU
    payload has no `IsruModel`, so `excavate` cannot manufacture anything."""
    asset = _asset("dig", capabilities=["excavation.bucket"], modes=["idle", "excavate"])
    assert mode_for_asset(asset) == "excavate"


def test_only_declared_modes_are_ever_commanded() -> None:
    """A lander declares comms but only `idle`/`deploy`; it must not be told to `downlink`."""
    asset = _asset(
        "lander",
        capabilities=["comms.direct_to_earth", "carrier.dispenser"],
        modes=["idle", "deploy"],
    )
    assert mode_for_asset(asset) in {"idle", "deploy"}


def test_an_asset_declaring_no_modes_falls_back_to_idle() -> None:
    asset = Asset.model_validate(
        {
            "identity": {"id": "bare", "name": "bare", "version": "0.1.0", "kind": "test"},
            "root_frame": "body",
            "bodies": [_BODY],
        }
    )
    assert mode_for_asset(asset) == "idle"


def test_the_policy_conforms_to_cores_protocol_and_is_order_stable(make_observation: Any) -> None:
    policy = CapabilityModePolicy(modes={"b": "excavate", "a": "prospect"})
    assert isinstance(policy, Policy)
    observations = {"b": make_observation("b"), "a": make_observation("a")}

    batch = check_policy(policy, observations, DecisionContext())

    assert [a.agent_id for a in batch.actions] == ["a", "b"]  # sorted, not iteration order
    assert [a.mode.mode for a in batch.actions if a.mode] == ["prospect", "excavate"]


def test_an_unpinned_agent_is_commanded_idle_rather_than_skipped(make_observation: Any) -> None:
    policy = CapabilityModePolicy(modes={"known": "prospect"})

    batch = policy.decide(
        {"known": make_observation("known"), "stranger": make_observation("stranger")},
        DecisionContext(),
    )

    assert len(batch.actions) == 2  # every observed agent gets a decision
    assert {a.agent_id: a.mode.mode for a in batch.actions if a.mode} == {
        "known": "prospect",
        "stranger": "idle",
    }


def test_the_policy_is_deterministic(make_observation: Any) -> None:
    policy = CapabilityModePolicy(modes={"a": "prospect"})
    observations = {"a": make_observation("a")}
    first = policy.decide(observations, DecisionContext())
    second = policy.decide(observations, DecisionContext())
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_mode_table_keys_on_the_sadf_identity() -> None:
    asset = _asset("rover", capabilities=["prospecting.nir"], modes=["idle", "prospect"])
    assert mode_table({"astro-mine.fleet.rover": asset}) == {"astro-mine.fleet.rover": "prospect"}


@pytest.fixture
def make_observation() -> Any:
    from astro_mine.core.messages.model import Observation, StateSample, Transform, Vec3
    from astro_mine.core.units import MOON_BODY_FIXED

    def _make(agent_id: str) -> Observation:
        return Observation(
            agent_id=agent_id,
            tick=0,
            sim_time_s=0.0,
            self_state=StateSample(
                agent_id=agent_id,
                frame=MOON_BODY_FIXED,
                pose=Transform(
                    translation_m=Vec3(x=0.0, y=0.0, z=0.0),
                    rotation_quat_xyzw={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                ),
            ),
        )

    return _make
