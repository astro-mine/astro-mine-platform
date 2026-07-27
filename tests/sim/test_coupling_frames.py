"""RM-P0-SIM-03/04 + RFC-0002 — the SPICE-backed cross-frame coupling boundary.

The gap this closes: ``FrameBridge`` used to raise ``FrameBridgeError`` unconditionally on any
inertial↔body-fixed handoff, so an orbital+surface co-simulation — the anchor scenario's relay
orbiter over its surface swarm — was not expressible at all. These tests prove the rotation now
resolves through ``astro-mine-spice`` (the shared SPICE foundation, never a Sim-local
re-derivation),
that an orbital↔surface coupled scenario runs end-to-end without a ``FrameBridgeError``, and that a
kernel-less pool still fails **loudly** rather than degrading into an identity rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.core.messages.model import Quat, StateSample, Transform, Vec3
from astro_mine.core.units import INERTIAL_J2000, MOON_BODY_FIXED, Epoch, TimeScale
from astro_mine.sim.coupling import (
    CoupledEngine,
    CouplingBoundary,
    FrameBridge,
    FrameBridgeError,
    SpiceFrameBridge,
)
from astro_mine.sim.engines import (
    MobilityEngine,
    RegimeEngine,
    mobility_engine_factory,
    orbital_engine_factory,
)
from astro_mine.sim.runtime import (
    AgentSpec,
    MobilityDynamics,
    OrbitalDynamics,
    RngStreams,
    Scenario,
    run_episode,
)
from astro_mine.spice import frame_transform
from tests.sim.conftest import SPICE_EPOCH

#: A low lunar orbit radius (m) and the circular speed there — the relay's initial state.
_ORBIT_R_M = 1_837_400.0
_MOON_MU = 4.902800118e12
_ORBIT_V_MPS = math.sqrt(_MOON_MU / _ORBIT_R_M)


def _sample(
    frame: object = INERTIAL_J2000,
    *,
    position: tuple[float, float, float] = (_ORBIT_R_M, 0.0, 0.0),
    velocity: tuple[float, float, float] | None = (0.0, _ORBIT_V_MPS, 0.0),
) -> StateSample:
    return StateSample(
        agent_id="relay",
        frame=frame,  # type: ignore[arg-type]
        pose=Transform(
            translation_m=Vec3(x=position[0], y=position[1], z=position[2]),
            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        linear_velocity_mps=None
        if velocity is None
        else Vec3(x=velocity[0], y=velocity[1], z=velocity[2]),
        battery_soc_j=100.0,
        mode="relay",
    )


# --- the bridge itself -----------------------------------------------------------


def test_the_default_bridge_is_the_shared_spice_realization() -> None:
    # RFC-0002: Sim resolves frame geometry through astro-mine-spice, never a Sim-local
    # re-derivation. The coupler's default bridge *is* that realization.
    assert isinstance(SpiceFrameBridge(), FrameBridge)
    assert isinstance(CoupledEngine({"a": _StubEngine()})._frame_bridge, SpiceFrameBridge)


def test_bridge_rotates_pose_and_velocity_into_the_body_fixed_frame(
    spice_bridge: SpiceFrameBridge,
) -> None:
    inertial = _sample()
    bridged = spice_bridge.bridge(inertial, MOON_BODY_FIXED, SPICE_EPOCH)

    assert bridged.frame == MOON_BODY_FIXED  # the sample now *says* which frame it is in
    # The rotation is exactly the one the shared SPICE foundation resolves — not an approximation.
    rotation = frame_transform(INERTIAL_J2000, MOON_BODY_FIXED, SPICE_EPOCH)
    expected_p = rotation @ np.array([_ORBIT_R_M, 0.0, 0.0])
    expected_v = rotation @ np.array([0.0, _ORBIT_V_MPS, 0.0])
    got_p = bridged.pose.translation_m
    got_v = bridged.linear_velocity_mps
    assert got_v is not None
    assert (got_p.x, got_p.y, got_p.z) == pytest.approx(tuple(expected_p), rel=1e-12)
    assert (got_v.x, got_v.y, got_v.z) == pytest.approx(tuple(expected_v), rel=1e-12)
    # A rotation is rigid: it moves the vector but never changes its magnitude (a real transform,
    # not a rescale — the orbiter is still in orbit after the handoff).
    assert math.dist((got_p.x, got_p.y, got_p.z), (0.0, 0.0, 0.0)) == pytest.approx(_ORBIT_R_M)
    # And the non-kinematic state rides through untouched.
    assert bridged.battery_soc_j == 100.0 and bridged.mode == "relay"


def test_bridge_is_identity_when_the_frames_already_match(
    spice_bridge: SpiceFrameBridge,
) -> None:
    # The common same-body Phase-0 case must stay a no-op (and must not need SPICE at all).
    surface = _sample(MOON_BODY_FIXED)
    assert spice_bridge.bridge(surface, MOON_BODY_FIXED, SPICE_EPOCH) is surface


def test_bridge_carries_the_attitude_into_the_target_frame(
    spice_bridge: SpiceFrameBridge,
) -> None:
    # The pose quaternion maps body -> source frame; after bridging it must map body -> target,
    # so the attitude travels with the pose rather than being silently reused in the wrong frame.
    bridged = spice_bridge.bridge(_sample(), MOON_BODY_FIXED, SPICE_EPOCH)
    q = bridged.pose.rotation_quat_xyzw
    assert math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2) == pytest.approx(1.0)  # still a unit quat
    assert (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 1.0)  # and genuinely rotated, not passed through


def test_bridge_is_time_dependent_a_rotating_frame_moves() -> None:
    # The whole point of a *rotating* frame: the same inertial state maps to different body-fixed
    # coordinates at different epochs. A bridge that ignored the epoch would be silently wrong.
    bridge = SpiceFrameBridge()
    later = Epoch(tdb_seconds=SPICE_EPOCH.tdb_seconds + 6.0 * 3600.0, scale=TimeScale.TDB)
    a = bridge.bridge(_sample(), MOON_BODY_FIXED, SPICE_EPOCH).pose.translation_m
    b = bridge.bridge(_sample(), MOON_BODY_FIXED, later).pose.translation_m
    assert (a.x, a.y, a.z) != pytest.approx((b.x, b.y, b.z))


test_bridge_is_time_dependent_a_rotating_frame_moves = pytest.mark.usefixtures("spice_kernels")(
    test_bridge_is_time_dependent_a_rotating_frame_moves
)


def test_an_unfurnished_kernel_pool_fails_loudly_it_never_assumes_an_identity() -> None:
    # Degrade, don't lie (spice.md §2.5): with no orientation kernel furnished, SPICE cannot know
    # where MOON_ME points, so the bridge must raise — never quietly hand back an unrotated state.
    with pytest.raises(FrameBridgeError, match=r"J2000.*MOON_ME"):
        SpiceFrameBridge().bridge(_sample(), MOON_BODY_FIXED, SPICE_EPOCH)


# --- the orbital <-> surface coupled scenario ------------------------------------
# The canonical cross-frame boundary: **one asset modelled in two regimes** — a lander propagating
# in the inertial frame whose state is handed down to the body-fixed surface engine (the RFC-0001
# descent handoff, and the pairing that used to be inexpressible). It mirrors the repo's existing
# shared-agent idiom (the excavator that lives in both the mobility and granular sub-engines).


def _lander_orbital(agent_id: str) -> Scenario:
    """The lander as the orbital engine sees it: inertial frame, two-body dynamics."""
    return Scenario(
        name="descent-orbital",
        seed=5,
        dt_s=60.0,
        horizon_steps=4,
        start_epoch=SPICE_EPOCH,
        agents=(
            AgentSpec(
                agent_id=agent_id,
                frame=INERTIAL_J2000,
                initial_position_m=(_ORBIT_R_M, 0.0, 0.0),
                velocity_mps=(0.0, _ORBIT_V_MPS, 0.0),
                battery_soc_j=1.0e6,
                dynamics=OrbitalDynamics(mu_m3_s2=_MOON_MU),
            ),
        ),
    )


def _surface_swarm(agent_id: str) -> Scenario:
    """The same lander as the surface engine sees it — body-fixed frame — plus a rover."""
    mobility = MobilityDynamics(mass_kg=250.0, max_speed_mps=0.5, max_traction_n=200.0)
    return Scenario(
        name="descent-surface",
        seed=5,
        dt_s=60.0,
        horizon_steps=4,
        start_epoch=SPICE_EPOCH,
        agents=(
            AgentSpec(
                agent_id=agent_id,
                frame=MOON_BODY_FIXED,
                initial_position_m=(0.0, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=mobility,
            ),
            AgentSpec(
                agent_id="rover",
                frame=MOON_BODY_FIXED,
                initial_position_m=(10.0, 0.0, 0.0),
                battery_soc_j=1.0e6,
                dynamics=mobility,
            ),
        ),
    )


def _descent_engine(
    frame_bridge: FrameBridge | None = None,
) -> tuple[CoupledEngine, MobilityEngine]:
    """An orbital sub-engine (inertial) coupled to a surface sub-engine (body-fixed), exchanging the
    lander's pose across the frame boundary after each macro step.

    Returns the coupler **and the surface sub-engine**, so a test can assert on the state the
    consumer actually received — the coupler's own export deduplicates a shared agent to the first
    sub-engine (the orbital one), which would hide whether the handoff landed."""
    rng = RngStreams(5)
    surface = mobility_engine_factory(_surface_swarm("lander"), rng)
    coupled = CoupledEngine(
        {
            "orbital": orbital_engine_factory(_lander_orbital("lander"), rng),
            "surface": surface,
        },
        boundaries=(CouplingBoundary("descent", "orbital", "surface", ("lander",)),),
        frame_bridge=frame_bridge,
        start_epoch=SPICE_EPOCH,
    )
    return coupled, surface


def test_an_orbital_surface_coupled_scenario_runs_without_a_frame_bridge_error(
    spice_kernels: object,
) -> None:
    # THE acceptance criterion: an inertial-frame producer handing state to a body-fixed consumer.
    # Before RFC-0002 adoption this raised FrameBridgeError unconditionally, so orbital+surface
    # co-simulation was impossible.
    engine, surface = _descent_engine()

    engine.advance(60.0)  # would have raised FrameBridgeError before

    # The boundary really exchanged, into the *consumer's* frame: the surface engine's lander now
    # sits at the orbital lander's body-fixed projection — a full orbit radius from where it started
    # — and that projection is exactly what SPICE resolves. Real, correctly-rotated state crossed.
    landed = surface.export_coupling_state().by_agent["lander"]
    assert landed.frame == MOON_BODY_FIXED
    p = landed.pose.translation_m
    assert math.dist((p.x, p.y, p.z), (0.0, 0.0, 0.0)) == pytest.approx(_ORBIT_R_M, rel=1e-6)

    rotation = frame_transform(INERTIAL_J2000, MOON_BODY_FIXED, engine.epoch)
    orbital = engine.export_coupling_state().by_agent["lander"].pose.translation_m
    expected = rotation @ np.array([orbital.x, orbital.y, orbital.z])
    assert (p.x, p.y, p.z) == pytest.approx(tuple(expected), rel=1e-9)

    # and the residual records the discontinuity the boundary corrected.
    residual = next(r for r in engine.residuals if r.agent_id == "lander")
    assert residual.boundary == "descent" and residual.position_residual_m > 0.0


def test_the_cross_frame_boundary_fails_loudly_with_no_kernels_furnished() -> None:
    # No fixture: the pool is empty, so SPICE cannot resolve where MOON_ME points. The coupler must
    # raise rather than exchange an unrotated pose — the pre-RFC-0002 contract, preserved.
    coupled, _surface = _descent_engine()
    with pytest.raises(FrameBridgeError, match=r"J2000.*MOON_ME"):
        coupled.advance(60.0)


def test_the_orbital_surface_episode_is_reproducible_end_to_end(spice_kernels: object) -> None:
    # Determinism is a hard requirement (conventions.md §11): adopting SPICE must not introduce a
    # non-reproducible input. SPICE geometry is a pure function of (kernels, epoch), so it does not
    # — the same seed still reproduces the trace byte-for-byte across the cross-frame boundary.
    scenario = _surface_swarm("lander")  # the Simulator's agent roster

    def factory(_scenario: Scenario, _rng: RngStreams) -> CoupledEngine:
        return _descent_engine()[0]

    first = run_episode(scenario, engine_factory=factory)
    second = run_episode(scenario, engine_factory=factory)
    assert first.content_hash == second.content_hash
    assert len(first.frames) == scenario.horizon_steps + 1  # it really ran to the horizon


def test_a_host_may_inject_its_own_frame_bridge() -> None:
    # The bridge is a seam, not a hard-wired dependency: a host with its own kernel discipline (or a
    # test) injects one, and the co-simulation machinery is untouched.
    class _NullBridge:
        def bridge(self, sample: StateSample, target_frame: object, epoch: Epoch) -> StateSample:
            return sample.model_copy(update={"frame": target_frame})

    engine, surface = _descent_engine(frame_bridge=_NullBridge())
    engine.advance(60.0)  # no kernels furnished, yet no error: the injected bridge answered
    # The injected bridge did no rotation, so the consumer received the raw inertial coordinates —
    # proof the coupler routed the handoff through *this* bridge and not through SPICE.
    p = surface.export_coupling_state().by_agent["lander"].pose.translation_m
    orbital = engine.export_coupling_state().by_agent["lander"].pose.translation_m
    assert (p.x, p.y, p.z) == pytest.approx((orbital.x, orbital.y, orbital.z))


class _StubEngine:
    """A minimal RegimeEngine, only to construct a CoupledEngine in the default-bridge test."""

    @property
    def descriptor(self) -> object:
        from astro_mine.sim.engines import KINEMATIC_ENGINE_DESCRIPTOR

        return KINEMATIC_ENGINE_DESCRIPTOR

    def apply_actions(self, actions: object) -> None: ...
    def advance(self, dt_s: float) -> None: ...

    def export_coupling_state(self) -> object:
        from astro_mine.sim.engines import CouplingState

        return CouplingState(sim_time_s=0.0, samples=())

    def import_coupling_state(self, state: object) -> None: ...
    def retire(self, agent_ids: object) -> None: ...


def test_the_stub_engine_satisfies_the_regime_engine_protocol() -> None:
    assert isinstance(_StubEngine(), RegimeEngine)
