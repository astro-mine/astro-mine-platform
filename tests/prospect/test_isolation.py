"""RM-P0-PROSPECT-05 — ground-truth/belief isolation (a leak is a security-class defect).

Proves both acceptance criteria (prospect.md §9; LUNAR-FR-002, LUNAR-SR-005, LUNAR-DR-005):

- **AC1** — no agent-facing Env-API path reaches ground truth: a :class:`BeliefField`, a Core
  :class:`Observation` (the agent-facing surface), and the synthetic observations drawn from the
  truth all pass :func:`assert_isolated`; a view that *does* hold the sealed field fails it. This
  test fails CI if isolation regresses.
- **AC2** — only access-gated, non-agent code paths read the sealed field: minting, revealing, and
  observing the truth all require Core's ``GROUND_TRUTH_ACCESS`` capability; an ungated caller is
  refused with :class:`IsolationError`.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import (
    Observation,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS, CapabilityTag
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.belief import BeliefField, GroundTruthField, sample_ground_truth
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.isolation import (
    GROUND_TRUTH_ACCESS,
    SEALED_MARKER,
    IsolationError,
    assert_agent_safe_capabilities,
    assert_isolated,
    require_ground_truth_access,
)
from astro_mine.prospect.priors import SPECIES, UNIT, load_prior

_CENTER = (0.0, 0.0, 0.0)
_GRANT = (GROUND_TRUTH_ACCESS,)  # the privileged (Sim-side) capability grant


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _truth(seed: int = 0) -> GroundTruthField:
    return sample_ground_truth(load_prior(grid=_grid()), seed=seed, capabilities=_GRANT)


def _belief() -> BeliefField:
    return BeliefField.from_prior(load_prior(grid=_grid()))


def _observation(*, sensors: tuple[SensorReading, ...] = ()) -> Observation:
    """A Core agent-facing :class:`Observation` — the Env-API surface a policy actually sees."""
    pose = Transform(
        translation_m=Vec3(x=0.0, y=0.0, z=0.0),
        rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    state = StateSample(agent_id="rover-1", frame=MOON_BODY_FIXED, pose=pose)
    return Observation(
        tick=0, sim_time_s=0.0, agent_id="rover-1", self_state=state, sensors=list(sensors)
    )


# --- the capability tag is Core's, gated, and wired to the field -----------------------------


def test_ground_truth_access_is_a_core_gated_tag() -> None:
    assert GROUND_TRUTH_ACCESS is CapabilityTag.GROUND_TRUTH_ACCESS
    assert GROUND_TRUTH_ACCESS in GATED_CAPABILITY_TAGS


def test_ground_truth_field_declares_the_required_capability() -> None:
    assert getattr(GroundTruthField, SEALED_MARKER) is CapabilityTag.GROUND_TRUTH_ACCESS


# --- AC2: reading the sealed field is capability-gated ---------------------------------------


def test_require_ground_truth_access_admits_the_grant_and_refuses_others() -> None:
    require_ground_truth_access([GROUND_TRUTH_ACCESS])  # privileged: no raise
    with pytest.raises(IsolationError, match=r"ground_truth_access.*none"):
        require_ground_truth_access(())
    with pytest.raises(IsolationError, match="ground_truth_access"):
        require_ground_truth_access([CapabilityTag.MOBILITY_WHEELED])


def test_minting_truth_requires_the_capability() -> None:
    with pytest.raises(IsolationError, match="ground_truth_access"):
        sample_ground_truth(load_prior(grid=_grid()), seed=0, capabilities=())


def test_reveal_is_gated_and_returns_a_read_only_array() -> None:
    gt = _truth()
    with pytest.raises(IsolationError):
        gt.reveal(capabilities=())
    realization = gt.reveal(capabilities=_GRANT)
    assert realization.shape == (8, 8)
    with pytest.raises(ValueError, match=r"read-only|assignment"):
        realization[0, 0] = 99.0


def test_observe_is_gated() -> None:
    gt = _truth()
    with pytest.raises(IsolationError):
        gt.observe([_CENTER], noise_sigma=0.01, seed=1, capabilities=())
    obs = gt.observe([_CENTER], noise_sigma=0.01, seed=1, capabilities=_GRANT)
    assert len(obs) == 1


def test_agent_safe_capabilities_rejects_every_gated_tag() -> None:
    assert_agent_safe_capabilities([CapabilityTag.MOBILITY_WHEELED, CapabilityTag.STAGING])
    for gated in GATED_CAPABILITY_TAGS:
        with pytest.raises(IsolationError, match="gated"):
            assert_agent_safe_capabilities([CapabilityTag.MOBILITY_WHEELED, gated])


# --- AC1: ground truth is unreachable from the agent-facing surface ---------------------------


def test_belief_field_is_isolated() -> None:
    assert assert_isolated(_belief()) is None


def test_belief_updated_from_synthetic_observations_is_isolated() -> None:
    # The realistic loop: privileged code draws observations from the truth and feeds the belief;
    # the resulting posterior carries the readings, never a handle to the truth they came from.
    truth = _truth()
    readings = truth.observe(
        [_CENTER, (500.0, -500.0, 0.0)], noise_sigma=0.05, seed=2, capabilities=_GRANT
    )
    belief = _belief().update(readings)
    assert assert_isolated(belief) is None


def test_core_observation_surface_is_isolated() -> None:
    reading = SensorReading(
        sensor="neutron", values=[1.2], unit=UNIT, resource_species=SPECIES, noise_sigma=0.05
    )
    assert assert_isolated(_observation(sensors=(reading,))) is None
    assert assert_isolated(reading) is None


def test_posterior_summary_is_isolated() -> None:
    # FieldDistribution is a slotted dataclass with a quantile mapping — exercises the __slots__
    # and Mapping traversal paths of the reachability walk.
    assert assert_isolated(_belief().posterior(_CENTER)) is None


def test_primitives_and_class_references_are_isolated() -> None:
    # Leaf values, and a reference to the *class* (not an instance) of the sealed field, are safe.
    assert assert_isolated(42) is None
    assert assert_isolated({"truth_type": GroundTruthField}) is None


# --- red team: any reachable sealed field must fail, however it is hidden ---------------------


def test_the_sealed_field_itself_fails_isolation() -> None:
    with pytest.raises(IsolationError, match="sealed ground truth is reachable"):
        assert_isolated(_truth())


def test_a_privately_stashed_handle_fails_isolation() -> None:
    truth = _truth()

    class LeakyView:
        def __init__(self) -> None:
            self._secret = truth  # hidden behind a private attribute — still a leak

    with pytest.raises(IsolationError):
        assert_isolated(LeakyView())


def test_a_nested_handle_fails_isolation() -> None:
    truth = _truth()
    with pytest.raises(IsolationError):
        assert_isolated({"observations": [truth]})  # buried in a mapping + list
    with pytest.raises(IsolationError):
        assert_isolated([_belief(), (truth,)])  # buried in a tuple beside a clean belief


# --- the walk is bounded and cycle-safe ------------------------------------------------------


def test_cycles_do_not_hang_the_walk() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)  # self-reference
    assert assert_isolated(cyclic) is None


def test_unset_slots_are_skipped() -> None:
    class Slotted:
        __slots__ = ("maybe",)  # declared but never assigned

    assert assert_isolated(Slotted()) is None


def test_depth_bound_stops_the_walk() -> None:
    truth = _truth()
    # Beyond the depth bound the walk stops, so a truth buried deeper than max_depth is not seen.
    # (Realistic agent views nest far shallower; this only documents the bound's behavior.)
    nested: object = truth
    for _ in range(3):
        nested = [nested]
    assert assert_isolated(nested, max_depth=2) is None
