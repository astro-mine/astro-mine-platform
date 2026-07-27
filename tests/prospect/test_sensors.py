"""Per-instrument sensor likelihoods — the ``sensors/`` extension point (prospect.md §3, §6).

Proves the acceptance criteria of the per-instrument-likelihood gap (LUNAR-FR-002; scenario §6):

- **A registered plugin seam** selects a likelihood by name *or* by Core plugin manifest, rather
  than only the one hard-coded Gaussian point model.
- **Neutron / NIR / GPR / drill each apply their own footprint and depth response** — not a scalar
  ``noise_sigma``. The load-bearing consequence is asserted, not just the parameters: conditioning a
  belief on a *surface-only* NIR reading and on a *drill assay* of the same cell yields honestly
  different posteriors, because a surface instrument is weak evidence about a buried column.
- **``BeliefField.update()`` applies the model an observation is tagged with**, and an *untagged*
  observation still conditions exactly as it did before this seam existed (no silent regression).
- **Sim consumes the same models with no duplicated implementation**: the forward path emits a Core
  ``SensorReading``, and the belief adapts that very record back under the same likelihood object.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.messages.model import SensorReading
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import CapabilityTag, SensorKind
from astro_mine.prospect.belief import BeliefField, FieldObservation, sample_ground_truth
from astro_mine.prospect.belief.observation import load_observations
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS, IsolationError, assert_isolated
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.sensors import (
    DEFAULT_LIKELIHOOD_NAME,
    REFERENCE_COLUMN_DEPTH_M,
    DepthResponse,
    SensorLikelihood,
    build_likelihood_manifest,
    get_likelihood,
    is_registered,
    likelihood_from_manifest,
    list_likelihoods,
    register_likelihood,
    resolve_likelihood,
)

_GRANT = (GROUND_TRUTH_ACCESS,)
_CENTER = (0.0, 0.0, 0.0)
_ANCHOR_SET = ("neutron_spectrometer", "nir_reflectance", "gpr", "drill_assay")


def _grid() -> FieldGrid:
    """The prospecting-scale grid (250 m cells, like the anchor prior's)."""
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _fine_grid() -> FieldGrid:
    """A dense local survey of a single cold trap — 1 m cells, where a rover footprint *resolves*.

    These are rover-borne instruments, so their footprints are metres while the anchor prospecting
    grid is 250 m per cell: at prospecting scale a footprint is sub-cell and correctly degenerates
    to a point measurement (it is the *depth response* that separates instruments there). Footprint
    averaging is load-bearing at this resolution, which is where it is tested.
    """
    return FieldGrid(min_x_m=-8.0, min_y_m=-8.0, max_x_m=8.0, max_y_m=8.0, n_rows=16, n_cols=16)


def _prior():  # type: ignore[no-untyped-def]
    return load_prior(grid=_grid())


def _truth():  # type: ignore[no-untyped-def]
    return sample_ground_truth(_prior(), seed=0, capabilities=_GRANT)


# --- AC1: a registered plugin seam, selectable by name or manifest -------------------------------


def test_the_anchor_instrument_set_is_registered() -> None:
    registered = list_likelihoods()
    for name in (*_ANCHOR_SET, DEFAULT_LIKELIHOOD_NAME):
        assert name in registered
        assert is_registered(name)


def test_a_likelihood_is_selected_by_name() -> None:
    assert get_likelihood("gpr").kind is SensorKind.GPR
    assert get_likelihood("nir_reflectance").kind is SensorKind.NIR_SPECTROMETER
    assert get_likelihood("drill_assay").capability is CapabilityTag.PROSPECTING_DRILL_ASSAY
    with pytest.raises(ValueError, match="unknown sensor likelihood"):
        get_likelihood("telepathy")


def test_an_untagged_observation_resolves_to_the_default_point_model() -> None:
    default = resolve_likelihood(None)
    assert default.name == DEFAULT_LIKELIHOOD_NAME
    # The reduction that keeps the pre-instrument path byte-identical: no footprint, unit gain.
    assert default.footprint_sigma_m == 0.0
    assert default.gain == pytest.approx(1.0)


def test_a_likelihood_round_trips_through_a_core_plugin_manifest() -> None:
    # The registry seam prospect.md §3 requires: a likelihood is a Core `observation_model` plugin,
    # so Sim resolves the *exact* instrument a scenario pinned, by manifest, not by import.
    original = get_likelihood("neutron_spectrometer")
    manifest = build_likelihood_manifest(original)

    assert manifest.kind is PluginKind.OBSERVATION_MODEL
    assert manifest.attributes["sensor_kind"] == SensorKind.NEUTRON_SPECTROMETER.value
    assert manifest.provenance is not None
    assert manifest.provenance.digest.startswith("sha256:")

    assert likelihood_from_manifest(manifest) == original


def test_a_manifest_of_the_wrong_kind_is_refused() -> None:
    manifest = build_likelihood_manifest(get_likelihood("gpr"))
    other = manifest.model_copy(update={"kind": PluginKind.RESOURCE_FIELD_BACKEND})
    with pytest.raises(ValueError, match="not a sensor likelihood"):
        likelihood_from_manifest(other)
    stripped = manifest.model_copy(update={"attributes": {}})
    with pytest.raises(ValueError, match="carries no"):
        likelihood_from_manifest(stripped)


def test_a_third_party_instrument_registers_with_no_change_to_the_conditioner() -> None:
    exotic = SensorLikelihood(
        name="test_deep_sounder",
        kind=SensorKind.GPR,
        capability=CapabilityTag.PROSPECTING_GPR,
        footprint_sigma_m=3.0,
        depth=DepthResponse(top_m=1.0, bottom_m=5.0),
        noise_sigma=0.05,
    )
    register_likelihood(exotic)
    try:
        assert resolve_likelihood("test_deep_sounder") is exotic
        # And it conditions a belief straight away — the seam, not a special case.
        obs = FieldObservation(
            x_m=0.0, y_m=0.0, value=0.1, noise_sigma=0.05, likelihood="test_deep_sounder"
        )
        assert BeliefField.from_prior(_prior()).update([obs]).variance(_CENTER) > 0.0
        with pytest.raises(ValueError, match="already registered"):
            register_likelihood(exotic)
    finally:  # keep the module-scoped registry clean for the other tests
        import astro_mine.prospect.sensors as sensors_mod

        del sensors_mod._LIKELIHOODS["test_deep_sounder"]


def test_an_agent_facing_likelihood_may_not_declare_a_gated_capability() -> None:
    sealed = SensorLikelihood(
        name="test_oracle",
        kind=SensorKind.DRILL_ASSAY,
        capability=CapabilityTag.GROUND_TRUTH_ACCESS,  # a reserved/gated tag
        footprint_sigma_m=0.0,
        depth=DepthResponse(top_m=0.0, bottom_m=1.0),
        noise_sigma=0.01,
    )
    with pytest.raises(IsolationError, match="gated"):
        build_likelihood_manifest(sealed)


# --- AC2: each instrument applies its own footprint + depth response -----------------------------


def test_the_instruments_have_genuinely_different_depth_gains() -> None:
    neutron = get_likelihood("neutron_spectrometer").gain
    nir = get_likelihood("nir_reflectance").gain
    gpr = get_likelihood("gpr").gain
    drill = get_likelihood("drill_assay").gain

    # NIR sees only the desiccated surface lag: it *under*-reads the buried column, badly.
    assert nir < 0.3
    # GPR and the drill reach past the lag into buried ice: they over-read it.
    assert gpr > 1.0
    assert drill > 1.0
    # The neutron spectrometer integrates (surface-weighted) the reference column: near unity, and
    # it is the instrument the field's own quantity is defined by.
    assert 0.7 < neutron < 1.0
    assert get_likelihood(DEFAULT_LIKELIHOOD_NAME).gain == pytest.approx(1.0)


def test_the_instruments_have_genuinely_different_footprints() -> None:
    grid = _fine_grid()
    neutron = get_likelihood("neutron_spectrometer").footprint_weights(grid, _CENTER)
    drill = get_likelihood("drill_assay").footprint_weights(grid, _CENTER)

    # Both are averaging kernels (they sum to 1), but the neutron footprint spreads over the
    # neighbourhood while the drill is essentially a point sample of its own cell.
    assert neutron.sum() == pytest.approx(1.0)
    assert drill.sum() == pytest.approx(1.0)
    assert np.count_nonzero(neutron > 1e-6) > 4 * np.count_nonzero(drill > 1e-6)


def test_a_footprint_below_the_cell_size_degenerates_to_a_point_measurement() -> None:
    # The honest scale behavior: on a 250 m prospecting grid a metre-scale rover footprint is
    # sub-cell, so it reduces to the bilinear point stencil rather than pretending to resolve
    # structure the grid cannot represent. (The depth response still applies at every scale.)
    coarse = get_likelihood("neutron_spectrometer").footprint_weights(_grid(), _CENTER)
    point = get_likelihood(DEFAULT_LIKELIHOOD_NAME).footprint_weights(_grid(), _CENTER)
    np.testing.assert_allclose(coarse, point)


def test_a_broad_footprint_informs_a_wider_neighbourhood_than_a_point_sensor() -> None:
    grid = _fine_grid()
    neutron = get_likelihood("neutron_spectrometer")
    drill = get_likelihood("drill_assay")
    ls = 1.0
    w_neutron = neutron.conditioning_weights(grid, _CENTER, correlation_length_m=ls)
    w_drill = drill.conditioning_weights(grid, _CENTER, correlation_length_m=ls)

    # The kernel is peak-normalized at the *reading's* location (which need not be a cell centre),
    # so cell weights never exceed 1. Footprint and correlation length add in quadrature, so the
    # broad instrument's single reading reaches much further into the neighbourhood.
    assert w_neutron.max() <= 1.0
    assert w_drill.max() <= 1.0
    assert float(w_neutron.sum()) > 2.0 * float(w_drill.sum())


def test_a_point_sensors_conditioning_weights_are_exactly_the_beliefs_own_rbf() -> None:
    # The reduction that keeps the default path unchanged: with no footprint, the effective kernel
    # is the belief's RBF weight, term for term — no quadrature drift, no "almost".
    grid = _grid()
    ls = 200.0
    weights = get_likelihood(DEFAULT_LIKELIHOOD_NAME).conditioning_weights(
        grid, (100.0, -50.0, 0.0), correlation_length_m=ls
    )
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)
    dist2 = (gx - 100.0) ** 2 + (gy + 50.0) ** 2
    np.testing.assert_array_equal(weights, np.exp(-dist2 / (2.0 * ls * ls)).ravel())


def test_depth_response_gain_is_the_sensitivity_weighted_profile_ratio() -> None:
    # A flat, full-reference-column response is unit gain *by construction* — that is what makes the
    # default likelihood a faithful stand-in for the pre-instrument scalar-sigma model.
    flat = DepthResponse(top_m=0.0, bottom_m=REFERENCE_COLUMN_DEPTH_M, attenuation_length_m=None)
    assert flat.gain(get_likelihood(DEFAULT_LIKELIHOOD_NAME).profile) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="bottom_m must be strictly below top_m"):
        DepthResponse(top_m=2.0, bottom_m=1.0)


def test_precision_about_the_field_is_scaled_by_the_depth_gain() -> None:
    # The crux of the honest-uncertainty story: a *precise* surface reading is still *weak* evidence
    # about the buried column, because the depth response rescales precision, not just the value.
    nir = get_likelihood("nir_reflectance")
    drill = get_likelihood("drill_assay")
    sigma = 0.01  # the same instrument noise for both, to isolate the depth-response effect
    assert nir.precision(sigma) < 0.1 * drill.precision(sigma)


# --- AC3: BeliefField.update() applies the model the observation is tagged with -------------------


def test_the_belief_applies_the_matching_model_per_observation() -> None:
    prior = _prior()
    base = BeliefField.from_prior(prior)
    prior_variance = base.variance(_CENTER)

    def conditioned(name: str) -> float:
        obs = FieldObservation(x_m=0.0, y_m=0.0, value=0.08, noise_sigma=0.01, likelihood=name)
        return base.update([obs]).variance(_CENTER)

    nir_variance = conditioned("nir_reflectance")
    drill_variance = conditioned("drill_assay")

    # Both readings are equally precise *as readings* (the same sigma=0.01 at the same cell), so the
    # only thing separating them is the depth response — and it separates them by an order of
    # magnitude: the drill assay collapses the belief's uncertainty about the buried column while
    # the surface-only NIR barely dents it. A single scalar-sigma model would have made these two
    # indistinguishable, and would have been over-confident about the NIR one.
    assert drill_variance < 0.2 * nir_variance
    assert nir_variance < prior_variance  # it *is* evidence — just weak evidence

    # With each instrument's own nominal noise (the realistic comparison, drill being far quieter),
    # the gap is far starker still.
    def with_own_noise(name: str) -> float:
        model = get_likelihood(name)
        obs = FieldObservation(
            x_m=0.0, y_m=0.0, value=0.08, noise_sigma=model.noise_sigma, likelihood=name
        )
        return base.update([obs]).variance(_CENTER)

    assert with_own_noise("drill_assay") < 0.02 * with_own_noise("nir_reflectance")


def test_an_untagged_observation_conditions_exactly_as_before_the_seam() -> None:
    # No silent regression: an untagged reading resolves to the zero-footprint, unit-gain model, for
    # which the arithmetic is the pre-instrument RBF precision fusion, term for term.
    prior = _prior()
    grid = prior.metadata.grid
    assert grid is not None
    belief = BeliefField.from_prior(prior)
    obs = FieldObservation(x_m=100.0, y_m=-50.0, value=0.09, noise_sigma=0.02)
    posterior = belief.update([obs])

    # The pre-instrument arithmetic, written out term for term — and asserted **exactly** (not
    # approximately), because the claim is that the default path is unchanged, not merely close.
    ls = belief.length_scale
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)
    dist2 = (gx - obs.x_m) ** 2 + (gy - obs.y_m) ** 2
    weights = np.exp(-dist2 / (2.0 * ls * ls))

    contrib = weights * (1.0 / (obs.noise_sigma * obs.noise_sigma))
    prior_precision = 1.0 / prior.variance
    expected_variance = 1.0 / (prior_precision + contrib)
    expected_mean = expected_variance * (prior_precision * prior.mean + contrib * obs.value)

    np.testing.assert_array_equal(posterior.variance_grid(), expected_variance)
    np.testing.assert_array_equal(posterior.mean_grid(), expected_mean)


def test_an_unknown_likelihood_tag_fails_loudly_rather_than_silently_defaulting() -> None:
    obs = FieldObservation(x_m=0.0, y_m=0.0, value=0.1, noise_sigma=0.01, likelihood="no-such")
    with pytest.raises(ValueError, match="unknown sensor likelihood"):
        BeliefField.from_prior(_prior()).update([obs])


def test_the_likelihood_tag_is_part_of_the_beliefs_content_address() -> None:
    # Two logs identical but for the instrument that produced them are *different* posteriors, so a
    # replayed log that lost its tags cannot silently pass as the same belief.
    base = BeliefField.from_prior(_prior())
    a = base.update(
        [FieldObservation(x_m=0.0, y_m=0.0, value=0.08, noise_sigma=0.01, likelihood="gpr")]
    )
    b = base.update(
        [FieldObservation(x_m=0.0, y_m=0.0, value=0.08, noise_sigma=0.01, likelihood="drill_assay")]
    )
    assert a.content_hash != b.content_hash


def test_an_observation_csv_carries_the_instrument_tag() -> None:
    rows = [
        "x_m,y_m,value,noise_sigma,likelihood",
        "0,0,0.08,0.01,drill_assay",
        "100,100,0.04,0.02,",
    ]
    log = load_observations(rows)
    assert log[0].likelihood == "drill_assay"
    assert log[1].likelihood is None  # an empty cell means "untagged", not an error
    with pytest.raises(ValueError, match="unknown sensor likelihood"):
        load_observations(["x_m,y_m,value,noise_sigma,likelihood", "0,0,0.1,0.01,telepathy"])


# --- AC4: Sim consumes the *same* models, with no duplicated implementation -----------------------


def test_the_forward_model_emits_a_core_sensor_reading_sim_can_put_on_the_wire() -> None:
    truth = _truth()
    values = truth.reveal(capabilities=_GRANT)
    gpr = get_likelihood("gpr")

    reading = gpr.sense(truth.metadata, values, _CENTER, rng=np.random.default_rng(1))

    assert isinstance(reading, SensorReading)
    assert reading.sensor == "gpr"  # the instrument names itself, so the belief can resolve it back
    assert reading.unit == truth.unit
    assert reading.resource_species == truth.species
    assert reading.noise_sigma == gpr.noise_sigma


def test_a_sim_rendered_reading_conditions_the_belief_under_the_same_model() -> None:
    # The whole point of the seam: Sim renders through `sense`, the belief adapts that very record
    # back with `from_sensor_reading`, and the instrument is resolved automatically from the tag.
    truth = _truth()
    values = truth.reveal(capabilities=_GRANT)
    reading = get_likelihood("nir_reflectance").sense(
        truth.metadata, values, _CENTER, rng=np.random.default_rng(3)
    )
    observation = FieldObservation.from_sensor_reading(reading, position=_CENTER)

    assert observation.likelihood == "nir_reflectance"  # resolved from the reading, not passed in
    belief = BeliefField.from_prior(_prior()).update([observation])
    assert belief.variance(_CENTER) > 0.0
    assert assert_isolated(belief) is None  # and it is still agent-safe


def test_a_reading_from_an_unregistered_sensor_stays_untagged() -> None:
    reading = SensorReading(sensor="some-vendor-widget", values=[0.1], noise_sigma=0.02)
    assert FieldObservation.from_sensor_reading(reading, position=_CENTER).likelihood is None


def test_the_noise_free_forward_model_is_the_footprint_average_times_the_gain() -> None:
    truth = _truth()
    values = truth.reveal(capabilities=_GRANT)
    gpr = get_likelihood("gpr")

    weights = gpr.footprint_weights(_grid(), _CENTER)
    expected = gpr.gain * float(np.dot(weights, values.ravel()))
    assert gpr.expected_reading(truth.metadata, values, _CENTER) == pytest.approx(expected)

    # And with rng=None the rendered reading is exactly that expectation (no hidden noise).
    reading = gpr.sense(truth.metadata, values, _CENTER, rng=None)
    assert reading.values[0] == pytest.approx(expected)


def test_ground_truth_observe_renders_through_the_named_instrument() -> None:
    truth = _truth()
    for name in _ANCHOR_SET:
        obs = truth.observe([_CENTER], seed=1, capabilities=_GRANT, likelihood=name)
        assert obs[0].likelihood == name
        assert obs[0].noise_sigma == get_likelihood(name).noise_sigma  # the instrument's own noise
        assert assert_isolated(obs) is None  # the readings never carry a handle to the truth


def test_the_default_observe_path_is_the_truths_own_interpolated_value() -> None:
    # An untagged observe is the zero-footprint point model, so its noise-free reading is exactly
    # the sealed field's bilinear value at that position — the pre-instrument behavior, unchanged.
    truth = _truth()
    point = (250.0, -125.0, 0.0)
    default = resolve_likelihood(None)
    values = truth.reveal(capabilities=_GRANT)
    assert default.expected_reading(truth.metadata, values, point) == pytest.approx(
        truth.mean(point)
    )


def test_observe_still_rejects_a_nonpositive_noise_override() -> None:
    with pytest.raises(ValueError, match="noise_sigma must be positive"):
        _truth().observe([_CENTER], noise_sigma=0.0, seed=0, capabilities=_GRANT)
