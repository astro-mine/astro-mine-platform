"""Spatially-correlated realizations in the sealed ground-truth path (RM-P1-PROSPECT-10).

Proves the acceptance criteria of the correlated-realization gap (prospect.md §5; RM-P0-PROSPECT-04;
RM-P0-PROSPECT-05; LUNAR-FR-002):

- **The correlated backends are wired in** — ``sample_ground_truth`` draws through the GMRF and
  generative ``realize()`` methods, and the result carries genuine spatial structure (measured,
  not assumed: the lag-1 autocorrelation of a correlated draw is far above an independent one's).
- **The independent path remains the default** — an existing call is byte-for-byte unchanged.
- **The seal is not weakened** — minting, revealing, and observing a correlated truth are still
  capability-gated, and a correlated truth is just as unreachable from an agent-facing view.
- **Determinism** — the same ``(prior, seed, backend)`` yields a byte-identical realization.

It also pins the property that makes the whole thing honest: because each backend's draw is
**standardized** before the prior scales it, a correlated truth has the *same per-cell marginals* as
an independent one. Only the spatial structure changes — not the distribution the truth is a
realization of. A truth whose marginals had drifted from its prior would silently invalidate the
calibration gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.prospect.belief import (
    DEFAULT_REALIZATION,
    REALIZATION_KINDS,
    BeliefField,
    GroundTruthField,
    sample_ground_truth,
)
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS, IsolationError, assert_isolated
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.priors.recipe import Prior

_GRANT = (GROUND_TRUTH_ACCESS,)
_CENTER = (0.0, 0.0, 0.0)
_CORRELATED = ("gmrf", "generative")


def _grid() -> FieldGrid:
    # Small enough for the GMRF's dense-Cholesky realize + O(N) marginal-variance extraction.
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=12, n_cols=12
    )


def _prior() -> Prior:
    return load_prior(grid=_grid())


def _truth(kind: str = DEFAULT_REALIZATION, *, seed: int = 3) -> GroundTruthField:
    return sample_ground_truth(_prior(), seed=seed, capabilities=_GRANT, realization=kind)


def _lag1_autocorrelation(field: np.ndarray) -> float:
    """The mean lag-1 (nearest-neighbour) spatial autocorrelation of a standardized field.

    Near 0 for per-cell independent noise; strongly positive for a spatially-correlated field. This
    is the measurement that decides whether the realization actually *has* the structure claimed —
    the difference between a hard prospecting problem and a trivially easy one.
    """
    z = (field - field.mean()) / field.std()
    horizontal = float(np.mean(z[:, :-1] * z[:, 1:]))
    vertical = float(np.mean(z[:-1, :] * z[1:, :]))
    return 0.5 * (horizontal + vertical)


# --- AC1: the correlated backends are wired in, and they actually correlate ----------------------


def test_the_realization_backends_are_selectable() -> None:
    assert REALIZATION_KINDS == ("independent", "gmrf", "generative")
    assert DEFAULT_REALIZATION == "independent"
    for kind in REALIZATION_KINDS:
        assert _truth(kind).realization_kind == kind


#: The lag-1 autocorrelation band each backend must land in, at a 500 m correlation length on a
#: 167 m grid — a *band*, not a floor, so that over-correlation fails too (a backend that smoothed
#: the truth into mush would otherwise sail through a one-sided threshold).
#:
#: **Where the GMRF number comes from — analytically, not by fitting the code.** 500 m / 166.67 m
#: makes the requested practical range ``rho = 3`` cells, so ``kappa = sqrt(8 nu)/rho = sqrt(8)/3``
#: for the ``nu = 1`` field the alpha = 2 SPDE operator realizes. Continuum Matern nu = 1 at one
#: cell is ``kappa * K_1(kappa) = 0.626``; the 5-point stencil discretizes that to an exact lattice
#: value of 0.599 (both pinned exactly, without sampling, in ``test_backends_gmrf.py``). This
#: estimator subtracts the *sample* mean of a single 12x12 draw, which biases it down by a few
#: hundredths, so a seeded single draw lands near 0.50. (Under the alpha = 1 operator — the bug this
#: replaces — it was 0.22, and the threshold here was a limp 0.15.)
#:
#: **The generative backend is higher, and that is not a defect.** Both backends now read
#: ``correlation_length_m`` as the same thing (the practical range: correlation ~0.1 at that
#: separation — see ``test_both_backends_structure_the_truth_over_the_requested_range``), but their
#: covariance *shapes* genuinely differ: a Gaussian-smoothing kernel yields an infinitely smooth
#: field, a Matern nu = 1 field is not smooth. Equal range, more short-lag correlation. They are the
#: same length scale, not the same model.
_LAG1_BAND = {"gmrf": (0.45, 0.70), "generative": (0.55, 0.85)}


@pytest.mark.parametrize("kind", _CORRELATED)
def test_a_correlated_backend_produces_a_spatially_correlated_truth(kind: str) -> None:
    prior = _prior()
    independent = sample_ground_truth(prior, seed=3, capabilities=_GRANT).reveal(
        capabilities=_GRANT
    )
    correlated = sample_ground_truth(
        prior, seed=3, capabilities=_GRANT, realization=kind, correlation_length_m=500.0
    ).reveal(capabilities=_GRANT)

    # The independent draw has essentially no neighbour correlation; the correlated ones do. This is
    # the whole point of the gap: sealed truth now carries the spatial structure the belief backends
    # model, rather than being white noise around the prior mean (a far easier, dishonest problem).
    assert abs(_lag1_autocorrelation(independent)) < 0.1
    low, high = _LAG1_BAND[kind]
    assert low < _lag1_autocorrelation(correlated) < high


def test_an_unknown_realization_backend_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown ground-truth realization backend"):
        sample_ground_truth(_prior(), seed=0, capabilities=_GRANT, realization="crystal_ball")


def _ensemble_correlation_at(kind: str, lag: int, *, n_seeds: int = 32) -> float:
    """The mean correlation between cells ``lag`` apart, measured **across seeds** (unbiased).

    Not within a single draw: standardizing one small field by its own sample mean and std removes
    the very long-range signal being measured, biasing every correlation downward by roughly the
    field-average correlation. The honest estimator for "how far does the structure reach" is the
    *ensemble* one — each cell standardized over the seeds, which is exactly the distribution the
    sealed truth is a draw from. Deterministic: a fixed set of seeds.
    """
    prior = _prior()
    kw = {} if kind == "independent" else {"correlation_length_m": 500.0}
    draws = np.stack(
        [
            sample_ground_truth(prior, seed=s, capabilities=_GRANT, realization=kind, **kw).reveal(
                capabilities=_GRANT
            )
            for s in range(n_seeds)
        ]
    )
    z = (draws - draws.mean(axis=0)) / draws.std(axis=0)
    across = float(np.mean(z[:, :, :-lag] * z[:, :, lag:]))
    down = float(np.mean(z[:, :-lag, :] * z[:, lag:, :]))
    return 0.5 * (across + down)


def test_both_backends_structure_the_truth_over_the_requested_range() -> None:
    """Both backends now read ``correlation_length_m`` as the same thing — the **practical range**.

    The requested 500 m is 3 cells on this grid. At that separation both backends' correlation has
    decayed to ~0.1 (the practical-range convention: Matern nu = 1 puts 0.14 there, a Gaussian
    kernel 0.10), and by twice the range both are essentially uncorrelated. Previously they did
    *not* agree: the generative backend read the number as its smoothing kernel's **sigma**, whose
    practical range is 3.03x longer, so it structured the truth over three times the length asked
    for while the alpha = 1 GMRF structured it over a third of it — an ~9x disagreement between two
    backends handed an identical number.

    They are still not interchangeable, and this test does not pretend otherwise: at *one cell* the
    Gaussian-smoothed field is markedly more correlated than the Matern one (0.77 vs 0.59). Same
    range, different covariance shapes — infinitely smooth against nu = 1. What is now true, and is
    what the acceptance criterion asks for, is that the two are on the same length scale.
    """
    at_range = {kind: _ensemble_correlation_at(kind, 3) for kind in REALIZATION_KINDS}
    lag1 = {kind: _ensemble_correlation_at(kind, 1) for kind in REALIZATION_KINDS}
    beyond = {kind: _ensemble_correlation_at(kind, 6) for kind in REALIZATION_KINDS}

    for kind in _CORRELATED:
        # Correlated out to the requested range, and gone by twice it.
        assert 0.05 < at_range[kind] < 0.25, f"{kind} at the range: {at_range[kind]}"
        assert beyond[kind] < 0.20, f"{kind} beyond twice the range: {beyond[kind]}"
    # And the two agree with each other at the range — the shared convention, in one assertion.
    assert abs(at_range["gmrf"] - at_range["generative"]) < 0.10

    # The independent draw carries no structure at any lag (the control).
    assert abs(at_range["independent"]) < 0.10
    assert abs(lag1["independent"]) < 0.10

    # The shapes still differ, and honestly so: Matern nu = 1 against an infinitely smooth Gaussian.
    assert 0.50 < lag1["gmrf"] < 0.68  # exact lattice value 0.599
    assert 0.70 < lag1["generative"] < 0.86  # the Gaussian kernel's exp(-h^2/4s^2) = 0.774
    assert lag1["generative"] > lag1["gmrf"]


def test_the_correlation_length_is_honoured() -> None:
    # A longer correlation length means a smoother truth — the knob is real, not decorative.
    short = sample_ground_truth(
        _prior(), seed=5, capabilities=_GRANT, realization="gmrf", correlation_length_m=150.0
    ).reveal(capabilities=_GRANT)
    long = sample_ground_truth(
        _prior(), seed=5, capabilities=_GRANT, realization="gmrf", correlation_length_m=900.0
    ).reveal(capabilities=_GRANT)
    assert _lag1_autocorrelation(long) > _lag1_autocorrelation(short)


# --- AC2: the independent path remains the default, and is unchanged -----------------------------


def test_the_default_is_the_independent_per_cell_draw_unchanged() -> None:
    prior = _prior()
    grid = prior.metadata.grid
    assert grid is not None

    # The exact Phase-0 arithmetic: a seeded per-cell standard-normal draw, scaled by the prior's
    # per-cell sigma, offset by its mean, clipped at the physical floor.
    z = np.random.default_rng(11).standard_normal((grid.n_rows, grid.n_cols))
    expected = np.clip(prior.mean + np.sqrt(prior.variance) * z, 0.0, None)

    truth = sample_ground_truth(prior, seed=11, capabilities=_GRANT)  # no `realization` argument
    np.testing.assert_array_equal(truth.reveal(capabilities=_GRANT), expected)
    assert truth.realization_kind == "independent"


# --- the honesty property: a correlated truth keeps the prior's marginals -------------------------


@pytest.mark.parametrize("kind", _CORRELATED)
def test_a_correlated_realization_preserves_the_priors_marginals(kind: str) -> None:
    """The marginal at a cell is the prior's — the *ensemble* over seeds says so.

    Note this must be measured across realizations, not across cells within one: in a spatially
    correlated field the cells are (by construction) *not* independent samples of the marginal, so
    the spatial average of a single draw is one sample of a random variable with a wide spread. It
    is the per-cell distribution over seeds that must match the prior — and it is what the
    calibration gate depends on.
    """
    prior = _prior()
    sigma = np.sqrt(prior.variance)
    cell = np.unravel_index(int(np.argmax(prior.mean / sigma)), prior.mean.shape)
    seeds = range(100, 112)

    def residuals(kind: str) -> np.ndarray:
        draws = [
            sample_ground_truth(
                prior,
                seed=s,
                capabilities=_GRANT,
                realization=kind,
                **({} if kind == "independent" else {"correlation_length_m": 500.0}),
            ).reveal(capabilities=_GRANT)[cell]
            for s in seeds
        ]
        return (np.asarray(draws) - prior.mean[cell]) / sigma[cell]

    baseline = residuals("independent")
    correlated = residuals(kind)

    # Each backend's draw is standardized (zero mean, unit marginal variance) before the prior
    # scales it, so a correlated truth is a realization of the *same* distribution as the
    # independent one — only its spatial structure differs. Compared against the independent draw's
    # own ensemble (which shares the same physical-floor clipping), the marginals agree.
    assert abs(float(correlated.mean() - baseline.mean())) < 0.6
    assert 0.5 < float(correlated.std(ddof=1)) < 1.7


# --- AC4: determinism by seed + backend ----------------------------------------------------------


@pytest.mark.parametrize("kind", REALIZATION_KINDS)
def test_the_same_seed_and_backend_give_a_byte_identical_realization(kind: str) -> None:
    a, b = _truth(kind, seed=4), _truth(kind, seed=4)
    first = a.reveal(capabilities=_GRANT)
    second = b.reveal(capabilities=_GRANT)
    np.testing.assert_array_equal(first, second)
    assert first.tobytes() == second.tobytes()  # byte-identical, not merely close
    assert a.content_hash == b.content_hash


@pytest.mark.parametrize("kind", REALIZATION_KINDS)
def test_a_different_seed_gives_a_different_realization(kind: str) -> None:
    a, b = _truth(kind, seed=1), _truth(kind, seed=2)
    assert not np.array_equal(a.reveal(capabilities=_GRANT), b.reveal(capabilities=_GRANT))
    assert a.content_hash != b.content_hash


def test_the_backend_is_part_of_the_content_address() -> None:
    # The same (prior, seed) through two backends are *different* sealed truths, so a Bench scenario
    # that pins a content hash pins the model that produced the field, not just the seed.
    hashes = {kind: _truth(kind, seed=9).content_hash for kind in REALIZATION_KINDS}
    assert len(set(hashes.values())) == len(REALIZATION_KINDS)


# --- AC3: the seal is not weakened by any of this ------------------------------------------------


@pytest.mark.parametrize("kind", _CORRELATED)
def test_minting_a_correlated_truth_is_still_capability_gated(kind: str) -> None:
    with pytest.raises(IsolationError, match="ground_truth_access"):
        sample_ground_truth(_prior(), seed=0, capabilities=(), realization=kind)


@pytest.mark.parametrize("kind", _CORRELATED)
def test_revealing_and_observing_a_correlated_truth_are_still_gated(kind: str) -> None:
    truth = _truth(kind)
    with pytest.raises(IsolationError):
        truth.reveal(capabilities=())
    with pytest.raises(IsolationError):
        truth.observe([_CENTER], noise_sigma=0.01, seed=1, capabilities=())
    # And the revealed array is still immutable.
    with pytest.raises(ValueError, match=r"read-only|assignment"):
        truth.reveal(capabilities=_GRANT)[0, 0] = 99.0


@pytest.mark.parametrize("kind", _CORRELATED)
def test_a_correlated_truth_is_unreachable_from_the_agent_facing_belief(kind: str) -> None:
    # The RM-P0-PROSPECT-05 reachability contract, extended over the correlated path: the belief a
    # correlated truth's observations drive carries the readings, never a handle to the truth — and
    # the sealed field itself still fails the isolation walk.
    truth = _truth(kind)
    readings = truth.observe(
        [_CENTER, (500.0, -500.0, 0.0)], noise_sigma=0.05, seed=2, capabilities=_GRANT
    )
    belief = BeliefField.from_prior(_prior()).update(readings)

    assert assert_isolated(belief) is None
    assert assert_isolated(readings) is None
    with pytest.raises(IsolationError, match="sealed ground truth is reachable"):
        assert_isolated(truth)


@pytest.mark.parametrize("kind", _CORRELATED)
def test_the_correlated_backend_object_is_not_retained_on_the_sealed_field(kind: str) -> None:
    # The backend is used to shape the draw and then dropped: a live GMRF/generative field carrying
    # the truth's structure must not survive on (or beside) the sealed object.
    truth = _truth(kind)
    held = [type(v).__name__ for v in vars(truth).values()]
    assert "GMRFField" not in held
    assert "GenerativeEnsembleField" not in held


@pytest.mark.parametrize("kind", _CORRELATED)
def test_a_correlated_truth_is_still_a_degenerate_resource_field(kind: str) -> None:
    truth = _truth(kind)
    assert truth.variance(_CENTER) == 0.0  # the truth carries no uncertainty
    assert truth.quantile(_CENTER, 0.1) == pytest.approx(truth.mean(_CENTER))
    assert float(truth.reveal(capabilities=_GRANT).min()) >= 0.0  # the physical floor still holds
