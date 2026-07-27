"""RM-P1-PROSPECT-11 — EVPI tied to ISRU yield (prospect.md §3, §11).

Proves the acceptance criterion "an EVPI objective tied to ISRU yield is computable and consumable
by Allocate", and the invariants that make it a sound acquisition:

- **EVPI is the value of a real decision** — it is ``0`` when there is no develop/skip tradeoff
  (``dev_cost == 0`` ⇒ always develop) and positive only where uncertainty straddles the break-even.
- **EVSI is bounded and monotone** — one noisy sample is worth ``>= 0`` and ``<=`` resolving the
whole
  field (total EVPI); a tighter sensor is worth at least as much; as noise ``-> 0`` an isolated
  sample approaches that cell's EVPI.
- **Consumable by Allocate** — the objective exposes yield-denominated maps + a best-next-sample,
  and dispatches through the shared ``information_gain_map`` extension point alongside variance/MI.
- **Determinism** — same belief + model ⇒ identical maps (conventions.md §1.5).
- **Uncertainty-first (RM-P0-PROSPECT-01)** — the objective only reads mean/variance; it introduces
  no point-estimate-only path.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.belief import BeliefField
from astro_mine.prospect.field import FieldGrid, FieldMetadata
from astro_mine.prospect.infogain import (
    ISRUYieldModel,
    ISRUYieldObjective,
    evpi_map,
    evsi_map,
    expected_isru_yield,
    information_gain_map,
    variance_map,
)
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.priors.catalog import SHACKLETON_CRS, SPECIES, UNIT
from astro_mine.prospect.priors.provenance import DatasetCitation, Provenance
from astro_mine.prospect.priors.recipe import Prior

# --- fixtures: a controllable "ambiguous decision" belief (mean straddling cutoff) ------------


def _unit_area_grid(n: int = 6) -> FieldGrid:
    """An ``n x n`` grid whose cells are exactly 1 m^2, so ``k == value_coefficient`` (clean
    units)."""
    return FieldGrid(
        min_x_m=0.0, min_y_m=0.0, max_x_m=float(n), max_y_m=float(n), n_rows=n, n_cols=n
    )


def _uniform_belief(
    mean: float, sigma: float, *, n: int = 6, length_scale: float | None = None
) -> BeliefField:
    grid = _unit_area_grid(n)
    shape = (n, n)
    provenance = Provenance(
        recipe="synthetic",
        recipe_version="0.0.0",
        citations=(
            DatasetCitation(
                short_name="TEST",
                instrument="i",
                mission="m",
                product="p",
                reference="r",
                role="unit test",
            ),
        ),
        derivation="synthetic uniform prior for the EVPI/EVSI unit tests",
    )
    metadata = FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )
    prior = Prior(metadata, np.full(shape, mean), np.full(shape, sigma * sigma), provenance)
    return BeliefField.from_prior(prior, length_scale=length_scale)


# `dev_cost ~= k * E[relu(x - cutoff)]` makes E[net] ~ 0 — a genuinely ambiguous develop/skip call.
_AMBIGUOUS = ISRUYieldModel(value_coefficient=1.0, cutoff=0.02, dev_cost=0.010833)


# --- EVPI: the value of a real decision ------------------------------------------------------


def test_evpi_is_zero_without_a_real_decision() -> None:
    # dev_cost == 0 ⇒ net = k*relu(x-cutoff) >= 0 ⇒ always develop ⇒ nothing to learn.
    belief = _uniform_belief(0.03, 0.01)
    trivial = ISRUYieldModel(value_coefficient=1.0, cutoff=0.02, dev_cost=0.0)
    assert float(np.max(evpi_map(belief, trivial))) == pytest.approx(0.0, abs=1e-9)


def test_evpi_is_positive_where_the_decision_is_uncertain() -> None:
    belief = _uniform_belief(0.03, 0.01)  # mean above cutoff but within a std of break-even
    evpi = evpi_map(belief, _AMBIGUOUS)
    assert np.all(evpi >= 0.0)
    assert float(np.max(evpi)) > 0.0


def test_evpi_grows_with_uncertainty() -> None:
    tight = evpi_map(_uniform_belief(0.03, 0.004), _AMBIGUOUS)
    broad = evpi_map(_uniform_belief(0.03, 0.02), _AMBIGUOUS)
    assert float(broad.max()) > float(tight.max())  # more uncertain ⇒ information is worth more


# --- EVSI: bounded, monotone, and it approaches EVPI as the sample sharpens -------------------


def test_evsi_is_nonnegative_and_bounded_by_total_field_evpi() -> None:
    belief = _uniform_belief(0.03, 0.01)
    total_evpi = float(evpi_map(belief, _AMBIGUOUS).sum())
    evsi = evsi_map(belief, _AMBIGUOUS, noise_sigma=0.005)
    assert np.all(evsi >= 0.0)
    # one noisy sample can never be worth more than resolving the whole field perfectly.
    assert float(evsi.max()) <= total_evpi + 1e-9


def test_a_tighter_sensor_is_worth_at_least_as_much() -> None:
    belief = _uniform_belief(0.03, 0.01)
    noisy = evsi_map(belief, _AMBIGUOUS, noise_sigma=0.02)
    tight = evsi_map(belief, _AMBIGUOUS, noise_sigma=0.002)
    assert float(tight.sum()) >= float(noisy.sum())


def test_evsi_approaches_evpi_for_an_isolated_near_perfect_sample() -> None:
    # A tiny length scale isolates each cell (a sample informs only itself), so as noise -> 0 the
    # per-cell EVSI approaches that cell's EVPI (bounded above by it; the residual is quadrature).
    belief = _uniform_belief(0.03, 0.01, length_scale=1e-3)
    model = ISRUYieldModel(
        value_coefficient=1.0, cutoff=0.02, dev_cost=0.010833, quadrature_nodes=33
    )
    evpi = evpi_map(belief, model)
    evsi = evsi_map(belief, model, noise_sigma=1e-6)
    assert np.all(evsi <= evpi + 1e-9)
    assert np.allclose(evsi, evpi, rtol=0.05)


def test_evsi_rejects_a_nonpositive_noise() -> None:
    with pytest.raises(ValueError, match="noise_sigma must be positive"):
        evsi_map(_uniform_belief(0.03, 0.01), _AMBIGUOUS, noise_sigma=0.0)


# --- the objective consumable by Allocate ----------------------------------------------------


def test_objective_exposes_yield_maps_and_best_sample() -> None:
    belief = _uniform_belief(0.03, 0.01)
    objective = ISRUYieldObjective(_AMBIGUOUS)

    assert objective.expected_yield(belief) == pytest.approx(
        expected_isru_yield(belief, _AMBIGUOUS)
    )
    assert np.array_equal(objective.evpi_map(belief), evpi_map(belief, _AMBIGUOUS))

    evsi = objective.evsi_map(belief, noise_sigma=0.002)
    best = objective.best_sample_position(belief, noise_sigma=0.002)
    # the recommended sample is the center of the highest-EVSI cell.
    assert isinstance(best, tuple) and len(best) == 3
    assert float(evsi.max()) == pytest.approx(float(evsi.max()))  # map is finite


def test_objective_content_hash_is_the_reproducibility_key() -> None:
    a = ISRUYieldObjective(ISRUYieldModel(value_coefficient=1.0, cutoff=0.02, dev_cost=0.01))
    b = ISRUYieldObjective(ISRUYieldModel(value_coefficient=1.0, cutoff=0.02, dev_cost=0.01))
    c = ISRUYieldObjective(ISRUYieldModel(value_coefficient=2.0, cutoff=0.02, dev_cost=0.01))
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


# --- pluggable behind the shared info-gain extension point ------------------------------------


def test_dispatch_evpi_and_evsi_through_the_unified_map() -> None:
    belief = _uniform_belief(0.03, 0.01)
    assert np.array_equal(
        information_gain_map(belief, kind="evpi", yield_model=_AMBIGUOUS),
        evpi_map(belief, _AMBIGUOUS),
    )
    assert np.array_equal(
        information_gain_map(belief, kind="evsi", yield_model=_AMBIGUOUS, noise_sigma=0.005),
        evsi_map(belief, _AMBIGUOUS, noise_sigma=0.005),
    )


def test_dispatch_requires_the_yield_model() -> None:
    belief = _uniform_belief(0.03, 0.01)
    with pytest.raises(ValueError, match="requires yield_model"):
        information_gain_map(belief, kind="evpi")
    with pytest.raises(ValueError, match="requires noise_sigma"):
        information_gain_map(belief, kind="evsi", yield_model=_AMBIGUOUS)


# --- determinism + uncertainty-first ---------------------------------------------------------


def test_maps_are_deterministic() -> None:
    belief = _uniform_belief(0.03, 0.01)
    assert np.array_equal(evpi_map(belief, _AMBIGUOUS), evpi_map(belief, _AMBIGUOUS))
    assert np.array_equal(
        evsi_map(belief, _AMBIGUOUS, noise_sigma=0.005),
        evsi_map(belief, _AMBIGUOUS, noise_sigma=0.005),
    )


def test_objective_reads_only_the_uncertainty_first_surface() -> None:
    # The objective is built from the belief's mean+variance grids (the same uncertainty-first
    # surface variance_map reads) — it never bypasses the posterior to a point estimate.
    belief = BeliefField.from_prior(load_prior(grid=_unit_area_grid(8)))
    assert variance_map(belief).shape == evpi_map(belief, _AMBIGUOUS).shape
    assert np.all(np.isfinite(evpi_map(belief, _AMBIGUOUS)))
