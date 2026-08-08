"""Sealed GroundTruthField — a fixed, seeded realization of a resource field (prospect.md §2.2, §9).

The "true" field for a scenario: a single deterministic realization sampled from a
:class:`~astro_mine.prospect.priors.recipe.Prior`, held immutable (read-only arrays) and
content-addressed so a Bench scenario pins it exactly (prospect.md §5). It is a *distinct type*
from :class:`~astro_mine.prospect.belief.field.BeliefField` — both implement the one Core
``ResourceField`` contract, but ground truth is consumed by Sim, **never reached through the belief
path** (prospect.md §9).

The sealed field is **access-gated** (RM-P0-PROSPECT-05): minting it (:func:`sample_ground_truth`),
revealing its raw values (:meth:`GroundTruthField.reveal`), and drawing observations from it
(:meth:`GroundTruthField.observe`) all require the Core ``GROUND_TRUTH_ACCESS`` capability — a
privilege Sim's sensor model holds and agent code does not. The distributional ``ResourceField``
accessors (``mean``/``variance``/``quantile``/``sample``) stay ungated so the field still satisfies
the Core contract for its privileged holder; the guarantee that an *agent* never holds a
:class:`GroundTruthField` in the first place is the reachability contract test
:func:`~astro_mine.prospect.isolation.assert_isolated`. The type advertises itself to that checker
via the ``REQUIRED_CAPABILITY`` marker (:data:`~astro_mine.prospect.isolation.SEALED_MARKER`).

Being the sealed *truth* (not a posterior), it queries as a **degenerate** field: the realized
value with zero variance. Its forward sensor model, :meth:`GroundTruthField.observe`, draws noisy
:class:`~astro_mine.prospect.belief.observation.FieldObservation`\\ s from it through the shared
per-instrument sensor likelihoods (:mod:`astro_mine.prospect.sensors`) — the synthetic sensor hits
Sim feeds back to a belief (LUNAR-FR-002).

**Realization backends** (:data:`REALIZATION_KINDS`). The Phase-0 realization was a per-cell draw
from the prior's *independent* Gaussians: faithful to a prior that carries no stored spatial
covariance, but a strictly easier — and dishonest — prospecting problem than the one the P1 belief
backends model, because real ice is spatially structured. :func:`sample_ground_truth` now selects
the correlated-realization backends those P1 backends ship (RM-P1-PROSPECT-10):
:meth:`~astro_mine.prospect.backends.gmrf.GMRFField.realize` (SPDE/Matern structure) and
:meth:`~astro_mine.prospect.backends.generative.GenerativeEnsembleField.realize` (a flow-warped
correlated draw). ``"independent"`` stays the default, so no existing scenario shifts.

Every backend draws a **standardized** field (zero mean, unit marginal variance) which is then
mapped onto the prior's own per-cell mean and standard deviation. The realization therefore gains
spatial structure while **preserving the prior's marginals exactly** — a truth whose marginals had
drifted from the prior it was drawn from would silently invalidate the calibration gate
(:mod:`astro_mine.prospect.calibration`). All backends are deterministic in ``seed``, and the
selected backend is folded into :attr:`GroundTruthField.content_hash`, so a Bench scenario pins the
realization *and* the model that produced it.

**The seal is unchanged.** Selecting a backend does not widen the gate: minting, revealing, and
observing still require ``GROUND_TRUTH_ACCESS``; the correlated backend object is used to produce an
array and then dropped, never retained on the field, so no un-gated handle to a truth-shaped field
escapes; and the result is the same sealed, marker-carrying type that
:func:`~astro_mine.prospect.isolation.assert_isolated` refuses to find on an agent-facing view.

Backlog: RM-P0-PROSPECT-04, RM-P1-PROSPECT-10 —
astro-mine-prospect#4
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata
from astro_mine.prospect.isolation import require_ground_truth_access
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.sensors import is_registered, resolve_likelihood

__all__ = [
    "DEFAULT_REALIZATION",
    "REALIZATION_KINDS",
    "GroundTruthField",
    "sample_ground_truth",
]

#: The realization backends :func:`sample_ground_truth` can draw a sealed truth through.
#:
#: - ``"independent"`` — a per-cell draw from the prior's independent Gaussians: the Phase-0 model,
#:   and the default, so no existing scenario changes behavior.
#: - ``"gmrf"`` — :meth:`~astro_mine.prospect.backends.gmrf.GMRFField.realize`: a draw from the
#:   SPDE/Matern sparse-precision field, the principled correlated path for a lattice domain.
#: - ``"generative"`` — the generative backend's ``realize`` over a spatially-correlated latent: the
#:   same structure, pushed through its learned monotone flow, for non-Gaussian truth.
REALIZATION_KINDS: tuple[str, ...] = ("independent", "gmrf", "generative")

#: The default realization backend — the Phase-0 independent per-cell draw (backward compatible).
DEFAULT_REALIZATION = "independent"


class GroundTruthField(BaseResourceField):
    """A sealed, immutable realization of a resource field — the scenario's ground truth.

    Construct via :func:`sample_ground_truth` (a seeded draw from a prior), not directly. The
    realization is held read-only and the field queries degenerately (the true value, zero
    variance), delegating its bilinear queries to an internal
    :class:`~astro_mine.prospect.backends.grid.GridField`. :attr:`content_hash` content-addresses
    the sealed truth (prior + seed + realization). The raw realization is read back through the
    capability-gated :meth:`reveal`, never a bare attribute (RM-P0-PROSPECT-05).
    """

    #: Reading this field's sealed values requires the Core ``GROUND_TRUTH_ACCESS`` capability. The
    #: marker (read by :func:`~astro_mine.prospect.isolation.assert_isolated`) also declares the
    #: field a sealed type that must never be reachable from an agent view (prospect.md §9).
    REQUIRED_CAPABILITY: ClassVar[CapabilityTag] = CapabilityTag.GROUND_TRUTH_ACCESS

    def __init__(
        self,
        metadata: FieldMetadata,
        realization: NDArray[np.float64],
        *,
        seed: int,
        prior_hash: str,
        realization_kind: str = DEFAULT_REALIZATION,
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("GroundTruthField requires metadata.grid (a FieldGrid spatial domain)")
        shape = (grid.n_rows, grid.n_cols)
        if realization.shape != shape:
            raise ValueError(f"realization must have grid shape {shape}; got {realization.shape}")
        super().__init__(metadata)
        self._realization = np.ascontiguousarray(realization, dtype=np.float64)
        self._realization.flags.writeable = False
        self._seed = seed
        self._prior_hash = prior_hash
        self._realization_kind = realization_kind
        # The sealed truth is exact: a degenerate (zero-variance) field reuses the grid backend's
        # bilinear interpolation and degenerate quantile/sample handling.
        self._field = GridField(metadata, self._realization, np.zeros(shape, dtype=np.float64))

    def reveal(self, *, capabilities: Iterable[CapabilityTag]) -> NDArray[np.float64]:
        """The sealed per-cell true values ``(n_rows, n_cols)``, read-only — **capability-gated**.

        Returns the raw realization only to a caller presenting ``GROUND_TRUTH_ACCESS`` (Sim's
        sensor model, the #7 calibration harness); any other caller raises
        :class:`~astro_mine.prospect.isolation.IsolationError`. The returned array is immutable.
        """
        require_ground_truth_access(capabilities)
        return self._realization

    @property
    def seed(self) -> int:
        """The seed the realization was drawn under (a reproducibility key)."""
        return self._seed

    @property
    def realization_kind(self) -> str:
        """The realization backend this truth was drawn through (see :data:`REALIZATION_KINDS`).

        Part of the reproducibility key alongside :attr:`seed`: the same ``(prior, seed, kind)``
        always yields the same sealed realization, and a *different* kind yields a different one.
        Both are pinned by :attr:`content_hash`.
        """
        return self._realization_kind

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._field.mean(position)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._field.variance(position)  # zero — the truth is exact

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return self._field.quantile(position, q)  # degenerate → the true value for every q

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return self._field.sample(position, n=n, seed=seed)  # the true value repeated

    def observe(
        self,
        positions: Sequence[Position],
        *,
        noise_sigma: float | None = None,
        seed: int,
        capabilities: Iterable[CapabilityTag],
        time_s: float = 0.0,
        sensor: str = "synthetic",
        likelihood: str | None = None,
    ) -> tuple[FieldObservation, ...]:
        """Draw noisy observations of the truth at ``positions`` — **the** forward sensor model.

        The synthetic sensor hits that drive a belief update (prospect.md §6; LUNAR-FR-002),
        rendered through the *shared* :class:`~astro_mine.prospect.sensors.SensorLikelihood`: the
        instrument's spatial **footprint** is averaged over the sealed realization, its
        **depth-response gain** applied, and seeded ``N(0, s**2)`` noise added. Because the belief
        conditions the resulting observation back under the very same likelihood object
        (:meth:`~astro_mine.prospect.belief.field.BeliefField.update`), Sim's forward model and
        Prospect's inverse model **cannot drift apart** — there is one implementation, not two.

        ``likelihood`` names a registered instrument (``"neutron_spectrometer"``,
        ``"nir_reflectance"``, ``"gpr"``, ``"drill_assay"``, ...); ``None`` uses the zero-footprint,
        unit-gain point model, for which a reading is exactly ``truth(position) + N(0, s**2)`` — the
        pre-instrument behavior. ``noise_sigma`` defaults to the instrument's own nominal noise.
        ``seed`` is independent of the realization's seed, so one truth supports many observation
        campaigns.

        Reading the truth is **capability-gated**: ``capabilities`` must carry
        ``GROUND_TRUTH_ACCESS`` (Sim's sensor model holds it), else
        :class:`~astro_mine.prospect.isolation.IsolationError` is raised. The observations returned
        carry no handle to this field — they are agent-safe by construction (prospect.md §9).
        """
        require_ground_truth_access(capabilities)
        # Resolve the instrument once: named explicitly, or — failing that — by a ``sensor``
        # provenance tag that happens to name a registered one. Resolving it in a single place is
        # what keeps the forward model and the tag the belief conditions under from disagreeing.
        name = likelihood if likelihood is not None else (sensor if is_registered(sensor) else None)
        model = resolve_likelihood(name)
        if noise_sigma is not None and noise_sigma <= 0.0:
            raise ValueError(f"noise_sigma must be positive, got {noise_sigma}")
        rng = np.random.default_rng(seed)
        return tuple(
            FieldObservation.from_sensor_reading(
                model.sense(
                    self._metadata,
                    self._realization,
                    p,
                    rng=rng,
                    noise_sigma=noise_sigma,
                    sensor=sensor,
                ),
                position=p,
                time_s=time_s,
                likelihood=name,
            )
            for p in positions
        )

    @property
    def content_hash(self) -> str:
        """A content address over the prior, seed, realization backend, arrays, and metadata.

        The realization backend is part of the address: the same ``(prior, seed)`` drawn through
        ``"independent"`` and through ``"gmrf"`` are *different* sealed truths and must not collide
        — a Bench scenario pins the exact field it scored against (prospect.md §5).
        """
        digest = hashlib.sha256()
        digest.update(self._prior_hash.encode("utf-8"))
        digest.update(str(self._seed).encode("utf-8"))
        digest.update(self._realization_kind.encode("utf-8"))
        digest.update(self._metadata.model_dump_json().encode("utf-8"))
        digest.update(self._realization.tobytes())
        return digest.hexdigest()


def sample_ground_truth(
    prior: Prior,
    *,
    seed: int,
    capabilities: Iterable[CapabilityTag],
    realization: str = DEFAULT_REALIZATION,
    correlation_length_m: float | None = None,
    **backend_config: Any,
) -> GroundTruthField:
    """Draw a sealed :class:`GroundTruthField` from ``prior`` under ``seed`` (deterministic).

    The realization is ``prior.mean + sqrt(prior.variance) * z``, clipped to the physical floor (a
    resource concentration cannot be negative), where ``z`` is a **standardized** — zero-mean,
    unit-marginal-variance — draw from the selected ``realization`` backend
    (:data:`REALIZATION_KINDS`):

    - ``"independent"`` (default) — ``z`` is per-cell independent standard normal: the Phase-0
      model, byte-for-byte unchanged, so no existing scenario shifts;
    - ``"gmrf"`` — ``z`` comes from :meth:`~astro_mine.prospect.backends.gmrf.GMRFField.realize`, a
      draw from the SPDE/Matern sparse-precision field, standardized by its own exact marginal
      variances;
    - ``"generative"`` — ``z`` comes from
      :meth:`~astro_mine.prospect.backends.generative.GenerativeEnsembleField.realize` over a
      spatially-correlated latent, so the draw carries the backend's flow shape as well as the
      spatial structure.

    Because ``z`` is standardized, **the prior's per-cell marginals are preserved exactly**
    whichever backend is chosen: the truth gains spatial structure, not a different distribution.

    The correlated backends' structure scale is ``correlation_length_m`` (default: 15% of the
    larger grid extent, matching the GMRF backend's own default), and both read it the same way —
    as the **practical range**, the separation at which correlation has decayed to ~0.1. What they
    do *not* share is the covariance shape: the GMRF's is Matérn nu = 1 and the generative backend's
    is a Gaussian-smoothed (infinitely smooth) latent, so at one cell's separation the latter is the
    more strongly correlated of the two even at an identical range. Same length scale, different
    models — which is the point of having both. ``backend_config`` passes any remaining
    backend-specific knobs through. The same ``(prior, seed, realization)`` always yields a
    byte-identical field (conventions.md §1.5).

    Minting the truth oracle is **capability-gated**: ``capabilities`` must carry
    ``GROUND_TRUTH_ACCESS`` — only privileged, non-agent code (scenario setup, Sim) obtains the
    sealed field; otherwise :class:`~astro_mine.prospect.isolation.IsolationError` is raised. The
    backend used to shape the draw is discarded once the array is produced, so nothing but the
    sealed field itself ever holds the realization.
    """
    require_ground_truth_access(capabilities)
    grid = prior.metadata.grid
    assert grid is not None  # a Prior always carries its grid (Prior.__init__ enforces it)
    z = _standardized_draw(
        prior,
        grid,
        kind=realization,
        seed=seed,
        correlation_length_m=correlation_length_m,
        **backend_config,
    )
    values = np.clip(prior.mean + np.sqrt(prior.variance) * z, 0.0, None)
    return GroundTruthField(
        prior.metadata,
        values,
        seed=seed,
        prior_hash=prior.content_hash,
        realization_kind=realization,
    )


def _standardized_draw(
    prior: Prior,
    grid: FieldGrid,
    *,
    kind: str,
    seed: int,
    correlation_length_m: float | None,
    **backend_config: Any,
) -> NDArray[np.float64]:
    """A seeded, zero-mean, unit-marginal-variance ``(n_rows, n_cols)`` draw from ``kind``.

    Standardizing here — rather than letting each backend impose its own marginals — is what lets
    the realization backend be swapped without changing *what the truth is a realization of*: only
    the spatial correlation structure of the draw changes, never the prior's per-cell distribution.
    """
    if kind == "independent":
        draw = np.random.default_rng(seed).standard_normal((grid.n_rows, grid.n_cols))
        return np.asarray(draw, dtype=np.float64)

    extent = max(grid.max_x_m - grid.min_x_m, grid.max_y_m - grid.min_y_m)
    length_m = 0.15 * extent if correlation_length_m is None else float(correlation_length_m)
    unit = prior.metadata  # a unit-marginal backend over the same species/unit/CRS/grid

    if kind == "gmrf":
        from astro_mine.prospect.backends.gmrf import GMRFField

        gmrf = GMRFField.from_prior(
            unit,
            prior_mean=0.0,
            prior_variance=1.0,
            correlation_length_m=length_m,
            seed=seed,
            **backend_config,
        )
        # The GMRF's marginal variances are not uniform (the lattice rim sees fewer neighbours), so
        # dividing by the *exact* per-cell marginal standard deviation is what makes this draw truly
        # standard normal at every cell — and hence marginal-preserving once the prior scales it.
        return np.asarray(gmrf.realize(seed=seed) / np.sqrt(gmrf.marginal_variance_grid()))

    if kind == "generative":
        from astro_mine.prospect.backends.generative import GenerativeEnsembleField

        generative = GenerativeEnsembleField.from_prior(
            unit, prior_mean=0.0, prior_variance=1.0, seed=seed, **backend_config
        )
        # Its base is the constant N(0, 1) prior and its flow is standardized by construction, so a
        # warped draw over a correlated latent is already zero-mean / unit-variance at every cell.
        return generative.realize(seed=seed, correlation_length_m=length_m)

    raise ValueError(
        f"unknown ground-truth realization backend {kind!r}; known kinds are {REALIZATION_KINDS}"
    )
