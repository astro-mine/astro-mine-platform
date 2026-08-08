"""Per-instrument sensor likelihoods — the ``sensors/`` extension point (prospect.md §3, §6).

The **one** observation model a scenario's instruments are described by, used in both directions so
Sim's forward sensor simulation and Prospect's belief updating cannot drift apart:

- :class:`SensorLikelihood` — an instrument's footprint, depth response, and noise model, frozen and
  content-addressable (see :mod:`~astro_mine.prospect.sensors._likelihood` for the measurement model
  and why a depth-response *gain* — not just a scalar sigma — is what keeps a belief honest);
- a **registry** — :func:`register_likelihood` / :func:`get_likelihood` / :func:`list_likelihoods`,
  so an observation names its instrument (:attr:`~astro_mine.prospect.belief.observation.\\
  FieldObservation.likelihood`) and :meth:`~astro_mine.prospect.belief.field.BeliefField.update`
  resolves the matching model — no hard-coded instrument list in the conditioner;
- a **Core plugin manifest** seam — :func:`build_likelihood_manifest` /
  :func:`likelihood_from_manifest` (kind ``observation_model``), registered on the
  ``astro_mine.providers`` entry point so Sim resolves the pinned instrument model without importing
  this package by name (conventions.md §1.1).

The anchor scenario's set (scenario §6) — ``neutron_spectrometer``, ``nir_reflectance``, ``gpr``,
``drill_assay`` — plus the ``point_gaussian`` default, are registered at import
(:mod:`~astro_mine.prospect.sensors._catalog`). A third-party instrument registers alongside them
with **no change** to the conditioner, the field service, or Core.

Backlog: prospect.md §3, §6; LUNAR-FR-002 —
astro-mine-prospect#31
"""

from __future__ import annotations

from astro_mine.prospect.sensors._catalog import BUILTIN_LIKELIHOODS
from astro_mine.prospect.sensors._likelihood import (
    DEFAULT_LIKELIHOOD_NAME,
    DEFAULT_PROFILE,
    REFERENCE_COLUMN_DEPTH_M,
    DepthResponse,
    SensorLikelihood,
    VerticalProfile,
)
from astro_mine.prospect.sensors._manifest import (
    OBSERVATION_MODEL_ENTRY_POINT,
    OBSERVATION_MODEL_INTERFACE,
    OBSERVATION_MODEL_INTERFACE_VERSION,
    build_likelihood_manifest,
    likelihood_content_hash,
    likelihood_from_manifest,
)

__all__ = [
    "BUILTIN_LIKELIHOODS",
    "DEFAULT_LIKELIHOOD_NAME",
    "DEFAULT_PROFILE",
    "OBSERVATION_MODEL_ENTRY_POINT",
    "OBSERVATION_MODEL_INTERFACE",
    "OBSERVATION_MODEL_INTERFACE_VERSION",
    "REFERENCE_COLUMN_DEPTH_M",
    "DepthResponse",
    "SensorLikelihood",
    "VerticalProfile",
    "build_likelihood_manifest",
    "default_likelihood",
    "get_likelihood",
    "is_registered",
    "likelihood_content_hash",
    "likelihood_from_manifest",
    "list_likelihoods",
    "register_likelihood",
    "resolve_likelihood",
]

_LIKELIHOODS: dict[str, SensorLikelihood] = {}


def register_likelihood(likelihood: SensorLikelihood, *, replace: bool = False) -> None:
    """Register ``likelihood`` under its :attr:`~SensorLikelihood.name` (loud on a duplicate).

    The registration seam of the ``sensors/`` extension point: a new instrument becomes selectable
    by name — from an observation tag, a scenario spec, or a resolved plugin manifest — with no edit
    to the belief conditioner. ``replace=True`` overrides an existing registration (a scenario
    re-tuning a built-in instrument's noise/footprint for its own fleet).
    """
    if not replace and likelihood.name in _LIKELIHOODS:
        raise ValueError(
            f"sensor likelihood {likelihood.name!r} is already registered "
            "(pass replace=True to override it)"
        )
    _LIKELIHOODS[likelihood.name] = likelihood


def get_likelihood(name: str) -> SensorLikelihood:
    """Resolve a registered sensor likelihood by ``name`` (``ValueError`` if unknown)."""
    try:
        return _LIKELIHOODS[name]
    except KeyError:
        known = ", ".join(sorted(_LIKELIHOODS)) or "(none)"
        raise ValueError(
            f"unknown sensor likelihood {name!r}; registered likelihoods are: {known}"
        ) from None


def resolve_likelihood(name: str | None) -> SensorLikelihood:
    """Resolve ``name``, or the :func:`default_likelihood` when it is ``None``.

    The conditioner's single lookup: an observation carrying no instrument tag is conditioned under
    the zero-footprint, unit-gain :data:`DEFAULT_LIKELIHOOD_NAME` model, which reproduces the
    Phase-0 scalar-sigma behavior exactly.
    """
    return default_likelihood() if name is None else get_likelihood(name)


def default_likelihood() -> SensorLikelihood:
    """The default (zero-footprint, unit-gain) likelihood an untagged observation is read under."""
    return _LIKELIHOODS[DEFAULT_LIKELIHOOD_NAME]


def is_registered(name: str) -> bool:
    """Whether ``name`` names a registered sensor likelihood."""
    return name in _LIKELIHOODS


def list_likelihoods() -> tuple[str, ...]:
    """The names of all registered sensor likelihoods, sorted."""
    return tuple(sorted(_LIKELIHOODS))


for _builtin in BUILTIN_LIKELIHOODS:
    register_likelihood(_builtin)
