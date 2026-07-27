"""Resource-field API v0.1 — the contract (prospect.md §2/§3).

The Core-owned, uncertainty-first probabilistic resource-field contract: a query surface
over a (position[, epoch]) that returns *distributions*, not guesses. Both Prospect's
ground-truth realizations and its evolving belief/posterior fields implement this one
contract (prospect.md §2.2), so a planner runs identically against a synthetic world and a
live estimate. The geostatistical backend — GP, GMRF, deep-generative, grid — is a plugin
behind this Core interface (prospect.md §2.7); Core owns only the *shape*.

Two properties are load-bearing and deliberately encoded in the *absence* of methods:

- **No point-estimate-only accessor.** Every query carries uncertainty: ``variance`` /
  ``quantile`` / ``sample`` sit alongside ``mean``, and :meth:`ResourceField.posterior`
  returns a :class:`~astro_mine.core.resource.model.FieldDistribution` that always pairs a
  mean with a calibrated uncertainty (prospect.md §2.1 — "a point-estimate-only API is
  forbidden").
- **No ground-truth accessor.** This agent-facing surface exposes no way to read the sealed
  ground-truth field; ground-truth isolation is the package's key safety property
  (prospect.md §9). A ``GroundTruthField`` is a distinct, access-gated type that *implements*
  this contract for Sim, never reached through it by a policy.

Core defines no physics, no inference, and no IO — those are Prospect's.
:func:`~astro_mine.core.resource.check_resource_field` is the consumer-driven contract test
an implementor runs in its own CI.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_mine.core.resource.model import FieldDistribution, Position
from astro_mine.core.units import Epoch, ReferenceFrame

__all__ = ["ResourceField"]


@runtime_checkable
class ResourceField(Protocol):
    """A probabilistic resource field over a world's spatial domain (prospect.md §3).

    Implemented identically by ground-truth and belief variants. Every spatial query is in
    :attr:`frame` (SI metres); an absolute ``epoch`` is an optional
    :class:`~astro_mine.core.units.Epoch` for time-varying fields. Queries are distributional
    by construction — there is no method that returns a value without its uncertainty.
    """

    @property
    def species(self) -> str:
        """The resource species the field models (e.g. ``water_equivalent_hydrogen``)."""
        ...

    @property
    def unit(self) -> str:
        """The field's SI unit token (e.g. ``mass_fraction``)."""
        ...

    @property
    def frame(self) -> ReferenceFrame:
        """The reference frame queried positions resolve in (the field's CRS binding)."""
        ...

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        """The posterior mean at ``position`` (paired with :meth:`variance`)."""
        ...

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        """The posterior variance at ``position`` — the calibrated uncertainty."""
        ...

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        """The value at quantile level ``q`` (in ``[0, 1]``) at ``position``."""
        ...

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        """Draw ``n`` seeded samples of the field at ``position`` (a Monte-Carlo realization)."""
        ...

    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        """The full distributional summary at ``position`` — mean, uncertainty, and quantiles."""
        ...
