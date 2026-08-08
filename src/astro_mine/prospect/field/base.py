"""BaseResourceField — the shared base making ground-truth and belief identical (prospect.md §2.2).

Sealed ground-truth realizations and evolving belief posteriors are *distinct types with the
same interface* (prospect.md §2.2). This base is where that "same interface" is made literal:
it holds the field :class:`~astro_mine.prospect.field.metadata.FieldMetadata`, derives the
Core :class:`~astro_mine.core.resource.ResourceField` contract's ``species``/``unit``/``frame``
from it, and composes the uncertainty-first :meth:`~BaseResourceField.posterior` from the
subclass's distributional accessors — so there is, by construction, no point-estimate-only path
through a Prospect field (prospect.md §2.1). Concrete fields and inference backends land later
(RM-P0-PROSPECT-02/04); this item delivers only the shared contract surface.

Backlog: RM-P0-PROSPECT-01 — astro-mine-prospect#1
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from astro_mine.core.resource import FieldDistribution, Position
from astro_mine.core.units import Epoch, ReferenceFrame
from astro_mine.prospect.field.metadata import FieldMetadata

__all__ = ["DEFAULT_QUANTILES", "BaseResourceField"]

#: The quantile levels the default :meth:`BaseResourceField.posterior` reports — a symmetric
#: 90% credible interval around the median. A subclass MAY report more, never fewer.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


class BaseResourceField(ABC):
    """Shared base for every Prospect resource field — ground-truth and belief alike.

    A concrete subclass implements only the four distributional accessors
    (:meth:`mean`, :meth:`variance`, :meth:`quantile`, :meth:`sample`); the metadata-backed
    ``species``/``unit``/``frame`` properties and the uncertainty-first :meth:`posterior` are
    provided here, so the two variants satisfy the Core contract *identically*. Because the
    accessors are abstract, a partial field that exposes only a point estimate cannot be
    instantiated — the point-estimate-only API is forbidden structurally (prospect.md §2.1).
    """

    def __init__(self, metadata: FieldMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> FieldMetadata:
        """The field's species/unit and CRS/grid metadata."""
        return self._metadata

    @property
    def species(self) -> str:
        """The resource species the field models (Core ResourceField contract)."""
        return self._metadata.species

    @property
    def unit(self) -> str:
        """The field's SI unit token (Core ResourceField contract)."""
        return self._metadata.unit

    @property
    def frame(self) -> ReferenceFrame:
        """The reference frame queried positions resolve in (the CRS binding)."""
        return self._metadata.frame

    @abstractmethod
    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        """The posterior mean at ``position`` (always paired with :meth:`variance`)."""
        ...

    @abstractmethod
    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        """The posterior variance at ``position`` — the calibrated uncertainty."""
        ...

    @abstractmethod
    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        """The value at quantile level ``q`` (in ``[0, 1]``) at ``position``."""
        ...

    @abstractmethod
    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        """Draw ``n`` seeded samples of the field at ``position``."""
        ...

    def posterior(self, position: Position, *, epoch: Epoch | None = None) -> FieldDistribution:
        """The uncertainty-first distributional summary at ``position``.

        Composed from the subclass accessors: the mean paired with the variance plus the
        :data:`DEFAULT_QUANTILES`. A subclass MAY override for a richer summary, but never to
        drop the uncertainty — the return is always a
        :class:`~astro_mine.core.resource.FieldDistribution` pairing a mean with a variance.
        """
        return FieldDistribution(
            mean=self.mean(position, epoch=epoch),
            variance=self.variance(position, epoch=epoch),
            quantiles={q: self.quantile(position, q, epoch=epoch) for q in DEFAULT_QUANTILES},
            species=self._metadata.species,
            unit=self._metadata.unit,
        )
