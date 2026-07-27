"""Provenance for dataset-derived priors: cited sources + a content hash (prospect.md §2.4, §5).

"Priors are sourced and cited, not invented" (prospect.md principle 4): every default prior records
the public datasets it derives from and a deterministic content hash, so a field with a prior is
never an uncited research artifact. :class:`DatasetCitation` is one cited public dataset;
:class:`Provenance` is the record a fitted prior carries — the recipe, its citations, the
derivation, and the numeric knobs — content-addressed for reproducibility (conventions.md §5).

Backlog: RM-P0-PROSPECT-03 — https://github.com/astro-mine/astro-mine-prospect/issues/3
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DatasetCitation", "Provenance"]


class _Model(BaseModel):
    """Frozen base for the provenance models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetCitation(_Model):
    """A cited public dataset a prior is derived from (prospect.md §2.4).

    ``short_name`` is the instrument/dataset tag (e.g. ``LCROSS``); ``instrument`` and ``mission``
    name the source; ``product`` is the PDS product / DOI / URL identifier; ``reference`` the
    published derivation citation; ``role`` what the dataset contributes to the fit.

    ``source_hash`` is the content hash of the *ingested raster* — ``None`` for the Phase-0
    parametric recipe, which cites each dataset's **published characterization** rather than
    ingesting the raster. The real raster-ingest recipe (RM-P1-PROSPECT-12, #11) fills it in.
    """

    short_name: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    mission: str = Field(min_length=1)
    product: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_hash: str | None = None


class Provenance(_Model):
    """The provenance a derived prior carries: the recipe, its cited datasets, and a content hash.

    ``recipe`` + ``recipe_version`` identify the deterministic derivation; ``citations`` the public
    datasets it draws on (at least one — an uncited prior is forbidden); ``derivation`` a one-line
    human summary; ``params`` the recipe's numeric knobs, so the fit is reconstructable.
    :attr:`content_hash` is a stable SHA-256 digest over the canonicalized record — the content
    address that makes a prior reproducible from its cited inputs (conventions.md §5).
    """

    recipe: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    citations: tuple[DatasetCitation, ...] = Field(min_length=1)
    derivation: str = Field(min_length=1)
    params: Mapping[str, float] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """A deterministic SHA-256 over the canonicalized provenance (no clock, no environment)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
