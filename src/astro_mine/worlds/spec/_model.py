"""WorldSpec — the declarative description of a world (RM-P0-WORLDS-07; worlds.md §3, §5).

A :class:`WorldSpec` names a *place* and how to derive it: the body CRS, the region/resolution,
the source DEM (referenced by content hash), and which derived layers (regolith / illumination /
thermal) are enabled with what parameters. It is authored as **YAML, validated by JSON Schema**
(Pydantic v2 supplies both the validation and ``model_json_schema``); its **canonical JSON** form
is the load-bearing artifact for content-addressing, so the spec hash is independent of YAML
formatting. A ``WorldSpec`` plus its resolved component hashes is what the bundle hashes into a
content-addressed world ID (worlds.md §5; :mod:`~astro_mine.worlds.spec._bundle`).

Backlog: RM-P0-WORLDS-07 — astro-mine-worlds#7
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.core.units import PlanetaryCRS, require_crs

__all__ = [
    "JSON_SCHEMA_DIALECT",
    "WORLDSPEC_SCHEMA_ID",
    "LayerSpec",
    "Region",
    "SourceRef",
    "WorldSpec",
]

#: The absolute ``$id`` a ``WorldSpec`` JSON Schema is referenced by — public, append-only API
#: (RFC-0009 rule 1). The namespace mirrors the two schemas Worlds already publishes under
#: ``schemas.astro-mine.org/worlds/…``; the ``v0.1`` segment is the schema's version, which moves
#: only when the format does.
WORLDSPEC_SCHEMA_ID = "https://schemas.astro-mine.org/worlds/spec/v0.1/worldspec.schema.json"

#: The JSON Schema dialect the generated document conforms to. Stated explicitly so a validator
#: does not have to guess, and so a dialect change is a visible diff rather than a silent one.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class _Model(BaseModel):
    """Frozen base for the spec models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Region(_Model):
    """A CRS-projected rectangular region (metres) at a target resolution.

    The half-open box ``[min, max)`` in the world CRS's projected metres, plus the intended
    ground sample distance. The realized grid (width/height/transform) comes from the terrain
    product; this is the *declared* extent the spec is authored against.
    """

    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float
    resolution_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_extent(self) -> Region:
        if self.max_x_m <= self.min_x_m or self.max_y_m <= self.min_y_m:
            raise ValueError("region max_x/max_y must exceed min_x/min_y")
        return self


class SourceRef(_Model):
    """A reference to a source input dataset (e.g. the LOLA DEM), cited by content hash.

    ``content_hash`` pins the exact input so a world is reproducible from its sources; it is
    ``None`` only for a synthetic/illustrative source (e.g. the CI stand-in DEM).
    """

    id: str = Field(min_length=1)
    content_hash: str | None = None
    description: str | None = None


class LayerSpec(_Model):
    """Which derived layers the world enables, and their parameters.

    Terrain is always present (it establishes the grid). Each of the others is opt-in: a value of
    ``None``/empty means the layer is not part of this world. The bundle build is given the matching
    products and validates them against this declaration.
    """

    regolith_prior: str | None = None
    illumination_n_azimuth: int | None = Field(default=None, gt=0)
    psr_semantics: str | None = None
    thermal_classes: tuple[str, ...] = ()
    #: Which Sun-visibility backend the world uses (RM-P1-WORLDS-10; worlds.md §11) —
    #: ``"horizon"`` (the precomputed map) | ``"raycast_cpu"`` | ``"raycast_gpu"`` |
    #: ``"surrogate:<name>"`` (a learned field surrogate). ``None`` selects the horizon default, so
    #: an existing spec keeps its behaviour; a non-default backend is folded into the world hash.
    illumination_backend: str | None = None
    # The PSR-mask-determining illumination parameters (issue #36). The world bundle folds the
    # SPICE-derived PSR mask into ``world_hash``, but the mask is a function of these parameters, so
    # recording them here makes ``spec_hash`` — hence ``world_hash`` — reproducible from the
    # declaration alone (worlds.md §10 determinism gate; §5 / conventions.md §5 provenance). Each
    # defaults to ``None`` (the model/SPICE default), so a spec that omits one is unconstrained and
    # existing worlds keep their behaviour; ``IlluminationModel.from_spec`` reads them back.
    #: The horizon frame the mask is computed in (``"grid"`` | ``"topocentric"``; the
    #: :class:`~astro_mine.worlds.illumination.HorizonFrame` value). ``grid`` resolves one
    #: region-centre Sun across the raster; ``topocentric`` resolves it per cell — the two disagree
    #: on shadow at the region edges, so the frame must be declared for the mask to be reproducible.
    illumination_horizon_frame: str | None = None
    #: The horizon search cap (metres): how far terrain may rise to block the Sun. ``None`` selects
    #: the model default (the grid half-extent).
    illumination_max_radius_m: float | None = Field(default=None, gt=0.0)
    #: The SPICE aberration correction applied to the Sun geometry (e.g. ``"NONE"``, ``"LT+S"``).
    #: ``None`` selects the SPICE default; it shifts the Sun elevation/azimuth, hence the mask.
    illumination_abcorr: str | None = None
    #: PSR sampling window start (ISO-8601 UTC). The mask is "never sunlit over the sampled window",
    #: so the window start/duration and step define it.
    psr_start: str | None = None
    #: PSR sampling window duration (days) and step (hours) — with :attr:`psr_start`, the window and
    #: cadence the shadow OR is accumulated over.
    psr_days: float | None = Field(default=None, gt=0.0)
    psr_step_hours: float | None = Field(default=None, gt=0.0)


class WorldSpec(_Model):
    """The declarative description of a world — authored as YAML, hashed as canonical JSON.

    Construct from YAML (:meth:`from_yaml` / :meth:`from_yaml_text`) or directly; serialize back
    with :meth:`to_yaml`. :attr:`spec_hash` content-addresses the declaration; the full world hash
    (declaration + resolved component hashes + toolchain) is computed by the bundle build.
    """

    world_id: str = Field(min_length=1)
    version: str = "0.1.0"
    crs: PlanetaryCRS
    region: Region
    source_dem: SourceRef
    layers: LayerSpec = Field(default_factory=LayerSpec)
    #: A fixed reference datetime (ISO-8601) stamped onto STAC items — deterministic by design
    #: (never wall-clock), so the catalog is byte-reproducible.
    reference_datetime: str = "2025-01-01T00:00:00Z"
    description: str | None = None

    @model_validator(mode="after")
    def _require_explicit_crs(self) -> WorldSpec:
        # Adopt Core's require_crs at the authoring/emit boundary (RM-P1-WORLDS-17, RM-P1-CORE-08):
        # rule 4 (a present, finite, positive-radius CRS) and rule 6 (an Earth datum/projection
        # marker — WGS84 / EPSG:4326 / urn:ogc:def:crs:OGC — is a defaulting bug on a non-EARTH
        # body) are enforced HERE, so an implicit Earth CRS is rejected before world.json is written
        # rather than by View at ingest (conventions.md §5; RFC-0007 Motivation §5).
        require_crs(self.crs)
        return self

    @classmethod
    def from_yaml_text(cls, text: str) -> WorldSpec:
        """Parse a :class:`WorldSpec` from a YAML document string."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("WorldSpec YAML must be a mapping at the top level")
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorldSpec:
        """Load a :class:`WorldSpec` from a YAML file."""
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        """Serialize to a stable (key-sorted) YAML document."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, allow_unicode=True)

    @property
    def canonical_json(self) -> str:
        """The canonical, key-sorted, compact JSON form — the basis for content-addressing."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def spec_hash(self) -> str:
        """A deterministic ``sha256:`` digest over the canonical declaration."""
        return f"sha256:{hashlib.sha256(self.canonical_json.encode('utf-8')).hexdigest()}"

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The JSON Schema for authoring/validating a WorldSpec (worlds.md §5, §11).

        **Self-identifying.** The schema carries its ``$id`` and declares its dialect, so it can be
        referenced, cached and ``$ref``-ed by absolute name rather than by whatever path a consumer
        happened to fetch it from — RFC-0009's "one name" rule, in the namespace Worlds's
        illumination schemas already use. Without an ``$id`` a published schema is a file, not a
        contract: two copies at two paths are indistinguishable and neither can be pointed at.

        **Derived, not authored.** The Pydantic model is the source of truth here — the opposite of
        Core, whose hand-written JSON Schemas are authoritative and whose models mirror them. So
        this is generated on demand, and the copy shipped as package data
        (:mod:`astro_mine.worlds.spec._schema`) is checked against it by a test rather than
        maintained by hand.
        """
        schema = dict(cls.model_json_schema())
        # Prepended rather than appended: `$schema` and `$id` conventionally lead a schema
        # document, and the shipped file is written from this dict, so the order is what a reader
        # sees on disk.
        return {
            "$schema": JSON_SCHEMA_DIALECT,
            "$id": WORLDSPEC_SCHEMA_ID,
            **schema,
        }
