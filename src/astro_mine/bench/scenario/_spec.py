# SPDX-License-Identifier: Apache-2.0
"""ScenarioSpec — the versioned, content-addressed benchmark task (bench.md §3, §5).

A :class:`ScenarioSpec` *is* the task: it pins the Core interface version, references the
Worlds/Fleet/Prospect/Link content it runs against **by content hash**, and fixes the seeds,
episode/horizon, termination, metric set, and per-submission budgets. Authored as JSON/YAML and
validated by JSON Schema (Pydantic v2 supplies both validation and ``model_json_schema``); its
**canonical JSON** form is the load-bearing artifact for content-addressing, so its
``spec_hash`` is independent of authoring formatting. Changing *any* pinned input changes the
hash — the task is frozen and the result is reproducible (bench.md §2; conventions.md §5).

The house pattern is Worlds' ``WorldSpec`` (``astro_mine.worlds.spec``): a frozen,
``extra="forbid"`` Pydantic base, content referenced by hash (Worlds' ``SourceRef``), and a
``sha256:`` canonical digest.

Backlog: RM-P0-BENCH-01 — astro-mine-bench#1
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astro_mine.bench.scenario._hash import canonical_json, content_hash, normalize_sha256
from astro_mine.core.compat import parse_version
from astro_mine.core.units import Epoch

__all__ = [
    "BudgetSpec",
    "ContentPins",
    "ContentRef",
    "EpisodeSpec",
    "LatLonRegion",
    "MetricRef",
    "PlacementSpec",
    "ScenarioSpec",
    "ScoringSpec",
    "SeedSet",
    "SitePlacement",
    "TerminationSpec",
]


class _Model(BaseModel):
    """Frozen base for the spec models: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContentRef(_Model):
    """A reference to referenced content (a world, an asset, a resource field, a link plan).

    ``content_hash`` pins the exact input by its ``sha256:`` digest so a scenario is reproducible
    from its content; ``id`` is the stable content identity (e.g. a Fleet ``identity.id`` or a
    Worlds ``world_id``). The hash is normalized to the platform ``sha256:<hex>`` form, accepting
    either the prefixed (Worlds/Fleet) or bare-hex (Prospect) producer conventions.
    """

    id: str = Field(min_length=1)
    content_hash: str
    description: str | None = None

    @field_validator("content_hash")
    @classmethod
    def _normalize_hash(cls, value: str) -> str:
        return normalize_sha256(value)


class ContentPins(_Model):
    """The content a scenario pins by hash: one world, its fleet, and optional fields/comms.

    ``world`` and at least one ``fleet`` asset are required — a scenario has a place and robots.
    ``prospect`` (resource fields) is optional (a pure-mobility task may omit it); ``link`` is
    optional and pins the comms model — a published, content-addressed Link ``ContactPlan`` bundle
    — for scenarios whose task depends on contact windows (LUNAR-TR-003).
    """

    world: ContentRef
    fleet: tuple[ContentRef, ...] = Field(min_length=1)
    prospect: tuple[ContentRef, ...] = ()
    link: ContentRef | None = None


class SeedSet(_Model):
    """The scenario's random seeds: a public dev set plus an optional held-out commitment.

    ``public`` seeds are disclosed for development. The embargoed held-out set (disclosed only at
    evaluation time — bench.md §9) is bound here as ``heldout_commit``, a ``sha256:`` commitment to
    the sealed seeds, so it influences the spec hash without leaking the seeds. The seeds themselves
    land with the anchor scenario (BENCH-02).
    """

    public: tuple[int, ...] = Field(min_length=1)
    heldout_commit: str | None = None

    @field_validator("heldout_commit")
    @classmethod
    def _normalize_commit(cls, value: str | None) -> str | None:
        return None if value is None else normalize_sha256(value)


class EpisodeSpec(_Model):
    """*When* the episode runs and how long: a start epoch, a step horizon, a sim-time cap.

    ``start_epoch`` is the instant the episode's clock begins, as a typed Core
    :class:`~astro_mine.core.units.Epoch` (RFC-0007: epochs travel on the wire with their scale, not
    as bare floats).

    It belongs to the **task**, not the runner, and its absence was a real defect. A scenario that
    pins a ``ContactPlan`` pins a *plan over a window of time*; whether that plan applies at all
    depends on **when the episode runs**. With no epoch in the spec, Sim fell back to its own
    default (``J2000_EPOCH``, TDB 0.0) while the anchor's plan covers a 24 h window at TDB 946728000
    (2030-01-01) — thirty years apart. Every contact interval was therefore inactive at every tick,
    `earth_contact` was false forever, and ``comms_robustness`` scored a confident **0.0** rather
    than failing. A number with the shape of a result and none of the content (astro-mine-bench#48).

    ``None`` means the runner picks the epoch — the pre-existing behaviour, and correct for a
    scenario that pins no time-dependent content. A scenario that pins a ``link`` ref should declare
    it, and Sim refuses a plan whose window does not cover the episode rather than silently scoring
    zero.
    """

    horizon_steps: int = Field(gt=0)
    max_sim_seconds: float | None = Field(default=None, gt=0.0)
    start_epoch: Epoch | None = None


class TerminationSpec(_Model):
    """When an episode ends early: named termination conditions and an optional hard step cap.

    An empty spec means the episode terminates at the :class:`EpisodeSpec` horizon. Conditions are
    named predicates evaluated by the harness (BENCH-04); this spec only pins *which* apply.
    """

    conditions: tuple[str, ...] = ()
    max_steps: int | None = Field(default=None, gt=0)


class MetricRef(_Model):
    """A reference to a scored metric plugin by name and interface version.

    The metric *implementations* (units/direction/uncertainty) are BENCH-03 plugins; a scenario
    only pins which metrics score it, at which version.
    """

    name: str = Field(min_length=1)
    version: str = "0.1.0"

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        parse_version(value)  # fail loud on a non-semver metric version
        return value


class BudgetSpec(_Model):
    """Per-submission resource budgets (bench.md §3): wall-clock, sim-step, and compute caps.

    Each is optional; ``None`` means unbounded (e.g. the always-works local tier — bench.md §6).
    """

    wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    sim_steps: int | None = Field(default=None, gt=0)
    compute_units: float | None = Field(default=None, gt=0.0)


class SitePlacement(_Model):
    """Where one pinned fleet asset stands, in the world's body-fixed frame.

    ``asset`` is the :class:`ContentRef` ``id`` of a pinned fleet asset (e.g.
    ``excavator``), so placement is keyed by the same vocabulary the content
    pins and the [Link](link.md) contact plan's nodes use — one robot, one name, everywhere.

    Coordinates are **planetocentric** and describe the asset's *body origin on the terrain*,
    not an antenna phase centre: ``elevation_m`` is terrain height above the body's mean radius,
    and any mast height belongs to the asset's own SADF. ``elevation_m`` may be omitted, in
    which case the runner samples the pinned world's terrain at ``(lat, lon)`` — the DEM is
    already pinned by hash, so the result is reproducible without restating it here.
    """

    asset: str = Field(min_length=1)
    lat_deg: float = Field(ge=-90.0, le=90.0)
    lon_deg: float = Field(ge=-180.0, lt=360.0)
    elevation_m: float | None = None

    @field_validator("lon_deg")
    @classmethod
    def _normalize_longitude(cls, value: float) -> float:
        """Fold east longitude into ``[0, 360)`` so one site has exactly one content address."""
        return value % 360.0


class LatLonRegion(_Model):
    """A body-fixed latitude/longitude window, as inclusive ``(min, max)`` degree bounds.

    Longitude bounds are *not* wrapped: a window crossing the prime meridian is authored as
    ``lon_deg: [350.0, 370.0]``, which keeps ``min < max`` true and the region unambiguous.
    """

    lat_deg: tuple[float, float]
    lon_deg: tuple[float, float]

    @field_validator("lat_deg")
    @classmethod
    def _check_latitude(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not -90.0 <= low < high <= 90.0:
            raise ValueError(f"lat_deg must be increasing bounds within [-90, 90], got {value!r}")
        return value

    @field_validator("lon_deg")
    @classmethod
    def _check_longitude(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not low < high:
            raise ValueError(f"lon_deg must be increasing bounds, got {value!r}")
        if high - low > 360.0:
            raise ValueError(f"lon_deg spans more than a full revolution: {value!r}")
        return value


class PlacementSpec(_Model):
    """Where the swarm stands — the siting the scenario pins (bench.md §5).

    Without this block a runner is free to invent a layout, and Sim's does: it spreads surface
    assets on a fixed-radius ring whose angular jitter is derived from ``scenario_hash``
    (``astro_mine.sim.bench._scenario._layout``). That makes placement a function of the content
    digest, so a re-pin *moves the swarm* and every position-dependent metric moves with it —
    and it contradicts the pinned contact plan, which is computed against a deliberate siting.
    Pinning placement here makes siting a property of the **task**, as reproducible as the
    content it runs against (CX-REPRO; conventions.md §5).

    Assets a scenario pins but does not place are left to the runner's own layout, so a
    scenario may site the assets whose position it cares about and leave the rest.
    """

    sites: tuple[SitePlacement, ...] = Field(min_length=1)

    @field_validator("sites")
    @classmethod
    def _check_unique_assets(cls, value: tuple[SitePlacement, ...]) -> tuple[SitePlacement, ...]:
        seen = [site.asset for site in value]
        duplicates = sorted({asset for asset in seen if seen.count(asset) > 1})
        if duplicates:
            raise ValueError(f"placement sites must be unique per asset; repeated: {duplicates}")
        return value


class ScoringSpec(_Model):
    """Scenario-pinned scoring parameters for the belief-quality and discovery metrics.

    Bench owns these — they are not a Core schema — and they have always been expressible only
    as constructor arguments to a runner, never as part of the task. That leaves their defaults
    load-bearing in a way the scenario cannot see or override, and two of those defaults are
    actively hostile: ``characterized_variance_threshold`` defaults to ``0.0``, which no
    posterior variance can satisfy, so ``psr_area_characterized`` reports a confident ``0.0``
    m² rather than abstaining; and ``discovery_threshold`` defaults to ``0.0``, which any valid
    non-negative reading trips at tick 0, so ``discovery_latency`` reports a discovery that did
    not happen. Every field here is therefore ``None`` by default, meaning *"the scenario does
    not pin this"* — distinct from a pinned zero, and distinct from a runner's fallback.

    ``psr_region`` states the permanently-shadowed extent **geometrically** rather than as a set
    of cell ids. The metric consumes opaque cell ids, but no cell-id convention is shared
    between Bench and [Prospect](prospect.md) yet (astro-mine-sim#66), so pinning ids here would
    freeze a convention that has not been chosen. A region is convention-independent: whatever
    resolves the region to cells is the same code that builds the belief history, so the two
    agree by construction.
    """

    psr_region: LatLonRegion | None = None
    cell_area_m2: float | None = Field(default=None, gt=0.0)
    #: A PSR cell counts as characterized once its posterior variance is at or below this.
    #: Must be strictly positive — ``information_gain`` treats a non-positive variance as an
    #: error, so a threshold of zero is unsatisfiable rather than merely strict.
    characterized_variance_threshold: float | None = Field(default=None, gt=0.0)
    #: A sensor reading of the discovery species at or above this counts as a detection.
    discovery_threshold: float | None = Field(default=None, gt=0.0)


class ScenarioSpec(_Model):
    """The declarative benchmark task — authored as JSON/YAML, hashed as canonical JSON.

    :attr:`spec_hash` content-addresses the whole declaration; the resolved, Core-validated
    scenario identity is computed by :func:`~astro_mine.bench.scenario._resolve.resolve_scenario`.
    """

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    spec_version: str = "0.1.0"
    description: str | None = None
    #: The Core interface versions the scenario is pinned to (interface name → SemVer), e.g.
    #: ``{"env": "0.1.0", "messages": "0.1.0"}``. Validated for well-formedness here and for
    #: compatibility with the installed Core at resolve time (``astro_mine.core.compat``).
    core_interface: dict[str, str]
    #: The Core **schema set** the scenario is pinned to, by digest —
    #: ``astro_mine.core.SCHEMA_DIGEST`` (``VERSIONING.md`` §4.1). ``core_interface`` above cannot
    #: do this job: every interface is frozen at ``0.1.0`` through Phase 3, so it is constant across
    #: every Core revision and carries no information. The schema digest *is* the contract identity
    #: — two Core revisions with materially different schemas have different digests — and
    #: :func:`~astro_mine.bench.scenario.resolve_scenario` fails loud when the pin disagrees with
    #: the installed Core. Optional: a scenario may omit it (an older spec, or one authored against
    #: a non-Python binding), and then reproducibility rests on the toolchain lockfile alone — an
    #: *environment* pin, not a *contract* pin (``VERSIONING.md`` §4, mechanism 1).
    core_schema_digest: str | None = None
    content: ContentPins
    seeds: SeedSet
    episode: EpisodeSpec
    termination: TerminationSpec = Field(default_factory=TerminationSpec)
    metrics: tuple[MetricRef, ...] = Field(min_length=1)
    budgets: BudgetSpec = Field(default_factory=BudgetSpec)
    #: Where the pinned assets stand. Omitted means the runner picks a layout — see
    #: :class:`PlacementSpec` for why that is not a neutral default.
    placement: PlacementSpec | None = None
    #: Scenario-pinned scoring parameters (:class:`ScoringSpec`). Omitted means every metric
    #: falls back to its runner-side default, including the two hostile ones.
    scoring: ScoringSpec | None = None

    @field_validator("spec_version")
    @classmethod
    def _check_spec_version(cls, value: str) -> str:
        parse_version(value)
        return value

    @field_validator("core_schema_digest")
    @classmethod
    def _normalize_schema_digest(cls, value: str | None) -> str | None:
        return None if value is None else normalize_sha256(value)

    @field_validator("core_interface")
    @classmethod
    def _check_interface_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("core_interface must pin at least one Core interface version")
        for name, version in value.items():
            if not name:
                raise ValueError("core_interface keys must be non-empty interface names")
            parse_version(version)  # fail loud on a non-semver interface version
        return value

    @model_validator(mode="after")
    def _check_placement_references_pinned_fleet(self) -> ScenarioSpec:
        """Every placed asset must be one the scenario actually pins.

        A site naming an asset outside ``content.fleet`` is silently inert otherwise — the
        runner places what it resolves, and a typo'd or stale id simply never matches. Failing
        here makes that a spec error instead of an unexplained default layout.
        """
        if self.placement is None:
            return self
        pinned = {ref.id for ref in self.content.fleet}
        unknown = sorted({site.asset for site in self.placement.sites} - pinned)
        if unknown:
            raise ValueError(
                f"placement sites reference assets the scenario does not pin: {unknown}; "
                f"pinned fleet assets are {sorted(pinned)}"
            )
        return self

    def content_refs(self) -> tuple[ContentRef, ...]:
        """Every pinned content reference, in a stable order (world, fleet, prospect, link)."""
        refs: list[ContentRef] = [self.content.world, *self.content.fleet, *self.content.prospect]
        if self.content.link is not None:
            refs.append(self.content.link)
        return tuple(refs)

    @property
    def canonical_json(self) -> str:
        """The canonical, key-sorted, compact JSON form — the basis for content-addressing.

        **Defaulted fields are excluded.** A spec that does not exercise an optional block
        serializes — and therefore hashes — as though the block did not exist, so *adding* an
        optional field to this model leaves every existing scenario's ``spec_hash`` unchanged.
        Without this, appending a field would re-identify every historical spec (``model_dump``
        emits ``None`` defaults), which contradicts bench.md §8: the zoo grows by adding
        immutable specs *"so historical leaderboards never need recomputation"*.

        The consequence to author against: a field explicitly set to its own default is
        indistinguishable from an omitted one, and **changing a declared default silently
        re-identifies every spec that omits that field**. New optional fields therefore default
        to ``None`` — a stable sentinel that never carries meaning — rather than to a live value.
        """
        return canonical_json(self.model_dump(mode="json", exclude_defaults=True))

    @property
    def spec_hash(self) -> str:
        """A deterministic ``sha256:`` digest over the canonical declaration.

        Digests exactly what :attr:`canonical_json` serializes — see there for why defaulted
        fields are excluded from the content address.
        """
        return content_hash(self.model_dump(mode="json", exclude_defaults=True))

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The JSON Schema for authoring/validating a ScenarioSpec (bench.md §5)."""
        return cls.model_json_schema()
