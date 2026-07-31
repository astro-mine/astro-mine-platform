"""Zarr field/prior storage — the parametric / ensemble / quantile encodings (prospect.md §5).

prospect.md §5 (and conventions.md §5; ``LUNAR-DR-003``) name **Zarr** as *the* field store:
chunked, cloud-native, self-describing, and — crucially — carrying a **distribution axis**, so a
stored field is a distribution rather than a grid of numbers. Three encodings are specified, and a
field is tagged with the one it is in:

- **parametric** — ``mean`` + ``variance`` arrays. The default for the GP / GMRF / grid backends,
  whose per-point posterior is a closed-form Gaussian.
- **ensemble** — a stacked ``realization`` axis of N samples. The encoding for the deep-generative,
  non-Gaussian backend, whose marginal has no closed form.
- **quantile** — a ``quantile`` axis plus its levels. The compact encoding for downstream
  consumption: a planner that wants a P10/P50/P90, not a full posterior.

A field **round-trips through the encoding it was tagged with** (:class:`FieldArchive`): what is
written as an ensemble reads back as an ensemble, not as a mean/variance summary of one. That is the
whole point of tagging the uncertainty representation — a reader that silently collapsed an ensemble
to two moments would quietly discard the multimodality the ensemble exists to carry.

**Content addressing.** A Zarr store is a *directory* and Hub distributes *blobs*, so
:func:`serialize_zarr` packs the store into the same **deterministic tar** the ``.npy`` bundle uses
(sorted members; zeroed mtime/uid/gid): one content-addressed layer, one stable digest, byte-stable
across machines (hub.md §2.1). Both layer media types resolve through the *same* ``from_bundle``
entry point (:mod:`astro_mine.prospect.publish._bundle`), so a consumer that pulls a Zarr-encoded
field and one that pulls a tar-encoded field call identical code.

``zarr`` is an **optional** dependency (the ``zarr`` extra). The dependency-light ``.npy``-tar path
needs only Core + numpy, so the offline local tier (``LUNAR-TR-004``) never pulls it; this module
imports it lazily and says so plainly if it is absent.

Backlog: prospect.md §5, §10 — https://github.com/astro-mine/astro-mine-prospect/issues/33
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.field.metadata import FieldMetadata
from astro_mine.prospect.priors.provenance import Provenance
from astro_mine.prospect.priors.recipe import Prior

__all__ = [
    "DEFAULT_LEVELS",
    "ENCODINGS",
    "ZARR_MEDIA_TYPE",
    "Encoding",
    "FieldArchive",
    "archive_from_zarr_bytes",
    "quantile_grids",
    "read_zarr",
    "serialize_zarr",
    "write_zarr",
]

#: The OCI layer media type of a Zarr-encoded resource field — a deterministic tar of the store.
#: Distinct from the ``.npy`` bundle's media type, so a manifest declares which encoding it ships
#: and ``from_bundle`` resolves either one without guessing.
ZARR_MEDIA_TYPE = "application/vnd.astro-mine.resource-field.zarr.v1.tar"

#: The uncertainty representations a stored field may be tagged with (prospect.md §5).
Encoding = Literal["parametric", "ensemble", "quantile"]
ENCODINGS: tuple[Encoding, ...] = ("parametric", "ensemble", "quantile")

#: The Zarr array names and store attribute keys — the field-store schema, v1.
_MEAN = "mean"
_VARIANCE = "variance"
_REALIZATION = "realization"
_QUANTILE = "quantile"
_QUANTILE_LEVEL = "quantile_level"
_ATTR_ENCODING = "uncertainty_representation"
_ATTR_BACKEND = "backend"
_ATTR_METADATA = "field_metadata"
_ATTR_PROVENANCE = "provenance"
_ATTR_SPECIES = "species"
_ATTR_UNIT = "unit"

#: The quantile levels :func:`quantile_grids` reports by default — a P5/P25/P50/P75/P95 summary,
#: matching :data:`~astro_mine.prospect.field.base.DEFAULT_QUANTILES` so the compact encoding and a
#: live ``posterior()`` agree on which levels a consumer gets.
DEFAULT_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

#: The standard-normal interquartile range: ``z(0.75) - z(0.25)``. The IQR-to-sigma factor used when
#: a *quantile*-encoded field is read back as a Gaussian prior (see :meth:`FieldArchive.as_prior`).
_IQR_TO_SIGMA = 1.3489795003921634


def _zarr() -> Any:
    """Import ``zarr`` lazily, with an actionable message when the optional extra is missing."""
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - only reachable without the optional extra
        raise ImportError(
            "Zarr field storage needs the optional 'zarr' extra: "
            "pip install astro-mine-platform[prospect-zarr]"
        ) from exc
    return zarr


@dataclass(frozen=True)
class FieldArchive:
    """A resource field as it is **stored**: the arrays, the encoding they are in, and provenance.

    The in-memory form of a Zarr store. It is deliberately *not* a ``ResourceField``: a store holds
    a distribution in one of three encodings, and collapsing that into a queryable field is a lossy
    decision the **consumer** makes (:meth:`as_prior` / :meth:`as_field`), not one the reader makes
    silently on their behalf.

    Build one with :meth:`parametric`, :meth:`ensemble_encoded`, or :meth:`quantile_encoded`; each
    encoding's invariants are checked on construction, so an archive is always well-formed.
    """

    metadata: FieldMetadata
    provenance: Provenance
    encoding: Encoding
    backend: str = "grid"
    mean: NDArray[np.float64] | None = None
    variance: NDArray[np.float64] | None = None
    realizations: NDArray[np.float64] | None = None
    quantiles: NDArray[np.float64] | None = None
    quantile_levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        grid = self.metadata.grid
        if grid is None:
            raise ValueError("a FieldArchive requires metadata.grid (a FieldGrid spatial domain)")
        shape = (grid.n_rows, grid.n_cols)
        if self.encoding == "parametric":
            if self.mean is None or self.variance is None:
                raise ValueError("a parametric archive requires both mean and variance arrays")
            if self.mean.shape != shape or self.variance.shape != shape:
                raise ValueError(
                    f"parametric mean/variance must have grid shape {shape}; "
                    f"got mean {self.mean.shape}, variance {self.variance.shape}"
                )
        elif self.encoding == "ensemble":
            if self.realizations is None:
                raise ValueError("an ensemble archive requires a realizations array")
            if self.realizations.ndim != 3 or self.realizations.shape[1:] != shape:
                raise ValueError(
                    f"ensemble realizations must have shape (n, {shape[0]}, {shape[1]}); "
                    f"got {self.realizations.shape}"
                )
        elif self.encoding == "quantile":
            if self.quantiles is None or self.quantile_levels is None:
                raise ValueError("a quantile archive requires quantiles and their levels")
            if self.quantiles.shape != (len(self.quantile_levels), *shape):
                raise ValueError(
                    f"quantiles must have shape ({len(self.quantile_levels)}, {shape[0]}, "
                    f"{shape[1]}); got {self.quantiles.shape}"
                )
            levels = np.asarray(self.quantile_levels, dtype=np.float64)
            if not bool(np.all(np.diff(levels) > 0.0)) or levels[0] <= 0.0 or levels[-1] >= 1.0:
                raise ValueError(
                    "quantile levels must be strictly increasing within (0, 1); "
                    f"got {self.quantile_levels}"
                )
        else:
            raise ValueError(f"unknown encoding {self.encoding!r}; known encodings are {ENCODINGS}")

    # --- constructors: one per encoding ---------------------------------------------------------

    @classmethod
    def parametric(cls, prior: Prior, *, backend: str = "grid") -> FieldArchive:
        """The **parametric** encoding of ``prior``: its per-cell ``mean`` + ``variance``."""
        return cls(
            metadata=prior.metadata,
            provenance=prior.provenance,
            encoding="parametric",
            backend=backend,
            mean=np.asarray(prior.mean, dtype=np.float64),
            variance=np.asarray(prior.variance, dtype=np.float64),
        )

    @classmethod
    def ensemble_encoded(
        cls,
        metadata: FieldMetadata,
        provenance: Provenance,
        realizations: NDArray[np.float64],
        *,
        backend: str = "generative",
    ) -> FieldArchive:
        """The **ensemble** encoding: a stacked ``(n, n_rows, n_cols)`` realization axis.

        The honest encoding for a non-Gaussian field: the realizations *are* the uncertainty, and no
        two moments summarize them without loss — a bimodal "rich lens or barren" belief has a mean
        that nothing believes in.
        """
        return cls(
            metadata=metadata,
            provenance=provenance,
            encoding="ensemble",
            backend=backend,
            realizations=np.asarray(realizations, dtype=np.float64),
        )

    @classmethod
    def quantile_encoded(
        cls,
        metadata: FieldMetadata,
        provenance: Provenance,
        quantiles: NDArray[np.float64],
        levels: Sequence[float],
        *,
        backend: str = "grid",
    ) -> FieldArchive:
        """The **quantile** encoding: a ``(k, n_rows, n_cols)`` axis plus its ``k`` levels."""
        return cls(
            metadata=metadata,
            provenance=provenance,
            encoding="quantile",
            backend=backend,
            quantiles=np.asarray(quantiles, dtype=np.float64),
            quantile_levels=tuple(float(q) for q in levels),
        )

    # --- consumption ----------------------------------------------------------------------------

    def as_prior(self) -> Prior:
        """Collapse the archive into a Gaussian :class:`Prior` — an explicitly lossy read.

        A parametric archive is exact. An **ensemble** yields the empirical per-cell mean/variance,
        discarding the shape — so call it only when a Gaussian summary is genuinely what is wanted.
        A **quantile** archive yields the median as the mean and recovers the variance from the
        interquartile range under a Gaussian assumption, so a quantile-encoded field can still seed
        a belief. That assumption is stated here rather than buried, because it is exactly the kind
        of silent Gaussianization that makes an uncertainty story dishonest.
        """
        if self.encoding == "parametric":
            assert self.mean is not None and self.variance is not None  # __post_init__ checked
            return Prior(self.metadata, self.mean, self.variance, self.provenance)
        if self.encoding == "ensemble":
            assert self.realizations is not None
            return Prior(
                self.metadata,
                self.realizations.mean(axis=0),
                self.realizations.var(axis=0),
                self.provenance,
            )
        assert self.quantiles is not None and self.quantile_levels is not None
        levels = np.asarray(self.quantile_levels, dtype=np.float64)
        lower = int(np.argmin(np.abs(levels - 0.25)))
        upper = int(np.argmin(np.abs(levels - 0.75)))
        median = int(np.argmin(np.abs(levels - 0.5)))
        if lower == upper:
            raise ValueError(
                "a quantile archive needs distinct lower/upper levels to recover a variance; "
                f"got levels {self.quantile_levels}"
            )
        sigma = (self.quantiles[upper] - self.quantiles[lower]) / _IQR_TO_SIGMA
        return Prior(self.metadata, self.quantiles[median], sigma**2, self.provenance)

    def as_field(self) -> GridField:
        """Reopen the archive as a live, queryable :class:`GridField` (through :meth:`as_prior`)."""
        return self.as_prior().as_field()

    def attrs(self) -> dict[str, str]:
        """The self-describing attributes stamped onto the store — never a bare grid of numbers.

        Mirrors the backends' own ``zarr_attrs()`` (RM-P1-PROSPECT-10) and adds the full
        :class:`FieldMetadata` (species / unit / frame / CRS / grid) and the cited
        :class:`~astro_mine.prospect.priors.provenance.Provenance`, so a store carries its own
        georeference and lineage (``LUNAR-TR-001``, ``LUNAR-DR-004``) and no consumer has to be told
        out of band what it is holding.
        """
        return {
            _ATTR_ENCODING: self.encoding,
            _ATTR_BACKEND: self.backend,
            _ATTR_SPECIES: self.metadata.species,
            _ATTR_UNIT: self.metadata.unit,
            _ATTR_METADATA: self.metadata.model_dump_json(),
            _ATTR_PROVENANCE: self.provenance.model_dump_json(),
        }


# --- store IO ------------------------------------------------------------------------------------


def write_zarr(archive: FieldArchive, path: str | Path) -> Path:
    """Write ``archive`` to a Zarr store at ``path`` (created or overwritten); returns the path."""
    zarr = _zarr()
    root = zarr.open_group(store=str(path), mode="w")
    root.attrs.update(archive.attrs())
    if archive.encoding == "parametric":
        assert archive.mean is not None and archive.variance is not None
        _put(root, _MEAN, archive.mean)
        _put(root, _VARIANCE, archive.variance)
    elif archive.encoding == "ensemble":
        assert archive.realizations is not None
        _put(root, _REALIZATION, archive.realizations)
    else:
        assert archive.quantiles is not None and archive.quantile_levels is not None
        _put(root, _QUANTILE, archive.quantiles)
        _put(root, _QUANTILE_LEVEL, np.asarray(archive.quantile_levels, dtype=np.float64))
    return Path(path)


def read_zarr(path: str | Path) -> FieldArchive:
    """Read a Zarr store into a :class:`FieldArchive`, **in the encoding it was written in**."""
    zarr = _zarr()
    root = zarr.open_group(store=str(path), mode="r")
    attrs = dict(root.attrs)
    encoding = attrs.get(_ATTR_ENCODING)
    if encoding not in ENCODINGS:
        raise ValueError(
            f"the Zarr store at {path} declares uncertainty_representation {encoding!r}; "
            f"expected one of {ENCODINGS} (prospect.md §5)"
        )
    metadata = FieldMetadata.model_validate_json(attrs[_ATTR_METADATA])
    provenance = Provenance.model_validate_json(attrs[_ATTR_PROVENANCE])
    backend = str(attrs.get(_ATTR_BACKEND, "grid"))
    if encoding == "parametric":
        return FieldArchive(
            metadata=metadata,
            provenance=provenance,
            encoding="parametric",
            backend=backend,
            mean=_get(root, _MEAN),
            variance=_get(root, _VARIANCE),
        )
    if encoding == "ensemble":
        return FieldArchive.ensemble_encoded(
            metadata, provenance, _get(root, _REALIZATION), backend=backend
        )
    levels = tuple(float(q) for q in _get(root, _QUANTILE_LEVEL))
    return FieldArchive.quantile_encoded(
        metadata, provenance, _get(root, _QUANTILE), levels, backend=backend
    )


def serialize_zarr(archive: FieldArchive) -> bytes:
    """Serialize ``archive`` into the content-addressed :data:`ZARR_MEDIA_TYPE` layer bytes.

    The Zarr store is written to a scratch directory and packed into a **deterministic** tar (sorted
    members, zeroed mtime/uid/gid, USTAR format), so the same archive yields byte-identical layer
    bytes on any machine — hence a reproducible OCI digest (hub.md §2.1; ``LUNAR-DR-004``).
    """
    with TemporaryDirectory() as tmp:
        store = Path(tmp) / "field.zarr"
        write_zarr(archive, store)
        members = {
            p.relative_to(store).as_posix(): p.read_bytes()
            for p in sorted(store.rglob("*"))
            if p.is_file()
        }
    return _deterministic_tar(members)


def archive_from_zarr_bytes(data: bytes) -> FieldArchive:
    """Rebuild a :class:`FieldArchive` from :func:`serialize_zarr` bytes — the inverse."""
    with TemporaryDirectory() as tmp:
        store = Path(tmp) / "field.zarr"
        store.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - always a file here
                    continue
                target = store / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(extracted.read())
        return read_zarr(store)


def quantile_grids(
    prior: Prior, levels: Sequence[float] = DEFAULT_LEVELS
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    """The ``(k, n_rows, n_cols)`` quantile stack of a Gaussian ``prior`` at ``levels``.

    The bridge from the parametric encoding to the compact quantile one. It is monotone in
    ``levels`` by construction — the normal inverse-CDF is — which is the invariant the property
    tests pin.
    """
    ordered = tuple(float(q) for q in levels)
    sigma = np.sqrt(prior.variance)
    stack = np.stack([prior.mean + NormalDist().inv_cdf(q) * sigma for q in ordered], axis=0)
    return np.asarray(stack, dtype=np.float64), ordered


def _put(root: Any, name: str, values: NDArray[np.float64]) -> None:
    """Create a float64 Zarr array ``name`` holding ``values`` (C-contiguous, whole-array chunk)."""
    array = root.create_array(
        name, shape=values.shape, dtype="float64", chunks=values.shape, overwrite=True
    )
    array[...] = np.ascontiguousarray(values, dtype=np.float64)


def _get(root: Any, name: str) -> NDArray[np.float64]:
    """Read a Zarr array by ``name`` into a C-contiguous float64 numpy array."""
    return np.ascontiguousarray(root[name][...], dtype=np.float64)


def _deterministic_tar(members: Mapping[str, bytes]) -> bytes:
    """A reproducible USTAR tar of ``members``: sorted names, zeroed mtime/uid/gid/owner.

    The same canonicalization :mod:`astro_mine.prospect.publish._bundle` applies to the ``.npy``
    bundle, so both layer types are content-addressed under identical rules.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
