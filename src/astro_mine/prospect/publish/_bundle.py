"""The content-addressed resource-field bundle: serialize a prior, reopen a live field.

The wire form of a published belief prior (RM-P1-PROSPECT-13): a **deterministic** tar carrying the
per-cell ``mean``/``variance`` arrays, the
:class:`~astro_mine.prospect.field.metadata.FieldMetadata` (species/unit/CRS/grid), and the cited
:class:`~astro_mine.prospect.priors.provenance.Provenance` — everything needed to rebuild the
field, and *only* the public belief prior. The sealed
:class:`~astro_mine.prospect.belief.ground_truth.GroundTruthField` is **never** serialized here
(``RM-P0-PROSPECT-05`` — a leak is a security-class defect); this module only ever sees a
:class:`~astro_mine.prospect.priors.recipe.Prior`.

The tar is canonical — members in sorted order, ``mtime``/``uid``/``gid`` zeroed, USTAR format — so
the same prior yields byte-identical bundle bytes on any machine or checkout, hence a stable OCI
layer digest (the reproducibility contract, hub.md §2.1).

:func:`from_bundle` is the **entry-point factory** (``astro_mine.providers`` →
``resource_field_backend``): a consumer resolves a field through Hub + the Core manifest and calls
this to rebuild a live :class:`~astro_mine.prospect.backends.grid.GridField` from the pulled bytes,
**without re-running the recipe** and **without importing** :mod:`astro_mine.prospect` by name — so
Sim/Bench never take a Prospect import to open a field (prospect.md §3; conventions.md §1.1).

Backlog: RM-P1-PROSPECT-13 — astro-mine-prospect#23
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.registry import PluginManifest
from astro_mine.core.resource import ResourceField
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.field.metadata import FieldMetadata
from astro_mine.prospect.priors.recipe import Prior

# The Zarr layer's media type only — importing it does NOT import `zarr` (that module resolves the
# optional dependency lazily), so `from_bundle` stays a Core + numpy path (LUNAR-TR-004).
from astro_mine.prospect.publish._zarr import ZARR_MEDIA_TYPE

__all__ = [
    "BUNDLE_MEDIA_TYPE",
    "MEAN_MEMBER",
    "METADATA_MEMBER",
    "PROVENANCE_MEMBER",
    "PROVIDER_ENTRY_POINT",
    "RESOURCE_FIELD_INTERFACE",
    "RESOURCE_FIELD_INTERFACE_VERSION",
    "VARIANCE_MEMBER",
    "bundle_digest",
    "from_bundle",
    "prior_from_bundle",
    "serialize_bundle",
]

#: The single OCI layer's media type — the resource-field bundle tar (hub.md §3). A consumer keys
#: the pulled ``layers`` mapping by media type and looks this one up.
BUNDLE_MEDIA_TYPE = "application/vnd.astro-mine.resource-field.bundle.v1.tar"

#: The Core interface the published backend implements, and the version it is built against — the
#: input to Core's load-time version negotiation (the manifest's ``core_interfaces`` entry).
RESOURCE_FIELD_INTERFACE = "resource_field"
RESOURCE_FIELD_INTERFACE_VERSION = "0.1.0"

#: The ``astro_mine.providers`` entry-point name the factory registers under, so a consumer resolves
#: it by name without importing this package (the narrow-waist decoupling, conventions.md §1.1).
PROVIDER_ENTRY_POINT = "resource_field_backend"

#: The tar member names (fixed — the bundle schema v1).
MEAN_MEMBER = "mean.npy"
VARIANCE_MEMBER = "variance.npy"
METADATA_MEMBER = "metadata.json"
PROVENANCE_MEMBER = "provenance.json"

_REQUIRED_MEMBERS = frozenset({MEAN_MEMBER, VARIANCE_MEMBER, METADATA_MEMBER, PROVENANCE_MEMBER})


def serialize_bundle(prior: Prior) -> bytes:
    """Serialize *prior* to the deterministic resource-field bundle tar (the public belief prior).

    Emits exactly the four members of the bundle schema — ``mean.npy``, ``variance.npy``,
    ``metadata.json``, ``provenance.json`` — from the prior's public arrays and metadata. It reads
    only :class:`~astro_mine.prospect.priors.recipe.Prior`; the sealed ground-truth realization has
    no path into this function (``RM-P0-PROSPECT-05``). The bytes are byte-stable across machines
    and checkouts, so the OCI layer digest is reproducible.
    """
    members = {
        MEAN_MEMBER: _npy_bytes(prior.mean),
        VARIANCE_MEMBER: _npy_bytes(prior.variance),
        METADATA_MEMBER: _canonical_json(prior.metadata.model_dump(mode="json")),
        PROVENANCE_MEMBER: _canonical_json(prior.provenance.model_dump(mode="json")),
    }
    return _deterministic_tar(members)


def from_bundle(manifest: PluginManifest, layers: Mapping[str, bytes]) -> ResourceField:
    """Rebuild a live :class:`ResourceField` from a pulled bundle — the entry-point factory.

    ``layers`` is the pulled OCI layer set keyed by media type. **Either** field encoding resolves
    through this one entry point (prospect.md §5):

    - a :data:`~astro_mine.prospect.publish._zarr.ZARR_MEDIA_TYPE` layer — the Zarr store, in
      whichever of the parametric / ensemble / quantile encodings it was tagged with — is preferred
      when present, since it is the format the architecture specifies;
    - a :data:`BUNDLE_MEDIA_TYPE` layer — the dependency-light ``.npy`` + JSON tar — otherwise, so
      the offline local tier keeps resolving a field with only Core + numpy installed
      (``LUNAR-TR-004``: no Zarr, no network, no account).

    Either way the field is rebuilt **without re-running the recipe** into a
    :class:`~astro_mine.prospect.backends.grid.GridField`. ``manifest`` is accepted for the Core
    factory contract (a consumer has already version-negotiated and verified it); the field content
    lives entirely in the layers. Raises :class:`ValueError` if no known layer, or any required
    member, is missing.
    """
    if ZARR_MEDIA_TYPE in layers:
        from astro_mine.prospect.publish._zarr import archive_from_zarr_bytes

        return archive_from_zarr_bytes(layers[ZARR_MEDIA_TYPE]).as_field()
    try:
        tar_bytes = layers[BUNDLE_MEDIA_TYPE]
    except KeyError:
        available = ", ".join(sorted(layers)) or "(none)"
        raise ValueError(
            f"no resource-field layer found in the pulled layers (expected "
            f"{BUNDLE_MEDIA_TYPE!r} or {ZARR_MEDIA_TYPE!r}); available media types: {available}"
        ) from None
    members = _read_tar(tar_bytes)
    missing = _REQUIRED_MEMBERS - members.keys()
    if missing:
        raise ValueError(f"resource-field bundle is missing required members: {sorted(missing)}")
    metadata = FieldMetadata.model_validate_json(members[METADATA_MEMBER])
    mean = _load_npy(members[MEAN_MEMBER])
    variance = _load_npy(members[VARIANCE_MEMBER])
    return GridField(metadata, mean, variance)


def prior_from_bundle(bundle: bytes) -> Prior:
    """Rebuild the :class:`Prior` (mean/variance + metadata + cited provenance) from bundle bytes.

    The inverse of :func:`serialize_bundle`: unpacks the four members and reconstructs a live
    :class:`~astro_mine.prospect.priors.recipe.Prior`, carrying its provenance — so the content
    address round-trips (the rebuilt prior's ``content_hash`` equals the original's). Unlike
    :func:`from_bundle` (which yields a queryable :class:`GridField`), this recovers the prior
    itself, so the distributed field service can reseed a ``BeliefField`` on a client and replay the
    streamed observation log against it (RM-P1-PROSPECT-11).
    """
    from astro_mine.prospect.priors.provenance import Provenance

    members = _read_tar(bundle)
    missing = _REQUIRED_MEMBERS - members.keys()
    if missing:
        raise ValueError(f"resource-field bundle is missing required members: {sorted(missing)}")
    metadata = FieldMetadata.model_validate_json(members[METADATA_MEMBER])
    provenance = Provenance.model_validate_json(members[PROVENANCE_MEMBER])
    mean = _load_npy(members[MEAN_MEMBER])
    variance = _load_npy(members[VARIANCE_MEMBER])
    return Prior(metadata, mean, variance, provenance)


def bundle_digest(data: bytes) -> str:
    """The ``sha256:<hex>`` content address of a bundle's bytes (matches the OCI layer digest)."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# --- serialization helpers -------------------------------------------------------------------


def _npy_bytes(arr: NDArray[np.float64]) -> bytes:
    """A ``.npy`` byte string of *arr* — deterministic (C-contiguous ``<f8``, no pickle)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.float64), allow_pickle=False)
    return buf.getvalue()


def _canonical_json(obj: Any) -> bytes:
    """Canonical UTF-8 JSON — sorted keys, no whitespace — so the bytes are order-stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _deterministic_tar(members: Mapping[str, bytes]) -> bytes:
    """A reproducible USTAR tar of *members*: sorted names, zeroed mtime/uid/gid/owner."""
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


def _read_tar(data: bytes) -> dict[str, bytes]:
    """Unpack a bundle tar to a ``name -> bytes`` mapping (regular files only)."""
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is not None:  # always true for a regular file (guards mypy's Optional)
                out[member.name] = extracted.read()
    return out


def _load_npy(data: bytes) -> NDArray[np.float64]:
    """Load a ``.npy`` byte string into a C-contiguous ``float64`` array (never unpickling)."""
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    return np.ascontiguousarray(arr, dtype=np.float64)
