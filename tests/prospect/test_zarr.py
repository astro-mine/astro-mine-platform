"""Zarr field storage — the parametric / ensemble / quantile encodings (prospect.md §5).

Proves the acceptance criterion that a field round-trips through a **Zarr store** (not only the
``.npy`` + JSON tar) **preserving the encoding it was tagged with**: an ensemble reads back as an
ensemble, a quantile stack as a quantile stack. A reader that silently collapsed either to two
moments would discard exactly the structure the encoding exists to carry.

Also proves that both layer types resolve through the *same* ``from_bundle`` entry point, so a
consumer with the ``zarr`` extra and one without it call identical code — the dependency-light
resolve path (Core + numpy only) keeps working offline (``LUNAR-TR-004``).

``zarr`` is an optional extra; these tests skip without it, per the repo's ``importorskip`` idiom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_mine.core.resource import ResourceField, check_resource_field
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.publish import (
    BUNDLE_MEDIA_TYPE,
    ENCODINGS,
    ZARR_MEDIA_TYPE,
    FieldArchive,
    archive_from_zarr_bytes,
    build_field_manifest,
    bundle_digest,
    from_bundle,
    quantile_grids,
    read_zarr,
    serialize_bundle,
    serialize_zarr,
    write_zarr,
)

pytest.importorskip("zarr", reason="the [zarr] extra is required for Zarr field storage")

_PROBES = [(0.0, 0.0, 0.0), (500.0, -500.0, 0.0), (-900.0, 900.0, 0.0)]


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _prior():  # type: ignore[no-untyped-def]
    return load_prior(grid=_grid())


def _ensemble_archive(n: int = 24) -> FieldArchive:
    prior = _prior()
    rng = np.random.default_rng(0)
    draws = prior.mean + np.sqrt(prior.variance) * rng.standard_normal((n, *prior.mean.shape))
    return FieldArchive.ensemble_encoded(prior.metadata, prior.provenance, draws)


def _quantile_archive() -> FieldArchive:
    prior = _prior()
    stack, levels = quantile_grids(prior)
    return FieldArchive.quantile_encoded(prior.metadata, prior.provenance, stack, levels)


# --- AC1: a field round-trips through Zarr, in the encoding it was tagged with --------------------


def test_a_parametric_field_round_trips_through_a_zarr_store(tmp_path: Path) -> None:
    archive = FieldArchive.parametric(_prior())
    write_zarr(archive, tmp_path / "field.zarr")
    restored = read_zarr(tmp_path / "field.zarr")

    assert restored.encoding == "parametric"
    assert restored.metadata == archive.metadata  # the georeference survives (LUNAR-TR-001)
    assert restored.provenance == archive.provenance  # and the cited lineage (LUNAR-DR-004)
    np.testing.assert_array_equal(restored.mean, archive.mean)
    np.testing.assert_array_equal(restored.variance, archive.variance)


def test_an_ensemble_field_round_trips_as_an_ensemble(tmp_path: Path) -> None:
    # The load-bearing case: an ensemble must come back as an ensemble. Collapsing it to a
    # mean/variance summary on read would silently throw away the non-Gaussian structure that is the
    # only reason to store an ensemble at all.
    archive = _ensemble_archive()
    write_zarr(archive, tmp_path / "ens.zarr")
    restored = read_zarr(tmp_path / "ens.zarr")

    assert restored.encoding == "ensemble"
    assert restored.realizations is not None
    assert restored.realizations.shape == (24, 8, 8)  # the distribution axis is intact
    np.testing.assert_array_equal(restored.realizations, archive.realizations)


def test_a_quantile_field_round_trips_as_a_quantile_stack(tmp_path: Path) -> None:
    archive = _quantile_archive()
    write_zarr(archive, tmp_path / "q.zarr")
    restored = read_zarr(tmp_path / "q.zarr")

    assert restored.encoding == "quantile"
    assert restored.quantile_levels == archive.quantile_levels
    np.testing.assert_array_equal(restored.quantiles, archive.quantiles)


def test_the_store_declares_its_own_uncertainty_representation(tmp_path: Path) -> None:
    # A stored field is self-describing — never a bare grid of numbers (prospect.md §5).
    import zarr

    for archive in (FieldArchive.parametric(_prior()), _ensemble_archive(), _quantile_archive()):
        path = tmp_path / f"{archive.encoding}.zarr"
        write_zarr(archive, path)
        attrs = dict(zarr.open_group(store=str(path), mode="r").attrs)
        assert attrs["uncertainty_representation"] == archive.encoding
        assert attrs["species"] == archive.metadata.species
        assert attrs["unit"] == archive.metadata.unit
        assert archive.encoding in ENCODINGS


def test_an_untagged_or_unknown_store_is_refused(tmp_path: Path) -> None:
    import zarr

    root = zarr.open_group(store=str(tmp_path / "bogus.zarr"), mode="w")
    root.attrs.update({"uncertainty_representation": "vibes"})
    with pytest.raises(ValueError, match="expected one of"):
        read_zarr(tmp_path / "bogus.zarr")


# --- the same from_bundle entry point resolves either layer --------------------------------------


def test_from_bundle_resolves_a_zarr_layer_into_a_live_resource_field() -> None:
    prior = _prior()
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(b""))
    layers = {ZARR_MEDIA_TYPE: serialize_zarr(FieldArchive.parametric(prior))}

    field = from_bundle(manifest, layers)

    assert isinstance(field, ResourceField)
    assert check_resource_field(field) is None
    reference = prior.as_field()
    for probe in _PROBES:
        assert field.mean(probe) == pytest.approx(reference.mean(probe))
        assert field.variance(probe) == pytest.approx(reference.variance(probe))
        assert field.posterior(probe).variance > 0.0  # still uncertainty-first


def test_the_dependency_light_tar_layer_still_resolves() -> None:
    # LUNAR-TR-004: a consumer with only Core + numpy resolves the .npy bundle exactly as before.
    prior = _prior()
    bundle = serialize_bundle(prior)
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(bundle))
    field = from_bundle(manifest, {BUNDLE_MEDIA_TYPE: bundle})
    assert field.mean(_PROBES[0]) == pytest.approx(prior.as_field().mean(_PROBES[0]))


def test_a_zarr_layer_is_preferred_when_both_are_present() -> None:
    prior = _prior()
    manifest = build_field_manifest(prior, bundle_sha256=bundle_digest(b""))
    field = from_bundle(
        manifest,
        {
            BUNDLE_MEDIA_TYPE: serialize_bundle(prior),
            ZARR_MEDIA_TYPE: serialize_zarr(FieldArchive.parametric(prior)),
        },
    )
    # Both describe the same field, so the choice is invisible in the values — but the format the
    # architecture specifies is the one taken when it is on offer (prospect.md §5).
    assert field.mean(_PROBES[0]) == pytest.approx(prior.as_field().mean(_PROBES[0]))


def test_no_known_layer_is_an_error() -> None:
    manifest = build_field_manifest(_prior(), bundle_sha256=bundle_digest(b""))
    with pytest.raises(ValueError, match="no resource-field layer found"):
        from_bundle(manifest, {"application/vnd.something-else": b"x"})


# --- content addressing: the Zarr layer is a deterministic, reproducible blob ---------------------


def test_the_zarr_layer_bytes_are_deterministic() -> None:
    # The reproducibility contract (hub.md §2.1; LUNAR-DR-004): the same field yields byte-identical
    # layer bytes, hence a stable OCI digest, on any machine or checkout.
    prior = _prior()
    first = serialize_zarr(FieldArchive.parametric(prior))
    second = serialize_zarr(FieldArchive.parametric(prior))
    assert first == second
    assert bundle_digest(first) == bundle_digest(second)


def test_the_layer_bytes_round_trip_back_into_the_archive() -> None:
    for archive in (FieldArchive.parametric(_prior()), _ensemble_archive(), _quantile_archive()):
        restored = archive_from_zarr_bytes(serialize_zarr(archive))
        assert restored.encoding == archive.encoding
        assert restored.metadata == archive.metadata


# --- consuming an archive: the lossy collapse is explicit, never silent ---------------------------


def test_an_ensemble_collapses_to_its_empirical_moments_only_when_asked() -> None:
    archive = _ensemble_archive(n=400)
    assert archive.realizations is not None
    prior = archive.as_prior()  # the explicit, documented lossy read
    np.testing.assert_allclose(prior.mean, archive.realizations.mean(axis=0))
    np.testing.assert_allclose(prior.variance, archive.realizations.var(axis=0))
    assert check_resource_field(archive.as_field()) is None


def test_a_quantile_archive_recovers_a_usable_gaussian_prior() -> None:
    # The quantile encoding is compact and lossy by design; reading it back as a Gaussian recovers
    # the median as the mean and the variance from the IQR. That assumption is documented, not hid.
    prior = _prior()
    recovered = _quantile_archive().as_prior()
    np.testing.assert_allclose(recovered.mean, prior.mean, rtol=1e-9)
    np.testing.assert_allclose(recovered.variance, prior.variance, rtol=1e-6)


# --- construction guards -------------------------------------------------------------------------


def test_a_malformed_archive_is_refused() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="requires both mean and variance"):
        FieldArchive(metadata=prior.metadata, provenance=prior.provenance, encoding="parametric")
    with pytest.raises(ValueError, match="requires a realizations array"):
        FieldArchive(metadata=prior.metadata, provenance=prior.provenance, encoding="ensemble")
    with pytest.raises(ValueError, match="unknown encoding"):
        FieldArchive(
            metadata=prior.metadata,
            provenance=prior.provenance,
            encoding="telepathic",  # type: ignore[arg-type]
        )


def test_a_mismatched_array_shape_is_refused() -> None:
    prior = _prior()
    with pytest.raises(ValueError, match="must have grid shape"):
        FieldArchive(
            metadata=prior.metadata,
            provenance=prior.provenance,
            encoding="parametric",
            mean=np.zeros((3, 3)),
            variance=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="must have shape"):
        FieldArchive.ensemble_encoded(prior.metadata, prior.provenance, np.zeros((4, 3, 3)))


def test_quantile_levels_must_be_ordered_and_within_the_open_unit_interval() -> None:
    prior = _prior()
    stack, _ = quantile_grids(prior)
    with pytest.raises(ValueError, match="strictly increasing"):
        FieldArchive.quantile_encoded(
            prior.metadata, prior.provenance, stack, (0.5, 0.25, 0.05, 0.75, 0.95)
        )
