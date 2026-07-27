"""Illumination/PSR regression against the published LOLA reference (worlds.md §10; issue #41).

Two tiers, because the real inputs cannot exist in CI:

- **Offline (runs everywhere, including CI).** The committed reference document and the committed
  result artifact are checked against each other: the recorded run must be inside its own stated
  error budget, and must have been produced with the harness configuration the reference declares
  it is comparable under. This is what stops the committed evidence from silently rotting into a
  claim nobody can check — and it exercises the same :func:`validate_psr` comparison the real run
  uses, against a synthetic mask.
- **Real data (``@pytest.mark.realdata``, skipped by default).** Re-runs the actual comparison:
  builds the illumination model on the **real LOLA-derived terrain product** with the **real NAIF
  kernels** and fails if the error exceeds the budget. The 3.7 GB DEM and the NAIF kernel set are
  neither in the repo nor fetched by CI (which runs offline against the synthetic fixture in
  ``conftest.py``), so it skips unless ``ASTRO_MINE_WORLDS_REAL_TERRAIN`` and
  ``ASTRO_MINE_WORLDS_METAKERNEL`` point at them::

      ASTRO_MINE_WORLDS_REAL_TERRAIN=files/data/shackleton-0.3.0/terrain \\
      ASTRO_MINE_WORLDS_METAKERNEL=files/data/spice/metakernel.tm \\
      uv run pytest -m realdata

Default CI stays green and offline: the marked test skips, and nothing else here touches the
network or a multi-gigabyte raster.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.worlds.illumination import (
    VALIDATION_SCHEMA,
    HorizonFrame,
    IlluminationModel,
    PsrEpochSemantics,
    PsrReference,
    PsrResult,
    psr_statistics,
    validate_psr,
)

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"
REFERENCE_PATH = VALIDATION_DIR / "shackleton_psr.reference.json"
RESULT_PATH = VALIDATION_DIR / "shackleton_psr.result.json"

_SECONDS_PER_DAY = 86_400.0
#: The harness keys the committed run must have used — the ones that actually determine the mask.
_HARNESS_KEYS = (
    "resolution_m",
    "n_azimuth",
    "max_radius_m",
    "abcorr",
    "horizon_frame",
    "semantics",
    "start",
    "duration_days",
    "step_hours",
)


@pytest.fixture(scope="module")
def reference() -> PsrReference:
    return PsrReference.load(REFERENCE_PATH)


@pytest.fixture(scope="module")
def committed_result() -> dict:
    doc: dict = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return doc


# --- the committed reference + artifact (offline; runs in CI) -----------------------


def test_reference_document_is_well_formed(reference: PsrReference) -> None:
    assert reference.region == "shackleton-de-gerlache"
    assert 0.0 < reference.psr_area_fraction < 1.0
    assert reference.tolerance_area_fraction > 0.0
    assert "LOLA" in reference.source and "Mazarico" in reference.source
    # Every mask-determining harness parameter is pinned: a PSR fraction without the window that
    # defines "permanent" is not a comparable number (worlds.md §11's PSR-epoch open question).
    for key in _HARNESS_KEYS:
        assert key in reference.harness, key


def test_committed_result_is_inside_the_stated_error_budget(
    reference: PsrReference, committed_result: dict
) -> None:
    """The committed artifact is evidence of a real run — assert it actually passes its budget."""
    assert committed_result["schema"] == VALIDATION_SCHEMA
    assert committed_result["region"] == reference.region
    comparison = committed_result["comparison"]
    result = committed_result["result"]

    assert comparison["passed"] is True
    assert comparison["error_area_fraction"] <= reference.tolerance_area_fraction
    assert comparison["error_area_fraction"] == pytest.approx(
        abs(result["psr_area_fraction"] - reference.psr_area_fraction), abs=1e-9
    )
    # It was run on the REAL DEM, not the CI stand-in: the anchor grid is 1264x1264 at 120 m.
    assert result["n_cells"] == 1264 * 1264
    assert result["resolution_m"] == 120.0
    assert result["illumination_hash"].startswith("sha256:")
    assert result["psr_hash"].startswith("sha256:")


def test_committed_result_used_the_reference_harness(
    reference: PsrReference, committed_result: dict
) -> None:
    """A result graded against the reference must have been produced under its configuration."""
    for key in _HARNESS_KEYS:
        assert committed_result["harness"][key] == reference.harness[key], key
    expected_epochs = int(reference.harness["duration_days"] * 24 / reference.harness["step_hours"])
    assert committed_result["result"]["n_epochs"] == pytest.approx(expected_epochs, abs=1)
    assert committed_result["result"]["semantics"] == reference.harness["semantics"]


# --- the comparison kernel (offline) ------------------------------------------------


def _synthetic_psr(fraction: float, *, height: int = 10, width: int = 10) -> PsrResult:
    mask = np.zeros((height, width), dtype=np.bool_)
    mask.reshape(-1)[: round(fraction * height * width)] = True
    window = EpochWindow(
        start=Epoch(tdb_seconds=0.0, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=_SECONDS_PER_DAY, scale=TimeScale.TDB),
    )
    return PsrResult(
        mask=mask,
        ever_lit_fraction=1.0 - fraction,
        void_mask=np.zeros((height, width), dtype=np.bool_),
        window=window,
        step_s=3600.0,
        n_epochs=24,
        semantics=PsrEpochSemantics.SEASONAL,
        illumination_hash="sha256:test",
    )


def test_psr_statistics_reports_fraction_and_area() -> None:
    stats = psr_statistics(_synthetic_psr(0.25), resolution_m=100.0)
    assert stats["psr_area_fraction"] == pytest.approx(0.25)
    assert stats["n_psr_cells"] == 25
    assert stats["n_cells"] == 100
    assert stats["psr_area_km2"] == pytest.approx(25 * 100.0 * 100.0 / 1e6)
    assert stats["n_void_cells"] == 0


def test_validate_psr_passes_inside_and_fails_outside_the_budget(reference: PsrReference) -> None:
    inside = validate_psr(
        _synthetic_psr(reference.psr_area_fraction), reference, resolution_m=120.0
    )
    assert inside.passed
    assert inside.error_area_fraction == pytest.approx(0.0, abs=0.01)

    outside = validate_psr(_synthetic_psr(0.9), reference, resolution_m=120.0)
    assert not outside.passed
    assert outside.error_area_fraction > reference.tolerance_area_fraction
    artifact = outside.to_artifact()
    assert artifact["comparison"]["passed"] is False
    assert artifact["schema"] == VALIDATION_SCHEMA


def test_reference_rejects_a_foreign_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.reference.json"
    bad.write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        PsrReference.load(bad)


# --- the real-data regression (marker-gated; skipped in offline CI) ------------------


@pytest.mark.realdata
@pytest.mark.skipif(
    not (
        os.environ.get("ASTRO_MINE_WORLDS_REAL_TERRAIN")
        and os.environ.get("ASTRO_MINE_WORLDS_METAKERNEL")
    ),
    reason=(
        "needs the real LOLA terrain product + NAIF kernels; set "
        "ASTRO_MINE_WORLDS_REAL_TERRAIN and ASTRO_MINE_WORLDS_METAKERNEL "
        "(neither is in the repo or in offline CI)"
    ),
)
def test_real_lola_psr_is_within_the_published_error_budget(reference: PsrReference) -> None:
    """Re-run the published-reference comparison on the real DEM; fail if it exceeds the budget."""
    from astro_mine.spice import epoch_from_utc, kernel_pool
    from astro_mine.worlds.terrain import TerrainModel

    terrain = TerrainModel.open(os.environ["ASTRO_MINE_WORLDS_REAL_TERRAIN"])
    harness = reference.harness
    resolution_m = float(terrain.manifest["grid"]["resolution_m"])
    assert resolution_m == pytest.approx(float(harness["resolution_m"]))

    with kernel_pool(os.environ["ASTRO_MINE_WORLDS_METAKERNEL"]):
        model = IlluminationModel(
            terrain,
            n_azimuth=int(harness["n_azimuth"]),
            max_radius_m=float(harness["max_radius_m"]),
            abcorr=str(harness["abcorr"]),
            horizon_frame=HorizonFrame(harness["horizon_frame"]),
            # Optional, and the difference between a ~90-minute run and an ~18-second one: point it
            # at a published bundle's illumination/horizon.zarr to adopt the skyline instead of
            # re-deriving it (issue #46). Not a shortcut — the store is validated against the
            # parameters resolved above and rejected if they disagree. Unset, the test recomputes.
            horizon_store=os.environ.get("ASTRO_MINE_WORLDS_HORIZON_STORE"),
        )
        start = epoch_from_utc(str(harness["start"]))
        end = Epoch(
            tdb_seconds=start.tdb_seconds + float(harness["duration_days"]) * _SECONDS_PER_DAY,
            scale=TimeScale.TDB,
        )
        psr = model.psr_mask(
            EpochWindow(start=start, end=end),
            float(harness["step_hours"]) * 3600.0,
            semantics=PsrEpochSemantics(harness["semantics"]),
        )

    validation = validate_psr(psr, reference, resolution_m=resolution_m)
    assert validation.passed, (
        f"PSR area fraction {validation.psr_area_fraction:.4f} is "
        f"{validation.error_area_fraction:.4f} from the published reference "
        f"{reference.psr_area_fraction:.4f}, outside the "
        f"{reference.tolerance_area_fraction:.4f} budget"
    )
    # ...and it reproduces the committed artifact exactly — the run is deterministic, so a change
    # that silently moves the PSR mask is caught even when it stays inside the budget.
    committed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))["result"]
    assert validation.n_psr_cells == committed["n_psr_cells"]
    assert validation.psr_area_fraction == pytest.approx(committed["psr_area_fraction"], abs=1e-9)
    # The illumination_hash is asserted too, which it could not be while the hash folded in the
    # toolchain — a benign dependency bump used to fail this for no physical reason (issue #46).
    # Now it is a hash of the skyline and the parameters that determine it, so it pins the physics
    # and nothing else: this is the same digest a consumer gets from the published bundle.
    assert validation.illumination_hash == committed["illumination_hash"]
